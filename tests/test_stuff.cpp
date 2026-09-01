

#include <iostream>
#include <mpi.h>
#include <random>

#include "Kokkos_Clamp.hpp"
#include "terra/fe/wedge/integrands.hpp"
// clang-format off
#include "terra/grid/grid_types.hpp"
#include "terra/fe/wedge/kernel_helpers.hpp"
// clang-format on
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/vector_q1.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::Grid3DDataScalar;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;

using terra::fe::wedge::num_nodes_per_wedge;
using terra::fe::wedge::num_nodes_per_wedge_surface;
using terra::fe::wedge::num_wedges_per_hex_cell;

using terra::fe::wedge::atomically_add_local_wedge_scalar_coefficients;
using terra::fe::wedge::forward_map;
using terra::fe::wedge::wedge_surface_physical_coords;

using uint_t  = unsigned int;
using ScalarT = double;

using Vec3 = dense::Vec< double, 3 >;

KOKKOS_INLINE_FUNCTION
double tripleProduct( const Vec3& a, const Vec3& b, const Vec3& c )
{
    // return dot(a, cross(b, c));
    return a.dot( b.cross( c ) );
}

// Barycentric coords (l0,l1,l2), l0+l1+l2=1, s.t. l0*A+l1*B+l2*C is parallel to P.
// Works for any unit vectors A,B,C,P -- exact, closed-form, O(1).
KOKKOS_INLINE_FUNCTION
Vec3 gnomonicBarycentric( const Vec3& A, const Vec3& B, const Vec3& C, const Vec3& P )
{
    double l0  = tripleProduct( P, B, C );
    double l1  = tripleProduct( A, P, C );
    double l2  = tripleProduct( A, B, P );
    double sum = l0 + l1 + l2;
    return Vec3{ l0 / sum, l1 / sum, l2 / sum };
}

KOKKOS_INLINE_FUNCTION
bool diamondPointToUV(
    const Vec3& V00,
    const Vec3& V10,
    const Vec3& V01,
    const Vec3& V11,
    Vec3        P,
    double&     u,
    double&     v )
{
    P.normalize();

    // Which side of the diagonal (V00-V11) is P on?
    Vec3   diag_normal = V00.cross( V11 );
    double side        = diag_normal.dot( P );
    double ref         = diag_normal.dot( V10 ); // V10 defines the "lower" side

    if ( side * ref >= 0.0 )
    {
        // Lower-right triangle: (V00, V10, V11)
        auto l = gnomonicBarycentric( V00, V10, V11, P );
        u      = l( 1 ) + l( 2 );
        v      = l( 2 );
    }
    else
    {
        // Upper-left triangle: (V00, V11, V01)
        auto l = gnomonicBarycentric( V00, V11, V01, P );
        u      = l( 1 );
        v      = l( 1 ) + l( 2 );
    }
    return ( u >= -1e-9 && u <= 1 + 1e-9 && v >= -1e-9 && v <= 1 + 1e-9 );
}

KOKKOS_INLINE_FUNCTION
dense::Vec< ScalarT, 3 > sphericalDiamondCentroid( const dense::Vec< ScalarT, 3 > vertices[4] )
{
    dense::Vec< ScalarT, 3 > c;

    for ( int i_v = 0; i_v < 4; i_v++ )
    {
        auto v = vertices[i_v];

        c( 0 ) += v( 0 );
        c( 1 ) += v( 1 );
        c( 2 ) += v( 2 );
    }

    c( 0 ) /= 4.0;
    c( 1 ) /= 4.0;
    c( 2 ) /= 4.0;

    c.normalize();

    return c;
}

KOKKOS_INLINE_FUNCTION
bool isPointInSphericalPolygon(
    const dense::Vec< ScalarT, 3 > vertices[4],
    const dense::Vec< ScalarT, 3 > interior_point,
    dense::Vec< ScalarT, 3 >       point,
    double                         tol = 1e-9 )
{
    point.normalize();

    for ( size_t i = 0; i < 4; ++i )
    {
        const dense::Vec< ScalarT, 3 >& A      = vertices[i];
        const dense::Vec< ScalarT, 3 >& B      = vertices[( i + 1 ) % 4];
        dense::Vec< ScalarT, 3 >        normal = A.cross( B );

        double ref_side = normal.dot( interior_point );
        double pt_side  = normal.dot( point );

        // Point must be on the same side as the interior reference point
        // (with a small tolerance for points essentially on the edge)
        if ( ref_side > 0.0 && pt_side < -tol )
            return false;
        if ( ref_side < 0.0 && pt_side > tol )
            return false;
    }
    return true;
}

template < typename ScalarT = double >
struct TestCellInterpolator
{
    uint_t num_subdomains_;
    uint_t num_nodes_per_side_laterally_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    grid::Grid4DDataScalar< ScalarT > data_;
    grid::Grid4DDataScalar< ScalarT > data_sub_;

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        dense::Vec< ScalarT, 3 > point;

        point( 0 ) = grid_( local_subdomain_id, x_cell, y_cell, 0 );
        point( 1 ) = grid_( local_subdomain_id, x_cell, y_cell, 1 );
        point( 2 ) = grid_( local_subdomain_id, x_cell, y_cell, 2 );

        data_( local_subdomain_id, x_cell, y_cell, r_cell ) = -1;

        dense::Vec< ScalarT, 3 > subdomain_extent[4];

        for ( int i = 0; i < num_subdomains_; ++i )
        {
            subdomain_extent[0]( 0 ) = grid_( i, 0, 0, 0 );
            subdomain_extent[0]( 1 ) = grid_( i, 0, 0, 1 );
            subdomain_extent[0]( 2 ) = grid_( i, 0, 0, 2 );

            subdomain_extent[1]( 0 ) = grid_( i, num_nodes_per_side_laterally_ - 1, 0, 0 );
            subdomain_extent[1]( 1 ) = grid_( i, num_nodes_per_side_laterally_ - 1, 0, 1 );
            subdomain_extent[1]( 2 ) = grid_( i, num_nodes_per_side_laterally_ - 1, 0, 2 );

            subdomain_extent[2]( 0 ) =
                grid_( i, num_nodes_per_side_laterally_ - 1, num_nodes_per_side_laterally_ - 1, 0 );
            subdomain_extent[2]( 1 ) =
                grid_( i, num_nodes_per_side_laterally_ - 1, num_nodes_per_side_laterally_ - 1, 1 );
            subdomain_extent[2]( 2 ) =
                grid_( i, num_nodes_per_side_laterally_ - 1, num_nodes_per_side_laterally_ - 1, 2 );

            subdomain_extent[3]( 0 ) = grid_( i, 0, num_nodes_per_side_laterally_ - 1, 0 );
            subdomain_extent[3]( 1 ) = grid_( i, 0, num_nodes_per_side_laterally_ - 1, 1 );
            subdomain_extent[3]( 2 ) = grid_( i, 0, num_nodes_per_side_laterally_ - 1, 2 );

            auto subdomain_centroid = sphericalDiamondCentroid( subdomain_extent );

            if ( isPointInSphericalPolygon( subdomain_extent, subdomain_centroid, point ) )
            {
                data_( local_subdomain_id, x_cell, y_cell, r_cell ) = i;
                break;
            }
        }

        dense::Vec< ScalarT, 3 > V00 = subdomain_extent[0];
        dense::Vec< ScalarT, 3 > V10 = subdomain_extent[1];
        dense::Vec< ScalarT, 3 > V01 = subdomain_extent[3];
        dense::Vec< ScalarT, 3 > V11 = subdomain_extent[2];

        double u{}, v{};

        diamondPointToUV( V00, V10, V01, V11, point, u, v );

        // int u_i = static_cast< int >( Kokkos::floor( u * ( num_nodes_per_side_laterally_ - 1 ) ) );
        // int v_i = static_cast< int >( Kokkos::floor( v * ( num_nodes_per_side_laterally_ - 1 ) ) );

        // u_i = Kokkos::clamp< int >( u_i, 0, num_nodes_per_side_laterally_ - 2 );
        // v_i = Kokkos::clamp< int >( v_i, 0, num_nodes_per_side_laterally_ - 2 );

        // int u_i_plus_1 = u_i + 1;
        // int v_i_plus_1 = v_i + 1;

        // data_sub_( local_subdomain_id, x_cell, y_cell, r_cell ) = local_subdomain_id;

        // Gather surface points for each wedge.
        // dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        // wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        // const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // ScalarT r_val =
        //     forward_map( wedge_phy_surf[0][0], wedge_phy_surf[0][1], wedge_phy_surf[0][1], r_1, r_2, 0.0, 0.0, -1.0 )
        //         .norm();

        // dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        // for ( int i = 0; i < num_nodes_per_wedge; i++ )
        // {
        //     dst[0]( i ) = r_val;
        //     dst[1]( i ) = r_val;
        // }

        // {
        //     Kokkos::atomic_max( &data_( local_subdomain_id, x_cell, y_cell, r_cell ), r_val );

        //     Kokkos::atomic_max( &data_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), r_val );
        //     Kokkos::atomic_max( &data_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), r_val );
        //     Kokkos::atomic_max( &data_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), r_val );

        //     Kokkos::atomic_max( &data_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell ), r_val );
        //     Kokkos::atomic_max( &data_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), r_val );
        //     Kokkos::atomic_max( &data_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), r_val );

        //     Kokkos::atomic_max( &data_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 ), r_val );
        // }

        // {
        //     atomically_add_local_wedge_scalar_coefficients( data_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
        // }
    }
};

template < typename CoordGridType, typename UVGridType >
void buildDiamondLookupTable(
    const CoordGridType& points,
    const Vec3&          V00,
    const Vec3&          V10,
    const Vec3&          V01,
    const Vec3&          V11,
    UVGridType&          u_grid,
    UVGridType&          v_grid,
    int                  N )
{
    for ( int i = 0; i <= N; ++i )
    {
        for ( int j = 0; j <= N; ++j )
        {
            double u{}, v{};

            Vec3 point{ points( 0, i, j, 0 ), points( 0, i, j, 1 ), points( 0, i, j, 2 ) };

            diamondPointToUV( V00, V10, V01, V11, point, u, v );
            u_grid( 0, i, j ) = u;
            v_grid( 0, i, j ) = v;
        }
    }
}

template < typename Vec3 = dense::Vec< double, 3 > >
double geodesicDistance( const Vec3& p, const Vec3& q )
{
    Vec3   c          = p.cross( q );
    double cross_norm = std::sqrt( c.dot( c ) );
    double dot_val    = p.dot( q );
    return std::atan2( cross_norm, dot_val );
}

template < typename CoordsGridType, typename UVGridType >
std::tuple< int, int, int, int > queryDiamondLookup(
    const CoordsGridType& coords_host,
    const UVGridType&     u_grid,
    const UVGridType&     v_grid,
    const Vec3&           V00,
    const Vec3&           V10,
    const Vec3&           V01,
    const Vec3&           V11,
    const Vec3            P,
    int                   N,
    int                   refine_radius = 2 )
{
    double u{}, v{};
    bool   conversion = diamondPointToUV( V00, V10, V01, V11, P, u, v );

    int seed_i = std::clamp( (int) std::floor( u * N ), 0, N );
    int seed_j = std::clamp( (int) std::floor( v * N ), 0, N );

    int    best_i = seed_i, best_j = seed_j;
    double du0 = u_grid( 0, seed_i, seed_j ) - u, dv0 = v_grid( 0, seed_i, seed_j ) - v;
    double best_d2 = du0 * du0 + dv0 * dv0;

    if ( !conversion )
    {
        Kokkos::abort( "Conversion failed" );
    }

    // std::cout << "u = " << u << std::endl;
    // std::cout << "v = " << v << std::endl;

    // std::cout << "best_i = " << best_i << std::endl;
    // std::cout << "best_j = " << best_j << std::endl;
    // std::cout << "best_d2 = " << best_d2 << std::endl;

    for ( int di = -refine_radius; di <= refine_radius; ++di )
    {
        for ( int dj = -refine_radius; dj <= refine_radius; ++dj )
        {
            int ci = seed_i + di, cj = seed_j + dj;
            if ( ci < 0 || ci > N || cj < 0 || cj > N )
                continue;
            double du = u_grid( 0, ci, cj ) - u;
            double dv = v_grid( 0, ci, cj ) - v;
            double d2 = du * du + dv * dv;
            if ( d2 < best_d2 )
            {
                best_d2 = d2;
                best_i  = ci;
                best_j  = cj;
            }
        }
    }

    int bound_i{}, bound_i_plus_1{}, bound_j{}, bound_j_plus_1{};

    {
        for ( int i = best_i - 1; i <= best_i; ++i )
        {
            for ( int j = best_j - 1; j <= best_j; ++j )
            {
                if ( i < 0 || i > N - 1 || j < 0 || j > N - 1 )
                    continue;

                Vec3 point00{ coords_host( 0, i, j, 0 ), coords_host( 0, i, j, 1 ), coords_host( 0, i, j, 2 ) };

                Vec3 point10{
                    coords_host( 0, i + 1, j, 0 ), coords_host( 0, i + 1, j, 1 ), coords_host( 0, i + 1, j, 2 ) };

                Vec3 point01{
                    coords_host( 0, i, j + 1, 0 ), coords_host( 0, i, j + 1, 1 ), coords_host( 0, i, j + 1, 2 ) };

                Vec3 point11{
                    coords_host( 0, i + 1, j + 1, 0 ),
                    coords_host( 0, i + 1, j + 1, 1 ),
                    coords_host( 0, i + 1, j + 1, 2 ) };

                Vec3 points[4] = { point00, point10, point11, point01 };

                Vec3 centroid = ( point00 + point10 + point01 + point11 ) * 0.25;
                centroid.normalize();

                if ( isPointInSphericalPolygon( points, centroid, P ) )
                {
                    bound_i        = i;
                    bound_i_plus_1 = i + 1;
                    bound_j        = j;
                    bound_j_plus_1 = j + 1;
                }
            }
        }
    }

    return { bound_i, bound_i_plus_1, bound_j, bound_j_plus_1 };
}

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    using ScalarType = double;

    const int level = 8;

    const auto domain = DistributedDomain::create_uniform( level, level, 0.5, 1.0, 0, 0 );

    const auto max_level = domain.domain_info().subdomain_max_refinement_level();
    std::cout << "Max level: " << max_level << std::endl;

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    auto subdomain_coords_host = Kokkos::create_mirror_view( subdomain_shell_coords );
    Kokkos::deep_copy( subdomain_coords_host, subdomain_shell_coords );

    Grid3DDataScalar< double > subdomain_u(
        "subdomain_u",
        domain.subdomains().size(),
        domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        domain.domain_info().subdomain_num_nodes_per_side_laterally() );

    Grid3DDataScalar< double > subdomain_v(
        "subdomain_v",
        domain.subdomains().size(),
        domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        domain.domain_info().subdomain_num_nodes_per_side_laterally() );

    auto subdomain_u_host = Kokkos::create_mirror_view( subdomain_u );
    // Kokkos::deep_copy( subdomain_u_host, subdomain_u );

    auto subdomain_v_host = Kokkos::create_mirror_view( subdomain_v );
    // Kokkos::deep_copy( subdomain_v_host, subdomain_v );

    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > subid( "subid", domain, mask_data );
    VectorQ1Scalar< ScalarType > rank_( "rank", domain, mask_data );

    auto T_grid    = T.grid_data();
    auto rank_grid = rank_.grid_data();

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    int num_subdomains = domain.subdomains().size();

    int num_nodes_per_side_laterally = domain.domain_info().subdomain_num_nodes_per_side_laterally();

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "subdomain_coords_host(1, 0, 0, 0): " << subdomain_coords_host( 1, 0, 0, 0 ) << std::endl;
    std::cout << "subdomain_coords_host(1, 0, 0, 1): " << subdomain_coords_host( 1, 0, 0, 1 ) << std::endl;
    std::cout << "subdomain_coords_host(1, 0, 0, 2): " << subdomain_coords_host( 1, 0, 0, 2 ) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "subdomain_coords_host(1, n, 0, 0): "
              << subdomain_coords_host( 1, num_nodes_per_side_laterally - 1, 0, 0 ) << std::endl;
    std::cout << "subdomain_coords_host(1, n, 0, 1): "
              << subdomain_coords_host( 1, num_nodes_per_side_laterally - 1, 0, 1 ) << std::endl;
    std::cout << "subdomain_coords_host(1, n, 0, 2): "
              << subdomain_coords_host( 1, num_nodes_per_side_laterally - 1, 0, 2 ) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "subdomain_coords_host(1, n, n, 0): "
              << subdomain_coords_host( 1, num_nodes_per_side_laterally - 1, num_nodes_per_side_laterally - 1, 0 )
              << std::endl;
    std::cout << "subdomain_coords_host(1, n, n, 1): "
              << subdomain_coords_host( 1, num_nodes_per_side_laterally - 1, num_nodes_per_side_laterally - 1, 1 )
              << std::endl;
    std::cout << "subdomain_coords_host(1, n, n, 2): "
              << subdomain_coords_host( 1, num_nodes_per_side_laterally - 1, num_nodes_per_side_laterally - 1, 2 )
              << std::endl;
    std::cout << std::endl;
    std::cout << "subdomain_coords_host(1, 0, n, 0): "
              << subdomain_coords_host( 1, 0, num_nodes_per_side_laterally - 1, 0 ) << std::endl;
    std::cout << "subdomain_coords_host(1, 0, n, 1): "
              << subdomain_coords_host( 1, 0, num_nodes_per_side_laterally - 1, 1 ) << std::endl;
    std::cout << "subdomain_coords_host(1, 0, n, 2): "
              << subdomain_coords_host( 1, 0, num_nodes_per_side_laterally - 1, 2 ) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "num_nodes_per_side_laterally: " << num_nodes_per_side_laterally << std::endl;

    Vec3 V00{
        subdomain_coords_host( 0, 0, 0, 0 ), subdomain_coords_host( 0, 0, 0, 1 ), subdomain_coords_host( 0, 0, 0, 2 ) };
    Vec3 V01{
        subdomain_coords_host( 0, 0, num_nodes_per_side_laterally - 1, 0 ),
        subdomain_coords_host( 0, 0, num_nodes_per_side_laterally - 1, 1 ),
        subdomain_coords_host( 0, 0, num_nodes_per_side_laterally - 1, 2 ) };
    Vec3 V10{
        subdomain_coords_host( 0, num_nodes_per_side_laterally - 1, 0, 0 ),
        subdomain_coords_host( 0, num_nodes_per_side_laterally - 1, 0, 1 ),
        subdomain_coords_host( 0, num_nodes_per_side_laterally - 1, 0, 2 ) };
    Vec3 V11{
        subdomain_coords_host( 0, num_nodes_per_side_laterally - 1, num_nodes_per_side_laterally - 1, 0 ),
        subdomain_coords_host( 0, num_nodes_per_side_laterally - 1, num_nodes_per_side_laterally - 1, 1 ),
        subdomain_coords_host( 0, num_nodes_per_side_laterally - 1, num_nodes_per_side_laterally - 1, 2 ) };

    buildDiamondLookupTable(
        subdomain_coords_host,
        V00,
        V10,
        V01,
        V11,
        subdomain_u_host,
        subdomain_v_host,
        num_nodes_per_side_laterally - 1 );

    // for ( int i = 0; i < num_nodes_per_side_laterally; ++i )
    // {
    //     for ( int j = 0; j < num_nodes_per_side_laterally; ++j )
    //     {
    //         auto [u_i, v_i] = queryDiamondLookup(
    //             subdomain_u_host,
    //             subdomain_v_host,
    //             V00,
    //             V10,
    //             V01,
    //             V11,
    //             Vec3{
    //                 subdomain_coords_host( 0, i, j, 0 ),
    //                 subdomain_coords_host( 0, i, j, 1 ),
    //                 subdomain_coords_host( 0, i, j, 2 )
    //             },
    //             num_nodes_per_side_laterally - 1, 5 );

    //         if( i != u_i || j != v_i )
    //         {
    //             std::cout << "Query result for (" << i << ", " << j << "): u_i = " << u_i << ", v_i = " << v_i << std::endl;
    //         }
    //     }
    //     // std::cout << std::endl;
    // }

    std::random_device                       rd;
    std::mt19937                             gen( rd() );
    std::uniform_int_distribution< int >     distrib( 0, num_nodes_per_side_laterally - 2 );
    std::uniform_real_distribution< double > distrib_real( 0, 1 );

    // int query_i = distrib( gen );
    // // int query_i = num_nodes_per_side_laterally - 1;

    // int query_j = distrib( gen );
    // // int query_j = num_nodes_per_side_laterally - 1;

    int n_tests = 100;

    std::cout << "Test started!" << std::endl;

    while(n_tests--)
    {
        int query_i = distrib( gen );
        int query_j = distrib( gen );

        Vec3 query_boundary_xx = {
            subdomain_coords_host( 0, query_i, query_j, 0 ),
            subdomain_coords_host( 0, query_i, query_j, 1 ),
            subdomain_coords_host( 0, query_i, query_j, 2 ) };

        Vec3 query_boundary_xy = {
            subdomain_coords_host( 0, query_i, query_j + 1, 0 ),
            subdomain_coords_host( 0, query_i, query_j + 1, 1 ),
            subdomain_coords_host( 0, query_i, query_j + 1, 2 ) };

        Vec3 query_boundary_yx = {
            subdomain_coords_host( 0, query_i + 1, query_j, 0 ),
            subdomain_coords_host( 0, query_i + 1, query_j, 1 ),
            subdomain_coords_host( 0, query_i + 1, query_j, 2 ) };

        Vec3 query_boundary_yy = {
            subdomain_coords_host( 0, query_i + 1, query_j + 1, 0 ),
            subdomain_coords_host( 0, query_i + 1, query_j + 1, 1 ),
            subdomain_coords_host( 0, query_i + 1, query_j + 1, 2 ) };

        double rand_u = distrib_real( gen );
        double rand_v = distrib_real( gen );

        Vec3 query_point = ( 1.0 - rand_u ) * ( 1.0 - rand_v ) * query_boundary_xx +
                        rand_u * ( 1.0 - rand_v ) * query_boundary_yx + ( 1.0 - rand_u ) * rand_v * query_boundary_xy +
                        rand_u * rand_v * query_boundary_yy;

        query_point.normalize();

        auto [u_i, u_i_plus_1, v_i, v_i_plus_1] = queryDiamondLookup(
            subdomain_coords_host, subdomain_u_host, subdomain_v_host, V00, V10, V01, V11, query_point, num_nodes_per_side_laterally - 1, 21 );

        if( u_i != query_i || v_i != query_j)
        {
            std::cout << "rand_u: " << rand_u << ", rand_v: " << rand_v << std::endl << std::endl;

            std::cout << "Query result for (" << query_i << ", " << query_j << "): u_i = " << u_i
                    << ", u_i_plus_1 = " << u_i_plus_1 << ", v_i = " << v_i << ", v_i_plus_1 = " << v_i_plus_1 << std::endl;
        }
    }

    std::cout << "If there are no logs, then test passed *_*, else... bruh :(" << std::endl;

    // Kokkos::parallel_for(
    //     "test_T",
    //     grid::shell::local_domain_md_range_policy_nodes( domain ),
    //     TestCellInterpolator(
    //         num_subdomains,
    //         num_nodes_per_side_laterally,
    //         subdomain_shell_coords,
    //         subdomain_radii,
    //         T.grid_data(),
    //         subid.grid_data() )
    //     // KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
    //     //     T_grid( sd, x, y, r ) = static_cast< ScalarType >( sd );
    //     //     // rank_grid(sd, x, y, r) = static_cast<ScalarType>(rank);
    //     // }
    // );
    // Kokkos::fence();

    // io::XDMFOutput< ScalarType > xdmf_output( "./output/", domain, subdomain_shell_coords, subdomain_radii );

    // xdmf_output.add( T.grid_data() );
    // xdmf_output.add( subid.grid_data() );
    // // xdmf_output.add( rank_.grid_data() );

    // xdmf_output.write( 0 );

    return 0;
}