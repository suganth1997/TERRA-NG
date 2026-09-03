

#include <iostream>
#include <mpi.h>
#include <random>

#include "terra/fe/wedge/integrands.hpp"
// clang-format off
#include <Kokkos_Core.hpp>
#include <Kokkos_Clamp.hpp>

#include "terra/grid/grid_types.hpp"
#include "terra/fe/wedge/kernel_helpers.hpp"
// clang-format on
#include "terra/dense/mat.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/linalg/solvers/pcg.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/util/table.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/vector_q1.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;

using terra::fe::wedge::num_nodes_per_wedge;
using terra::fe::wedge::num_nodes_per_wedge_surface;
using terra::fe::wedge::num_wedges_per_hex_cell;

using terra::fe::wedge::atomically_add_local_wedge_scalar_coefficients;
using terra::fe::wedge::extract_local_wedge_scalar_coefficients;
using terra::fe::wedge::forward_map;
using terra::fe::wedge::wedge_surface_physical_coords;

using uint_t  = unsigned int;
using ScalarT = double;

using terra::fe::wedge::num_nodes_per_wedge_surface;
using terra::fe::wedge::num_wedges_per_hex_cell;
using terra::fe::wedge::wedge_surface_physical_coords;

using Vec3 = dense::Vec< double, 3 >;

KOKKOS_INLINE_FUNCTION
double tripleProduct( const Vec3& a, const Vec3& b, const Vec3& c )
{
    // return dot(a, cross(b, c));
    return a.dot( b.cross( c ) );
}

KOKKOS_INLINE_FUNCTION
dense::Mat< double, 3, 3 > getRotationMatrix( Vec3 axis, double angleRadians )
{
    axis.normalize(); // Ensure the axis is a unit vector

    double x = axis( 0 );
    double y = axis( 1 );
    double z = axis( 2 );

    // 2. Precompute sine, cosine, and (1 - cos)
    double c = Kokkos::cos( angleRadians );
    double s = Kokkos::sin( angleRadians );
    double t = 1.0 - c;

    // 3. Construct the matrix rows
    dense::Mat< double, 3, 3 > R{};

    R( 0, 0 ) = c + x * x * t;
    R( 0, 1 ) = x * y * t - z * s;
    R( 0, 2 ) = x * z * t + y * s;

    R( 1, 0 ) = y * x * t + z * s;
    R( 1, 1 ) = c + y * y * t;
    R( 1, 2 ) = y * z * t - x * s;

    R( 2, 0 ) = z * x * t - y * s;
    R( 2, 1 ) = z * y * t + x * s;
    R( 2, 2 ) = c + z * z * t;

    return R;
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
dense::Vec< ScalarT, 3 > sphericalTriangleCentroid( const dense::Vec< ScalarT, 3 > vertices[3] )
{
    dense::Vec< ScalarT, 3 > c;

    for ( int i_v = 0; i_v < 3; i_v++ )
    {
        auto v = vertices[i_v];

        c( 0 ) += v( 0 );
        c( 1 ) += v( 1 );
        c( 2 ) += v( 2 );
    }

    c( 0 ) /= 3.0;
    c( 1 ) /= 3.0;
    c( 2 ) /= 3.0;

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

KOKKOS_INLINE_FUNCTION
bool isPointInSphericalTriangle(
    const dense::Vec< ScalarT, 3 > vertices[3],
    const dense::Vec< ScalarT, 3 > interior_point,
    dense::Vec< ScalarT, 3 >       point,
    double                         tol = 1e-9 )
{
    point.normalize();

    for ( size_t i = 0; i < 3; ++i )
    {
        const dense::Vec< ScalarT, 3 >& A      = vertices[i];
        const dense::Vec< ScalarT, 3 >& B      = vertices[( i + 1 ) % 3];
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

KOKKOS_INLINE_FUNCTION
dense::Vec< int, 4 > queryDiamondLookup(
    const Grid3DDataVec< double, 3 > coords,
    const Grid3DDataScalar< double > u_grid,
    const Grid3DDataScalar< double > v_grid,
    int                              i_subdomain,
    const Vec3                       P,
    int                              N,
    int                              refine_radius = 2 )
{
    Vec3 V00{ coords( i_subdomain, 0, 0, 0 ), coords( i_subdomain, 0, 0, 1 ), coords( i_subdomain, 0, 0, 2 ) };
    Vec3 V01{ coords( i_subdomain, 0, N, 0 ), coords( i_subdomain, 0, N, 1 ), coords( i_subdomain, 0, N, 2 ) };
    Vec3 V10{ coords( i_subdomain, N, 0, 0 ), coords( i_subdomain, N, 0, 1 ), coords( i_subdomain, N, 0, 2 ) };
    Vec3 V11{ coords( i_subdomain, N, N, 0 ), coords( i_subdomain, N, N, 1 ), coords( i_subdomain, N, N, 2 ) };

    double u{}, v{};
    bool   conversion = diamondPointToUV( V00, V10, V01, V11, P, u, v );

    int seed_i = Kokkos::clamp( (int) Kokkos::floor( u * N ), 0, N );
    int seed_j = Kokkos::clamp( (int) Kokkos::floor( v * N ), 0, N );

    int    best_i = seed_i, best_j = seed_j;
    double du0 = u_grid( i_subdomain, seed_i, seed_j ) - u, dv0 = v_grid( i_subdomain, seed_i, seed_j ) - v;
    double best_d2 = du0 * du0 + dv0 * dv0;

    if ( !conversion )
    {
        Kokkos::abort( "Conversion failed" );
    }

    for ( int di = -refine_radius; di <= refine_radius; ++di )
    {
        for ( int dj = -refine_radius; dj <= refine_radius; ++dj )
        {
            int ci = seed_i + di, cj = seed_j + dj;
            if ( ci < 0 || ci > N || cj < 0 || cj > N )
                continue;
            double du = u_grid( i_subdomain, ci, cj ) - u;
            double dv = v_grid( i_subdomain, ci, cj ) - v;
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

                Vec3 point00{
                    coords( i_subdomain, i, j, 0 ), coords( i_subdomain, i, j, 1 ), coords( i_subdomain, i, j, 2 ) };

                Vec3 point10{
                    coords( i_subdomain, i + 1, j, 0 ),
                    coords( i_subdomain, i + 1, j, 1 ),
                    coords( i_subdomain, i + 1, j, 2 ) };

                Vec3 point01{
                    coords( i_subdomain, i, j + 1, 0 ),
                    coords( i_subdomain, i, j + 1, 1 ),
                    coords( i_subdomain, i, j + 1, 2 ) };

                Vec3 point11{
                    coords( i_subdomain, i + 1, j + 1, 0 ),
                    coords( i_subdomain, i + 1, j + 1, 1 ),
                    coords( i_subdomain, i + 1, j + 1, 2 ) };

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

KOKKOS_INLINE_FUNCTION
int find_interval( const Grid2DDataScalar< double >& arr, const int n, int i_sub, const double query )
{
    if ( query <= arr( i_sub, 0 ) )
        return 0;
    if ( query >= arr( i_sub, n - 1 ) )
        return n - 2;

    int lo = 0;
    int hi = n - 1;

    while ( hi - lo > 1 )
    {
        int mid = lo + ( hi - lo ) / 2;
        if ( arr( i_sub, mid ) <= query )
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }

    return lo;
}

template < typename ScalarT = double >
struct TestCellInterpolator
{
    uint_t num_subdomains_;
    uint_t num_nodes_radially_;
    uint_t num_nodes_per_side_laterally_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    grid::Grid3DDataScalar< ScalarT > data_u_grid_;
    grid::Grid3DDataScalar< ScalarT > data_v_grid_;

    grid::Grid4DDataScalar< ScalarT > data_T_new_;
    grid::Grid4DDataScalar< ScalarT > data_T_old_;

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        auto dof_coords = grid::shell::coords( local_subdomain_id, x_cell, y_cell, r_cell, grid_, radii_ );

        auto rotation_matrix = getRotationMatrix( Vec3{ -1.0, 0.0, 0.0 }, M_PI / 8 ); // To move particles

        auto rotated_dof_coords = rotation_matrix * dof_coords;

        dense::Vec< ScalarT, 3 > point = rotated_dof_coords.normalized();

        // point( 0 ) = grid_( local_subdomain_id, x_cell, y_cell, 0 );
        // point( 1 ) = grid_( local_subdomain_id, x_cell, y_cell, 1 );
        // point( 2 ) = grid_( local_subdomain_id, x_cell, y_cell, 2 );

        double r_point = rotated_dof_coords.norm();
        // double r_point = radii_( local_subdomain_id, r_cell );

        int i_subdomain = -1;

        for ( int i = 0; i < num_subdomains_; ++i )
        {
            dense::Vec< ScalarT, 3 > subdomain_extent[4];

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
                i_subdomain = i;
                break;
            }
        }

        // i_subdomain = local_subdomain_id;

        dense::Vec< int, 4 > idx = queryDiamondLookup(
            grid_, data_u_grid_, data_v_grid_, i_subdomain, point, num_nodes_per_side_laterally_ - 1, 17 );

        int r_i = find_interval( radii_, num_nodes_radially_, i_subdomain, radii_( local_subdomain_id, r_cell ) );

        // int x_cell_ = x_cell > num_nodes_per_side_laterally_ - 2 ? num_nodes_per_side_laterally_ - 2 : x_cell;
        // int y_cell_ = y_cell > num_nodes_per_side_laterally_ - 2 ? num_nodes_per_side_laterally_ - 2 : y_cell;
        // int r_cell_ = r_cell > num_nodes_radially_ - 2 ? num_nodes_radially_ - 2 : r_cell;

        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, i_subdomain, idx( 0 ), idx( 2 ) );

        dense::Vec< ScalarT, 6 > T_old[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( T_old, i_subdomain, idx( 0 ), idx( 2 ), r_i, data_T_old_ );

        for ( int i_wedge = 0; i_wedge < num_wedges_per_hex_cell; ++i_wedge )
        {
            dense::Vec< ScalarT, 3 > wedge_centroid = sphericalTriangleCentroid( wedge_phy_surf[i_wedge] );

            if ( isPointInSphericalTriangle( wedge_phy_surf[i_wedge], wedge_centroid, point ) )
            {
                auto jac = terra::dense::Mat< double, 3, 3 >::from_col_vecs(
                    wedge_phy_surf[i_wedge][0], wedge_phy_surf[i_wedge][1], wedge_phy_surf[i_wedge][2] );
                auto jac_inv = jac.inv();

                dense::Vec< ScalarT, 3 > barycentric = jac_inv * point;

                double xi  = barycentric( 1 );
                double eta = barycentric( 2 );

                double zeta = -1 + 2.0 * ( r_point - radii_( i_subdomain, r_i ) ) /
                                       ( radii_( i_subdomain, r_i + 1 ) - radii_( i_subdomain, r_i ) );

                double T_val = 0.0;

                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                {
                    double shape_i = terra::fe::wedge::shape( i, xi, eta, zeta );

                    T_val += T_old[i_wedge]( i ) * shape_i;
                }

                data_T_new_( local_subdomain_id, x_cell, y_cell, r_cell ) = T_val;

                break;
            }
        }
    }
};

template < typename CoordGridType, typename UVGridType >
void buildDiamondLookupTable(
    const CoordGridType& points,
    UVGridType&          u_grid,
    UVGridType&          v_grid,
    int                  N_subdomains,
    int                  N )
{
    for ( int i_sub = 0; i_sub < N_subdomains; ++i_sub )
    {
        Vec3 V00{ points( i_sub, 0, 0, 0 ), points( i_sub, 0, 0, 1 ), points( i_sub, 0, 0, 2 ) };
        Vec3 V01{ points( i_sub, 0, N, 0 ), points( i_sub, 0, N, 1 ), points( i_sub, 0, N, 2 ) };
        Vec3 V10{ points( i_sub, N, 0, 0 ), points( i_sub, N, 0, 1 ), points( i_sub, N, 0, 2 ) };
        Vec3 V11{ points( i_sub, N, N, 0 ), points( i_sub, N, N, 1 ), points( i_sub, N, N, 2 ) };

        for ( int i = 0; i <= N; ++i )
        {
            for ( int j = 0; j <= N; ++j )
            {
                double u{}, v{};

                Vec3 point{ points( i_sub, i, j, 0 ), points( i_sub, i, j, 1 ), points( i_sub, i, j, 2 ) };

                diamondPointToUV( V00, V10, V01, V11, point, u, v );
                u_grid( i_sub, i, j ) = u;
                v_grid( i_sub, i, j ) = v;
            }
        }
    }
}

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    using ScalarType = double;

    const int level = 6;

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

    auto subdomain_radii_host = Kokkos::create_mirror_view( subdomain_radii );
    Kokkos::deep_copy( subdomain_radii_host, subdomain_radii );

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
    VectorQ1Scalar< ScalarType > T_new( "T_new", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_mass( "T_mass", domain, mask_data );
    VectorQ1Scalar< ScalarType > subid( "subid", domain, mask_data );
    VectorQ1Scalar< ScalarType > rank_( "rank", domain, mask_data );

    // kernels::common::set_constant(k.grid_data(), 1.0);

    auto T_new_grid = T_new.grid_data();
    auto T_grid      = T.grid_data();
    auto rank_grid   = rank_.grid_data();

    Kokkos::parallel_for(
        "k_interpolate",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int local_subdomain_id, const int x, const int y, const int r ) {
            const dense::Vec< ScalarType, 3 > c =
                grid::shell::coords( local_subdomain_id, x, y, r, subdomain_shell_coords, subdomain_radii );

            if ( c( 0 ) * c( 0 ) + ( c( 1 ) - 0.75 ) * ( c( 1 ) - 0.75 ) + c( 2 ) * c( 2 ) < 0.1 * 0.1 )
            {
                T_grid( local_subdomain_id, x, y, r )     = 1.0;
                T_new_grid( local_subdomain_id, x, y, r ) = 1.0;
            }
            else
            {
                T_grid( local_subdomain_id, x, y, r )     = 0.0;
                T_new_grid( local_subdomain_id, x, y, r ) = 0.0;
            }
        } );
    Kokkos::fence();

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    int num_subdomains               = domain.subdomains().size();
    int num_nodes_per_side_laterally = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    int num_nodes_radially           = domain.domain_info().subdomain_num_nodes_radially();

    std::cout << "num_subdomains:               " << num_subdomains << std::endl;
    std::cout << "num_nodes_per_side_laterally: " << num_nodes_per_side_laterally << std::endl;
    std::cout << "num_nodes_radially:           " << num_nodes_radially << std::endl;

    // for(int i_rad = 0; i_rad < num_nodes_radially; ++i_rad)
    // {
    //     std::cout << "Radial node " << i_rad << ": radius = " << subdomain_radii_host( 0, i_rad ) << std::endl;
    // }

    buildDiamondLookupTable(
        subdomain_coords_host, subdomain_u_host, subdomain_v_host, num_subdomains, num_nodes_per_side_laterally - 1 );

    Kokkos::deep_copy( subdomain_u, subdomain_u_host );
    Kokkos::deep_copy( subdomain_v, subdomain_v_host );

    io::XDMFOutput< ScalarType > xdmf_output( "./output/", domain, subdomain_shell_coords, subdomain_radii );

    xdmf_output.add( T_grid );
    xdmf_output.add( T_new_grid );

    xdmf_output.write( 0 );

    // const int n_timesteps = 8;

    // for(int i_timestep = 0; i_timestep < n_timesteps; ++i_timestep)
    // {

    Kokkos::parallel_for(
        "test_idx_lookup",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        TestCellInterpolator(
            num_subdomains,
            num_nodes_radially,
            num_nodes_per_side_laterally,
            subdomain_shell_coords,
            subdomain_radii,
            subdomain_u,
            subdomain_v,
            T_new_grid,
            T_grid ) );
    Kokkos::fence();

    // kernels::common::lincomb( T_old_grid, 0.0, 1.0, T_grid );

    xdmf_output.write( 1 );

    // }

    return 0;
}