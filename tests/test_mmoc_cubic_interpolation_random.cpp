// Test: evaluate_cubic_scalar on a real shell, with every dof turned into a randomly displaced particle.
//
// test_mmoc_cubic_interpolation pins the reconstruction down on a hand-built 5x5x5 index block. This is the
// other half: the real grid, the real point location, and a sample at *every* node of every diamond. Each node
// is displaced by a random vector drawn uniformly from a ball of radius 0.75 h around it, the displaced point
// is located exactly as the transport locates a departure point, and the reconstruction of a smooth Q1 field
// there is compared against the analytic value. One sweep therefore meets every cell of the mesh at an
// arbitrary position inside it, including every diamond edge and every 5-valent corner. It is the
// interpolation error of the scheme with nothing else mixed in: no timestep, no trajectory, no accumulation.
//
// The fields live in the grid's own Grid4DDataScalar / Grid5DDataScalar containers and the sweep runs as a
// Kokkos::parallel_for over the dofs in the default execution space, so on a CUDA build every sample is
// located and evaluated on the GPU -- one thread per departure point, neighbouring threads searching
// different subdomains, walking different numbers of steps and sliding different stencil windows, which is
// the divergence the transport kernel actually runs into. Only the per-sample results come back; every
// assertion is made on the host, where a failure can name the dof that caused it.
//
// Four evaluators run on each sample, so the table says where the error comes from:
//
//   q1             the second-order evaluation the cubic replaces -- the control.
//   cubic          width 4, unclipped: the raw tensor-product reconstruction.
//   cubic+clip     width 4, clipped into the containing cell's nodal range (Bermejo-Staniforth).
//   quintic+clip   width 6 and clipped -- what MmocTransport is configured with by default.
//
// and the whole sweep runs twice, over two fields that differ only in which coordinates they are smooth in
// (see FieldKind). That split is what the test is for, because the two answers are not the same:
//
//   * On a field that is smooth in the reconstruction's *own* coordinates, the cubic is third order (measured
//     3.34 between levels 3 and 4) and 29x better than Q1. The evaluator does what it claims.
//
//   * On a field that is smooth in *physical* space -- which is what the transport actually carries -- it is
//     second order (measured 1.73) and only 2.4x better than Q1. The cubic's order does not survive the trip
//     through the coordinates the point arrives with.
//
// The cap is lateral, not radial. Replacing the physical field by one that depends on |x| alone restores
// order 3.25; replacing it by one that depends on the direction alone leaves 1.94. Radially the evaluation is
// already clean, because radial_coords_from_radius re-derives the radial coordinate from the point's true
// radius; laterally there is no such correction. The `lateral map` row measures what is left: it reconstructs
// the unit-sphere node map with the very same evaluator at the sample's own lateral index coordinates and
// compares it against the sample's own direction. Between levels 3 and 4 that mismatch falls 1.6e-3 -> 5.6e-4
// while the cubic's error on the physical field falls 1.4e-3 -> 4.2e-4 -- the same size, and neither of them
// third order. The reconstruction is being asked for a value at a point the nodal map only locates to second
// order, and no interpolation order can recover that. It is the lateral analogue of the rho-versus-|x| bias
// documented on radial_coords_from_radius, and it is the thing to fix if the cubic's third order is ever
// wanted end to end.
//
// So the asserted quantities are: third order for the reconstruction in its own coordinates, second order and
// a clear margin over Q1 end to end, and the mismatch above remaining the dominant term in that gap. Errors
// are reported as an rms, an rms over the samples whose cell sits within two cells of a diamond edge (where
// the stencil is slid inwards and goes one-sided), and a max. The max is printed rather than asserted: it is
// owned by the few cells at the diamond corners, where the index map is irregular by construction.
//
// The figures quoted above were measured on the host; a device run may move the last digit, since nvcc
// contracts multiply-adds into FMAs where the host compiler does not. Nothing asserted here is that sharp.
//
// Single rank only (registered at np=1): with no ghost layer, a point displaced off a diamond has to be found
// in another *local* subdomain, which needs all ten present.

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <mpi.h>

#include "fe/wedge/sl/point_location.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;
using ScalarType = double;
using Vec3       = dense::Vec< ScalarType, 3 >;

using fe::wedge::sl::IndexBounds;
using fe::wedge::sl::WedgeCell;

namespace
{

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        std::cout << "  FAIL: " << what << std::endl;
    }
}

std::string fmt( const ScalarType x )
{
    std::ostringstream os;
    os << std::scientific << std::setprecision( 2 ) << x;
    return os.str();
}

/// @brief The interpolated formula: smooth, non-polynomial, with extrema inside the domain.
///
/// Read in two different coordinate systems, see \ref FieldKind. Its three arguments are of comparable range
/// either way, so the two readings have derivatives of comparable size and their errors may be compared.
KOKKOS_INLINE_FUNCTION ScalarType smooth_formula( const ScalarType a, const ScalarType b, const ScalarType c )
{
    return 1.0 + Kokkos::sin( 3.0 * a ) * Kokkos::cos( 2.0 * b + 0.4 ) + 0.5 * Kokkos::exp( -2.0 * c * c );
}

/// @brief Which coordinates the sampled field is a smooth function of.
///
/// `Physical` is the honest end-to-end question -- a smooth function of x, sampled at the nodes, recovered at
/// a point in space -- and measures the whole departure-point evaluation: location, coordinates and
/// reconstruction together. `Index` is the same formula read in the coordinates the reconstruction itself
/// works in (the lateral node indices, normalised by the diamond side, and the cone radius), so the
/// coordinates the point arrives with are exact by construction and only the interpolation error is left.
/// Running both separates the two.
enum class FieldKind
{
    Physical,
    Index
};

const char* field_kind_name( const FieldKind kind )
{
    return ( kind == FieldKind::Physical ) ? "physical-space field" : "index-space field";
}

/// @brief One evaluator configuration; `width == 0` selects Q1.
struct Variant
{
    const char* name;
    int         width;
    bool        clip;
};

constexpr int num_variants           = 4;
const Variant variants[num_variants] = { { "q1", 0, false },
                                         { "cubic", fe::wedge::sl::cubic_stencil_size, false },
                                         { "cubic+clip", fe::wedge::sl::cubic_stencil_size, true },
                                         { "quintic+clip", fe::wedge::sl::quintic_stencil_size, true } };

/// @brief Error accumulator for one variant, split into all samples and the near-edge ones.
struct Stats
{
    ScalarType max_err     = 0.0;
    ScalarType sum_sq      = 0.0;
    ScalarType sum_sq_edge = 0.0;
    long long  n           = 0;
    long long  n_edge      = 0;
    int        worst[4]    = { 0, 0, 0, 0 }; // the dof (subdomain, x, y, r) that produced max_err
    int        worst_edge  = 0;              // how many cells that sample sat from the diamond edge

    void add( const ScalarType err, const int edge_dist, const int sd, const int x, const int y, const int r )
    {
        sum_sq += err * err;
        ++n;
        if ( edge_dist < 2 )
        {
            sum_sq_edge += err * err;
            ++n_edge;
        }
        if ( err > max_err )
        {
            max_err    = err;
            worst[0]   = sd;
            worst[1]   = x;
            worst[2]   = y;
            worst[3]   = r;
            worst_edge = edge_dist;
        }
    }

    ScalarType rms() const { return ( n > 0 ) ? std::sqrt( sum_sq / static_cast< ScalarType >( n ) ) : 0.0; }
    ScalarType rms_edge() const
    {
        return ( n_edge > 0 ) ? std::sqrt( sum_sq_edge / static_cast< ScalarType >( n_edge ) ) : 0.0;
    }
};

struct LevelResult
{
    Stats     stats[num_variants];
    Stats     map_mismatch; // |reconstructed nodal direction - the sample's own direction|, see below
    long long samples = 0;
    long long escaped = 0; // displaced points no local subdomain claimed
    long long clipped = 0; // samples where clip_to_cell actually moved the value
};

/// @brief Displaces every dof of the shell into a ball around it and interpolates the field back at the
///        displaced point.
///
/// `perturb_fraction` is measured in cell widths, so the sample sits at the same relative position inside its
/// cell at every level and the errors below may be compared across levels as a convergence study.
LevelResult run_level( const int level, const FieldKind kind, const ScalarType perturb_fraction )
{
    const ScalarType r_min = 0.5;
    const ScalarType r_max = 1.0;

    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
        level, level, r_min, r_max );

    // Device-resident, straight from the grid; the host mirrors are for building the fields and the sample.
    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    auto coords_shell_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_shell );
    auto coords_radii_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_radii );

    const int num_subdomains = static_cast< int >( domain.subdomains().size() );
    const int num_nodes_lat  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int num_nodes_rad  = domain.domain_info().subdomain_num_nodes_radially();

    const IndexBounds bounds{ num_nodes_lat, num_nodes_lat, num_nodes_rad };
    const auto        box     = fe::wedge::sl::corner_box_from_bounds( bounds );
    const auto        stencil = fe::wedge::sl::full_stencil_bounds( bounds );

    // No ghost layer on these plain subdomain fields, so every node carries usable geometry.
    grid::Grid3DDataScalar< uint8_t > valid( "valid", num_subdomains, num_nodes_lat, num_nodes_lat );
    Kokkos::deep_copy( valid, static_cast< uint8_t >( 1 ) );

    grid::Grid4DDataScalar< ScalarType > field(
        "field", num_subdomains, num_nodes_lat, num_nodes_lat, num_nodes_rad );

    // The unit-sphere node map, laid out so that the same evaluator can reconstruct it. It does not depend on
    // the radial index; the radial sweep over constant data is exact, so what comes back is the lateral map
    // alone. Reconstructing the geometry a field is defined on is how the lateral error below is measured.
    grid::Grid5DDataScalar< ScalarType > unit_map(
        "unit_map", num_subdomains, num_nodes_lat, num_nodes_lat, num_nodes_rad, 3 );

    auto field_h    = Kokkos::create_mirror_view( field );
    auto unit_map_h = Kokkos::create_mirror_view( unit_map );

    const auto node_position = [&]( const int sd, const int x, const int y, const int r ) {
        Vec3 p;
        for ( int d = 0; d < 3; ++d )
            p( d ) = coords_shell_h( sd, x, y, d ) * coords_radii_h( sd, r );
        return p;
    };

    // Normalises a lateral index onto [0, 1], so that the index-space field has a level-independent
    // wavelength and its error converges under refinement like the physical-space one.
    const ScalarType inv_span = 1.0 / static_cast< ScalarType >( num_nodes_lat - 1 );

    for ( int sd = 0; sd < num_subdomains; ++sd )
        for ( int x = 0; x < num_nodes_lat; ++x )
            for ( int y = 0; y < num_nodes_lat; ++y )
                for ( int r = 0; r < num_nodes_rad; ++r )
                {
                    const Vec3 p = node_position( sd, x, y, r );

                    for ( int d = 0; d < 3; ++d )
                        unit_map_h( sd, x, y, r, d ) = coords_shell_h( sd, x, y, d );

                    field_h( sd, x, y, r ) =
                        ( kind == FieldKind::Physical )
                            ? smooth_formula( p( 0 ), p( 1 ), p( 2 ) )
                            : smooth_formula( static_cast< ScalarType >( x ) * inv_span,
                                              static_cast< ScalarType >( y ) * inv_span,
                                              coords_radii_h( sd, r ) );
                }

    Kokkos::deep_copy( field, field_h );
    Kokkos::deep_copy( unit_map, unit_map_h );

    // ---- the sample ----------------------------------------------------------------------------------------
    // Drawn on the host and copied down, so the sample does not depend on the backend and a failing dof is
    // reproducible from the index printed with it.
    const int num_samples = num_subdomains * num_nodes_lat * num_nodes_lat * num_nodes_rad;

    grid::Grid2DDataScalar< ScalarType > points( "points", num_samples, 3 );
    auto                                 points_h = Kokkos::create_mirror_view( points );

    // The displacement radius, in units of the smallest cell dimension: laterally the shortest edge is the one
    // at the CMB, radially the layers are uniform here.
    const ScalarType h_lat = r_min * ( M_PI / 2.0 ) / static_cast< ScalarType >( num_nodes_lat - 1 );
    const ScalarType h_rad = ( r_max - r_min ) / static_cast< ScalarType >( num_nodes_rad - 1 );
    const ScalarType perturb_radius = perturb_fraction * std::min( h_lat, h_rad );

    std::mt19937_64                              rng( 0x9E3779B97F4A7C15ull + static_cast< uint64_t >( level ) );
    std::uniform_real_distribution< ScalarType > uniform( 0.0, 1.0 );

    for ( int idx = 0; idx < num_samples; ++idx )
    {
        const int sd  = idx / ( num_nodes_lat * num_nodes_lat * num_nodes_rad );
        const int rem = idx % ( num_nodes_lat * num_nodes_lat * num_nodes_rad );
        const int x   = rem / ( num_nodes_lat * num_nodes_rad );
        const int y   = ( rem / num_nodes_rad ) % num_nodes_lat;
        const int r   = rem % num_nodes_rad;

        // Uniform in the ball: a uniform direction with radius R * u^(1/3).
        const ScalarType z    = 2.0 * uniform( rng ) - 1.0;
        const ScalarType phi  = 2.0 * M_PI * uniform( rng );
        const ScalarType s    = std::sqrt( std::max( 0.0, 1.0 - z * z ) );
        const ScalarType dist = perturb_radius * std::cbrt( uniform( rng ) );

        const Vec3 X0 = node_position( sd, x, y, r );
        Vec3       X;
        X( 0 ) = X0( 0 ) + dist * s * std::cos( phi );
        X( 1 ) = X0( 1 ) + dist * s * std::sin( phi );
        X( 2 ) = X0( 2 ) + dist * z;

        // A dof on the CMB or on the surface is displaced straight through the boundary half the time. The
        // location would then be radially clamped while the analytic value is not, and the comparison would
        // measure the clamp rather than the interpolation; pulling the sample back onto the boundary shell
        // keeps it honest.
        const ScalarType radius = X.norm();
        const ScalarType inside = std::min( std::max( radius, r_min ), r_max );
        if ( inside != radius )
            X = X * ( inside / radius );

        for ( int d = 0; d < 3; ++d )
            points_h( idx, d ) = X( d );
    }
    Kokkos::deep_copy( points, points_h );

    // ---- the sweep -----------------------------------------------------------------------------------------
    // One thread per displaced dof. Per-sample results are stored rather than reduced: the kernel stays a
    // straight location-and-evaluate, and the statistics -- which need the dof a value came from, not just the
    // value -- are formed on the host below.
    grid::Grid2DDataScalar< ScalarType > errors( "errors", num_samples, num_variants );
    grid::Grid1DDataScalar< ScalarType > map_error( "map_error", num_samples );
    grid::Grid1DDataScalar< int >        edge_dist( "edge_dist", num_samples );
    grid::Grid1DDataScalar< int >        found( "found", num_samples );
    grid::Grid1DDataScalar< int >        clip_hit( "clip_hit", num_samples );

    // The variant table, in a form a device lambda can capture.
    Kokkos::Array< int, num_variants > width{};
    Kokkos::Array< int, num_variants > clip{};
    for ( int v = 0; v < num_variants; ++v )
    {
        width[v] = variants[v].width;
        clip[v]  = variants[v].clip ? 1 : 0;
    }

    // The same budgets the rest of the direct-location tests use.
    const int        direct_max_refinements = 2;
    const int        direct_max_walk_steps  = 4;
    const ScalarType eps                    = 1e-12;

    const int nlat = num_nodes_lat;
    const int nrad = num_nodes_rad;
    const int nsub = num_subdomains;

    Kokkos::parallel_for(
        "mmoc_cubic_random_samples",
        Kokkos::RangePolicy<>( 0, num_samples ),
        KOKKOS_LAMBDA( const int idx ) {
            const int sd = idx / ( nlat * nlat * nrad );

            Vec3 X;
            for ( int d = 0; d < 3; ++d )
                X( d ) = points( idx, d );

            // Exactly how the transport finds a departure point: the owning subdomain is not assumed.
            int        sd_found = -1;
            const auto res      = fe::wedge::sl::locate_point_in_local_subdomains(
                X, sd, nsub, coords_shell, coords_radii, box, bounds, direct_max_refinements,
                direct_max_walk_steps, eps, /*clamp_radially=*/true, r_min, r_max, valid, sd_found );

            found( idx ) = res.found ? 1 : 0;
            if ( !res.found )
                return;

            // And exactly how it evaluates there: the radial cell and zeta come from the point's true radius,
            // not from the wedge's parametric one.
            WedgeCell  cell = res.cell;
            ScalarType zeta = res.zeta;
            fe::wedge::sl::radial_coords_from_radius(
                sd_found, X.norm(), coords_radii, nrad - 1, r_min, r_max, cell.r, zeta );

            // The coordinates the evaluator will actually interpolate at.
            ScalarType u = 0.0, v = 0.0;
            fe::wedge::sl::wedge_lateral_index_coords( cell, res.xi, res.eta, u, v );
            const ScalarType rho =
                fe::wedge::sl::wedge_radius_from_zeta( sd_found, cell, coords_radii, zeta );

            const ScalarType exact = ( kind == FieldKind::Physical )
                                         ? smooth_formula( X( 0 ), X( 1 ), X( 2 ) )
                                         : smooth_formula( u * inv_span, v * inv_span, rho );

            edge_dist( idx ) = Kokkos::min( Kokkos::min( cell.x, cell.y ),
                                            Kokkos::min( nlat - 2 - cell.x, nlat - 2 - cell.y ) );

            ScalarType value[num_variants];
            for ( int vi = 0; vi < num_variants; ++vi )
            {
                value[vi] = ( width[vi] == 0 )
                                ? fe::wedge::sl::evaluate_q1_scalar(
                                      field, sd_found, cell, res.xi, res.eta, zeta )
                                : fe::wedge::sl::evaluate_cubic_scalar(
                                      field, sd_found, cell, res.xi, res.eta, zeta, coords_radii, stencil,
                                      valid, clip[vi] != 0, /*limit_slopes=*/false, width[vi] );

                errors( idx, vi ) = Kokkos::abs( value[vi] - exact );
            }

            // Variants 1 and 2 differ only in the clip, so this flags the samples it bites on.
            clip_hit( idx ) = ( Kokkos::abs( value[2] - value[1] ) > 1e-14 ) ? 1 : 0;

            // Where the nodal map itself puts the lateral index coordinates this sample arrived with, against
            // where the sample actually is. The reconstruction interpolates data that lives on the curved map,
            // while the coordinates come from the flat chord triangle of the located wedge, and the gap
            // between the two is an error in the *evaluation point* that no amount of interpolation order can
            // remove: it enters any field as |grad f| times this.
            const auto p_map = fe::wedge::sl::evaluate_cubic_vec< ScalarType, 3 >(
                unit_map, sd_found, cell, res.xi, res.eta, zeta, coords_radii, stencil, valid, false );

            const Vec3 dir      = X * ( 1.0 / X.norm() );
            ScalarType mismatch = 0.0;
            for ( int d = 0; d < 3; ++d )
                mismatch += ( p_map( d ) - dir( d ) ) * ( p_map( d ) - dir( d ) );
            map_error( idx ) = Kokkos::sqrt( mismatch );
        } );
    Kokkos::fence();

    // ---- the statistics ------------------------------------------------------------------------------------
    auto errors_h    = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, errors );
    auto map_error_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, map_error );
    auto edge_dist_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, edge_dist );
    auto found_h     = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, found );
    auto clip_hit_h  = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, clip_hit );

    LevelResult out;
    out.samples = num_samples;

    for ( int idx = 0; idx < num_samples; ++idx )
    {
        const int sd  = idx / ( num_nodes_lat * num_nodes_lat * num_nodes_rad );
        const int rem = idx % ( num_nodes_lat * num_nodes_lat * num_nodes_rad );
        const int x   = rem / ( num_nodes_lat * num_nodes_rad );
        const int y   = ( rem / num_nodes_rad ) % num_nodes_lat;
        const int r   = rem % num_nodes_rad;

        if ( found_h( idx ) == 0 )
        {
            ++out.escaped;
            continue;
        }

        for ( int v = 0; v < num_variants; ++v )
            out.stats[v].add( errors_h( idx, v ), edge_dist_h( idx ), sd, x, y, r );

        out.map_mismatch.add( map_error_h( idx ), edge_dist_h( idx ), sd, x, y, r );
        out.clipped += clip_hit_h( idx );
    }

    return out;
}

void print_level( const int level, const LevelResult& res )
{
    std::cout << "  level=" << level << "  samples=" << res.samples << "  escaped=" << res.escaped
              << "  clip changed the value on " << res.clipped << " samples" << std::endl;

    for ( int v = 0; v < num_variants; ++v )
    {
        const Stats& st = res.stats[v];
        std::cout << "    " << std::left << std::setw( 13 ) << variants[v].name << std::right
                  << "  rms=" << fmt( st.rms() ) << "  rms(near edge)=" << fmt( st.rms_edge() )
                  << "  max=" << fmt( st.max_err ) << " at dof (" << st.worst[0] << ", " << st.worst[1] << ", "
                  << st.worst[2] << ", " << st.worst[3] << "), " << st.worst_edge << " cells from the edge"
                  << std::endl;
    }

    std::cout << "    " << std::left << std::setw( 13 ) << "lateral map" << std::right
              << "  rms=" << fmt( res.map_mismatch.rms() )
              << "  rms(near edge)=" << fmt( res.map_mismatch.rms_edge() )
              << "  max=" << fmt( res.map_mismatch.max_err ) << "  (an error in the evaluation point, not in a"
              << " field value)" << std::endl;
}

/// @brief Sweeps one field kind at two levels, prints both tables and the observed convergence orders.
void sweep( const FieldKind  kind,
            const int        coarse,
            const int        fine,
            const ScalarType perturb_fraction,
            LevelResult&     res_fine_out,
            ScalarType ( &order )[num_variants] )
{
    std::cout << field_kind_name( kind ) << ":" << std::endl;

    const LevelResult res_coarse = run_level( coarse, kind, perturb_fraction );
    const LevelResult res_fine   = run_level( fine, kind, perturb_fraction );

    print_level( coarse, res_coarse );
    print_level( fine, res_fine );

    std::cout << "    observed order:";
    for ( int v = 0; v < num_variants; ++v )
    {
        order[v] = std::log2( res_coarse.stats[v].rms() / std::max( res_fine.stats[v].rms(), 1e-300 ) );
        std::cout << "  " << variants[v].name << "=" << std::fixed << std::setprecision( 2 ) << order[v];
    }
    std::cout << "  lateral map="
              << std::log2( res_coarse.map_mismatch.rms() / std::max( res_fine.map_mismatch.rms(), 1e-300 ) )
              << std::defaultfloat << std::endl;

    check( res_coarse.escaped == 0 && res_fine.escaped == 0,
           std::string( field_kind_name( kind ) ) + ": a displaced dof was not claimed by any subdomain" );

    res_fine_out = res_fine;
}

} // namespace

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    // Three quarters of a cell: far enough from the node that the reconstruction is doing real work, close
    // enough that the sample stays in the node's own neighbourhood.
    const ScalarType perturb_fraction = 10.0;

    const int coarse = 5;
    const int fine   = 6;

    LevelResult index_fine, physical_fine;
    ScalarType  index_order[num_variants], physical_order[num_variants];

    sweep( FieldKind::Index, coarse, fine, perturb_fraction, index_fine, index_order );
    sweep( FieldKind::Physical, coarse, fine, perturb_fraction, physical_fine, physical_order );

    // ---- the reconstruction itself, on the index-space field -------------------------------------------
    // Q1 is the control: it is second order, and the comparison only means something as long as it shows that.
    check( index_order[0] > 1.8, "Q1 is not second order, so nothing below is calibrated" );

    check( index_order[1] > 2.8, "the cubic reconstruction is not third order in its own coordinates" );
    check( index_fine.stats[1].rms() < 0.25 * index_fine.stats[0].rms(),
           "the cubic reconstruction is not clearly better than Q1 in its own coordinates" );

    // The stencil goes one-sided within two cells of a diamond edge, which costs accuracy but must not cost
    // so much that it falls back behind Q1 on the same samples.
    check( index_fine.stats[3].rms_edge() < index_fine.stats[0].rms_edge(),
           "the transport's evaluator is worse than Q1 near the diamond edges" );

    // ---- the whole evaluation, on the physical-space field ---------------------------------------------
    // Second order, not third: see the note at the top of the file. The reconstruction is still a worthwhile
    // constant-factor gain over Q1 and must stay one.
    check( physical_order[1] > 1.5, "the physical-space reconstruction lost even second order" );
    check( physical_fine.stats[1].rms() < 0.5 * physical_fine.stats[0].rms(),
           "the cubic reconstruction no longer beats Q1 on a physical-space field" );
    check( physical_fine.stats[3].rms() < physical_fine.stats[0].rms(),
           "the transport's evaluator is worse than Q1 overall" );

    // And the reason it is second order: the lateral index coordinates the sample arrives with agree with the
    // nodal map only to second order, which enters any field with a lateral gradient directly. The check is
    // that this really is the dominant term -- if the reconstruction error ever dropped well below it, the
    // explanation in the header would be wrong.
    check( physical_fine.map_mismatch.rms() > 0.2 * physical_fine.stats[1].rms(),
           "the lateral map mismatch no longer explains the second-order cap on the physical-space field" );

    if ( g_failures == 0 )
    {
        std::cout << "\ntest_mmoc_cubic_interpolation_random: PASSED" << std::endl;
    }
    else
    {
        std::cout << "\ntest_mmoc_cubic_interpolation_random: FAILED (" << g_failures << " checks)" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
