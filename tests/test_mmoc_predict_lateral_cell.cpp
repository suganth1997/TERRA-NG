// Test: predict_lateral_cell evaluated on the device at every lateral node of the shell mesh.
//
// Each node's own unit-sphere position is fed to the predictor of its subdomain, and the predicted index
// coordinates and cell are written out next to the node index they came from.

#include <iomanip>
#include <iostream>

#include "fe/wedge/sl/point_location.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;
using ScalarType = double;

using fe::wedge::sl::IndexBounds;

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int level = 2;

    const auto domain =
        grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );

    const int num_subdomains = static_cast< int >( domain.subdomains().size() );
    const int num_nodes_lat  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int num_nodes_rad  = domain.domain_info().subdomain_num_nodes_radially();

    const IndexBounds bounds{ num_nodes_lat, num_nodes_lat, num_nodes_rad };
    const auto        box = fe::wedge::sl::corner_box_from_bounds( bounds );

    const ScalarType eps = 1e-12;

    // Per node: u, v, and cell x, y, w.
    Kokkos::View< ScalarType**** > uv( "uv", num_subdomains, num_nodes_lat, num_nodes_lat, 2 );
    Kokkos::View< int**** >        cell( "cell", num_subdomains, num_nodes_lat, num_nodes_lat, 3 );

    Kokkos::parallel_for(
        "predict_lateral_cell",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_subdomains, num_nodes_lat, num_nodes_lat } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y ) {
            dense::Vec< ScalarType, 3 > X;
            for ( int d = 0; d < 3; ++d )
                X( d ) = coords_shell( sd, x, y, d );

            const auto pred = fe::wedge::sl::predict_lateral_cell( X, sd, coords_shell, box, bounds, eps );

            uv( sd, x, y, 0 )   = pred.u;
            uv( sd, x, y, 1 )   = pred.v;
            cell( sd, x, y, 0 ) = pred.cell.x;
            cell( sd, x, y, 1 ) = pred.cell.y;
            cell( sd, x, y, 2 ) = pred.cell.w;
        } );

    auto uv_h   = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, uv );
    auto cell_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, cell );

    // Full precision: a node sits on a cell corner, so round-off below the printed digits decides the cell.
    std::cout << std::setprecision( 17 );
    std::cout << "level=" << level << "  subdomains=" << num_subdomains << "  nodes/side=" << num_nodes_lat
              << std::endl;
    std::cout << "sd  node(x,y)  ->  u, v  ->  cell(x,y,w)" << std::endl;

    for ( int sd = 0; sd < num_subdomains; ++sd )
        for ( int x = 0; x < num_nodes_lat; ++x )
            for ( int y = 0; y < num_nodes_lat; ++y )
                std::cout << sd << "  (" << x << "," << y << ")  ->  " << uv_h( sd, x, y, 0 ) << ", "
                          << uv_h( sd, x, y, 1 ) << "  ->  (" << cell_h( sd, x, y, 0 ) << ","
                          << cell_h( sd, x, y, 1 ) << "," << cell_h( sd, x, y, 2 ) << ")" << std::endl;

    return 0;
}
