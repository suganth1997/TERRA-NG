

#include <iostream>
#include <mpi.h>

#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/linalg/vector_q1.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/io/xdmf.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    using ScalarType = double;

    const int level = 4;

    const auto domain = DistributedDomain::create_uniform( level, level, 0.5, 1.0, 1, 1);

    const auto max_level = domain.domain_info().subdomain_max_refinement_level();
    std::cout << "Max level: " << max_level << std::endl;

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > rank_( "rank", domain, mask_data );

    auto T_grid    = T.grid_data();
    auto rank_grid = rank_.grid_data();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Kokkos::parallel_for(
        "test_T",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            T_grid(sd, x, y, r) = static_cast<ScalarType>(sd);
            rank_grid(sd, x, y, r) = static_cast<ScalarType>(rank);
        } );
    Kokkos::fence();

    io::XDMFOutput< ScalarType > xdmf_output( "./output/", domain, subdomain_shell_coords, subdomain_radii );

    xdmf_output.add( T.grid_data() );
    xdmf_output.add( rank_.grid_data() );

    xdmf_output.write( 0 );

    return 0;
}