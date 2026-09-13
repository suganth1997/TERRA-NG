// Test: lateral_cell_from_index_coords maps continuous lateral index coordinates onto the containing wedge.
//
// Uses non-square bounds so that a swap of the x and y extents is caught.

#include <iostream>
#include <string>

#include "fe/wedge/sl/point_location.hpp"
#include "util/init.hpp"

using namespace terra;
using ScalarType = double;

using fe::wedge::sl::IndexBounds;
using fe::wedge::sl::lateral_cell_from_index_coords;

namespace
{

int g_failures = 0;

void expect_cell( const ScalarType u, const ScalarType v, const int x, const int y, const int w )
{
    const IndexBounds bounds{ 5, 7, 3 }; // last hex cell is (3, 5)

    const auto cell = lateral_cell_from_index_coords( u, v, bounds );

    if ( cell.x != x || cell.y != y || cell.w != w || cell.r != 0 )
    {
        ++g_failures;
        std::cout << "  FAIL: (u, v) = (" << u << ", " << v << "): got (" << cell.x << ", " << cell.y << ", w="
                  << cell.w << ", r=" << cell.r << "), expected (" << x << ", " << y << ", w=" << w << ")"
                  << std::endl;
    }
}

} // namespace

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    // Interior: lower-left and upper-right triangle of a hex cell.
    expect_cell( 1.2, 2.3, 1, 2, 0 );
    expect_cell( 1.8, 2.7, 1, 2, 1 );

    // On the anti-diagonal the lower-left triangle wins.
    expect_cell( 1.5, 2.5, 1, 2, 0 );

    // Nodes.
    expect_cell( 0.0, 0.0, 0, 0, 0 );
    expect_cell( 2.0, 3.0, 2, 3, 0 );

    // Last node in each direction belongs to the last cell.
    expect_cell( 4.0, 0.0, 3, 0, 0 );
    expect_cell( 0.0, 6.0, 0, 5, 0 );
    expect_cell( 4.0, 6.0, 3, 5, 1 );

    // Outside the index space: clamped onto the boundary cell, triangle facing the point.
    expect_cell( -0.3, 0.2, 0, 0, 0 );
    expect_cell( 10.5, 10.5, 3, 5, 1 );
    expect_cell( -2.0, 9.0, 0, 5, 1 );
    expect_cell( 7.0, -3.0, 3, 0, 0 );

    if ( g_failures == 0 )
    {
        std::cout << "test_mmoc_lateral_cell_from_index_coords: PASSED" << std::endl;
    }
    else
    {
        std::cout << "test_mmoc_lateral_cell_from_index_coords: FAILED (" << g_failures << " checks)" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
