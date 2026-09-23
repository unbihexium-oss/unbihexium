# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : tests/unit/test_terrain.py
# Title       : Tests of terrain derivatives, hydrology and visibility
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires pytest, NumPy and SciPy
# =============================================================================
#
# Abstract
# --------
# Checks unbihexium.terrain on surfaces with analytic answers: slope and
# aspect of tilted planes (including the borders), hillshade of flat
# ground, curvatures of paraboloids and ridges, TPI, TRI and roughness of
# small windows, filling of a pit, D8 directions and accumulation of a
# ramp, watersheds of a V-shaped valley, and visibility behind a wall.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Arrays.
import numpy as np

# Test framework.
import pytest

# Terrain functions under test.
from unbihexium import terrain


# Plane z = 0.5 east + 0.2 north on a 10 m grid (row 0 is north).
@pytest.fixture
def plane() -> np.ndarray:
    # Row and column indices.
    rows, cols = np.mgrid[0:8, 0:10].astype(float)
    # Easting and northing in metres.
    east, north = cols * 10.0, (7 - rows) * 10.0
    # Tilted plane.
    return 0.5 * east + 0.2 * north


# Slope and aspect of a plane are exact everywhere, including the borders.
def test_slope_aspect_plane(plane: np.ndarray) -> None:
    # Gradient magnitude sqrt(0.5^2 + 0.2^2).
    rise = np.hypot(0.5, 0.2)
    # Slope in degrees.
    assert np.allclose(terrain.slope(plane, 10.0), np.degrees(np.arctan(rise)))
    # Slope in percent.
    assert np.allclose(terrain.slope(plane, 10.0, units="percent"), 100 * rise)
    # The plane rises to the east-north-east, so it faces west-south-west.
    expected = np.degrees(np.arctan2(-0.5, -0.2)) % 360
    # Aspect everywhere.
    assert np.allclose(terrain.aspect(plane, 10.0), expected)
    # A plane rising to the north faces south (180 degrees).
    north_rising = np.repeat(np.arange(5.0)[::-1, None], 5, axis=1)
    # Aspect of that plane.
    assert np.allclose(terrain.aspect(north_rising), 180.0)
    # Flat ground has no aspect.
    assert np.isnan(terrain.aspect(np.zeros((4, 4)))).all()
    # Non-square cells: 0.5 per 10 m in x becomes 0.5 per 20 m.
    assert np.allclose(terrain.gradient(plane, (20.0, 10.0))[0], 0.25)


# Hillshade of flat ground and of a slope facing the sun.
def test_hillshade() -> None:
    # Flat ground under a sun at 45 degrees: 255 sin(45).
    assert np.allclose(terrain.hillshade(np.zeros((5, 5))), 255 * np.sin(np.radians(45)))
    # A 45-degree slope facing west (rising to the east) lit from the west at 45 degrees.
    ramp = np.tile(np.arange(6.0), (6, 1))
    # The light hits the slope perpendicularly.
    assert np.allclose(terrain.hillshade(ramp, 1.0, azimuth=270, altitude=45), 255.0)
    # Lit from the east, the slope is in its own shadow: cos(i) = cos(90) = 0.
    assert np.allclose(terrain.hillshade(ramp, 1.0, azimuth=90, altitude=45), 0.0, atol=1e-9)


# Curvatures of analytic surfaces.
def test_curvature() -> None:
    # Grid coordinates with 2 m cells.
    rows, cols = np.mgrid[0:11, 0:11].astype(float)
    # Coordinates in metres relative to the centre.
    x, y = (cols - 5) * 2.0, (5 - rows) * 2.0
    # Bowl z = x^2 + y^2: the Laplacian is 4, total curvature -4 (concave).
    assert terrain.total_curvature(x**2 + y**2, 2.0)[5, 5] == pytest.approx(-4.0)
    # Ridge descending to the east: z = -x - 0.1 y^2.
    profile, plan = terrain.curvature(-x - 0.1 * y**2, 2.0)
    # Straight profile along the ridge line.
    assert profile[5, 5] == pytest.approx(0.0, abs=1e-12)
    # Divergent contours: plan curvature -2 E = 0.2.
    assert plan[5, 5] == pytest.approx(0.2)
    # Slope steepening downhill (z = -x^2 east of the top) is convex: profile 2.
    assert terrain.curvature(-(x**2), 2.0)[0][5, 7] == pytest.approx(2.0)


# Window statistics of small patterns.
def test_tpi_tri_roughness_vrm(plane: np.ndarray) -> None:
    # A peak of 9 among zeros.
    peak = np.pad(np.array([[9.0]]), 2)
    # TPI of the peak: 9 minus the mean of its zero neighbours.
    assert terrain.tpi(peak)[2, 2] == pytest.approx(9.0)
    # A centre 1 among zeros.
    bump = np.pad(np.array([[1.0]]), 1)
    # Riley TRI: sqrt(8).
    assert terrain.tri(bump)[1, 1] == pytest.approx(np.sqrt(8.0))
    # Wilson TRI: mean absolute difference 1.
    assert terrain.tri(bump, method="wilson")[1, 1] == pytest.approx(1.0)
    # Roughness of 0 .. 8: 8.
    assert terrain.roughness(np.arange(9.0).reshape(3, 3))[1, 1] == pytest.approx(8.0)
    # A plane has parallel normals: VRM 0.
    assert np.allclose(terrain.vrm(plane, 10.0), 0.0, atol=1e-12)


# Filling a pit and routing flow on a ramp.
def test_fill_and_flow() -> None:
    # A pit of depth 9 in a plateau.
    pit = np.full((5, 5), 10.0)
    # Pit cell.
    pit[2, 2] = 1.0
    # Filled to the spill level.
    assert terrain.fill_depressions(pit)[2, 2] == pytest.approx(10.0)
    # With epsilon the flat ring around the pit rises by one step and the pit by two.
    filled = terrain.fill_depressions(pit, epsilon=0.01)
    # Ring cell and pit cell drain towards the border.
    assert filled[1, 1] == pytest.approx(10.01) and filled[2, 2] == pytest.approx(10.02)
    # A ramp descending to the east.
    ramp = np.tile(np.arange(5.0, 0.0, -1.0), (3, 1))
    # D8 codes: east (1) except the last column, which would drain off the grid (0).
    codes = terrain.flow_direction_d8(ramp)
    # Expected codes.
    assert codes.tolist() == [[1, 1, 1, 1, 0]] * 3
    # Accumulation counts the upstream cells of each row.
    assert terrain.flow_accumulation(codes).tolist() == [[0, 1, 2, 3, 4]] * 3
    # Weighted accumulation sums the upstream weights.
    assert terrain.flow_accumulation(codes, np.full((3, 5), 2.0))[0, 4] == pytest.approx(8.0)
    # Streams from an accumulation threshold.
    assert terrain.extract_streams(terrain.flow_accumulation(codes), 3).sum() == 6


# Diagonal drops are divided by the diagonal distance.
def test_d8_diagonal() -> None:
    # Centre 10 with a drop of 3 to the south and 4 to the south-east.
    dem = np.full((3, 3), 10.0)
    # South neighbour.
    dem[2, 1] = 7.0
    # South-east neighbour: drop 4 / sqrt(2) = 2.83 < 3.
    dem[2, 2] = 6.0
    # The centre drains south (code 4).
    assert terrain.flow_direction_d8(dem)[1, 1] == 4


# Watershed of a V-shaped valley draining to one outlet.
def test_watershed() -> None:
    # Row and column indices.
    rows, cols = np.mgrid[0:7, 0:7].astype(float)
    # Valley along column 3, falling towards row 6.
    dem = np.abs(cols - 3) * 2.0 + (6 - rows) + 1.0
    # Flow directions.
    codes = terrain.flow_direction_d8(dem)
    # Cells draining to the valley outlet at the bottom.
    basin = terrain.watershed(codes, (6, 3))
    # Side slopes drain diagonally into the valley (drop 3 / sqrt(2) > 2): all 49 cells.
    assert basin.all()
    # The outlet collects the other 48 cells.
    assert terrain.flow_accumulation(codes)[6, 3] == 48
    # Outlets outside the grid are rejected.
    with pytest.raises(ValueError):
        # Row 9 does not exist.
        terrain.watershed(codes, (9, 0))


# The wetness index is larger in the valley than on the slopes.
def test_twi() -> None:
    # Row and column indices.
    rows, cols = np.mgrid[0:15, 0:15].astype(float)
    # Valley along column 7.
    dem = np.abs(cols - 7) * 1.0 + (14 - rows) * 0.2
    # Wetness index.
    wet = terrain.twi(dem, 10.0)
    # The valley floor near the outlet is wetter than a ridge cell.
    assert wet[13, 7] > wet[13, 0]


# Visibility behind a wall and on flat ground.
def test_viewshed() -> None:
    # Flat ground: everything is visible.
    assert terrain.viewshed(np.zeros((20, 20)), (10, 10)).all()
    # A 100 m wall across column 10 of a flat strip.
    wall = np.zeros((3, 30))
    # Wall cells.
    wall[:, 10] = 100.0
    # Observer at the western end of the middle row.
    visible = terrain.viewshed(wall, (1, 0), 1.0)
    # The wall is visible, the ground behind it is not.
    assert visible[1, :11].all() and not visible[1, 11:].any()
    # A distance limit hides far cells even on flat ground.
    limited = terrain.viewshed(np.zeros((2, 30)), (0, 0), 1.0, max_distance=10.0)
    # Cells 0 .. 10 of the observer's row are visible.
    assert limited[0].sum() == 11
    # The observer must stand on the grid.
    with pytest.raises(ValueError):
        # Outside the grid.
        terrain.viewshed(wall, (5, 5))


# =============================================================================
# End of module tests/unit/test_terrain.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
