"""Tests for room_mode_optimizer.geometry."""
import numpy as np
import pytest
from room_mode_optimizer.geometry import (
    bisector_filter,
    min_wall_distances,
    points_in_polygon,
    polygon_area_perimeter,
    room_y_range_at_x,
)

# Simple 4x3 rectangle with origin at (0,0)
RECT = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]

# L-shape: 4x3 rectangle with a 2x1 notch cut from top-right
L_SHAPE = [
    (0.0, 0.0), (4.0, 0.0), (4.0, 2.0),
    (2.0, 2.0), (2.0, 3.0), (0.0, 3.0),
]


class TestPointsInPolygon:
    def test_center_inside_rect(self):
        pts = np.array([[2.0, 1.5]])
        assert points_in_polygon(pts, RECT)[0]

    def test_outside_rect(self):
        pts = np.array([[5.0, 1.5], [-1.0, 1.5], [2.0, 4.0]])
        result = points_in_polygon(pts, RECT)
        assert not result.any()

    def test_multiple_points(self):
        pts = np.array([
            [1.0, 1.0],   # inside
            [2.0, 2.0],   # inside
            [5.0, 5.0],   # outside
        ])
        result = points_in_polygon(pts, RECT)
        assert result[0] and result[1] and not result[2]

    def test_l_shape_inside(self):
        pts = np.array([
            [1.0, 1.0],   # inside main body
            [1.0, 2.5],   # inside the arm
        ])
        result = points_in_polygon(pts, L_SHAPE)
        assert result[0] and result[1]

    def test_l_shape_outside_notch(self):
        # Point in the notch area (top-right) should be outside
        pts = np.array([[3.0, 2.5]])
        result = points_in_polygon(pts, L_SHAPE)
        assert not result[0]


class TestPolygonAreaPerimeter:
    def test_rectangle_area(self):
        area, _ = polygon_area_perimeter(RECT)
        assert area == pytest.approx(12.0, abs=1e-6)

    def test_rectangle_perimeter(self):
        _, perim = polygon_area_perimeter(RECT)
        assert perim == pytest.approx(14.0, abs=1e-6)

    def test_l_shape_area(self):
        # 4x2 + 2x1 = 10
        area, _ = polygon_area_perimeter(L_SHAPE)
        assert area == pytest.approx(10.0, abs=1e-6)


class TestMinWallDistances:
    def test_center_of_rectangle(self):
        pts = np.array([[2.0, 1.5]])
        d = min_wall_distances(pts, RECT)
        assert d[0] == pytest.approx(1.5, abs=1e-6)

    def test_near_wall(self):
        pts = np.array([[0.1, 1.5]])
        d = min_wall_distances(pts, RECT)
        assert d[0] == pytest.approx(0.1, abs=1e-6)

    def test_corner_point(self):
        pts = np.array([[0.5, 0.5]])
        d = min_wall_distances(pts, RECT)
        assert d[0] == pytest.approx(0.5, abs=1e-6)


class TestBisectorFilter:
    def test_midpoint_passes(self):
        spk_l = (0.0, 0.0)
        spk_r = (0.0, 4.0)
        # Midpoint is (0, 2) - equidistant from both speakers
        pts = np.array([[0.0, 2.0]])
        result = bisector_filter(pts, spk_l, spk_r, tolerance=0.1)
        assert result[0]

    def test_off_center_fails(self):
        spk_l = (0.0, 0.0)
        spk_r = (0.0, 4.0)
        # Point closer to spk_l
        pts = np.array([[0.0, 0.5]])
        result = bisector_filter(pts, spk_l, spk_r, tolerance=0.1)
        assert not result[0]

    def test_within_tolerance(self):
        spk_l = (0.0, 0.0)
        spk_r = (0.0, 4.0)
        # Slightly off the bisector but within tolerance
        pts = np.array([[0.0, 2.05]])
        result = bisector_filter(pts, spk_l, spk_r, tolerance=0.15)
        assert result[0]


class TestRoomYRangeAtX:
    def test_rectangle_center(self):
        result = room_y_range_at_x(2.0, RECT)
        assert result is not None
        y_min, y_max = result
        assert y_min == pytest.approx(0.0, abs=1e-6)
        assert y_max == pytest.approx(3.0, abs=1e-6)

    def test_outside_room(self):
        result = room_y_range_at_x(5.0, RECT)
        assert result is None

    def test_l_shape_narrow_part(self):
        # At x=1.0, the L-shape spans full height 0..3
        result = room_y_range_at_x(1.0, L_SHAPE)
        assert result is not None
        y_min, y_max = result
        assert y_min == pytest.approx(0.0, abs=1e-6)
        assert y_max == pytest.approx(3.0, abs=1e-6)

    def test_l_shape_wide_part(self):
        # At x=3.0, the L-shape only spans 0..2
        result = room_y_range_at_x(3.0, L_SHAPE)
        assert result is not None
        y_min, y_max = result
        assert y_min == pytest.approx(0.0, abs=1e-6)
        assert y_max == pytest.approx(2.0, abs=1e-6)
