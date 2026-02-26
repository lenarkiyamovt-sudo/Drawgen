"""
Unit tests for stl_drawing.geometry.mesh_stats module.

Tests:
- Bounding box calculation
- Mesh statistics (area, volume, center)
- Edge counting
- Watertight detection
"""

import numpy as np
import pytest

from stl_drawing.geometry.mesh_stats import (
    BoundingBox,
    MeshStatistics,
    calculate_bounding_box,
    calculate_face_areas,
    calculate_surface_area,
    calculate_volume,
    calculate_center_of_mass,
    count_edges,
    calculate_mesh_statistics,
    get_oriented_bounding_box_axes,
    compare_mesh_statistics,
)


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_dimensions(self):
        """Test bounding box dimensions calculation."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([10.0, 20.0, 30.0])
        )
        dims = bbox.dimensions
        assert dims[0] == 10.0  # width
        assert dims[1] == 20.0  # height
        assert dims[2] == 30.0  # depth

    def test_center(self):
        """Test bounding box center calculation."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([10.0, 20.0, 30.0])
        )
        center = bbox.center
        assert center[0] == 5.0
        assert center[1] == 10.0
        assert center[2] == 15.0

    def test_volume(self):
        """Test bounding box volume calculation."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([2.0, 3.0, 4.0])
        )
        assert bbox.volume == 24.0

    def test_diagonal(self):
        """Test bounding box diagonal calculation."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([3.0, 4.0, 0.0])
        )
        assert bbox.diagonal == 5.0  # 3-4-5 triangle

    def test_max_min_dimension(self):
        """Test max and min dimension properties."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([5.0, 10.0, 15.0])
        )
        assert bbox.max_dimension == 15.0
        assert bbox.min_dimension == 5.0

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([10.0, 20.0, 10.0])
        )
        assert bbox.aspect_ratio == 2.0

    def test_contains_point(self):
        """Test point containment check."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([10.0, 10.0, 10.0])
        )
        assert bbox.contains_point(np.array([5.0, 5.0, 5.0]))
        assert not bbox.contains_point(np.array([15.0, 5.0, 5.0]))

    def test_intersects(self):
        """Test bounding box intersection check."""
        bbox1 = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([10.0, 10.0, 10.0])
        )
        bbox2 = BoundingBox(
            min_point=np.array([5.0, 5.0, 5.0]),
            max_point=np.array([15.0, 15.0, 15.0])
        )
        bbox3 = BoundingBox(
            min_point=np.array([20.0, 20.0, 20.0]),
            max_point=np.array([30.0, 30.0, 30.0])
        )
        assert bbox1.intersects(bbox2)
        assert not bbox1.intersects(bbox3)

    def test_expand(self):
        """Test bounding box expansion."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([10.0, 10.0, 10.0])
        )
        expanded = bbox.expand(5.0)
        assert np.allclose(expanded.min_point, [-5.0, -5.0, -5.0])
        assert np.allclose(expanded.max_point, [15.0, 15.0, 15.0])

    def test_to_dict(self):
        """Test dictionary serialization."""
        bbox = BoundingBox(
            min_point=np.array([0.0, 0.0, 0.0]),
            max_point=np.array([10.0, 10.0, 10.0])
        )
        d = bbox.to_dict()
        assert 'min' in d
        assert 'max' in d
        assert 'dimensions' in d
        assert 'volume' in d


class TestCalculateBoundingBox:
    """Tests for calculate_bounding_box function."""

    def test_simple_vertices(self):
        """Test bounding box for simple vertices."""
        vertices = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [-1, -2, -3],
        ], dtype=np.float64)

        bbox = calculate_bounding_box(vertices)

        assert np.allclose(bbox.min_point, [-1, -2, -3])
        assert np.allclose(bbox.max_point, [1, 2, 3])

    def test_empty_vertices(self):
        """Test bounding box for empty vertices."""
        vertices = np.array([], dtype=np.float64).reshape(0, 3)
        bbox = calculate_bounding_box(vertices)
        assert np.allclose(bbox.min_point, [0, 0, 0])

    def test_cube_vertices(self, cube_stl_path):
        """Test bounding box for cube."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        bbox = calculate_bounding_box(vertices)

        # Cube should be 10x10x10
        assert np.allclose(bbox.dimensions, [10, 10, 10], atol=0.1)


class TestFaceAreas:
    """Tests for calculate_face_areas function."""

    def test_unit_triangle(self):
        """Test area of unit right triangle."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        areas = calculate_face_areas(vertices, faces)

        assert len(areas) == 1
        assert np.isclose(areas[0], 0.5)

    def test_empty_faces(self):
        """Test with no faces."""
        vertices = np.array([[0, 0, 0]], dtype=np.float64)
        faces = np.array([], dtype=np.int32).reshape(0, 3)

        areas = calculate_face_areas(vertices, faces)
        assert len(areas) == 0


class TestSurfaceArea:
    """Tests for calculate_surface_area function."""

    def test_cube_surface_area(self, cube_stl_path):
        """Test surface area of cube."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        area = calculate_surface_area(vertices, faces)

        # 10x10x10 cube: 6 faces * 100 = 600 mm^2
        assert np.isclose(area, 600, rtol=0.01)


class TestVolume:
    """Tests for calculate_volume function."""

    def test_cube_volume(self, cube_stl_path):
        """Test volume of cube."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        volume = calculate_volume(vertices, faces)

        # 10x10x10 cube: 1000 mm^3
        assert np.isclose(abs(volume), 1000, rtol=0.01)

    def test_empty_mesh_volume(self):
        """Test volume of empty mesh."""
        vertices = np.array([], dtype=np.float64).reshape(0, 3)
        faces = np.array([], dtype=np.int32).reshape(0, 3)

        volume = calculate_volume(vertices, faces)
        assert volume == 0.0


class TestCenterOfMass:
    """Tests for calculate_center_of_mass function."""

    def test_centered_cube(self, cube_stl_path):
        """Test center of mass for centered cube."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        center = calculate_center_of_mass(vertices, faces)

        # Centered cube should have center at origin
        assert np.allclose(center, [0, 0, 0], atol=0.1)

    def test_single_triangle(self):
        """Test center of single triangle."""
        vertices = np.array([
            [0, 0, 0],
            [3, 0, 0],
            [0, 3, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        center = calculate_center_of_mass(vertices, faces)

        # Centroid of triangle
        assert np.allclose(center, [1, 1, 0])


class TestCountEdges:
    """Tests for count_edges function."""

    def test_single_triangle(self):
        """Test edge count for single triangle."""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        assert count_edges(faces) == 3

    def test_two_triangles_shared_edge(self):
        """Test edge count for two triangles sharing an edge."""
        # Two triangles sharing edge 0-1
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
        ], dtype=np.int32)
        # Edges: 0-1 (shared), 0-2, 1-2, 0-3, 1-3 = 5 unique edges
        assert count_edges(faces) == 5

    def test_empty_faces(self):
        """Test edge count for empty faces."""
        faces = np.array([], dtype=np.int32).reshape(0, 3)
        assert count_edges(faces) == 0


class TestMeshStatistics:
    """Tests for MeshStatistics class and calculate_mesh_statistics function."""

    def test_cube_statistics(self, cube_stl_path):
        """Test statistics for cube."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        stats = calculate_mesh_statistics(vertices, faces)

        assert stats.n_vertices == 8
        assert stats.n_faces == 12
        assert stats.is_watertight
        assert np.isclose(stats.surface_area, 600, rtol=0.01)
        assert np.isclose(abs(stats.volume), 1000, rtol=0.01)

    def test_statistics_summary(self, cube_stl_path):
        """Test summary generation."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        stats = calculate_mesh_statistics(vertices, faces)
        summary = stats.summary()

        assert "Vertices:" in summary
        assert "Faces:" in summary
        assert "Surface Area:" in summary

    def test_statistics_to_dict(self, cube_stl_path):
        """Test dictionary conversion."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        stats = calculate_mesh_statistics(vertices, faces)
        d = stats.to_dict()

        assert 'n_vertices' in d
        assert 'n_faces' in d
        assert 'bbox' in d
        assert 'surface_area_mm2' in d

    def test_fuel_statistics(self, fuel_stl_path):
        """Test statistics for fuel.stl (real model)."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(fuel_stl_path))

        stats = calculate_mesh_statistics(vertices, faces)

        assert stats.n_vertices > 1000
        assert stats.n_faces > 2000
        assert stats.surface_area > 0
        # Dimensions should be reasonable
        dims = stats.dimensions
        assert all(d > 0 for d in dims)

    def test_euler_characteristic_cube(self, cube_stl_path):
        """Test Euler characteristic for closed mesh."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        stats = calculate_mesh_statistics(vertices, faces)

        # For closed mesh: V - E + F = 2
        assert stats.euler_characteristic == 2


class TestOrientedBoundingBox:
    """Tests for get_oriented_bounding_box_axes function."""

    def test_obb_cube(self, cube_stl_path):
        """Test OBB for axis-aligned cube."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        center, axes, half_extents = get_oriented_bounding_box_axes(vertices)

        # Center should be near origin for centered cube
        assert np.allclose(center, [0, 0, 0], atol=0.1)
        # Half extents should be ~5 for 10x10x10 cube
        assert all(np.isclose(he, 5, rtol=0.1) for he in half_extents)

    def test_obb_empty(self):
        """Test OBB for empty vertices."""
        vertices = np.array([], dtype=np.float64).reshape(0, 3)
        center, axes, half_extents = get_oriented_bounding_box_axes(vertices)
        assert np.allclose(center, [0, 0, 0])


class TestCompareMeshStatistics:
    """Tests for compare_mesh_statistics function."""

    def test_comparison_report(self, cube_stl_path, cylinder_stl_path):
        """Test comparison report generation."""
        from stl_drawing.io.stl_loader import load_stl

        v1, f1 = load_stl(str(cube_stl_path))
        v2, f2 = load_stl(str(cylinder_stl_path))

        stats1 = calculate_mesh_statistics(v1, f1)
        stats2 = calculate_mesh_statistics(v2, f2)

        report = compare_mesh_statistics(
            stats1, stats2,
            name1="Cube", name2="Cylinder"
        )

        assert "Cube" in report
        assert "Cylinder" in report
        assert "Vertices" in report
        assert "Faces" in report
