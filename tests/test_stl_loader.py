"""
Unit tests for stl_drawing.io.stl_loader module.

Tests:
- Loading binary and ASCII STL files
- Format autodetection
- Vertex deduplication
- Error handling (missing files, empty files)
- Data integrity validation
"""

import numpy as np
import pytest

from stl_drawing.io.stl_loader import (
    load_stl,
    load_stl_with_info,
    detect_stl_format,
    STLLoadError,
    STLFormat,
    STLInfo,
)
from tests.conftest import assert_valid_vertices, assert_valid_faces


class TestLoadSTL:
    """Tests for load_stl function."""

    def test_load_fuel_stl(self, fuel_stl_path):
        """Test loading fuel.stl (complex geometry)."""
        vertices, faces = load_stl(str(fuel_stl_path))

        assert_valid_vertices(vertices)
        assert_valid_faces(faces, len(vertices))

        # fuel.stl has significant geometry
        assert len(vertices) > 100
        assert len(faces) > 100

    def test_load_shveller_stl(self, shveller_stl_path):
        """Test loading Shveller_16P.stl (stepped profile)."""
        vertices, faces = load_stl(str(shveller_stl_path))

        assert_valid_vertices(vertices)
        assert_valid_faces(faces, len(vertices))

        # Check approximate dimensions (751 x 160 x 64 mm in some order)
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        size = sorted(bbox_max - bbox_min)  # Sort to be axis-independent

        # Smallest dimension ~64mm, middle ~160mm, largest ~751mm
        assert 60 < size[0] < 70     # ~64mm
        assert 150 < size[1] < 170   # ~160mm
        assert 700 < size[2] < 800   # ~751mm

    def test_load_cube(self, cube_stl_path):
        """Test loading a simple cube."""
        vertices, faces = load_stl(str(cube_stl_path))

        assert_valid_vertices(vertices)
        assert_valid_faces(faces, len(vertices))

        # Cube should have 8 unique vertices
        assert len(vertices) == 8
        # Cube has 12 triangular faces (2 per side * 6 sides)
        assert len(faces) == 12

    def test_load_cylinder(self, cylinder_stl_path):
        """Test loading a cylinder."""
        vertices, faces = load_stl(str(cylinder_stl_path))

        assert_valid_vertices(vertices)
        assert_valid_faces(faces, len(vertices))

        # Cylinder should have many faces
        assert len(faces) > 50

    def test_load_ascii_stl(self, ascii_stl_path):
        """Test loading ASCII format STL."""
        vertices, faces = load_stl(str(ascii_stl_path))

        assert_valid_vertices(vertices)
        assert_valid_faces(faces, len(vertices))

        # Should load same as binary cube
        assert len(vertices) == 8
        assert len(faces) == 12

    def test_load_missing_file(self, tmp_stl_dir):
        """Test error handling for missing file."""
        with pytest.raises(STLLoadError, match="Файл не найден"):
            load_stl(str(tmp_stl_dir / "nonexistent.stl"))

    def test_load_empty_stl(self, empty_stl_path):
        """Test error handling for empty STL file."""
        with pytest.raises(STLLoadError, match="не содержит треугольников"):
            load_stl(str(empty_stl_path))

    def test_load_invalid_file(self, tmp_stl_dir):
        """Test error handling for invalid file content."""
        invalid_path = tmp_stl_dir / "invalid.stl"
        invalid_path.write_text("This is not a valid STL file")

        with pytest.raises(STLLoadError):
            load_stl(str(invalid_path))


class TestVertexDeduplication:
    """Tests for vertex deduplication logic."""

    def test_vertices_are_unique(self, cube_stl_path):
        """Test that all returned vertices are unique."""
        vertices, _ = load_stl(str(cube_stl_path))

        # Round to precision used in deduplication
        rounded = np.round(vertices, 6)
        unique_rows = np.unique(rounded, axis=0)

        assert len(unique_rows) == len(vertices)

    def test_face_indices_valid(self, cube_stl_path):
        """Test that all face indices reference valid vertices."""
        vertices, faces = load_stl(str(cube_stl_path))

        assert np.all(faces >= 0)
        assert np.all(faces < len(vertices))

    def test_no_degenerate_faces(self, cube_stl_path):
        """Test that no face has duplicate vertex indices."""
        _, faces = load_stl(str(cube_stl_path))

        for face in faces:
            assert len(set(face)) == 3, f"Degenerate face found: {face}"


class TestDataTypes:
    """Tests for correct data types."""

    def test_vertices_dtype(self, cube_stl_path):
        """Test vertices array data type."""
        vertices, _ = load_stl(str(cube_stl_path))
        assert vertices.dtype == np.float64

    def test_faces_dtype(self, cube_stl_path):
        """Test faces array data type."""
        _, faces = load_stl(str(cube_stl_path))
        assert faces.dtype == np.int32

    def test_vertices_shape(self, cube_stl_path):
        """Test vertices array shape."""
        vertices, _ = load_stl(str(cube_stl_path))
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3

    def test_faces_shape(self, cube_stl_path):
        """Test faces array shape."""
        _, faces = load_stl(str(cube_stl_path))
        assert faces.ndim == 2
        assert faces.shape[1] == 3


class TestGeometryIntegrity:
    """Tests for geometric integrity of loaded data."""

    def test_cube_dimensions(self, cube_stl_path):
        """Test cube has correct dimensions (10x10x10)."""
        vertices, _ = load_stl(str(cube_stl_path))

        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        size = bbox_max - bbox_min

        np.testing.assert_array_almost_equal(size, [10.0, 10.0, 10.0], decimal=4)

    def test_cube_centered(self, cube_stl_path):
        """Test cube is centered at origin."""
        vertices, _ = load_stl(str(cube_stl_path))

        center = vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [0.0, 0.0, 0.0], decimal=4)

    def test_cylinder_radius(self, cylinder_stl_path):
        """Test cylinder has correct radius."""
        vertices, _ = load_stl(str(cylinder_stl_path))

        # Get vertices in XY plane (exclude caps)
        z_values = vertices[:, 2]
        mid_z = (z_values.min() + z_values.max()) / 2
        side_mask = np.abs(vertices[:, 2] - mid_z) < 5.0  # within cylinder body

        if side_mask.sum() > 0:
            side_verts = vertices[side_mask]
            radii = np.sqrt(side_verts[:, 0]**2 + side_verts[:, 1]**2)
            # Should be approximately 5.0 (the radius)
            assert np.allclose(radii, 5.0, atol=0.1)


class TestFormatDetection:
    """Tests for STL format autodetection."""

    def test_detect_binary_format(self, cube_stl_path):
        """Test detection of binary STL format."""
        stl_format, _ = detect_stl_format(str(cube_stl_path))
        assert stl_format == STLFormat.BINARY

    def test_detect_ascii_format(self, ascii_stl_path):
        """Test detection of ASCII STL format."""
        stl_format, solid_name = detect_stl_format(str(ascii_stl_path))
        assert stl_format == STLFormat.ASCII
        assert solid_name == "cube"

    def test_detect_fuel_format(self, fuel_stl_path):
        """Test format detection on real model (fuel.stl)."""
        stl_format, _ = detect_stl_format(str(fuel_stl_path))
        assert stl_format in (STLFormat.BINARY, STLFormat.ASCII)

    def test_detect_missing_file(self, tmp_stl_dir):
        """Test error handling for missing file."""
        with pytest.raises(STLLoadError, match="Файл не найден"):
            detect_stl_format(str(tmp_stl_dir / "nonexistent.stl"))


class TestLoadWithInfo:
    """Tests for load_stl_with_info function."""

    def test_load_with_info_cube(self, cube_stl_path):
        """Test loading cube with metadata."""
        vertices, faces, info = load_stl_with_info(str(cube_stl_path))

        assert_valid_vertices(vertices)
        assert_valid_faces(faces, len(vertices))

        assert isinstance(info, STLInfo)
        assert info.n_triangles == 12
        assert info.n_unique_vertices == 8
        assert info.format == STLFormat.BINARY
        assert info.file_size_bytes > 0

    def test_load_with_info_ascii(self, ascii_stl_path):
        """Test loading ASCII format with metadata."""
        vertices, faces, info = load_stl_with_info(str(ascii_stl_path))

        assert info.format == STLFormat.ASCII
        assert info.solid_name == "cube"
        assert info.n_triangles == 12

    def test_stl_info_properties(self, cube_stl_path):
        """Test STLInfo computed properties."""
        _, _, info = load_stl_with_info(str(cube_stl_path))

        assert info.file_size_kb == info.file_size_bytes / 1024
        assert info.file_size_mb == info.file_size_bytes / (1024 * 1024)

    def test_load_with_info_fuel(self, fuel_stl_path):
        """Test loading fuel.stl with metadata."""
        vertices, faces, info = load_stl_with_info(str(fuel_stl_path))

        assert info.n_triangles > 1000
        assert info.n_unique_vertices > 100
        assert info.file_size_kb > 100  # fuel.stl is ~500KB
