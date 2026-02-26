"""
Unit tests for stl_drawing.io.validator module.

Tests:
- Manifold validation
- Closed mesh detection
- Degenerate face detection
- Validation report generation
"""

import numpy as np
import pytest

from stl_drawing.io.validator import (
    validate_mesh,
    validate_stl_file,
    ValidationReport,
    ValidationSeverity,
)


class TestValidateMesh:
    """Tests for validate_mesh function."""

    def test_valid_cube(self, cube_stl_path):
        """Test validation of a valid cube mesh."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        report = validate_mesh(vertices, faces)

        assert report.is_valid
        assert report.is_manifold
        assert report.is_closed
        assert not report.has_degenerate_faces
        assert report.n_vertices == 8
        assert report.n_faces == 12
        assert report.n_boundary_edges == 0
        assert report.n_non_manifold_edges == 0

    def test_valid_cylinder(self, cylinder_stl_path):
        """Test validation of a valid cylinder mesh."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cylinder_stl_path))

        report = validate_mesh(vertices, faces)

        assert report.is_valid
        assert report.is_manifold
        assert report.n_degenerate_faces == 0

    def test_fuel_stl_validation(self, fuel_stl_path):
        """Test validation of fuel.stl (real model)."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(fuel_stl_path))

        report = validate_mesh(vertices, faces)

        # Should be valid even if not perfect
        assert report.n_vertices > 0
        assert report.n_faces > 0
        # Check report generation works
        summary = report.summary()
        assert "Mesh Validation Report" in summary

    def test_shveller_stl_validation(self, shveller_stl_path):
        """Test validation of Shveller_16P.stl."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(shveller_stl_path))

        report = validate_mesh(vertices, faces)

        assert report.n_vertices > 0
        assert report.n_faces > 0


class TestValidateFile:
    """Tests for validate_stl_file convenience function."""

    def test_validate_cube_file(self, cube_stl_path):
        """Test file-based validation."""
        report = validate_stl_file(str(cube_stl_path))

        assert isinstance(report, ValidationReport)
        assert report.is_valid
        assert report.n_faces == 12

    def test_validate_fuel_file(self, fuel_stl_path):
        """Test file-based validation of fuel.stl."""
        report = validate_stl_file(str(fuel_stl_path))

        assert isinstance(report, ValidationReport)
        assert report.n_faces > 1000


class TestDegenerateFaces:
    """Tests for degenerate face detection."""

    def test_detect_degenerate_triangle(self):
        """Test detection of a degenerate (zero-area) triangle."""
        # Create a mesh with one degenerate face (collinear vertices)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],  # Collinear with 0 and 1
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        faces = np.array([
            [0, 1, 2],  # Degenerate (collinear)
            [0, 1, 3],  # Valid
            [0, 1, 4],  # Valid
        ], dtype=np.int32)

        report = validate_mesh(vertices, faces)

        assert report.has_degenerate_faces
        assert report.n_degenerate_faces == 1

    def test_no_degenerate_in_cube(self, cube_stl_path):
        """Test that a proper cube has no degenerate faces."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        report = validate_mesh(vertices, faces)

        assert not report.has_degenerate_faces
        assert report.n_degenerate_faces == 0


class TestNonManifoldDetection:
    """Tests for non-manifold edge detection."""

    def test_detect_non_manifold_edge(self):
        """Test detection of non-manifold edge (>2 faces sharing an edge)."""
        # Create a mesh where edge 0-1 is shared by 3 faces
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, -1, 0],
            [0.5, 0, 1],
        ], dtype=np.float64)

        faces = np.array([
            [0, 1, 2],  # Top face
            [0, 1, 3],  # Bottom face
            [0, 1, 4],  # Third face sharing edge 0-1
        ], dtype=np.int32)

        report = validate_mesh(vertices, faces)

        assert not report.is_manifold
        assert report.n_non_manifold_edges == 1
        assert any(i.code == "NON_MANIFOLD_EDGES" for i in report.issues)


class TestBoundaryEdges:
    """Tests for boundary edge (open mesh) detection."""

    def test_detect_open_mesh(self):
        """Test detection of boundary edges in an open mesh."""
        # Single triangle - all edges are boundary
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float64)

        faces = np.array([
            [0, 1, 2],
        ], dtype=np.int32)

        report = validate_mesh(vertices, faces)

        assert not report.is_closed
        assert report.n_boundary_edges == 3
        assert any(i.code == "BOUNDARY_EDGES" for i in report.issues)

    def test_closed_cube(self, cube_stl_path):
        """Test that a cube is detected as closed."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        report = validate_mesh(vertices, faces)

        assert report.is_closed
        assert report.n_boundary_edges == 0


class TestValidationReport:
    """Tests for ValidationReport class."""

    def test_report_summary(self, cube_stl_path):
        """Test report summary generation."""
        report = validate_stl_file(str(cube_stl_path))

        summary = report.summary()

        assert "Mesh Validation Report" in summary
        assert "Vertices: 8" in summary
        assert "Faces: 12" in summary
        assert "VALID" in summary

    def test_report_warnings_property(self):
        """Test warnings property filters correctly."""
        # Create mesh with boundary edges (warning)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        report = validate_mesh(vertices, faces)

        assert len(report.warnings) > 0
        assert all(w.severity == ValidationSeverity.WARNING for w in report.warnings)

    def test_report_errors_property(self):
        """Test errors property filters correctly."""
        # Create non-manifold mesh (error)
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)

        report = validate_mesh(vertices, faces)

        assert len(report.errors) > 0
        assert all(e.severity == ValidationSeverity.ERROR for e in report.errors)
