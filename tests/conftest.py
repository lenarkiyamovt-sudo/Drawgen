"""
Pytest configuration and fixtures for STL-to-ESKD Drawing Generator.

Provides:
- STL file fixtures (fuel.stl, Shveller_16P.stl)
- Temporary directory fixtures
- Common test utilities
- Snapshot testing support
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from stl import mesh as stl_mesh

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Update snapshot files instead of comparing",
    )
    parser.addoption(
        "--ignore-missing-snapshots",
        action="store_true",
        default=False,
        help="Skip tests if snapshot files are missing",
    )


@pytest.fixture
def update_snapshots(request) -> bool:
    """Whether to update snapshots instead of comparing."""
    return request.config.getoption("--update-snapshots", default=False)


@pytest.fixture
def ignore_missing_snapshots(request) -> bool:
    """Whether to skip tests with missing snapshots."""
    return request.config.getoption("--ignore-missing-snapshots", default=False)


# ============================================================================
# STL File Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def stl_dir() -> Path:
    """Return path to project root containing STL files."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fuel_stl_path(stl_dir: Path) -> Path:
    """Path to fuel.stl test model (complex geometry, 5 cylinders)."""
    path = stl_dir / "fuel.stl"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return path


@pytest.fixture(scope="session")
def shveller_stl_path(stl_dir: Path) -> Path:
    """Path to Shveller_16P.stl test model (stepped profile, 1 cylinder)."""
    path = stl_dir / "Shveller_16P.stl"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return path


@pytest.fixture(scope="session")
def fuel_stl_data(fuel_stl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load fuel.stl and return (vertices, faces) arrays."""
    from stl_drawing.io.stl_loader import load_stl
    return load_stl(str(fuel_stl_path))


@pytest.fixture(scope="session")
def shveller_stl_data(shveller_stl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load Shveller_16P.stl and return (vertices, faces) arrays."""
    from stl_drawing.io.stl_loader import load_stl
    return load_stl(str(shveller_stl_path))


# ============================================================================
# Temporary File Fixtures
# ============================================================================

@pytest.fixture
def tmp_stl_dir(tmp_path: Path) -> Path:
    """Temporary directory for STL files created during tests."""
    return tmp_path


@pytest.fixture
def tmp_svg_path(tmp_path: Path) -> Path:
    """Temporary path for SVG output."""
    return tmp_path / "output.svg"


# ============================================================================
# Geometry Fixtures - Simple Shapes for Unit Testing
# ============================================================================

@pytest.fixture
def cube_stl_path(tmp_stl_dir: Path) -> Path:
    """Create a simple cube STL for testing."""
    path = tmp_stl_dir / "cube.stl"
    _create_cube_stl(path, size=10.0)
    return path


@pytest.fixture
def cylinder_stl_path(tmp_stl_dir: Path) -> Path:
    """Create a simple cylinder STL for testing."""
    path = tmp_stl_dir / "cylinder.stl"
    _create_cylinder_stl(path, radius=5.0, height=20.0, segments=32)
    return path


@pytest.fixture
def plate_with_hole_stl_path(tmp_stl_dir: Path) -> Path:
    """Create a plate with a hole STL for testing."""
    path = tmp_stl_dir / "plate_with_hole.stl"
    _create_plate_with_hole_stl(path, plate_size=50.0, plate_height=5.0,
                                 hole_radius=10.0, segments=24)
    return path


@pytest.fixture
def empty_stl_path(tmp_stl_dir: Path) -> Path:
    """Create an empty (0 triangles) STL for error testing."""
    path = tmp_stl_dir / "empty.stl"
    m = stl_mesh.Mesh(np.zeros(0, dtype=stl_mesh.Mesh.dtype))
    m.save(str(path))
    return path


@pytest.fixture
def ascii_stl_path(tmp_stl_dir: Path) -> Path:
    """Create ASCII format STL for format detection testing."""
    path = tmp_stl_dir / "ascii_cube.stl"
    _create_cube_stl(path, size=10.0, binary=False)
    return path


# ============================================================================
# Helper Functions for Creating Test STL Files
# ============================================================================

def _create_cube_stl(path: Path, size: float = 10.0, binary: bool = True) -> None:
    """Create a cube STL file."""
    hs = size / 2
    vertices = np.array([
        [-hs, -hs, -hs], [+hs, -hs, -hs], [+hs, +hs, -hs], [-hs, +hs, -hs],  # bottom
        [-hs, -hs, +hs], [+hs, -hs, +hs], [+hs, +hs, +hs], [-hs, +hs, +hs],  # top
    ])
    faces = [
        # bottom
        [0, 1, 2], [0, 2, 3],
        # top
        [4, 6, 5], [4, 7, 6],
        # front
        [0, 5, 1], [0, 4, 5],
        # back
        [2, 7, 3], [2, 6, 7],
        # left
        [0, 3, 7], [0, 7, 4],
        # right
        [1, 5, 6], [1, 6, 2],
    ]
    triangles = np.array([[vertices[f[0]], vertices[f[1]], vertices[f[2]]] for f in faces])
    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        m.vectors[i] = tri

    if binary:
        m.save(str(path))
    else:
        # Write ASCII format manually (numpy-stl Mode may not be available)
        with open(str(path), 'w') as f:
            f.write("solid cube\n")
            for tri in triangles:
                # Compute normal
                v0, v1, v2 = tri
                e1, e2 = v1 - v0, v2 - v0
                normal = np.cross(e1, e2)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("    outer loop\n")
                for v in tri:
                    f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid cube\n")


def _create_cylinder_stl(path: Path, radius: float = 5.0, height: float = 20.0,
                          segments: int = 32) -> None:
    """Create a cylinder STL file."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    h2 = height / 2

    # Top and bottom circles
    top = np.column_stack([radius * np.cos(angles), radius * np.sin(angles),
                           np.full(segments, h2)])
    bot = np.column_stack([radius * np.cos(angles), radius * np.sin(angles),
                           np.full(segments, -h2)])

    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        # Side faces
        triangles.append([top[i], bot[i], top[j]])
        triangles.append([bot[i], bot[j], top[j]])
        # Top cap
        triangles.append([top[i], top[j], [0, 0, h2]])
        # Bottom cap
        triangles.append([bot[i], [0, 0, -h2], bot[j]])

    triangles = np.array(triangles)
    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        m.vectors[i] = tri
    m.save(str(path))


def _create_plate_with_hole_stl(path: Path, plate_size: float = 50.0,
                                 plate_height: float = 5.0, hole_radius: float = 10.0,
                                 segments: int = 24) -> None:
    """Create a rectangular plate with a central hole."""
    hs = plate_size / 2
    hh = plate_height / 2

    triangles = []

    # Plate corners
    corners = np.array([
        [-hs, -hs, -hh], [+hs, -hs, -hh], [+hs, +hs, -hh], [-hs, +hs, -hh],
        [-hs, -hs, +hh], [+hs, -hs, +hh], [+hs, +hs, +hh], [-hs, +hs, +hh],
    ])

    # Outer faces (simplified - without hole cut)
    faces = [
        [0, 4, 1], [1, 4, 5],  # front
        [2, 6, 3], [3, 6, 7],  # back
        [0, 3, 4], [3, 7, 4],  # left
        [1, 5, 2], [2, 5, 6],  # right
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
    ]
    for f in faces:
        triangles.append([corners[f[0]], corners[f[1]], corners[f[2]]])

    # Hole (cylinder inner surface)
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    for i in range(segments):
        j = (i + 1) % segments
        a1, a2 = angles[i], angles[j]
        p1 = [hole_radius * np.cos(a1), hole_radius * np.sin(a1), hh]
        p2 = [hole_radius * np.cos(a1), hole_radius * np.sin(a1), -hh]
        p3 = [hole_radius * np.cos(a2), hole_radius * np.sin(a2), hh]
        p4 = [hole_radius * np.cos(a2), hole_radius * np.sin(a2), -hh]
        # Reversed winding for inner surface
        triangles.append([p1, p3, p2])
        triangles.append([p2, p3, p4])

    triangles = np.array(triangles)
    m = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        m.vectors[i] = tri
    m.save(str(path))


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_valid_vertices(vertices: np.ndarray) -> None:
    """Assert that vertices array is valid."""
    assert isinstance(vertices, np.ndarray)
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert vertices.dtype == np.float64
    assert not np.any(np.isnan(vertices))
    assert not np.any(np.isinf(vertices))


def assert_valid_faces(faces: np.ndarray, n_vertices: int) -> None:
    """Assert that faces array is valid."""
    assert isinstance(faces, np.ndarray)
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert faces.dtype == np.int32
    assert np.all(faces >= 0)
    assert np.all(faces < n_vertices)


def assert_bbox_approx(vertices: np.ndarray, expected_size: Tuple[float, float, float],
                        tolerance: float = 0.01) -> None:
    """Assert that bounding box matches expected size within tolerance."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    actual_size = bbox_max - bbox_min
    for i, (actual, expected) in enumerate(zip(actual_size, expected_size)):
        assert abs(actual - expected) < expected * tolerance, \
            f"Axis {i}: expected {expected}, got {actual}"
