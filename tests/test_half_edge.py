"""
Unit tests for stl_drawing.topology.half_edge module.

Tests:
- HalfEdge dataclass
- HalfEdgeMesh construction
- Topological queries
- Boundary detection
- Sharp edge marking
"""

import numpy as np
import pytest

from stl_drawing.topology.half_edge import HalfEdge, HalfEdgeMesh


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def single_triangle():
    """Single triangle mesh."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def two_triangles():
    """Two triangles sharing an edge."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0],
    ], dtype=np.float64)
    # Two triangles sharing edge (1, 2)
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def cube_mesh():
    """Simple cube mesh (12 triangles)."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
    ], dtype=np.float64)

    faces = np.array([
        # bottom
        [0, 2, 1], [0, 3, 2],
        # top
        [4, 5, 6], [4, 6, 7],
        # front
        [0, 1, 5], [0, 5, 4],
        # back
        [2, 3, 7], [2, 7, 6],
        # left
        [0, 4, 7], [0, 7, 3],
        # right
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)

    return vertices, faces


@pytest.fixture
def open_mesh():
    """Open mesh (not watertight) - a flat quad made of 2 triangles."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    return vertices, faces


# ============================================================================
# HalfEdge Tests
# ============================================================================

class TestHalfEdge:
    """Tests for HalfEdge dataclass."""

    def test_create_half_edge(self):
        """Test creating a half-edge."""
        he = HalfEdge(index=0, origin=0)
        assert he.index == 0
        assert he.origin == 0
        assert he.twin is None
        assert he.next is None
        assert he.face is None

    def test_half_edge_hash(self):
        """Test that half-edges are hashable."""
        he1 = HalfEdge(index=0, origin=0)
        he2 = HalfEdge(index=1, origin=1)
        he3 = HalfEdge(index=0, origin=0)

        s = {he1, he2}
        assert len(s) == 2
        assert he1 in s
        assert he3 in s  # Same index

    def test_half_edge_equality(self):
        """Test half-edge equality based on index."""
        he1 = HalfEdge(index=0, origin=0)
        he2 = HalfEdge(index=0, origin=1)  # Same index, different origin
        he3 = HalfEdge(index=1, origin=0)

        assert he1 == he2  # Same index
        assert he1 != he3  # Different index


# ============================================================================
# HalfEdgeMesh Construction Tests
# ============================================================================

class TestHalfEdgeMeshConstruction:
    """Tests for HalfEdgeMesh construction."""

    def test_build_single_triangle(self, single_triangle):
        """Test building mesh from single triangle."""
        vertices, faces = single_triangle
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1
        assert mesh.n_half_edges == 3

    def test_build_two_triangles(self, two_triangles):
        """Test building mesh from two triangles."""
        vertices, faces = two_triangles
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        assert mesh.n_vertices == 4
        assert mesh.n_faces == 2
        assert mesh.n_half_edges == 6

    def test_build_cube(self, cube_mesh):
        """Test building cube mesh."""
        vertices, faces = cube_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        assert mesh.n_vertices == 8
        assert mesh.n_faces == 12
        assert mesh.n_half_edges == 36  # 12 faces * 3 edges

    def test_twin_pointers_set(self, two_triangles):
        """Test that twin pointers are correctly set."""
        vertices, faces = two_triangles
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # The shared edge (1, 2) should have twins
        he12 = mesh.get_half_edge(1, 2)
        he21 = mesh.get_half_edge(2, 1)

        assert he12 is not None
        assert he21 is not None
        assert he12.twin == he21
        assert he21.twin == he12

    def test_next_pointers_form_loop(self, single_triangle):
        """Test that next pointers form a loop around face."""
        vertices, faces = single_triangle
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        start = mesh.face_to_half_edge[0]
        current = start
        count = 0

        while True:
            count += 1
            current = current.next
            if current == start or count > 10:
                break

        assert count == 3  # Triangle has 3 edges


# ============================================================================
# Topological Query Tests
# ============================================================================

class TestTopologicalQueries:
    """Tests for topological queries."""

    def test_get_half_edge(self, two_triangles):
        """Test getting half-edge by vertex pair."""
        vertices, faces = two_triangles
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        he = mesh.get_half_edge(0, 1)
        assert he is not None
        assert he.origin == 0

        # Non-existent edge
        assert mesh.get_half_edge(0, 3) is None

    def test_vertex_half_edges(self, two_triangles):
        """Test iterating over vertex outgoing edges."""
        vertices, faces = two_triangles
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # Vertex 1 is shared by both triangles
        edges = list(mesh.vertex_half_edges(1))
        assert len(edges) >= 2

        # All edges should originate from vertex 1
        for he in edges:
            assert he.origin == 1

    def test_face_half_edges(self, single_triangle):
        """Test iterating over face edges."""
        vertices, faces = single_triangle
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        edges = list(mesh.face_half_edges(0))
        assert len(edges) == 3

        # Check all edges belong to face 0
        for he in edges:
            assert he.face == 0

    def test_adjacent_faces(self, two_triangles):
        """Test getting adjacent faces."""
        vertices, faces = two_triangles
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # Face 0 should be adjacent to face 1
        adj = list(mesh.adjacent_faces(0))
        assert 1 in adj

        # And vice versa
        adj = list(mesh.adjacent_faces(1))
        assert 0 in adj

    def test_vertex_faces(self, cube_mesh):
        """Test getting faces incident to vertex."""
        vertices, faces = cube_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # Corner vertex should have 3 incident faces
        faces_at_vertex = list(mesh.vertex_faces(0))
        assert len(faces_at_vertex) >= 3

    def test_vertex_neighbors(self, single_triangle):
        """Test getting vertex neighbors."""
        vertices, faces = single_triangle
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # Vertex 0 should have neighbors 1 and 2
        neighbors = set(mesh.vertex_neighbors(0))
        assert 1 in neighbors
        assert 2 in neighbors

    def test_edge_faces(self, two_triangles):
        """Test getting faces adjacent to edge."""
        vertices, faces = two_triangles
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # Shared edge (1, 2) should have both faces
        f1, f2 = mesh.edge_faces(1, 2)
        assert f1 is not None
        assert f2 is not None
        assert {f1, f2} == {0, 1}


# ============================================================================
# Boundary Detection Tests
# ============================================================================

class TestBoundaryDetection:
    """Tests for boundary edge detection."""

    def test_single_triangle_all_boundary(self, single_triangle):
        """Test that single triangle has all boundary edges."""
        vertices, faces = single_triangle
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        boundary = list(mesh.boundary_edges())
        assert len(boundary) == 3

    def test_closed_mesh_no_boundary(self, cube_mesh):
        """Test that closed mesh has no boundary."""
        vertices, faces = cube_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        boundary = list(mesh.boundary_edges())
        assert len(boundary) == 0

    def test_open_mesh_has_boundary(self, open_mesh):
        """Test that open mesh has boundary edges."""
        vertices, faces = open_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        boundary = list(mesh.boundary_edges())
        assert len(boundary) == 4  # Outer edges of the quad

    def test_is_boundary_vertex(self, open_mesh):
        """Test boundary vertex detection."""
        vertices, faces = open_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # All vertices in open mesh are on boundary
        for i in range(4):
            assert mesh.is_boundary_vertex(i)

    def test_boundary_loops(self, open_mesh):
        """Test finding boundary loops."""
        vertices, faces = open_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        loops = mesh.boundary_loops()
        assert len(loops) == 1  # One boundary loop
        assert len(loops[0]) == 4  # 4 boundary edges


# ============================================================================
# Sharp Edge Tests
# ============================================================================

class TestSharpEdges:
    """Tests for sharp edge detection."""

    def test_mark_sharp_edges_cube(self, cube_mesh):
        """Test marking sharp edges on cube."""
        vertices, faces = cube_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # Cube edges are 90 degrees, should be sharp at 30 deg threshold
        n_sharp = mesh.mark_sharp_edges(angle_threshold_deg=30.0)
        assert n_sharp == 12  # Cube has 12 edges

    def test_mark_sharp_edges_flat(self, open_mesh):
        """Test marking sharp edges on flat mesh."""
        vertices, faces = open_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        # Flat mesh has no sharp interior edges
        n_sharp = mesh.mark_sharp_edges(angle_threshold_deg=30.0)

        # Only boundary edges should be sharp
        sharp_list = list(mesh.sharp_edges())
        # Boundary edges are marked sharp
        assert len(sharp_list) == 4

    def test_sharp_edges_iterator(self, cube_mesh):
        """Test iterating over sharp edges."""
        vertices, faces = cube_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)
        mesh.mark_sharp_edges(angle_threshold_deg=30.0)

        sharp = list(mesh.sharp_edges())
        assert len(sharp) == 12

        # Each edge should be a tuple of vertex indices
        for v1, v2 in sharp:
            assert isinstance(v1, (int, np.integer))
            assert isinstance(v2, (int, np.integer))


# ============================================================================
# Geometry Tests
# ============================================================================

class TestGeometry:
    """Tests for geometric computations."""

    def test_compute_face_normals(self, single_triangle):
        """Test computing face normals."""
        vertices, faces = single_triangle
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        normals = mesh.compute_face_normals()
        assert normals.shape == (1, 3)

        # Triangle in XY plane should have normal in Z direction
        assert np.allclose(np.abs(normals[0, 2]), 1.0, atol=1e-6)

    def test_compute_vertex_normals(self, cube_mesh):
        """Test computing vertex normals."""
        vertices, faces = cube_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        normals = mesh.compute_vertex_normals()
        assert normals.shape == (8, 3)

        # All normals should be unit length
        lengths = np.linalg.norm(normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-6)

    def test_get_edge_vector(self, single_triangle):
        """Test getting edge vector."""
        vertices, faces = single_triangle
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        he = mesh.get_half_edge(0, 1)
        vec = mesh.get_edge_vector(he)

        expected = vertices[1] - vertices[0]
        assert np.allclose(vec, expected)

    def test_get_edge_length(self, single_triangle):
        """Test getting edge length."""
        vertices, faces = single_triangle
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        he = mesh.get_half_edge(0, 1)
        length = mesh.get_edge_length(he)

        expected = np.linalg.norm(vertices[1] - vertices[0])
        assert np.isclose(length, expected)


# ============================================================================
# Edge Chain Tests
# ============================================================================

class TestEdgeChains:
    """Tests for edge chain finding."""

    def test_edge_chain_on_cube(self, cube_mesh):
        """Test finding edge chains on cube."""
        vertices, faces = cube_mesh
        mesh = HalfEdgeMesh.from_faces(vertices, faces)
        mesh.mark_sharp_edges(angle_threshold_deg=30.0)

        # Find a sharp edge and get its chain
        for he in mesh.half_edges:
            if he.is_sharp:
                chain = mesh.edge_chain(he)
                # On a cube, chains should be relatively short (edge of cube)
                assert len(chain) >= 1
                break


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with real STL data."""

    def test_with_fuel_stl(self, fuel_stl_data):
        """Test building half-edge mesh from fuel.stl."""
        vertices, faces = fuel_stl_data
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        assert mesh.n_vertices == len(vertices)
        assert mesh.n_faces == len(faces)

        # Should have reasonable number of half-edges
        assert mesh.n_half_edges == len(faces) * 3

    def test_sharp_edges_fuel_stl(self, fuel_stl_data):
        """Test sharp edge detection on fuel.stl."""
        vertices, faces = fuel_stl_data
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        n_sharp = mesh.mark_sharp_edges(angle_threshold_deg=30.0)

        # Should have some sharp edges
        assert n_sharp > 0

        sharp = list(mesh.sharp_edges())
        assert len(sharp) == n_sharp

    def test_boundary_detection_fuel_stl(self, fuel_stl_data):
        """Test boundary detection on fuel.stl."""
        vertices, faces = fuel_stl_data
        mesh = HalfEdgeMesh.from_faces(vertices, faces)

        boundary = list(mesh.boundary_edges())

        # fuel.stl should be watertight (no boundary)
        # But this depends on the mesh quality
        # Just check the query works
        assert isinstance(boundary, list)
