"""
Half-edge mesh data structure for efficient topological queries.

The half-edge (or doubly-connected edge list) data structure provides O(1) access to:
- Adjacent faces from an edge
- All edges around a vertex
- All edges around a face
- Twin (opposite direction) half-edge

This is essential for operations like:
- Finding boundary edges (edges with no twin)
- Walking around vertices/faces
- Computing edge connectivity
- Feature edge detection

Usage:
    from stl_drawing.topology.half_edge import HalfEdgeMesh

    mesh = HalfEdgeMesh.from_faces(vertices, faces)

    # Get all edges around a vertex
    for he in mesh.vertex_half_edges(vertex_idx):
        print(f"Edge to vertex {he.target}")

    # Find boundary edges
    for he in mesh.boundary_edges():
        print(f"Boundary: {he.origin} -> {he.target}")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class HalfEdge:
    """A half-edge in the mesh.

    Each edge in the mesh is represented by two half-edges going in opposite
    directions. A half-edge stores:
    - origin: vertex index where this half-edge starts
    - target: vertex index where this half-edge ends (computed property)
    - twin: the half-edge going in the opposite direction (None for boundary)
    - next: the next half-edge around the same face
    - prev: the previous half-edge around the same face
    - face: the face this half-edge belongs to (None for boundary)
    """
    index: int  # Unique index of this half-edge
    origin: int  # Vertex index where this half-edge starts

    twin: Optional['HalfEdge'] = None  # Opposite half-edge (None if boundary)
    next: Optional['HalfEdge'] = None  # Next half-edge around face
    prev: Optional['HalfEdge'] = None  # Previous half-edge around face
    face: Optional[int] = None  # Face index (None if boundary half-edge)

    # Optional attributes for edge classification
    is_sharp: bool = False  # Is this a sharp/feature edge?
    is_boundary: bool = False  # Is this a boundary edge?

    @property
    def target(self) -> int:
        """Get the target vertex index."""
        if self.next is not None:
            return self.next.origin
        raise ValueError("Half-edge has no next pointer")

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        if isinstance(other, HalfEdge):
            return self.index == other.index
        return False


@dataclass
class HalfEdgeMesh:
    """Half-edge mesh data structure.

    Provides efficient topological queries on a triangle mesh.
    """
    vertices: NDArray[np.float64]  # (N, 3) vertex positions
    faces: NDArray[np.int32]  # (M, 3) face vertex indices

    half_edges: List[HalfEdge] = field(default_factory=list)
    vertex_to_half_edge: Dict[int, HalfEdge] = field(default_factory=dict)
    face_to_half_edge: Dict[int, HalfEdge] = field(default_factory=dict)

    # Edge lookup: (v1, v2) -> half-edge from v1 to v2
    _edge_map: Dict[Tuple[int, int], HalfEdge] = field(default_factory=dict)

    @classmethod
    def from_faces(
        cls,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int32],
    ) -> 'HalfEdgeMesh':
        """Build half-edge mesh from vertices and faces.

        Args:
            vertices: (N, 3) array of vertex positions
            faces: (M, 3) array of face vertex indices

        Returns:
            HalfEdgeMesh instance
        """
        mesh = cls(vertices=vertices, faces=faces)
        mesh._build_structure()
        return mesh

    def _build_structure(self) -> None:
        """Build the half-edge structure from faces."""
        n_faces = len(self.faces)
        he_index = 0
        edge_map: Dict[Tuple[int, int], HalfEdge] = {}

        # First pass: create half-edges for each face
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face
            face_vertices = [v0, v1, v2]

            # Create 3 half-edges for this triangle
            face_half_edges = []
            for i in range(3):
                origin = face_vertices[i]
                target = face_vertices[(i + 1) % 3]

                he = HalfEdge(
                    index=he_index,
                    origin=origin,
                    face=face_idx,
                )
                he_index += 1
                face_half_edges.append(he)

                # Store in edge map
                edge_map[(origin, target)] = he

            # Link next/prev pointers around the face
            for i in range(3):
                face_half_edges[i].next = face_half_edges[(i + 1) % 3]
                face_half_edges[i].prev = face_half_edges[(i - 1) % 3]

            # Store face -> half-edge mapping
            self.face_to_half_edge[face_idx] = face_half_edges[0]

            # Store vertex -> half-edge mapping (one per vertex)
            for he in face_half_edges:
                if he.origin not in self.vertex_to_half_edge:
                    self.vertex_to_half_edge[he.origin] = he

            self.half_edges.extend(face_half_edges)

        # Second pass: link twin pointers
        n_boundary = 0
        for (v1, v2), he in edge_map.items():
            twin_key = (v2, v1)
            if twin_key in edge_map:
                twin = edge_map[twin_key]
                he.twin = twin
                # twin.twin = he is set when processing twin_key
            else:
                # Boundary edge
                he.is_boundary = True
                n_boundary += 1

        self._edge_map = edge_map

        logger.debug(
            "Built half-edge mesh: %d vertices, %d faces, %d half-edges, %d boundary",
            len(self.vertices), n_faces, len(self.half_edges), n_boundary
        )

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self.faces)

    @property
    def n_edges(self) -> int:
        """Number of unique edges (half-edges / 2 + boundary edges)."""
        n_boundary = sum(1 for he in self.half_edges if he.is_boundary)
        return (len(self.half_edges) - n_boundary) // 2 + n_boundary

    @property
    def n_half_edges(self) -> int:
        """Number of half-edges."""
        return len(self.half_edges)

    def get_half_edge(self, v1: int, v2: int) -> Optional[HalfEdge]:
        """Get half-edge from v1 to v2.

        Args:
            v1: Origin vertex
            v2: Target vertex

        Returns:
            HalfEdge or None if not found
        """
        return self._edge_map.get((v1, v2))

    def vertex_half_edges(self, vertex: int) -> Iterator[HalfEdge]:
        """Iterate over all outgoing half-edges from a vertex.

        Args:
            vertex: Vertex index

        Yields:
            Half-edges originating from this vertex
        """
        # Collect all half-edges originating from this vertex
        # by scanning through all half-edges (simpler and more robust)
        for he in self.half_edges:
            if he.origin == vertex:
                yield he

    def face_half_edges(self, face: int) -> Iterator[HalfEdge]:
        """Iterate over half-edges around a face.

        Args:
            face: Face index

        Yields:
            Half-edges around the face (counter-clockwise)
        """
        start = self.face_to_half_edge.get(face)
        if start is None:
            return

        current = start
        while True:
            yield current
            current = current.next
            if current is None or current == start:
                break

    def boundary_edges(self) -> Iterator[HalfEdge]:
        """Iterate over boundary half-edges.

        Yields:
            Half-edges that have no twin (boundary edges)
        """
        for he in self.half_edges:
            if he.is_boundary:
                yield he

    def boundary_loops(self) -> List[List[HalfEdge]]:
        """Find all boundary loops.

        Returns:
            List of boundary loops, each loop is a list of half-edges
        """
        boundary_set = {he for he in self.half_edges if he.is_boundary}
        loops = []

        while boundary_set:
            # Start a new loop
            start = next(iter(boundary_set))
            loop = [start]
            boundary_set.remove(start)

            # Follow the boundary
            current = start
            while True:
                # Find next boundary edge from current.target
                target = current.target
                next_he = None

                for he in self.vertex_half_edges(target):
                    if he.is_boundary and he in boundary_set:
                        next_he = he
                        break

                if next_he is None or next_he == start:
                    break

                loop.append(next_he)
                boundary_set.remove(next_he)
                current = next_he

            loops.append(loop)

        return loops

    def adjacent_faces(self, face: int) -> Iterator[int]:
        """Get faces adjacent to a face (sharing an edge).

        Args:
            face: Face index

        Yields:
            Adjacent face indices
        """
        for he in self.face_half_edges(face):
            if he.twin is not None and he.twin.face is not None:
                yield he.twin.face

    def vertex_faces(self, vertex: int) -> Iterator[int]:
        """Get faces incident to a vertex.

        Args:
            vertex: Vertex index

        Yields:
            Face indices containing this vertex
        """
        seen = set()
        for he in self.vertex_half_edges(vertex):
            if he.face is not None and he.face not in seen:
                seen.add(he.face)
                yield he.face

    def vertex_neighbors(self, vertex: int) -> Iterator[int]:
        """Get neighboring vertices (connected by edge).

        Args:
            vertex: Vertex index

        Yields:
            Neighboring vertex indices
        """
        seen = set()
        # Outgoing edges: vertex -> neighbor
        for he in self.vertex_half_edges(vertex):
            if he.target not in seen:
                seen.add(he.target)
                yield he.target
        # Incoming edges: neighbor -> vertex (via prev of outgoing)
        for he in self.vertex_half_edges(vertex):
            if he.prev is not None:
                neighbor = he.prev.origin
                if neighbor not in seen:
                    seen.add(neighbor)
                    yield neighbor

    def is_boundary_vertex(self, vertex: int) -> bool:
        """Check if a vertex is on the boundary.

        Args:
            vertex: Vertex index

        Returns:
            True if vertex is on boundary
        """
        for he in self.vertex_half_edges(vertex):
            if he.is_boundary:
                return True
        return False

    def edge_faces(self, v1: int, v2: int) -> Tuple[Optional[int], Optional[int]]:
        """Get the two faces adjacent to an edge.

        Args:
            v1: First vertex
            v2: Second vertex

        Returns:
            Tuple of (face1, face2), either can be None for boundary edges
        """
        he = self.get_half_edge(v1, v2)
        if he is None:
            return (None, None)

        face1 = he.face
        face2 = he.twin.face if he.twin else None

        return (face1, face2)

    def compute_face_normals(self) -> NDArray[np.float64]:
        """Compute face normals.

        Returns:
            (M, 3) array of face normals
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)

        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths > 1e-10, lengths, 1.0)
        normals = normals / lengths

        return normals

    def compute_vertex_normals(self) -> NDArray[np.float64]:
        """Compute vertex normals by averaging adjacent face normals.

        Returns:
            (N, 3) array of vertex normals
        """
        face_normals = self.compute_face_normals()
        vertex_normals = np.zeros_like(self.vertices)

        # Accumulate face normals at each vertex
        for face_idx, face in enumerate(self.faces):
            for v in face:
                vertex_normals[v] += face_normals[face_idx]

        # Normalize
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        lengths = np.where(lengths > 1e-10, lengths, 1.0)
        vertex_normals = vertex_normals / lengths

        return vertex_normals

    def mark_sharp_edges(
        self,
        angle_threshold_deg: float = 30.0,
        face_normals: Optional[NDArray[np.float64]] = None,
    ) -> int:
        """Mark edges as sharp based on dihedral angle.

        Args:
            angle_threshold_deg: Angle threshold in degrees
            face_normals: Precomputed face normals (optional)

        Returns:
            Number of sharp edges marked
        """
        if face_normals is None:
            face_normals = self.compute_face_normals()

        cos_threshold = np.cos(np.radians(angle_threshold_deg))
        n_sharp = 0

        for he in self.half_edges:
            if he.twin is None:
                # Boundary edges are always sharp
                he.is_sharp = True
                n_sharp += 1
                continue

            if he.index > he.twin.index:
                # Only process each edge once
                continue

            # Get face normals
            n1 = face_normals[he.face]
            n2 = face_normals[he.twin.face]

            # Compute dihedral angle
            dot = np.dot(n1, n2)

            if dot < cos_threshold:
                he.is_sharp = True
                he.twin.is_sharp = True
                n_sharp += 1

        logger.debug("Marked %d sharp edges (threshold=%.1f°)", n_sharp, angle_threshold_deg)
        return n_sharp

    def sharp_edges(self) -> Iterator[Tuple[int, int]]:
        """Iterate over sharp edges.

        Yields:
            (v1, v2) vertex pairs for sharp edges
        """
        seen = set()
        for he in self.half_edges:
            if he.is_sharp:
                edge = (min(he.origin, he.target), max(he.origin, he.target))
                if edge not in seen:
                    seen.add(edge)
                    yield (he.origin, he.target)

    def edge_chain(self, start_he: HalfEdge) -> List[HalfEdge]:
        """Follow a chain of connected sharp edges.

        Args:
            start_he: Starting half-edge

        Returns:
            List of connected sharp half-edges
        """
        chain = [start_he]
        visited = {start_he.index}

        # Follow forward
        current = start_he
        while True:
            # Find next sharp edge from current.target
            next_he = None
            for he in self.vertex_half_edges(current.target):
                if he.is_sharp and he.index not in visited:
                    # Check it's not the twin
                    if he.twin is None or he.twin.index not in visited:
                        next_he = he
                        break

            if next_he is None:
                break

            chain.append(next_he)
            visited.add(next_he.index)
            if next_he.twin:
                visited.add(next_he.twin.index)
            current = next_he

        # Follow backward from start
        current = start_he
        while True:
            # Find previous sharp edge ending at current.origin
            prev_he = None
            for he in self.vertex_half_edges(current.origin):
                if he.twin and he.twin.is_sharp and he.twin.index not in visited:
                    prev_he = he.twin
                    break

            if prev_he is None:
                break

            chain.insert(0, prev_he)
            visited.add(prev_he.index)
            if prev_he.twin:
                visited.add(prev_he.twin.index)
            current = prev_he

        return chain

    def get_edge_vector(self, he: HalfEdge) -> NDArray[np.float64]:
        """Get the vector along a half-edge.

        Args:
            he: Half-edge

        Returns:
            3D vector from origin to target
        """
        return self.vertices[he.target] - self.vertices[he.origin]

    def get_edge_length(self, he: HalfEdge) -> float:
        """Get the length of a half-edge.

        Args:
            he: Half-edge

        Returns:
            Edge length
        """
        return float(np.linalg.norm(self.get_edge_vector(he)))
