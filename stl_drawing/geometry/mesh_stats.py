"""
Mesh statistics calculation module.

Provides:
- Bounding box calculation (AABB, OBB)
- Mesh statistics (vertices, faces, area, volume)
- Center of mass calculation
- Mesh dimensions and aspect ratios

Reference GOST: 2.307-2011 for dimension notation requirements.
"""

import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Axis-Aligned Bounding Box (AABB) for a mesh.

    Attributes:
        min_point: Minimum corner (x_min, y_min, z_min)
        max_point: Maximum corner (x_max, y_max, z_max)
    """
    min_point: NDArray[np.float64]
    max_point: NDArray[np.float64]

    @property
    def dimensions(self) -> NDArray[np.float64]:
        """Get box dimensions (width, height, depth)."""
        return self.max_point - self.min_point

    @property
    def width(self) -> float:
        """X-axis dimension."""
        return float(self.dimensions[0])

    @property
    def height(self) -> float:
        """Y-axis dimension."""
        return float(self.dimensions[1])

    @property
    def depth(self) -> float:
        """Z-axis dimension."""
        return float(self.dimensions[2])

    @property
    def center(self) -> NDArray[np.float64]:
        """Get box center point."""
        return (self.min_point + self.max_point) / 2

    @property
    def volume(self) -> float:
        """Get box volume."""
        dims = self.dimensions
        return float(dims[0] * dims[1] * dims[2])

    @property
    def diagonal(self) -> float:
        """Get box diagonal length."""
        return float(np.linalg.norm(self.dimensions))

    @property
    def max_dimension(self) -> float:
        """Get largest dimension."""
        return float(np.max(self.dimensions))

    @property
    def min_dimension(self) -> float:
        """Get smallest dimension."""
        return float(np.min(self.dimensions))

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (max/min dimension)."""
        min_dim = self.min_dimension
        if min_dim < 1e-10:
            return float('inf')
        return self.max_dimension / min_dim

    def contains_point(self, point: NDArray[np.float64]) -> bool:
        """Check if point is inside the bounding box."""
        return bool(
            np.all(point >= self.min_point) and
            np.all(point <= self.max_point)
        )

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if two bounding boxes intersect."""
        return bool(
            np.all(self.min_point <= other.max_point) and
            np.all(self.max_point >= other.min_point)
        )

    def expand(self, margin: float) -> 'BoundingBox':
        """Return expanded bounding box by margin on all sides."""
        return BoundingBox(
            min_point=self.min_point - margin,
            max_point=self.max_point + margin
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'min': self.min_point.tolist(),
            'max': self.max_point.tolist(),
            'dimensions': self.dimensions.tolist(),
            'center': self.center.tolist(),
            'volume': self.volume,
            'diagonal': self.diagonal,
        }


@dataclass
class MeshStatistics:
    """Comprehensive mesh statistics.

    Attributes:
        n_vertices: Number of unique vertices
        n_faces: Number of triangular faces
        n_edges: Number of edges (computed)
        bbox: Axis-aligned bounding box
        surface_area: Total surface area in mm^2
        volume: Mesh volume in mm^3 (signed, negative if inverted)
        center_of_mass: Geometric center of mass
        is_watertight: True if mesh is closed (no boundary edges)
        euler_characteristic: Euler characteristic V - E + F
    """
    n_vertices: int
    n_faces: int
    n_edges: int
    bbox: BoundingBox
    surface_area: float
    volume: float
    center_of_mass: NDArray[np.float64]
    is_watertight: bool = False
    euler_characteristic: int = 0
    face_areas: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Get mesh dimensions (width, height, depth) in mm."""
        dims = self.bbox.dimensions
        return (float(dims[0]), float(dims[1]), float(dims[2]))

    @property
    def avg_face_area(self) -> float:
        """Average face area."""
        if self.n_faces == 0:
            return 0.0
        return self.surface_area / self.n_faces

    @property
    def compactness(self) -> float:
        """Sphericity measure: how close to a sphere (0-1).

        Computed as 36*pi*V^2 / A^3 for a perfect sphere = 1.
        """
        if self.surface_area < 1e-10:
            return 0.0
        return (36 * np.pi * self.volume ** 2) / (self.surface_area ** 3)

    def summary(self) -> str:
        """Generate human-readable summary."""
        dims = self.dimensions
        lines = [
            "Mesh Statistics",
            "=" * 40,
            f"Vertices:     {self.n_vertices:,}",
            f"Faces:        {self.n_faces:,}",
            f"Edges:        {self.n_edges:,}",
            f"",
            f"Dimensions:   {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} mm",
            f"Diagonal:     {self.bbox.diagonal:.2f} mm",
            f"Bbox Volume:  {self.bbox.volume:.2f} mm^3",
            f"",
            f"Surface Area: {self.surface_area:.2f} mm^2",
            f"Mesh Volume:  {self.volume:.2f} mm^3",
            f"Watertight:   {'Yes' if self.is_watertight else 'No'}",
            f"",
            f"Center:       ({self.center_of_mass[0]:.2f}, "
            f"{self.center_of_mass[1]:.2f}, {self.center_of_mass[2]:.2f})",
            f"Euler char:   {self.euler_characteristic}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'n_vertices': self.n_vertices,
            'n_faces': self.n_faces,
            'n_edges': self.n_edges,
            'bbox': self.bbox.to_dict(),
            'surface_area_mm2': self.surface_area,
            'volume_mm3': self.volume,
            'center_of_mass': self.center_of_mass.tolist(),
            'is_watertight': self.is_watertight,
            'euler_characteristic': self.euler_characteristic,
            'dimensions_mm': list(self.dimensions),
        }


def calculate_bounding_box(vertices: NDArray[np.float64]) -> BoundingBox:
    """Calculate axis-aligned bounding box for vertices.

    Args:
        vertices: Nx3 array of vertex coordinates

    Returns:
        BoundingBox instance
    """
    if len(vertices) == 0:
        return BoundingBox(
            min_point=np.zeros(3),
            max_point=np.zeros(3)
        )

    return BoundingBox(
        min_point=np.min(vertices, axis=0),
        max_point=np.max(vertices, axis=0)
    )


def calculate_face_areas(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32]
) -> NDArray[np.float64]:
    """Calculate area of each triangular face.

    Uses cross product: area = 0.5 * |v1 x v2|

    Args:
        vertices: Nx3 array of vertices
        faces: Mx3 array of face indices

    Returns:
        Array of M face areas
    """
    if len(faces) == 0:
        return np.array([], dtype=np.float64)

    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0

    # Cross product
    cross = np.cross(e1, e2)

    # Area = 0.5 * |cross|
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return areas


def calculate_surface_area(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32]
) -> float:
    """Calculate total mesh surface area.

    Args:
        vertices: Nx3 array of vertices
        faces: Mx3 array of face indices

    Returns:
        Total surface area in mm^2
    """
    areas = calculate_face_areas(vertices, faces)
    return float(np.sum(areas))


def calculate_volume(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32]
) -> float:
    """Calculate signed mesh volume using divergence theorem.

    For a closed mesh, this gives the enclosed volume.
    Negative volume indicates inverted normals.

    Formula: V = (1/6) * sum(v0 . (v1 x v2))

    Args:
        vertices: Nx3 array of vertices
        faces: Mx3 array of face indices

    Returns:
        Signed volume in mm^3
    """
    if len(faces) == 0:
        return 0.0

    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Signed volume contribution from each face
    # V_i = (1/6) * v0 . (v1 x v2)
    cross = np.cross(v1, v2)
    signed_volumes = np.sum(v0 * cross, axis=1) / 6.0

    return float(np.sum(signed_volumes))


def calculate_center_of_mass(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32]
) -> NDArray[np.float64]:
    """Calculate geometric center of mass (centroid).

    Weighted by face areas for surface centroid.

    Args:
        vertices: Nx3 array of vertices
        faces: Mx3 array of face indices

    Returns:
        3D center of mass point
    """
    if len(faces) == 0:
        if len(vertices) == 0:
            return np.zeros(3)
        return np.mean(vertices, axis=0)

    # Face centroids
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    # Weight by face areas
    areas = calculate_face_areas(vertices, faces)
    total_area = np.sum(areas)

    if total_area < 1e-10:
        return np.mean(centroids, axis=0)

    weighted_centroid = np.sum(centroids * areas[:, np.newaxis], axis=0)
    return weighted_centroid / total_area


def count_edges(faces: NDArray[np.int32]) -> int:
    """Count unique edges in mesh.

    Args:
        faces: Mx3 array of face indices

    Returns:
        Number of unique edges
    """
    if len(faces) == 0:
        return 0

    # Extract all edges (sorted pairs)
    edges = []
    for i in range(3):
        j = (i + 1) % 3
        e = np.stack([faces[:, i], faces[:, j]], axis=1)
        edges.append(np.sort(e, axis=1))

    all_edges = np.vstack(edges)

    # Count unique edges
    unique_edges = np.unique(all_edges, axis=0)
    return len(unique_edges)


def calculate_mesh_statistics(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    check_watertight: bool = True,
) -> MeshStatistics:
    """Calculate comprehensive mesh statistics.

    Args:
        vertices: Nx3 array of vertices
        faces: Mx3 array of face indices
        check_watertight: Whether to check if mesh is closed

    Returns:
        MeshStatistics instance with all computed values

    Example:
        >>> stats = calculate_mesh_statistics(vertices, faces)
        >>> print(f"Mesh has {stats.n_faces} faces")
        >>> print(f"Dimensions: {stats.dimensions}")
    """
    n_vertices = len(vertices)
    n_faces = len(faces)
    n_edges = count_edges(faces)

    bbox = calculate_bounding_box(vertices)
    face_areas = calculate_face_areas(vertices, faces)
    surface_area = float(np.sum(face_areas))
    volume = calculate_volume(vertices, faces)
    center = calculate_center_of_mass(vertices, faces)

    # Check watertight (each edge should have exactly 2 faces)
    is_watertight = False
    if check_watertight and len(faces) > 0:
        # Build edge-face count
        from collections import Counter
        edge_count: Counter = Counter()
        for face in faces:
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted([face[i], face[j]]))
                edge_count[edge] += 1

        # Watertight if all edges have exactly 2 faces
        is_watertight = all(count == 2 for count in edge_count.values())

    # Euler characteristic: V - E + F
    euler = n_vertices - n_edges + n_faces

    stats = MeshStatistics(
        n_vertices=n_vertices,
        n_faces=n_faces,
        n_edges=n_edges,
        bbox=bbox,
        surface_area=surface_area,
        volume=volume,
        center_of_mass=center,
        is_watertight=is_watertight,
        euler_characteristic=euler,
        face_areas=face_areas,
    )

    logger.debug(
        "Mesh statistics calculated",
        extra={
            'vertices': n_vertices,
            'faces': n_faces,
            'surface_area': surface_area,
            'volume': volume,
        }
    )

    return stats


def get_oriented_bounding_box_axes(
    vertices: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Get oriented bounding box axes using PCA.

    Args:
        vertices: Nx3 array of vertices

    Returns:
        Tuple of (center, axes, half_extents)
        - center: OBB center point
        - axes: 3x3 matrix where rows are principal axes
        - half_extents: half-lengths along each axis
    """
    if len(vertices) == 0:
        return np.zeros(3), np.eye(3), np.zeros(3)

    # Center vertices
    center = np.mean(vertices, axis=0)
    centered = vertices - center

    # PCA for principal axes
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, idx].T  # Rows are principal axes

    # Project to find extents
    projected = centered @ axes.T
    half_extents = (np.max(projected, axis=0) - np.min(projected, axis=0)) / 2

    return center, axes, half_extents


def compare_mesh_statistics(
    stats1: MeshStatistics,
    stats2: MeshStatistics,
    name1: str = "Mesh 1",
    name2: str = "Mesh 2",
) -> str:
    """Generate comparison report between two meshes.

    Args:
        stats1: First mesh statistics
        stats2: Second mesh statistics
        name1: Name for first mesh
        name2: Name for second mesh

    Returns:
        Formatted comparison string
    """
    def fmt_diff(v1: float, v2: float) -> str:
        if v1 == 0:
            return "N/A"
        diff_pct = ((v2 - v1) / v1) * 100
        sign = "+" if diff_pct > 0 else ""
        return f"{sign}{diff_pct:.1f}%"

    lines = [
        f"Mesh Comparison: {name1} vs {name2}",
        "=" * 50,
        f"{'Property':<20} {name1:>12} {name2:>12} {'Diff':>10}",
        "-" * 50,
        f"{'Vertices':<20} {stats1.n_vertices:>12,} {stats2.n_vertices:>12,} "
        f"{fmt_diff(stats1.n_vertices, stats2.n_vertices):>10}",
        f"{'Faces':<20} {stats1.n_faces:>12,} {stats2.n_faces:>12,} "
        f"{fmt_diff(stats1.n_faces, stats2.n_faces):>10}",
        f"{'Surface Area':<20} {stats1.surface_area:>12.1f} {stats2.surface_area:>12.1f} "
        f"{fmt_diff(stats1.surface_area, stats2.surface_area):>10}",
        f"{'Volume':<20} {stats1.volume:>12.1f} {stats2.volume:>12.1f} "
        f"{fmt_diff(stats1.volume, stats2.volume):>10}",
    ]
    return "\n".join(lines)
