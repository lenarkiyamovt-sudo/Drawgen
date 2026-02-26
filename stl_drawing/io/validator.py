"""
STL Mesh Validation Module.

Performs integrity checks on loaded STL meshes:
- Manifold validation (each edge shared by exactly 2 faces)
- Degenerate triangle detection (zero area)
- Normal consistency check
- Boundary edge detection (open mesh)

Non-critical errors are reported as warnings and don't block processing.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation issue found in the mesh."""
    code: str
    severity: ValidationSeverity
    message: str
    count: int = 1
    details: List[int] = field(default_factory=list)  # face/edge indices

    def __str__(self) -> str:
        if self.count > 1:
            return f"[{self.severity.value.upper()}] {self.code}: {self.message} ({self.count} occurrences)"
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report for a mesh."""
    is_valid: bool
    is_manifold: bool
    is_closed: bool
    has_degenerate_faces: bool
    has_inconsistent_normals: bool

    n_vertices: int
    n_faces: int
    n_boundary_edges: int
    n_degenerate_faces: int
    n_non_manifold_edges: int

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Mesh Validation Report",
            f"=" * 40,
            f"Vertices: {self.n_vertices}",
            f"Faces: {self.n_faces}",
            f"",
            f"Manifold: {'Yes' if self.is_manifold else 'No'}",
            f"Closed: {'Yes' if self.is_closed else 'No'}",
            f"Degenerate faces: {self.n_degenerate_faces}",
            f"Non-manifold edges: {self.n_non_manifold_edges}",
            f"Boundary edges: {self.n_boundary_edges}",
        ]

        if self.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")

        lines.append("")
        lines.append(f"Overall: {'VALID' if self.is_valid else 'INVALID'}")

        return "\n".join(lines)


def _build_edge_map(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """Build edge-to-faces map.

    Args:
        faces: Array of shape (N, 3) with vertex indices

    Returns:
        Dict mapping (v1, v2) sorted tuple to list of face indices
    """
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for fi, face in enumerate(faces):
        for i in range(3):
            v1, v2 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v1, v2), max(v1, v2))
            edge_to_faces[edge].append(fi)

    return edge_to_faces


def _compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area of each face.

    Args:
        vertices: Array of shape (V, 3)
        faces: Array of shape (F, 3)

    Returns:
        Array of shape (F,) with face areas
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return areas


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute unit normal of each face.

    Args:
        vertices: Array of shape (V, 3)
        faces: Array of shape (F, 3)

    Returns:
        Array of shape (F, 3) with unit normals (zero for degenerate faces)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)

    return cross / norms


def validate_mesh(vertices: np.ndarray, faces: np.ndarray,
                  degenerate_area_threshold: float = 1e-10) -> ValidationReport:
    """Validate mesh integrity.

    Checks performed:
    1. Manifold: each edge shared by exactly 2 faces
    2. Closed: no boundary edges (edges with only 1 face)
    3. Degenerate faces: triangles with zero area
    4. Normal consistency: adjacent faces should have compatible normals

    Args:
        vertices: Array of shape (V, 3) with vertex coordinates
        faces: Array of shape (F, 3) with vertex indices
        degenerate_area_threshold: Minimum area to consider non-degenerate

    Returns:
        ValidationReport with all findings
    """
    n_vertices = len(vertices)
    n_faces = len(faces)
    issues: List[ValidationIssue] = []

    logger.debug("Validating mesh: %d vertices, %d faces", n_vertices, n_faces)

    # Build edge map
    edge_to_faces = _build_edge_map(faces)
    n_edges = len(edge_to_faces)

    # Check manifold and boundary edges
    boundary_edges: List[Tuple[int, int]] = []
    non_manifold_edges: List[Tuple[int, int]] = []

    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 1:
            boundary_edges.append(edge)
        elif len(face_list) > 2:
            non_manifold_edges.append(edge)

    is_manifold = len(non_manifold_edges) == 0
    is_closed = len(boundary_edges) == 0

    if boundary_edges:
        issues.append(ValidationIssue(
            code="BOUNDARY_EDGES",
            severity=ValidationSeverity.WARNING,
            message=f"Mesh has {len(boundary_edges)} boundary edges (not closed)",
            count=len(boundary_edges),
        ))
        logger.warning("Mesh has %d boundary edges", len(boundary_edges))

    if non_manifold_edges:
        issues.append(ValidationIssue(
            code="NON_MANIFOLD_EDGES",
            severity=ValidationSeverity.ERROR,
            message=f"Mesh has {len(non_manifold_edges)} non-manifold edges (>2 faces)",
            count=len(non_manifold_edges),
        ))
        logger.error("Mesh has %d non-manifold edges", len(non_manifold_edges))

    # Check degenerate faces
    areas = _compute_face_areas(vertices, faces)
    degenerate_mask = areas < degenerate_area_threshold
    n_degenerate = int(degenerate_mask.sum())
    has_degenerate = n_degenerate > 0

    if has_degenerate:
        degenerate_indices = np.where(degenerate_mask)[0].tolist()
        issues.append(ValidationIssue(
            code="DEGENERATE_FACES",
            severity=ValidationSeverity.WARNING,
            message=f"Mesh has {n_degenerate} degenerate faces (zero area)",
            count=n_degenerate,
            details=degenerate_indices[:10],  # First 10 indices
        ))
        logger.warning("Mesh has %d degenerate faces", n_degenerate)

    # Check normal consistency (optional, more detailed check)
    normals = _compute_face_normals(vertices, faces)
    inconsistent_count = 0

    # Check adjacent faces for consistent normals (simple heuristic)
    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 2:
            f1, f2 = face_list
            dot = np.dot(normals[f1], normals[f2])
            # If normals point in opposite directions for adjacent faces,
            # this might indicate inverted faces (dot < -0.5 is suspicious)
            if dot < -0.9:
                inconsistent_count += 1

    has_inconsistent_normals = inconsistent_count > n_edges * 0.1  # >10% suspicious

    if has_inconsistent_normals:
        issues.append(ValidationIssue(
            code="INCONSISTENT_NORMALS",
            severity=ValidationSeverity.WARNING,
            message=f"Mesh may have inconsistent normals ({inconsistent_count} suspicious edges)",
            count=inconsistent_count,
        ))
        logger.warning("Mesh may have %d edges with inconsistent normals", inconsistent_count)

    # Determine overall validity
    # Mesh is valid if it has no errors (warnings are OK)
    is_valid = len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0

    report = ValidationReport(
        is_valid=is_valid,
        is_manifold=is_manifold,
        is_closed=is_closed,
        has_degenerate_faces=has_degenerate,
        has_inconsistent_normals=has_inconsistent_normals,
        n_vertices=n_vertices,
        n_faces=n_faces,
        n_boundary_edges=len(boundary_edges),
        n_degenerate_faces=n_degenerate,
        n_non_manifold_edges=len(non_manifold_edges),
        issues=issues,
    )

    logger.info("Validation complete: %s", "VALID" if is_valid else "INVALID")
    return report


def validate_stl_file(filepath: str,
                      degenerate_area_threshold: float = 1e-10) -> ValidationReport:
    """Load and validate STL file.

    Convenience function that loads the file and runs validation.

    Args:
        filepath: Path to STL file
        degenerate_area_threshold: Minimum area to consider non-degenerate

    Returns:
        ValidationReport with all findings
    """
    from stl_drawing.io.stl_loader import load_stl

    vertices, faces = load_stl(filepath)
    return validate_mesh(vertices, faces, degenerate_area_threshold)
