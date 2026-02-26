"""
Rotation matrix utilities for 3D transformations.

Provides:
- Rotation3D class for representing and composing rotations
- Rotation matrix constructors from various representations
- Standard view rotation matrices
- Axis alignment utilities

Reference: ISO 841 for coordinate system conventions.
GOST 2.305-2008 specifies standard drawing views.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Axis(Enum):
    """Standard coordinate axes."""
    X = 0
    Y = 1
    Z = 2


class StandardView(Enum):
    """Standard orthographic views per GOST 2.305-2008.

    Views are named by the direction the observer looks FROM.
    """
    FRONT = "front"      # Observer at +Z, looks toward -Z
    BACK = "back"        # Observer at -Z, looks toward +Z
    TOP = "top"          # Observer at +Y, looks toward -Y
    BOTTOM = "bottom"    # Observer at -Y, looks toward +Y
    LEFT = "left"        # Observer at -X, looks toward +X
    RIGHT = "right"      # Observer at +X, looks toward -X


# View direction vectors (direction FROM observer TO object center)
VIEW_DIRECTIONS = {
    StandardView.FRONT: np.array([0.0, 0.0, -1.0]),
    StandardView.BACK: np.array([0.0, 0.0, 1.0]),
    StandardView.TOP: np.array([0.0, -1.0, 0.0]),
    StandardView.BOTTOM: np.array([0.0, 1.0, 0.0]),
    StandardView.LEFT: np.array([1.0, 0.0, 0.0]),
    StandardView.RIGHT: np.array([-1.0, 0.0, 0.0]),
}


@dataclass
class Rotation3D:
    """3D rotation represented as a rotation matrix.

    Attributes:
        matrix: 3x3 orthogonal rotation matrix (det = +1)
    """
    matrix: NDArray[np.float64]

    def __post_init__(self):
        """Validate rotation matrix."""
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {self.matrix.shape}")

    @classmethod
    def identity(cls) -> 'Rotation3D':
        """Create identity rotation (no rotation)."""
        return cls(np.eye(3))

    @classmethod
    def from_axis_angle(cls, axis: NDArray[np.float64], angle_rad: float) -> 'Rotation3D':
        """Create rotation from axis and angle (Rodrigues' formula).

        Args:
            axis: 3D unit vector for rotation axis
            angle_rad: Rotation angle in radians

        Returns:
            Rotation3D instance
        """
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        t = 1 - c

        x, y, z = axis
        matrix = np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
        ])
        return cls(matrix)

    @classmethod
    def from_euler_xyz(cls, angles_rad: Tuple[float, float, float]) -> 'Rotation3D':
        """Create rotation from Euler angles (XYZ convention).

        Args:
            angles_rad: (roll, pitch, yaw) in radians

        Returns:
            Rotation3D instance
        """
        roll, pitch, yaw = angles_rad

        # Rotation around X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ])

        # Rotation around Y
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])

        # Rotation around Z
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])

        # Combined rotation: R = Rz @ Ry @ Rx
        return cls(Rz @ Ry @ Rx)

    @classmethod
    def from_two_vectors(
        cls,
        vec_from: NDArray[np.float64],
        vec_to: NDArray[np.float64]
    ) -> 'Rotation3D':
        """Create rotation that transforms one vector to another.

        Args:
            vec_from: Source vector (will be normalized)
            vec_to: Target vector (will be normalized)

        Returns:
            Rotation3D that rotates vec_from to vec_to
        """
        a = np.asarray(vec_from, dtype=np.float64)
        b = np.asarray(vec_to, dtype=np.float64)
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)

        # Check if vectors are parallel
        dot = np.dot(a, b)
        if dot > 0.9999:
            return cls.identity()
        if dot < -0.9999:
            # 180 degree rotation around any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(a, perp)
            axis = axis / np.linalg.norm(axis)
            return cls.from_axis_angle(axis, np.pi)

        # General case: use Rodrigues' formula
        axis = np.cross(a, b)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        return cls.from_axis_angle(axis, angle)

    @classmethod
    def around_x(cls, angle_rad: float) -> 'Rotation3D':
        """Create rotation around X axis."""
        return cls.from_axis_angle(np.array([1, 0, 0]), angle_rad)

    @classmethod
    def around_y(cls, angle_rad: float) -> 'Rotation3D':
        """Create rotation around Y axis."""
        return cls.from_axis_angle(np.array([0, 1, 0]), angle_rad)

    @classmethod
    def around_z(cls, angle_rad: float) -> 'Rotation3D':
        """Create rotation around Z axis."""
        return cls.from_axis_angle(np.array([0, 0, 1]), angle_rad)

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply rotation to points.

        Args:
            points: Nx3 array of 3D points

        Returns:
            Rotated Nx3 array
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            return self.matrix @ points
        return points @ self.matrix.T

    def apply_to_normals(self, normals: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply rotation to normal vectors (same as apply for orthogonal matrix)."""
        return self.apply(normals)

    def compose(self, other: 'Rotation3D') -> 'Rotation3D':
        """Compose with another rotation: self * other.

        Result applies `other` first, then `self`.

        Args:
            other: Rotation to compose with

        Returns:
            New composed Rotation3D
        """
        return Rotation3D(self.matrix @ other.matrix)

    def inverse(self) -> 'Rotation3D':
        """Get inverse rotation.

        For orthogonal matrices, inverse = transpose.

        Returns:
            Inverse Rotation3D
        """
        return Rotation3D(self.matrix.T)

    @property
    def axis_angle(self) -> Tuple[NDArray[np.float64], float]:
        """Extract axis and angle from rotation matrix.

        Returns:
            (axis, angle_rad) tuple
        """
        # Angle from trace: trace(R) = 1 + 2*cos(theta)
        trace = np.trace(self.matrix)
        angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

        if abs(angle) < 1e-6:
            return np.array([1.0, 0.0, 0.0]), 0.0

        if abs(angle - np.pi) < 1e-6:
            # 180 degree rotation: axis from eigenvector
            eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
            idx = np.argmin(np.abs(eigenvalues - 1))
            axis = np.real(eigenvectors[:, idx])
            return axis / np.linalg.norm(axis), np.pi

        # General case: axis from skew-symmetric part
        axis = np.array([
            self.matrix[2, 1] - self.matrix[1, 2],
            self.matrix[0, 2] - self.matrix[2, 0],
            self.matrix[1, 0] - self.matrix[0, 1],
        ])
        axis = axis / (2 * np.sin(angle))
        return axis, angle

    def is_identity(self, tol: float = 1e-9) -> bool:
        """Check if rotation is identity."""
        return np.allclose(self.matrix, np.eye(3), atol=tol)

    def __matmul__(self, other: 'Rotation3D') -> 'Rotation3D':
        """Matrix multiplication operator."""
        return self.compose(other)


def get_view_rotation(view: StandardView) -> Rotation3D:
    """Get rotation matrix to transform to standard view orientation.

    After rotation, the object's "front" faces toward the camera (along -Z),
    with +Y pointing up and +X pointing right.

    Args:
        view: Standard view to rotate to

    Returns:
        Rotation3D for the view transformation
    """
    # View rotation matrices that transform world coordinates
    # to view coordinates where -Z is the view direction
    rotations = {
        StandardView.FRONT: Rotation3D.identity(),
        StandardView.BACK: Rotation3D.around_y(np.pi),
        StandardView.TOP: Rotation3D.around_x(-np.pi/2),
        StandardView.BOTTOM: Rotation3D.around_x(np.pi/2),
        StandardView.LEFT: Rotation3D.around_y(-np.pi/2),
        StandardView.RIGHT: Rotation3D.around_y(np.pi/2),
    }
    return rotations[view]


def align_axis_to_direction(
    source_axis: Union[Axis, NDArray[np.float64]],
    target_direction: NDArray[np.float64],
) -> Rotation3D:
    """Create rotation that aligns an axis to a target direction.

    Args:
        source_axis: Axis enum or 3D vector to align
        target_direction: Target direction vector

    Returns:
        Rotation3D that aligns source to target
    """
    if isinstance(source_axis, Axis):
        source = np.zeros(3)
        source[source_axis.value] = 1.0
    else:
        source = np.asarray(source_axis)

    return Rotation3D.from_two_vectors(source, target_direction)


def compute_optimal_view_rotation(
    vertices: NDArray[np.float64],
    faces: NDArray[np.float64],
    face_normals: NDArray[np.float64],
) -> Tuple[Rotation3D, StandardView]:
    """Compute rotation for optimal viewing direction.

    Evaluates all 6 standard views and returns rotation to the best one.
    The best view maximizes visible area and geometric complexity.

    Args:
        vertices: Nx3 vertex array
        faces: Mx3 face index array
        face_normals: Mx3 face normal array

    Returns:
        (rotation, best_view) tuple
    """
    best_score = -1.0
    best_view = StandardView.FRONT

    for view in StandardView:
        view_dir = VIEW_DIRECTIONS[view]
        score = _score_view(vertices, faces, face_normals, view_dir)

        if score > best_score:
            best_score = score
            best_view = view

    logger.debug(
        "Optimal view: %s (score=%.0f)",
        best_view.value, best_score
    )

    return get_view_rotation(best_view), best_view


def _score_view(
    vertices: NDArray[np.float64],
    faces: NDArray[np.float64],
    face_normals: NDArray[np.float64],
    view_dir: NDArray[np.float64],
) -> float:
    """Score a view direction by visible area."""
    dots = face_normals @ view_dir
    visible_mask = dots < 0

    if not np.any(visible_mask):
        return 0.0

    # Calculate projected area of visible faces
    v0 = vertices[faces[visible_mask, 0]]
    v1 = vertices[faces[visible_mask, 1]]
    v2 = vertices[faces[visible_mask, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.abs(cross @ view_dir) * 0.5

    return float(np.sum(areas))


def rotation_between_axes(
    from_axis: Axis,
    to_axis: Axis,
) -> Rotation3D:
    """Create rotation from one standard axis to another.

    Args:
        from_axis: Source axis
        to_axis: Target axis

    Returns:
        Rotation3D
    """
    if from_axis == to_axis:
        return Rotation3D.identity()

    from_vec = np.zeros(3)
    from_vec[from_axis.value] = 1.0

    to_vec = np.zeros(3)
    to_vec[to_axis.value] = 1.0

    return Rotation3D.from_two_vectors(from_vec, to_vec)


def orthonormalize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Orthonormalize a 3x3 matrix using Gram-Schmidt.

    Useful for fixing accumulated numerical errors in rotation matrices.

    Args:
        matrix: 3x3 matrix (nearly orthogonal)

    Returns:
        Orthonormalized 3x3 matrix
    """
    q, r = np.linalg.qr(matrix)
    # Ensure det = +1 (proper rotation, not reflection)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q
