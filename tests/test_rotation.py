"""
Unit tests for stl_drawing.orientation.rotation module.

Tests:
- Rotation3D class and constructors
- Axis-angle and Euler representations
- View rotations
- Composition and inversion
"""

import numpy as np
import pytest

from stl_drawing.orientation.rotation import (
    Rotation3D,
    Axis,
    StandardView,
    VIEW_DIRECTIONS,
    get_view_rotation,
    align_axis_to_direction,
    compute_optimal_view_rotation,
    rotation_between_axes,
    orthonormalize,
)


class TestRotation3D:
    """Tests for Rotation3D class."""

    def test_identity(self):
        """Test identity rotation."""
        rot = Rotation3D.identity()
        assert np.allclose(rot.matrix, np.eye(3))
        assert rot.is_identity()

    def test_from_axis_angle_z_90(self):
        """Test 90 degree rotation around Z axis."""
        rot = Rotation3D.from_axis_angle(np.array([0, 0, 1]), np.pi/2)

        # X -> Y
        result = rot.apply(np.array([1, 0, 0]))
        assert np.allclose(result, [0, 1, 0], atol=1e-10)

        # Y -> -X
        result = rot.apply(np.array([0, 1, 0]))
        assert np.allclose(result, [-1, 0, 0], atol=1e-10)

    def test_from_axis_angle_x_180(self):
        """Test 180 degree rotation around X axis."""
        rot = Rotation3D.from_axis_angle(np.array([1, 0, 0]), np.pi)

        # Y -> -Y
        result = rot.apply(np.array([0, 1, 0]))
        assert np.allclose(result, [0, -1, 0], atol=1e-10)

        # Z -> -Z
        result = rot.apply(np.array([0, 0, 1]))
        assert np.allclose(result, [0, 0, -1], atol=1e-10)

    def test_from_euler_xyz(self):
        """Test Euler angle rotation."""
        # 90 degrees around each axis in sequence
        rot = Rotation3D.from_euler_xyz((np.pi/2, 0, 0))

        # Should rotate Y to Z
        result = rot.apply(np.array([0, 1, 0]))
        assert np.allclose(result, [0, 0, 1], atol=1e-10)

    def test_from_two_vectors(self):
        """Test rotation from one vector to another."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])

        rot = Rotation3D.from_two_vectors(v1, v2)
        result = rot.apply(v1)
        assert np.allclose(result, v2, atol=1e-10)

    def test_from_two_vectors_parallel(self):
        """Test rotation between parallel vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([2, 0, 0])  # Same direction

        rot = Rotation3D.from_two_vectors(v1, v2)
        assert rot.is_identity()

    def test_from_two_vectors_antiparallel(self):
        """Test rotation between antiparallel vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])

        rot = Rotation3D.from_two_vectors(v1, v2)
        result = rot.apply(v1)
        assert np.allclose(result, v2, atol=1e-10)

    def test_around_x(self):
        """Test rotation around X axis convenience method."""
        rot = Rotation3D.around_x(np.pi/2)
        result = rot.apply(np.array([0, 1, 0]))
        assert np.allclose(result, [0, 0, 1], atol=1e-10)

    def test_around_y(self):
        """Test rotation around Y axis convenience method."""
        rot = Rotation3D.around_y(np.pi/2)
        result = rot.apply(np.array([1, 0, 0]))
        assert np.allclose(result, [0, 0, -1], atol=1e-10)

    def test_around_z(self):
        """Test rotation around Z axis convenience method."""
        rot = Rotation3D.around_z(np.pi/2)
        result = rot.apply(np.array([1, 0, 0]))
        assert np.allclose(result, [0, 1, 0], atol=1e-10)

    def test_apply_single_point(self):
        """Test applying rotation to single point."""
        rot = Rotation3D.around_z(np.pi/2)
        point = np.array([1, 0, 0])
        result = rot.apply(point)
        assert result.shape == (3,)

    def test_apply_multiple_points(self):
        """Test applying rotation to multiple points."""
        rot = Rotation3D.around_z(np.pi/2)
        points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ])
        result = rot.apply(points)
        assert result.shape == (3, 3)
        assert np.allclose(result[0], [0, 1, 0], atol=1e-10)
        assert np.allclose(result[1], [-1, 0, 0], atol=1e-10)

    def test_compose(self):
        """Test rotation composition."""
        rot1 = Rotation3D.around_z(np.pi/2)
        rot2 = Rotation3D.around_z(np.pi/2)
        composed = rot1.compose(rot2)

        # 90 + 90 = 180 degrees around Z
        result = composed.apply(np.array([1, 0, 0]))
        assert np.allclose(result, [-1, 0, 0], atol=1e-10)

    def test_matmul_operator(self):
        """Test @ operator for composition."""
        rot1 = Rotation3D.around_z(np.pi/4)
        rot2 = Rotation3D.around_z(np.pi/4)
        composed = rot1 @ rot2

        # Should equal 90 degree rotation
        result = composed.apply(np.array([1, 0, 0]))
        assert np.allclose(result, [0, 1, 0], atol=1e-10)

    def test_inverse(self):
        """Test rotation inverse."""
        rot = Rotation3D.around_z(np.pi/3)
        inv = rot.inverse()

        # rot * inv should be identity
        composed = rot.compose(inv)
        assert composed.is_identity()

    def test_axis_angle_extraction(self):
        """Test extracting axis and angle from rotation."""
        axis_in = np.array([0, 0, 1])
        angle_in = np.pi / 3

        rot = Rotation3D.from_axis_angle(axis_in, angle_in)
        axis_out, angle_out = rot.axis_angle

        assert np.allclose(abs(np.dot(axis_out, axis_in)), 1.0)
        assert np.isclose(angle_out, angle_in, atol=1e-10)

    def test_determinant_is_one(self):
        """Test that rotation matrices have determinant +1."""
        for angle in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
            rot = Rotation3D.around_z(angle)
            det = np.linalg.det(rot.matrix)
            assert np.isclose(det, 1.0)


class TestViewRotations:
    """Tests for standard view rotations."""

    def test_front_view_identity(self):
        """Test front view is identity."""
        rot = get_view_rotation(StandardView.FRONT)
        assert rot.is_identity()

    def test_back_view(self):
        """Test back view is 180 around Y."""
        rot = get_view_rotation(StandardView.BACK)
        # Should flip X and Z
        result = rot.apply(np.array([1, 0, 0]))
        assert np.allclose(result, [-1, 0, 0], atol=1e-10)

    def test_top_view(self):
        """Test top view rotation."""
        rot = get_view_rotation(StandardView.TOP)
        # Looking down from +Y, Z becomes the "up" on screen
        result = rot.apply(np.array([0, 0, 1]))
        # After top view rotation, Z should map to Y
        assert np.allclose(result, [0, 1, 0], atol=1e-10)

    def test_left_view(self):
        """Test left view rotation."""
        rot = get_view_rotation(StandardView.LEFT)
        # Left view: -90 deg around Y, Z -> -X
        result = rot.apply(np.array([0, 0, 1]))
        assert np.allclose(result, [-1, 0, 0], atol=1e-10)

    def test_all_views_are_rotations(self):
        """Test all view matrices are valid rotations."""
        for view in StandardView:
            rot = get_view_rotation(view)
            # Check orthogonality
            assert np.allclose(rot.matrix @ rot.matrix.T, np.eye(3))
            # Check determinant
            assert np.isclose(np.linalg.det(rot.matrix), 1.0)


class TestAxisAlignment:
    """Tests for axis alignment functions."""

    def test_align_x_to_y(self):
        """Test aligning X axis to Y direction."""
        rot = align_axis_to_direction(Axis.X, np.array([0, 1, 0]))
        result = rot.apply(np.array([1, 0, 0]))
        assert np.allclose(result, [0, 1, 0], atol=1e-10)

    def test_align_z_to_negative_x(self):
        """Test aligning Z axis to -X direction."""
        rot = align_axis_to_direction(Axis.Z, np.array([-1, 0, 0]))
        result = rot.apply(np.array([0, 0, 1]))
        assert np.allclose(result, [-1, 0, 0], atol=1e-10)

    def test_align_vector_to_direction(self):
        """Test aligning arbitrary vector to direction."""
        source = np.array([1, 1, 0]) / np.sqrt(2)
        target = np.array([0, 0, 1])
        rot = align_axis_to_direction(source, target)
        result = rot.apply(source)
        assert np.allclose(result, target, atol=1e-10)


class TestRotationBetweenAxes:
    """Tests for rotation_between_axes function."""

    def test_same_axis(self):
        """Test rotation between same axis is identity."""
        rot = rotation_between_axes(Axis.X, Axis.X)
        assert rot.is_identity()

    def test_x_to_y(self):
        """Test rotation from X to Y."""
        rot = rotation_between_axes(Axis.X, Axis.Y)
        result = rot.apply(np.array([1, 0, 0]))
        assert np.allclose(result, [0, 1, 0], atol=1e-10)

    def test_y_to_z(self):
        """Test rotation from Y to Z."""
        rot = rotation_between_axes(Axis.Y, Axis.Z)
        result = rot.apply(np.array([0, 1, 0]))
        assert np.allclose(result, [0, 0, 1], atol=1e-10)


class TestOrthonormalize:
    """Tests for orthonormalize function."""

    def test_already_orthogonal(self):
        """Test orthonormalizing already orthogonal matrix."""
        matrix = np.eye(3)
        result = orthonormalize(matrix)
        assert np.allclose(result, matrix)

    def test_with_numerical_errors(self):
        """Test fixing numerical errors in rotation matrix."""
        # Start with valid rotation, add small errors
        rot = Rotation3D.around_z(np.pi/4)
        noisy = rot.matrix + np.random.randn(3, 3) * 1e-8

        fixed = orthonormalize(noisy)

        # Should be orthogonal
        assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-10)
        # Should have det = +1
        assert np.isclose(np.linalg.det(fixed), 1.0)


class TestComputeOptimalView:
    """Tests for compute_optimal_view_rotation function."""

    def test_cube_any_view(self, cube_stl_path):
        """Test optimal view for cube (any view should work)."""
        from stl_drawing.io.stl_loader import load_stl
        vertices, faces = load_stl(str(cube_stl_path))

        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)

        rot, view = compute_optimal_view_rotation(vertices, faces, normals)

        # Should return a valid rotation
        assert isinstance(rot, Rotation3D)
        assert isinstance(view, StandardView)

    def test_flat_plate_top_view(self):
        """Test that flat plate prefers top/bottom view."""
        # Create flat plate in XZ plane
        vertices = np.array([
            [0, 0, 0], [10, 0, 0], [10, 0, 10], [0, 0, 10],
            [0, 1, 0], [10, 1, 0], [10, 1, 10], [0, 1, 10],
        ], dtype=np.float64)

        # Two triangles for top and bottom faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
        ], dtype=np.int32)

        # Compute normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)

        rot, view = compute_optimal_view_rotation(vertices, faces, normals)

        # For flat plate, top or bottom should have highest score
        assert view in [StandardView.TOP, StandardView.BOTTOM]


class TestViewDirections:
    """Tests for VIEW_DIRECTIONS constant."""

    def test_all_views_defined(self):
        """Test all standard views have directions defined."""
        for view in StandardView:
            assert view in VIEW_DIRECTIONS
            assert VIEW_DIRECTIONS[view].shape == (3,)

    def test_directions_are_unit(self):
        """Test all directions are unit vectors."""
        for view, direction in VIEW_DIRECTIONS.items():
            assert np.isclose(np.linalg.norm(direction), 1.0)

    def test_opposite_views(self):
        """Test opposite views have opposite directions."""
        pairs = [
            (StandardView.FRONT, StandardView.BACK),
            (StandardView.TOP, StandardView.BOTTOM),
            (StandardView.LEFT, StandardView.RIGHT),
        ]
        for v1, v2 in pairs:
            assert np.allclose(
                VIEW_DIRECTIONS[v1], -VIEW_DIRECTIONS[v2]
            )
