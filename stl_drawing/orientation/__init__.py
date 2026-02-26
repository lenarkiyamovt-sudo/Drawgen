"""Ориентация модели: PCA-выравнивание и выбор главного вида."""

from stl_drawing.orientation.rotation import (
    Rotation3D,
    Axis,
    StandardView,
    VIEW_DIRECTIONS,
    get_view_rotation,
    align_axis_to_direction,
    compute_optimal_view_rotation,
    rotation_between_axes,
)

__all__ = [
    "Rotation3D",
    "Axis",
    "StandardView",
    "VIEW_DIRECTIONS",
    "get_view_rotation",
    "align_axis_to_direction",
    "compute_optimal_view_rotation",
    "rotation_between_axes",
]
