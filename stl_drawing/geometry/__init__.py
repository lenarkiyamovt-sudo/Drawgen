"""Геометрические примитивы: треугольники, видимость, пространственный индекс."""

from stl_drawing.geometry.mesh_stats import (
    BoundingBox,
    MeshStatistics,
    calculate_bounding_box,
    calculate_mesh_statistics,
    calculate_surface_area,
    calculate_volume,
)

__all__ = [
    "BoundingBox",
    "MeshStatistics",
    "calculate_bounding_box",
    "calculate_mesh_statistics",
    "calculate_surface_area",
    "calculate_volume",
]
