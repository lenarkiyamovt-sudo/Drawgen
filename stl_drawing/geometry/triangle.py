"""
Операции над треугольниками в 2D.

Содержит:
- проверку принадлежности точки треугольнику (барицентрические координаты)
- интерполяцию Z-значения внутри треугольника
- вычисление масштабированных порогов точности
"""

from typing import Tuple

import numpy as np


def point_in_triangle(pt: np.ndarray, triangle_2d: np.ndarray) -> bool:
    """Проверить, лежит ли точка внутри треугольника (метод барицентрических координат).

    Args:
        pt: точка [x, y].
        triangle_2d: вершины треугольника формы (3, 2).

    Returns:
        True если точка внутри (включая границу).
    """
    a, b, c = triangle_2d
    v0 = c - a
    v1 = b - a
    v2 = pt - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0)


def interpolate_z(
    pt_2d: np.ndarray,
    triangle_2d: np.ndarray,
    triangle_z: np.ndarray,
) -> float:
    """Билинейная интерполяция Z-значения в точке внутри треугольника.

    Использует барицентрические координаты для взвешенного усреднения
    Z-значений в вершинах.

    Args:
        pt_2d: точка [x, y] для интерполяции.
        triangle_2d: вершины треугольника формы (3, 2).
        triangle_z: Z-значения в вершинах формы (3,).

    Returns:
        Интерполированное Z-значение, или np.inf если треугольник вырожден.
    """
    a, b, c = triangle_2d
    z_a, z_b, z_c = triangle_z

    v0 = c - a
    v1 = b - a
    v2 = pt_2d - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return np.inf

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v
    return w * z_a + u * z_c + v * z_b


def compute_scaled_tolerances(
    projected_vertices: np.ndarray,
    eps_depth: float,
    eps_segment: float,
) -> Tuple[float, float]:
    """Вычислить абсолютные пороги точности, масштабированные к размеру модели.

    Относительные пороги (EPS_DEPTH, EPS_SEGMENT из config) умножаются на
    характерный размер проекции, чтобы оставаться корректными при любом
    масштабе модели.

    Args:
        projected_vertices: проецированные вершины формы (N, 3), колонки XY + Z-глубина.
        eps_depth: относительный порог глубины (из config.EPS_DEPTH).
        eps_segment: относительный порог длины сегмента (из config.EPS_SEGMENT).

    Returns:
        (scaled_eps_depth, scaled_eps_segment) — абсолютные пороги.
    """
    bbox_min = np.min(projected_vertices[:, :2], axis=0)
    bbox_max = np.max(projected_vertices[:, :2], axis=0)
    bbox_size = bbox_max - bbox_min
    model_scale = float(max(bbox_size[0], bbox_size[1]))
    return eps_depth * model_scale, eps_segment * model_scale
