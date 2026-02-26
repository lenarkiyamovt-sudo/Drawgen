"""
Определение видимости рёбер и адаптивная дискретизация.

Содержит:
- VisibilityCache  — потокобезопасный кэш результатов видимости
- is_point_visible — проверка видимости проецированной точки
- adaptive_sampling — итеративная (нерекурсивная) дискретизация ребра
- process_edge_segments — точка входа для обработки одного ребра
"""

import math
import threading
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

from stl_drawing.config import BASE_MAX_RECURSION_DEPTH, EPS_DEPTH, EPS_SEGMENT
from stl_drawing.geometry.triangle import (
    compute_scaled_tolerances,
    interpolate_z,
    point_in_triangle,
)
from stl_drawing.geometry.spatial_index import query_rtree


# ---------------------------------------------------------------------------
# Кэш видимости
# ---------------------------------------------------------------------------

class VisibilityCache:
    """Потокобезопасный кэш результатов проверки видимости точек.

    Точки квантизируются по сетке grid_size для уменьшения числа
    уникальных ключей и ускорения поиска.
    """

    def __init__(self, grid_size: float = 1e-3) -> None:
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._grid_size = grid_size

    def _snap(self, coord: Tuple[float, float, float]) -> Tuple[float, float, float]:
        g = self._grid_size
        return tuple((c // g) * g for c in coord)

    def get(self, x: float, y: float, z: float) -> Optional[bool]:
        key = self._snap((x, y, z))
        with self._lock:
            return self._cache.get(key)

    def set(self, x: float, y: float, z: float, value: bool) -> None:
        key = self._snap((x, y, z))
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# Глобальный кэш (сбрасывается перед обработкой каждого вида)
visibility_cache = VisibilityCache(grid_size=EPS_DEPTH * 10)


# ---------------------------------------------------------------------------
# Проверка видимости точки
# ---------------------------------------------------------------------------

def is_point_visible(
    x: float,
    y: float,
    z: float,
    spatial_idx,
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    depth_sign: float,
    eps_depth: float,
) -> bool:
    """Проверить, видима ли проецированная точка (не перекрыта ли другой гранью).

    Алгоритм:
      1. Ищем треугольники, bbox которых содержит (x, y).
      2. Для каждого кандидата: если точка внутри треугольника,
         интерполируем Z грани.
      3. Если грань расположена «ближе к наблюдателю», чем точка —
         точка перекрыта → невидима.

    Args:
        x, y, z: координаты проецированной точки (XY — на плоскости вида, Z — глубина).
        spatial_idx: rtree-индекс граней.
        projected_vertices: проецированные вершины (N, 3).
        faces: индексы граней (M, 3).
        depth_sign: +1 — больший Z ближе к наблюдателю; -1 — меньший Z ближе.
        eps_depth: абсолютный порог глубины (уже масштабированный).

    Returns:
        True если точка видима.
    """
    cached = visibility_cache.get(x, y, z)
    if cached is not None:
        return cached

    pt_2d = np.array([x, y])
    candidate_faces = query_rtree(spatial_idx, (x, y, x, y))
    visible = True

    for face_id in candidate_faces:
        face = faces[face_id]
        tri_2d = projected_vertices[face, :2]
        tri_z = projected_vertices[face, 2]

        if point_in_triangle(pt_2d, tri_2d):
            z_face = interpolate_z(pt_2d, tri_2d, tri_z)
            # Грань перекрывает точку, если она находится между наблюдателем и точкой
            if (z_face - z) * depth_sign > eps_depth:
                visible = False
                break

    visibility_cache.set(x, y, z, visible)
    return visible


# ---------------------------------------------------------------------------
# Адаптивная дискретизация ребра (итеративная версия)
# ---------------------------------------------------------------------------

def _compute_dynamic_max_depth(
    edge: Tuple[int, int],
    projected_vertices: np.ndarray,
    scaled_eps_depth: float,
) -> int:
    """Вычислить максимальную глубину разбиения для данного ребра.

    Учитывает 2D-длину ребра и перепад глубины, чтобы детальнее
    дискретизировать длинные и/или «наклонённые» в глубину рёбра.
    """
    pA = projected_vertices[edge[0]]
    pB = projected_vertices[edge[1]]
    length_2d = norm(pB[:2] - pA[:2])
    delta_z = abs(pB[2] - pA[2])

    depth_for_length = (
        int(math.ceil(math.log2(length_2d / scaled_eps_depth)))
        if length_2d > scaled_eps_depth
        else 0
    )
    depth_for_z = 1 if delta_z > 10 * scaled_eps_depth else 0

    return BASE_MAX_RECURSION_DEPTH + max(depth_for_length, depth_for_z)


def adaptive_sampling(
    edge: Tuple[int, int],
    projected_vertices: np.ndarray,
    spatial_idx,
    faces: np.ndarray,
    depth_sign: float,
    scaled_eps_depth: float,
    scaled_eps_segment: float,
    max_depth: int,
) -> List[Tuple[float, float, bool]]:
    """Итеративная адаптивная дискретизация ребра по видимости.

    Разбивает параметрический отрезок [0, 1] ребра на сегменты с
    постоянной видимостью, рекурсивно (через явный стек) уточняя
    границы перехода видимый/невидимый.

    Args:
        edge: пара индексов вершин (i, j).
        projected_vertices: проецированные вершины с Z-глубиной (N, 3).
        spatial_idx: rtree-индекс для проверки видимости.
        faces: индексы граней (M, 3).
        depth_sign: знак глубины для текущего вида (+1 или -1).
        scaled_eps_depth: абсолютный порог глубины.
        scaled_eps_segment: минимальная длина 2D-сегмента для сохранения.
        max_depth: максимальная глубина дерева разбиения.

    Returns:
        Список кортежей (t_start, t_end, is_visible), где t_start/t_end
        — параметрические координаты на ребре [0, 1].
    """
    pA_3d = projected_vertices[edge[0]]
    pB_3d = projected_vertices[edge[1]]

    def interp(t: float) -> Tuple[np.ndarray, float]:
        xy = (1 - t) * pA_3d[:2] + t * pB_3d[:2]
        z = (1 - t) * pA_3d[2] + t * pB_3d[2]
        return xy, z

    def check_visible(t: float) -> Tuple[np.ndarray, float, bool]:
        xy, z = interp(t)
        vis = is_point_visible(
            xy[0], xy[1], z,
            spatial_idx, projected_vertices, faces,
            depth_sign, scaled_eps_depth,
        )
        return xy, z, vis

    def seg_length_2d(t0: float, t1: float) -> float:
        p0 = (1 - t0) * pA_3d[:2] + t0 * pB_3d[:2]
        p1 = (1 - t1) * pA_3d[:2] + t1 * pB_3d[:2]
        return float(norm(p1 - p0))

    # Стек: (t0, t1, pt0_data, pt1_data, depth)
    # pt_data = (xy, z, vis) или None — вычисляется лениво
    initial_pt0 = check_visible(0.0)
    initial_pt1 = check_visible(1.0)
    stack = [(0.0, 1.0, initial_pt0, initial_pt1, 0)]
    result: List[Tuple[float, float, bool]] = []

    while stack:
        t0, t1, pt0, pt1, depth = stack.pop()
        _, _, vis0 = pt0
        _, _, vis1 = pt1

        if depth >= max_depth:
            # Максимальная глубина: сохраняем сегмент если он достаточно длинный
            if seg_length_2d(t0, t1) >= scaled_eps_segment:
                result.append((t0, t1, vis0))
            continue

        mid_t = 0.5 * (t0 + t1)
        pt_mid = check_visible(mid_t)
        _, _, vis_mid = pt_mid

        if vis_mid == vis0 and vis_mid == vis1:
            # Весь сегмент одного типа — сохраняем целиком
            if seg_length_2d(t0, t1) >= scaled_eps_segment:
                result.append((t0, t1, vis0))
        else:
            # Переход видимости — разбиваем, правая половина в стек первой
            # (чтобы при pop() левая обрабатывалась первой → порядок сохраняется)
            stack.append((mid_t, t1, pt_mid, pt1, depth + 1))
            stack.append((t0, mid_t, pt0, pt_mid, depth + 1))

    # Сортируем результат по t0 (стек нарушает порядок при разбиении)
    result.sort(key=lambda s: s[0])
    return result


def process_edge_segments(
    edge: Tuple[int, int],
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    spatial_idx,
    depth_sign: float,
) -> List[Tuple[float, float, bool]]:
    """Обработать ребро: определить сегменты с постоянной видимостью.

    Высокоуровневая функция: вычисляет пороги, глубину разбиения и
    делегирует adaptive_sampling.

    Args:
        edge: пара индексов вершин.
        projected_vertices: проецированные вершины (N, 3).
        faces: индексы граней (M, 3).
        spatial_idx: rtree-индекс.
        depth_sign: +1 или -1 — знак глубины для данного вида.

    Returns:
        Список (t_start, t_end, is_visible).
    """
    scaled_eps_depth, scaled_eps_segment = compute_scaled_tolerances(
        projected_vertices, EPS_DEPTH, EPS_SEGMENT
    )
    max_depth = _compute_dynamic_max_depth(edge, projected_vertices, scaled_eps_depth)

    return adaptive_sampling(
        edge=edge,
        projected_vertices=projected_vertices,
        spatial_idx=spatial_idx,
        faces=faces,
        depth_sign=depth_sign,
        scaled_eps_depth=scaled_eps_depth,
        scaled_eps_segment=scaled_eps_segment,
        max_depth=max_depth,
    )
