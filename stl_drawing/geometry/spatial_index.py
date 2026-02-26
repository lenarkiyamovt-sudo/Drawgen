"""
Пространственный индекс для треугольников в 2D (rtree-обёртка).

Изолирует зависимость от библиотеки `rtree` и обеспечивает
потокобезопасный доступ к индексу.
"""

import threading
from typing import List

import numpy as np
from rtree import index


_rtree_lock = threading.Lock()


def build_rtree_index(projected_vertices: np.ndarray, faces: np.ndarray) -> index.Index:
    """Построить 2D R-tree индекс по ограничивающим прямоугольникам треугольников.

    Индекс позволяет быстро находить треугольники, bbox которых перекрывает
    запрашиваемую точку или область.

    Args:
        projected_vertices: проецированные вершины (N, 3), используются только XY (колонки 0-1).
        faces: индексы вершин граней (M, 3).

    Returns:
        Построенный rtree Index (2D).
    """
    props = index.Property()
    props.dimension = 2
    rtree_idx = index.Index(properties=props)

    for face_id, face in enumerate(faces):
        tri_2d = projected_vertices[face, :2]
        min_xy = tri_2d.min(axis=0)
        max_xy = tri_2d.max(axis=0)
        rtree_idx.insert(face_id, (min_xy[0], min_xy[1], max_xy[0], max_xy[1]))

    return rtree_idx


def query_rtree(spatial_idx: index.Index, bounds) -> List[int]:
    """Потокобезопасный запрос к rtree по прямоугольной области.

    Args:
        spatial_idx: индекс, построенный через build_rtree_index.
        bounds: (min_x, min_y, max_x, max_y).

    Returns:
        Список идентификаторов граней, bbox которых пересекается с bounds.
    """
    with _rtree_lock:
        return list(spatial_idx.intersection(bounds))
