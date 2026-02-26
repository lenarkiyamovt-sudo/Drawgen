"""
Проецирование модели на плоскость вида и обработка рёбер.

ViewProcessor хранит матрицы вида для 6 ортогональных проекций
и изометрии, выполняет:
  - трансформацию вершин в систему координат вида
  - детектирование силуэтных рёбер
  - обработку каждого ребра (видимость + сегментация)
  - сборку финальных линий с применением слияния и приоритета стилей
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

from stl_drawing.config import ENABLE_MERGE, ENABLE_PRIORITY
from stl_drawing.geometry.spatial_index import build_rtree_index
from stl_drawing.geometry.visibility import process_edge_segments, visibility_cache
from stl_drawing.geometry.triangle import compute_scaled_tolerances
from stl_drawing.config import EPS_DEPTH, EPS_SEGMENT
from stl_drawing.projection.line_ops import apply_style_priority, merge_collinear_segments

logger = logging.getLogger(__name__)

Edge = Tuple[int, int]
Segment2D = Tuple[np.ndarray, np.ndarray, str]  # (pA, pB, style)


# ---------------------------------------------------------------------------
# Матрицы вида (world → view space)
# Строки матрицы: [right, up, depth]
# Проекция на плоскость = первые 2 координаты результата умножения p @ M
# ---------------------------------------------------------------------------
VIEW_MATRICES: Dict[str, np.ndarray] = {
    "front": np.array([
        [1,  0,  0],
        [0,  1,  0],
        [0,  0,  1],
    ], dtype=float),
    "top": np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0],
    ], dtype=float),
    "right": np.array([
        [0,  0,  1],
        [0,  1,  0],
        [-1, 0,  0],
    ], dtype=float),
    "left": np.array([
        [0,  0, -1],
        [0,  1,  0],
        [1,  0,  0],
    ], dtype=float),
    "back": np.array([
        [-1, 0,  0],
        [0,  1,  0],
        [0,  0, -1],
    ], dtype=float),
    "bottom": np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=float),
    "isometric": np.array([
        [0.707, -0.579,  0.406],
        [0.707,  0.579, -0.406],
        [0.000,  0.574,  0.819],
    ], dtype=float),
}

# Вектора направления взгляда в мировой системе координат для каждого вида
VIEW_DIRECTIONS: Dict[str, np.ndarray] = {
    "front":    np.array([0.0,  0.0, -1.0]),
    "top":      np.array([0.0, -1.0,  0.0]),
    "right":    np.array([1.0,  0.0,  0.0]),
    "left":     np.array([-1.0, 0.0,  0.0]),
    "back":     np.array([0.0,  0.0,  1.0]),
    "bottom":   np.array([0.0,  1.0,  0.0]),
    "isometric": np.array([0.0, 0.0, -1.0]),
}


class ViewProcessor:
    """Процессор ортогональных видов: трансформация + hidden line removal.

    Args:
        vertices: вершины модели (N, 3).
        sharp_edges: острые и граничные рёбра для отрисовки.
        faces: грани (M, 3).
        edge_faces: словарь {edge → set(face_indices)} для силуэтного анализа.
        face_normals: нормали граней (M, 3).
        smooth_edges: плавные рёбра — кандидаты в силуэты.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        sharp_edges: List[Edge],
        faces: np.ndarray,
        edge_faces: Optional[Dict] = None,
        face_normals: Optional[np.ndarray] = None,
        smooth_edges: Optional[List[Edge]] = None,
    ) -> None:
        self.vertices = vertices
        self.sharp_edges = sharp_edges
        self.faces = faces
        self.edge_faces = edge_faces or {}
        self.face_normals = face_normals
        self.smooth_edges = smooth_edges or []
        logger.info("ViewProcessor инициализирован.")

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def process_view(
        self, view_type: str
    ) -> Tuple[np.ndarray, List[Tuple], List[Tuple]]:
        """Обработать один вид: спроецировать, определить видимость, собрать линии.

        Args:
            view_type: имя вида ("front", "top", "right", "left", "back", "bottom").

        Returns:
            (projected_vertices, visible_lines, hidden_lines):
              - projected_vertices (N, 3): XY-проекция + Z-глубина
              - visible_lines: список (pA, pB) видимых отрезков
              - hidden_lines:  список (pA, pB) скрытых отрезков
        """
        visibility_cache.clear()

        projected = self._project_vertices(view_type)
        spatial_idx = build_rtree_index(projected, self.faces)
        depth_sign = self._compute_depth_sign(view_type)

        silhouette_edges = self._get_silhouette_edges(
            VIEW_DIRECTIONS.get(view_type, VIEW_DIRECTIONS["front"])
        )
        all_edges = list(self.sharp_edges) + silhouette_edges

        segments = self._process_all_edges(all_edges, projected, spatial_idx, depth_sign)

        if ENABLE_MERGE:
            model_scale = self._model_scale(projected)
            segments = merge_collinear_segments(segments, model_scale)

        if ENABLE_PRIORITY:
            model_scale = self._model_scale(projected)
            segments = apply_style_priority(segments, model_scale)

        visible_lines = [(pA, pB) for pA, pB, st in segments if st == "visible"]
        hidden_lines  = [(pA, pB) for pA, pB, st in segments if st == "hidden"]

        return projected, visible_lines, hidden_lines

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _project_vertices(self, view_type: str) -> np.ndarray:
        """Спроецировать вершины в систему координат вида.

        Returns:
            Массив (N, 3): первые 2 колонки — XY на плоскости вида,
            третья — Z-глубина.
        """
        if view_type not in VIEW_MATRICES:
            raise ValueError(f"Неизвестный тип вида: {view_type!r}")
        M = VIEW_MATRICES[view_type]
        transformed = self.vertices @ M.T   # (N, 3)
        return np.hstack((transformed[:, :2], transformed[:, 2:3]))

    def _compute_depth_sign(self, view_type: str) -> float:
        """Определить знак глубины для данного вида.

        После применения матрицы вида, Z-координата вершины означает
        глубину. Знак показывает: +1 → больший Z ближе к наблюдателю,
        -1 → меньший Z ближе.
        """
        view_dir = VIEW_DIRECTIONS.get(view_type, np.array([0.0, 0.0, -1.0]))
        M = VIEW_MATRICES.get(view_type, VIEW_MATRICES["front"])
        view_dir_in_view = view_dir @ M.T
        z_comp = float(view_dir_in_view[2])
        if abs(z_comp) < 1e-9:
            logger.warning("Вид %r: z-компонента направления ≈ 0, depth_sign=1.", view_type)
            return 1.0
        return -1.0 if z_comp > 0 else 1.0

    def _get_silhouette_edges(self, view_dir: np.ndarray) -> List[Edge]:
        """Найти силуэтные рёбра для данного направления взгляда.

        Силуэтное ребро — ребро, смежное с двумя гранями, одна из которых
        обращена к наблюдателю, другая — от него.
        """
        if self.face_normals is None or not self.smooth_edges:
            return []

        silhouette: List[Edge] = []
        for edge in self.smooth_edges:
            edge_key: Edge = (min(edge), max(edge))
            if edge_key not in self.edge_faces:
                continue
            face_set = self.edge_faces[edge_key]
            if len(face_set) != 2:
                continue

            fi0, fi1 = list(face_set)
            d0 = float(np.dot(self.face_normals[fi0], view_dir))
            d1 = float(np.dot(self.face_normals[fi1], view_dir))

            # Противоположные знаки → классический силуэт
            if d0 * d1 < 0:
                silhouette.append(edge)
            # Одна грань почти касательная → «мягкий» силуэт
            elif (abs(d0) < 0.01 or abs(d1) < 0.01) and abs(d0 - d1) > 0.01:
                silhouette.append(edge)

        return silhouette

    def _process_all_edges(
        self,
        edges: List[Edge],
        projected: np.ndarray,
        spatial_idx,
        depth_sign: float,
    ) -> List[Segment2D]:
        """Обработать все рёбра: определить видимость и собрать 2D-отрезки."""
        result: List[Segment2D] = []

        for edge in edges:
            e_tup: Edge = (int(edge[0]), int(edge[1]))
            segs = process_edge_segments(
                e_tup, projected, self.faces, spatial_idx, depth_sign
            )
            p0 = projected[e_tup[0]]
            p1 = projected[e_tup[1]]

            for t_start, t_end, is_visible in segs:
                if t_start >= t_end:
                    continue
                pA = (1 - t_start) * p0[:2] + t_start * p1[:2]
                pB = (1 - t_end)   * p0[:2] + t_end   * p1[:2]
                style = "visible" if is_visible else "hidden"
                result.append((pA, pB, style))

        return result

    @staticmethod
    def _model_scale(projected: np.ndarray) -> float:
        """Вычислить характерный размер проекции (для масштабирования eps)."""
        bbox_min = projected[:, :2].min(axis=0)
        bbox_max = projected[:, :2].max(axis=0)
        return float(max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]))
