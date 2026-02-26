"""
Топологическая обработка 3D-сетки.

TopologyProcessor классифицирует рёбра на:
- острые (sharp) — угол между нормалями смежных граней превышает порог
- плавные (smooth) — кандидаты для силуэтных рёбер
- граничные (boundary) — принадлежащие только одной грани
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.linalg import norm

from stl_drawing.config import ENABLE_SHARP_EDGES, MIN_EDGE_LENGTH, SHARP_ANGLE_COS

logger = logging.getLogger(__name__)

# Тип: ребро = упорядоченная пара индексов вершин
Edge = Tuple[int, int]


class TopologyProcessor:
    """Обрабатывает топологию сетки: находит острые, плавные и граничные рёбра.

    Attributes:
        vertices: вершины сетки (N, 3).
        faces: грани сетки (M, 3).
        face_normals: нормали граней (M, 3).
        edge_faces: словарь {edge -> set(face_indices)}.
        sharp_edges: рёбра с острым двугранным углом.
        smooth_edges: рёбра с плавным переходом (кандидаты в силуэты).
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        self.vertices: np.ndarray = np.asarray(vertices, dtype=np.float32)
        self.faces: np.ndarray = np.asarray(faces, dtype=np.int32)
        self.edge_faces: Dict[Edge, Set[int]] = defaultdict(set)
        self.face_normals: Optional[np.ndarray] = None
        self.sharp_edges: List[Edge] = []
        self.smooth_edges: List[Edge] = []

        self._build_topology()
        logger.info("TopologyProcessor: %d рёбер.", len(self.edge_faces))

    # ------------------------------------------------------------------
    # Построение топологии
    # ------------------------------------------------------------------

    def _build_topology(self) -> None:
        """Заполнить edge_faces и вычислить нормали граней."""
        for face_idx, face in enumerate(self.faces):
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge: Edge = (min(v1, v2), max(v1, v2))
                self.edge_faces[edge].add(face_idx)

        self.face_normals = self._compute_face_normals()

    def _compute_face_normals(self) -> np.ndarray:
        """Вычислить единичные нормали для каждой грани."""
        v01 = self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]]
        v02 = self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]]
        raw = np.cross(v01, v02)
        lengths = norm(raw, axis=1)

        normals = np.zeros_like(raw)
        valid = lengths > 1e-12
        normals[valid] = raw[valid] / lengths[valid, None]
        normals[~valid] = [0.0, 0.0, 1.0]  # вырожденная грань → произвольная нормаль
        return normals.astype(np.float32)

    # ------------------------------------------------------------------
    # Классификация рёбер
    # ------------------------------------------------------------------

    def classify_edges(self) -> List[Edge]:
        """Классифицировать рёбра и вернуть список острых/граничных рёбер.

        Returns:
            Список острых и граничных рёбер (включаются в финальный чертёж).
            Плавные рёбра сохраняются в self.smooth_edges для силуэтного анализа.
        """
        edge_items = list(self.edge_faces.items())
        args = [
            (edge, face_set, self.face_normals, self.vertices)
            for edge, face_set in edge_items
        ]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._classify_single_edge, args))

        self.sharp_edges = []
        self.smooth_edges = []

        for edge, is_sharp in results:
            if is_sharp:
                self.sharp_edges.append(edge)
            else:
                self.smooth_edges.append(edge)

        logger.info(
            "Рёбра: %d острых, %d плавных.",
            len(self.sharp_edges),
            len(self.smooth_edges),
        )
        return self.sharp_edges

    @staticmethod
    def _classify_single_edge(
        args: Tuple[Edge, Set[int], np.ndarray, np.ndarray]
    ) -> Tuple[Edge, bool]:
        """Классифицировать одно ребро (вызывается в пуле потоков).

        Returns:
            (edge, is_sharp): True если ребро острое или граничное.
        """
        edge, face_set, face_normals, vertices = args

        if not ENABLE_SHARP_EDGES:
            return edge, True

        # Граничное ребро (только одна грань) — всегда включаем
        if len(face_set) != 2:
            return edge, True

        face_list = list(face_set)
        n0 = face_normals[face_list[0]]
        n1 = face_normals[face_list[1]]
        dot = float(np.dot(n0, n1))

        edge_length = float(norm(vertices[edge[1]] - vertices[edge[0]]))
        is_sharp = (dot < SHARP_ANGLE_COS) and (edge_length >= MIN_EDGE_LENGTH)
        return edge, is_sharp
