"""
Топологическая обработка 3D-сетки.

TopologyProcessor классифицирует рёбра на:
- острые (sharp) — угол между нормалями смежных граней превышает порог
- плавные (smooth) — кандидаты для силуэтных рёбер
- граничные (boundary) — принадлежащие только одной грани

Использует HalfEdgeMesh для эффективных топологических запросов.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.linalg import norm

from stl_drawing.config import ENABLE_SHARP_EDGES, MIN_EDGE_LENGTH, SHARP_ANGLE_DEGREES
from stl_drawing.topology.half_edge import HalfEdgeMesh

logger = logging.getLogger(__name__)

# Тип: ребро = упорядоченная пара индексов вершин
Edge = Tuple[int, int]


class TopologyProcessor:
    """Обрабатывает топологию сетки: находит острые, плавные и граничные рёбра.

    Использует HalfEdgeMesh для эффективных топологических операций.

    Attributes:
        vertices: вершины сетки (N, 3).
        faces: грани сетки (M, 3).
        face_normals: нормали граней (M, 3).
        edge_faces: словарь {edge -> set(face_indices)}.
        sharp_edges: рёбра с острым двугранным углом.
        smooth_edges: рёбра с плавным переходом (кандидаты в силуэты).
        half_edge_mesh: внутренняя half-edge структура данных.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        self.vertices: np.ndarray = np.asarray(vertices, dtype=np.float64)
        self.faces: np.ndarray = np.asarray(faces, dtype=np.int32)
        self.edge_faces: Dict[Edge, Set[int]] = defaultdict(set)
        self.face_normals: Optional[np.ndarray] = None
        self.sharp_edges: List[Edge] = []
        self.smooth_edges: List[Edge] = []

        # Создаём half-edge структуру
        self.half_edge_mesh: Optional[HalfEdgeMesh] = None

        self._build_topology()
        logger.info("TopologyProcessor: %d рёбер.", len(self.edge_faces))

    # ------------------------------------------------------------------
    # Построение топологии
    # ------------------------------------------------------------------

    def _build_topology(self) -> None:
        """Заполнить edge_faces и вычислить нормали граней.

        Использует HalfEdgeMesh для эффективных топологических запросов.
        """
        # Создать half-edge структуру
        self.half_edge_mesh = HalfEdgeMesh.from_faces(self.vertices, self.faces)

        # Для обратной совместимости заполнить edge_faces
        for face_idx, face in enumerate(self.faces):
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge: Edge = (min(v1, v2), max(v1, v2))
                self.edge_faces[edge].add(face_idx)

        # Вычислить нормали (использовать HalfEdgeMesh)
        self.face_normals = self.half_edge_mesh.compute_face_normals().astype(np.float32)

    # ------------------------------------------------------------------
    # Классификация рёбер
    # ------------------------------------------------------------------

    def classify_edges(self) -> List[Edge]:
        """Классифицировать рёбра и вернуть список острых/граничных рёбер.

        Использует HalfEdgeMesh.mark_sharp_edges() для эффективной классификации
        на основе двугранного угла между смежными гранями.

        Returns:
            Список острых и граничных рёбер (включаются в финальный чертёж).
            Плавные рёбра сохраняются в self.smooth_edges для силуэтного анализа.
        """
        self.sharp_edges = []
        self.smooth_edges = []

        if not ENABLE_SHARP_EDGES:
            # Все рёбра острые, если классификация отключена
            self.sharp_edges = list(self.edge_faces.keys())
            logger.info("Рёбра: %d (классификация отключена).", len(self.sharp_edges))
            return self.sharp_edges

        if self.half_edge_mesh is None:
            raise RuntimeError("HalfEdgeMesh не создан. Вызовите _build_topology() сначала.")

        # Пометить острые рёбра в half-edge структуре
        self.half_edge_mesh.mark_sharp_edges(
            angle_threshold_deg=SHARP_ANGLE_DEGREES,
            face_normals=self.face_normals,
        )

        # Собрать острые и плавные рёбра с фильтрацией по длине
        seen_edges: Set[Edge] = set()

        for he in self.half_edge_mesh.half_edges:
            v1, v2 = he.origin, he.target
            edge: Edge = (min(v1, v2), max(v1, v2))

            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            # Фильтр по минимальной длине ребра
            edge_length = float(norm(self.vertices[v2] - self.vertices[v1]))
            if edge_length < MIN_EDGE_LENGTH:
                continue

            if he.is_sharp:
                self.sharp_edges.append(edge)
            else:
                self.smooth_edges.append(edge)

        logger.info(
            "Рёбра: %d острых, %d плавных.",
            len(self.sharp_edges),
            len(self.smooth_edges),
        )
        return self.sharp_edges
