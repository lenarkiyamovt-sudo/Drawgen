"""
Загрузка и нормализация STL-файлов.

Единственная ответственность: прочитать файл и вернуть
дедуплицированные вершины + индексы граней.
"""

import logging
from typing import List, Tuple

import numpy as np
from stl import mesh

logger = logging.getLogger(__name__)


class STLLoadError(Exception):
    """Ошибка при загрузке или разборе STL-файла."""


def load_stl(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Загрузить STL-файл и вернуть уникальные вершины и грани.

    Дедупликация вершин выполняется по точному совпадению координат,
    округлённых до 6 знаков после запятой.

    Args:
        filepath: Путь к STL-файлу (бинарный или ASCII).

    Returns:
        vertices: массив формы (N, 3), float64 — уникальные вершины.
        faces:    массив формы (M, 3), int32  — индексы вершин каждого треугольника.

    Raises:
        STLLoadError: если файл не найден, повреждён или содержит 0 треугольников.
    """
    logger.info("Загрузка STL: %s", filepath)

    try:
        stl_mesh = mesh.Mesh.from_file(filepath)
    except FileNotFoundError:
        raise STLLoadError(f"Файл не найден: {filepath!r}")
    except Exception as exc:
        raise STLLoadError(f"Не удалось прочитать STL-файл {filepath!r}: {exc}") from exc

    if len(stl_mesh.vectors) == 0:
        raise STLLoadError(f"STL-файл {filepath!r} не содержит треугольников.")

    vertices: List[np.ndarray] = []
    faces: List[Tuple[int, int, int]] = []
    vertex_index: dict = {}

    for triangle in stl_mesh.vectors:
        face_indices = []
        for raw_vertex in triangle:
            # Округляем до 6 знаков, чтобы объединить вершины с небольшими
            # погрешностями численного представления из STL.
            key = tuple(round(float(c), 6) for c in raw_vertex)
            if key not in vertex_index:
                vertex_index[key] = len(vertices)
                vertices.append(np.array(key, dtype=np.float64))
            face_indices.append(vertex_index[key])
        faces.append(tuple(face_indices))

    vertices_arr = np.array(vertices, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int32)

    logger.info(
        "Загружено: %d уникальных вершин, %d граней.",
        len(vertices_arr),
        len(faces_arr),
    )
    return vertices_arr, faces_arr
