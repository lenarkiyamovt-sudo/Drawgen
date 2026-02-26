"""
Выбор лучшего главного вида и переориентация модели.

Оценивает все 6 аксиальных направлений по:
- площади видимых граней
- числу силуэтных рёбер
- числу различных нормалей (геометрическая сложность)

Выбирает направление с наибольшим score и переориентирует модель
так, чтобы выбранный «фронт» смотрел вдоль -Z (стандарт ЕСКД).
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def score_view_direction(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    view_dir: np.ndarray,
) -> Dict:
    """Оценить направление взгляда по информативности вида.

    Приоритет отдаётся видам с внутренними элементами (отверстия, рёбра,
    карманы), а не просто большой плоской поверхности.

    Score = visible_area × (1 + 0.3×n_silhouette + 0.5×n_distinct_normals)

    Args:
        vertices: вершины (N, 3).
        faces: грани (M, 3).
        face_normals: единичные нормали граней (M, 3).
        view_dir: единичный вектор направления взгляда.

    Returns:
        dict с ключами: score, visible_area, n_front_faces,
        n_silhouette, n_distinct_normals.
    """
    vd = view_dir / np.linalg.norm(view_dir)
    dot_products = face_normals @ vd  # (M,)

    # Грань «видна», если её нормаль направлена к наблюдателю (dot < 0)
    front_mask = dot_products < 0

    # Проецированная площадь видимых граней
    v0 = vertices[faces[front_mask, 0]]
    v1 = vertices[faces[front_mask, 1]]
    v2 = vertices[faces[front_mask, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    proj_areas = np.abs(cross @ vd) * 0.5
    visible_area = float(proj_areas.sum())

    # Силуэтные рёбра: граница между видимой и невидимой гранью
    n_silhouette = _count_silhouette_edges(faces, dot_products)

    # Число различных ориентаций нормалей (геометрическая сложность)
    front_normals = face_normals[front_mask]
    if len(front_normals) > 0:
        quantized = np.round(front_normals * 4) / 4  # квантизация ~15°
        n_distinct = len(set(map(tuple, quantized)))
    else:
        n_distinct = 0

    score = visible_area * (1.0 + 0.3 * n_silhouette + 0.5 * n_distinct)
    return {
        'score': score,
        'visible_area': visible_area,
        'n_front_faces': int(front_mask.sum()),
        'n_silhouette': n_silhouette,
        'n_distinct_normals': n_distinct,
    }


def _count_silhouette_edges(faces: np.ndarray, dot_products: np.ndarray) -> int:
    """Посчитать число силуэтных рёбер для данного вектора взгляда."""
    edge_face_map: Dict[Tuple, List[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        for j in range(3):
            ek = tuple(sorted((int(face[j]), int(face[(j + 1) % 3]))))
            edge_face_map[ek].append(fi)

    n_silhouette = 0
    for face_list in edge_face_map.values():
        if len(face_list) == 2:
            d0, d1 = dot_products[face_list[0]], dot_products[face_list[1]]
            if d0 * d1 < 0:
                n_silhouette += 1
    return n_silhouette


def select_best_front_and_reorient(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Выбрать лучший главный вид и переориентировать модель.

    После переориентации:
      - Выбранный «фронт» → направление -Z (камера смотрит вдоль +Z)
      - Наибольший из двух оставшихся размеров → X (ширина на чертеже)
      - Меньший оставшийся размер → Y (высота на чертеже)

    Args:
        vertices: вершины (N, 3).
        faces: грани (M, 3).
        face_normals: нормали граней (M, 3).

    Returns:
        (reoriented_vertices, reoriented_face_normals)
    """
    candidates = [
        ('front -Z', np.array([0.0,  0.0, -1.0])),
        ('front +Z', np.array([0.0,  0.0,  1.0])),
        ('front -X', np.array([-1.0, 0.0,  0.0])),
        ('front +X', np.array([1.0,  0.0,  0.0])),
        ('front -Y', np.array([0.0, -1.0,  0.0])),
        ('front +Y', np.array([0.0,  1.0,  0.0])),
    ]

    best_name, best_dir, best_score = None, None, -1.0
    for name, vdir in candidates:
        info = score_view_direction(vertices, faces, face_normals, vdir)
        logger.info(
            "  %-10s  area=%-10.0f  sil=%-4d  normals=%-3d  score=%.0f",
            name, info['visible_area'], info['n_silhouette'],
            info['n_distinct_normals'], info['score'],
        )
        if info['score'] > best_score:
            best_score = info['score']
            best_name = name
            best_dir = vdir.copy()

    logger.info("=> Лучший фронт: %s (score=%.0f)", best_name, best_score)

    # --- Строим матрицу вращения ---
    front_axis = int(np.argmax(np.abs(best_dir)))
    remaining = [i for i in range(3) if i != front_axis]
    bb = vertices.max(axis=0) - vertices.min(axis=0)

    # Наибольший из двух оставшихся → X (ширина)
    if bb[remaining[0]] >= bb[remaining[1]]:
        right_axis, up_axis = remaining[0], remaining[1]
    else:
        right_axis, up_axis = remaining[1], remaining[0]

    new_x = _unit_axis(right_axis)
    new_y = _unit_axis(up_axis)
    new_z = -best_dir  # камера смотрит вдоль -Z → new_Z = -view_dir

    # Гарантируем правую систему координат
    if np.dot(np.cross(new_x, new_y), new_z) < 0:
        new_y = -new_y

    R = np.stack([new_x, new_y, new_z], axis=0)  # (3, 3)

    new_verts = (vertices @ R.T).astype(np.float32)
    new_normals = (face_normals @ R.T).astype(np.float32)

    new_verts -= new_verts.min(axis=0)
    bb_new = new_verts.max(axis=0) - new_verts.min(axis=0)
    logger.info(
        "После переориентации: %.1f x %.1f x %.1f (X=ширина, Y=высота, Z=глубина)",
        *bb_new,
    )

    return new_verts, new_normals


def _unit_axis(axis_index: int) -> np.ndarray:
    """Вернуть единичный вектор мировой оси (0→X, 1→Y, 2→Z)."""
    v = np.zeros(3)
    v[axis_index] = 1.0
    return v
