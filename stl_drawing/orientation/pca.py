"""
PCA-ориентация модели по нормалям граней.

Алгоритм ищет естественные оси детали через взвешенную по площади
матрицу рассеяния нормалей граней. Это гораздо устойчивее, чем
вершинный PCA, для невыпуклых форм (L-профили, трубы, сборки).
"""

import logging
from typing import Tuple

import numpy as np

from stl_drawing.config import OBB_GAIN_THRESHOLD

logger = logging.getLogger(__name__)


def orient_model_by_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Выровнять модель по главным осям через PCA нормалей граней.

    Алгоритм:
      1. Вычислить нормали и площади всех треугольников.
      2. Построить матрицу рассеяния: M = Σ(area_i * n_i * n_i^T).
      3. Разложить M по собственным векторам → естественные оси детали.
      4. Применить вращение только если OBB значительно меньше AABB.

    Args:
        vertices: вершины (N, 3), float64.
        faces: грани (M, 3), int32.

    Returns:
        (oriented_vertices, rotation_matrix):
          - oriented_vertices (N, 3), float32 — вершины после вращения,
            сдвинутые так, чтобы min-угол bounding box оказался в начале координат.
          - rotation_matrix (3, 3) — применённая матрица вращения
            (np.eye(3) если PCA пропущен).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    aabb = verts.max(axis=0) - verts.min(axis=0)
    logger.info("AABB до ориентации: %.1f x %.1f x %.1f", *aabb)

    # --- Нормали и площади граней ---
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    cross_len = np.linalg.norm(cross, axis=1, keepdims=True)
    areas = (cross_len * 0.5).ravel()
    normals = cross / (cross_len + 1e-12)  # единичные нормали

    # --- Матрица рассеяния: M = (normals.T * areas) @ normals ---
    M = (normals.T * areas) @ normals  # (3, 3)

    eigenvalues, eigenvectors = np.linalg.eigh(M)  # возрастающий порядок
    order = np.argsort(eigenvalues)[::-1]           # убывающий порядок
    eigenvalues = eigenvalues[order]
    eigvecs = eigenvectors[:, order]                 # столбцы = собственные векторы

    # Правая система координат
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    _log_pca_alignment(eigvecs, eigenvalues)

    # --- Применять ли PCA? ---
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    oriented = centered @ eigvecs

    obb = oriented.max(axis=0) - oriented.min(axis=0)
    aabb_vol = float(np.prod(aabb))
    obb_vol = float(np.prod(obb))

    if aabb_vol > 0:
        ratio = obb_vol / aabb_vol
        logger.info("OBB/AABB объём: %.1f%%", 100.0 * ratio)

    if aabb_vol > 0 and obb_vol >= aabb_vol * OBB_GAIN_THRESHOLD:
        logger.info("PCA ПРОПУЩЕН: модель уже хорошо выровнена.")
        result = verts.copy()
        result -= result.min(axis=0)
        return result.astype(np.float32), np.eye(3)

    logger.info("PCA ПРИМЕНЁН: OBB значительно меньше AABB.")
    logger.info("AABB после PCA: %.1f x %.1f x %.1f", *obb)

    oriented -= oriented.min(axis=0)
    rotation_matrix = eigvecs.T
    return oriented.astype(np.float32), rotation_matrix


def _log_pca_alignment(eigvecs: np.ndarray, eigenvalues: np.ndarray) -> None:
    """Логировать соответствие главных компонент осям мировой системы."""
    axis_names = ("X", "Y", "Z")
    world_axes = np.eye(3)
    for i in range(3):
        pc = eigvecs[:, i]
        dots = [abs(float(np.dot(pc, world_axes[j]))) for j in range(3)]
        best_j = int(np.argmax(dots))
        angle_deg = float(np.degrees(np.arccos(np.clip(dots[best_j], 0.0, 1.0))))
        logger.info(
            "  PC%d -> ближайшая ось %s (угол=%.1f, eigenvalue=%.0f)",
            i, axis_names[best_j], angle_deg, eigenvalues[i],
        )
