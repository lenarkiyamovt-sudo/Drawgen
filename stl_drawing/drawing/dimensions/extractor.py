"""
Извлечение размеров из данных видов (ГОСТ 2.307-2011).

Три источника размеров:
  1. Габаритные размеры (из bbox каждого вида)
  2. Диаметры цилиндров (из CylinderDetector)
  3. Ступенчатые размеры (из кластеров параллельных рёбер)

Все размеры представляются как DimensionCandidate с каноническим ключом
для последующей дедупликации между видами.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from stl_drawing.projection.view_processor import VIEW_DIRECTIONS, VIEW_MATRICES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Маппинг видов на 3D-оси
# ---------------------------------------------------------------------------

# Какие 3D-оси отображаются на горизонтальную (X') и вертикальную (Y')
# оси каждого вида:
#   view_name → (3D_axis_for_horizontal, 3D_axis_for_vertical)
VIEW_AXIS_MAP: Dict[str, Tuple[str, str]] = {
    'front':  ('X', 'Y'),
    'back':   ('X', 'Y'),
    'top':    ('X', 'Z'),
    'bottom': ('X', 'Z'),
    'right':  ('Z', 'Y'),
    'left':   ('Z', 'Y'),
}


# ---------------------------------------------------------------------------
# Структура данных: кандидат размера
# ---------------------------------------------------------------------------

@dataclass
class DimensionCandidate:
    """Кандидат размера, извлечённый из одного вида.

    Attributes:
        dim_type: тип размера: "linear_horizontal", "linear_vertical", "diameter"
        value_mm: значение размера в мм модели
        canonical_key: уникальный ключ для дедупликации между видами
        view_name: имя вида, из которого извлечён
        anchor_a: начальная точка измеряемого элемента (координаты вида)
        anchor_b: конечная точка (координаты вида)
        preferred_side: предпочтительная сторона размещения
        priority: приоритет (ниже = важнее): 0=габарит, 1=диаметр, 2=ступень
        center: центр (для диаметров)
        radius: радиус (для диаметров)
    """
    dim_type: str
    value_mm: float
    canonical_key: str
    view_name: str
    anchor_a: Tuple[float, float]
    anchor_b: Tuple[float, float]
    preferred_side: str = 'bottom'
    priority: int = 0
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None


# ---------------------------------------------------------------------------
# Главная функция извлечения
# ---------------------------------------------------------------------------

def extract_dimensions(
    views_data: Dict[str, Dict],
    cylinders: List[dict],
    selected_views: List[str],
) -> List[DimensionCandidate]:
    """Извлечь все кандидаты размеров из данных видов.

    Args:
        views_data: словарь {view_name → view_data} с ключами
                    'visible', 'hidden', 'bbox', 'centerlines'.
        cylinders: список цилиндров из CylinderDetector.
        selected_views: список выбранных видов для чертежа.

    Returns:
        Список DimensionCandidate из всех видов.
    """
    candidates: List[DimensionCandidate] = []

    for view_name in selected_views:
        if view_name not in views_data:
            continue
        vd = views_data[view_name]
        bbox = vd['bbox']

        # 1. Габаритные размеры
        overall = _extract_overall_dims(view_name, bbox)
        candidates.extend(overall)

        # 2. Диаметры цилиндров
        cyl_dims = _extract_cylinder_diameters(
            view_name, cylinders, vd.get('centerlines', []), bbox)
        candidates.extend(cyl_dims)

        # 3. Ступенчатые размеры (из видимых рёбер)
        step_dims = _extract_step_dims(view_name, vd['visible'], bbox)
        candidates.extend(step_dims)

    logger.info("Извлечено %d кандидатов размеров из %d видов",
                len(candidates), len(selected_views))
    return candidates


# ---------------------------------------------------------------------------
# 1. Габаритные размеры
# ---------------------------------------------------------------------------

def _extract_overall_dims(
    view_name: str,
    bbox: Dict,
) -> List[DimensionCandidate]:
    """Извлечь габаритные размеры (ширина и высота) из bbox вида."""
    results = []
    axis_h, axis_v = VIEW_AXIS_MAP.get(view_name, ('?', '?'))

    width = bbox['width']
    height = bbox['height']

    if width > 1e-6:
        results.append(DimensionCandidate(
            dim_type='linear_horizontal',
            value_mm=width,
            canonical_key=f'overall_{axis_h}',
            view_name=view_name,
            anchor_a=(bbox['min_x'], bbox['min_y']),
            anchor_b=(bbox['max_x'], bbox['min_y']),
            preferred_side='bottom',
            priority=0,
        ))

    if height > 1e-6:
        results.append(DimensionCandidate(
            dim_type='linear_vertical',
            value_mm=height,
            canonical_key=f'overall_{axis_v}',
            view_name=view_name,
            anchor_a=(bbox['min_x'], bbox['min_y']),
            anchor_b=(bbox['min_x'], bbox['max_y']),
            preferred_side='left',
            priority=0,
        ))

    return results


# ---------------------------------------------------------------------------
# 2. Диаметры цилиндров
# ---------------------------------------------------------------------------

_AXIS_PERP_THRESHOLD = 0.95  # порог: ось ⊥ плоскости вида → перекрестие


def _extract_cylinder_diameters(
    view_name: str,
    cylinders: List[dict],
    centerlines: List[dict],
    bbox: Dict,
) -> List[DimensionCandidate]:
    """Извлечь размеры диаметров цилиндров.

    Для перекрестий (ось перпендикулярна виду) — размер Ø с лидером.
    Для осевых линий (ось параллельна виду) — линейный размер поперёк.
    """
    results = []
    if not cylinders or view_name not in VIEW_MATRICES:
        return results

    M = VIEW_MATRICES[view_name]
    view_dir = np.array(VIEW_DIRECTIONS[view_name], dtype=np.float64)

    for i, cyl in enumerate(cylinders):
        axis = np.asarray(cyl['axis'], dtype=np.float64)
        center_3d = np.asarray(cyl['center'], dtype=np.float64)
        radius = float(cyl['radius'])
        diameter = 2.0 * radius

        alignment = abs(float(axis @ view_dir))
        center_2d = (center_3d @ M.T)[:2]
        cx, cy = float(center_2d[0]), float(center_2d[1])

        canonical = f'cylinder_diameter_{i}'

        if alignment > _AXIS_PERP_THRESHOLD:
            # Перекрестие: диаметр виден как окружность → размер Ø
            results.append(DimensionCandidate(
                dim_type='diameter',
                value_mm=diameter,
                canonical_key=canonical,
                view_name=view_name,
                anchor_a=(cx - radius, cy),
                anchor_b=(cx + radius, cy),
                preferred_side='right',
                priority=1,
                center=(cx, cy),
                radius=radius,
            ))
        else:
            # Осевая линия: диаметр как линейный размер поперёк оси
            # Находим перпендикулярное направление к проекции оси
            axis_2d = (axis @ M.T)[:2]
            ax_len = float(np.linalg.norm(axis_2d))
            if ax_len < 1e-9:
                continue

            # Перпендикуляр к оси в плоскости вида
            perp = np.array([-axis_2d[1], axis_2d[0]]) / ax_len

            # Точки на противоположных сторонах цилиндра
            pa = center_2d + perp * radius
            pb = center_2d - perp * radius

            # Определить ориентацию размера
            if abs(perp[0]) > abs(perp[1]):
                dim_type = 'linear_horizontal'
                side = 'top'
            else:
                dim_type = 'linear_vertical'
                side = 'right'

            results.append(DimensionCandidate(
                dim_type=dim_type,
                value_mm=diameter,
                canonical_key=canonical,
                view_name=view_name,
                anchor_a=(float(pa[0]), float(pa[1])),
                anchor_b=(float(pb[0]), float(pb[1])),
                preferred_side=side,
                priority=1,
                center=(cx, cy),
                radius=radius,
            ))

    return results


# ---------------------------------------------------------------------------
# 3. Ступенчатые размеры
# ---------------------------------------------------------------------------

def _extract_step_dims(
    view_name: str,
    visible_lines: List[Tuple],
    bbox: Dict,
) -> List[DimensionCandidate]:
    """Извлечь ступенчатые размеры из кластеров параллельных видимых рёбер.

    Кластеризует горизонтальные и вертикальные рёбра, находит
    характерные промежуточные координаты (ступени) и генерирует размеры.
    """
    results = []
    axis_h, axis_v = VIEW_AXIS_MAP.get(view_name, ('?', '?'))

    width = bbox['width']
    height = bbox['height']

    # Пороги
    tol_h = height * 0.005 if height > 0 else 1e-3
    tol_v = width * 0.005 if width > 0 else 1e-3
    min_step_ratio = 0.10   # минимальный относительный размер ступени
    max_step_ratio = 0.90   # максимальный (исключаем ≈ overall)
    max_steps_per_axis = 4  # ограничение числа ступеней на ось

    # --- Горизонтальные рёбра → ступени по вертикали ---
    h_coords = _cluster_edge_coords(visible_lines, axis='horizontal', tol=tol_h)
    if len(h_coords) > 2:
        sorted_y = sorted(h_coords)
        step_cands = []
        for y_val in sorted_y:
            step_h = y_val - bbox['min_y']
            if step_h > height * min_step_ratio and step_h < height * max_step_ratio:
                step_cands.append((step_h, y_val))

        # Оставляем только наиболее значимые ступени (наибольшие расстояния)
        step_cands = _select_significant_steps(step_cands, max_steps_per_axis)

        for step_h, y_val in step_cands:
            canon_key = f'step_{axis_v}_{round(step_h, 2)}'
            results.append(DimensionCandidate(
                dim_type='linear_vertical',
                value_mm=step_h,
                canonical_key=canon_key,
                view_name=view_name,
                anchor_a=(bbox['min_x'], bbox['min_y']),
                anchor_b=(bbox['min_x'], y_val),
                preferred_side='left',
                priority=2,
            ))

    # --- Вертикальные рёбра → ступени по горизонтали ---
    v_coords = _cluster_edge_coords(visible_lines, axis='vertical', tol=tol_v)
    if len(v_coords) > 2:
        sorted_x = sorted(v_coords)
        step_cands = []
        for x_val in sorted_x:
            step_w = x_val - bbox['min_x']
            if step_w > width * min_step_ratio and step_w < width * max_step_ratio:
                step_cands.append((step_w, x_val))

        step_cands = _select_significant_steps(step_cands, max_steps_per_axis)

        for step_w, x_val in step_cands:
            canon_key = f'step_{axis_h}_{round(step_w, 2)}'
            results.append(DimensionCandidate(
                dim_type='linear_horizontal',
                value_mm=step_w,
                canonical_key=canon_key,
                view_name=view_name,
                anchor_a=(bbox['min_x'], bbox['min_y']),
                anchor_b=(x_val, bbox['min_y']),
                preferred_side='bottom',
                priority=2,
            ))

    return results


def _select_significant_steps(
    step_cands: List[Tuple[float, float]],
    max_count: int,
) -> List[Tuple[float, float]]:
    """Выбрать наиболее значимые ступени из списка.

    Значимость определяется наибольшим минимальным расстоянием до соседей
    (разреженность), что предпочитает равномерно распределённые ступени.
    """
    if len(step_cands) <= max_count:
        return step_cands

    # Сортируем по позиции
    step_cands.sort(key=lambda x: x[0])

    # Вычисляем значимость: минимальное расстояние до соседей
    scored = []
    for i, (val, coord) in enumerate(step_cands):
        dist_prev = val - step_cands[i - 1][0] if i > 0 else val
        dist_next = step_cands[i + 1][0] - val if i < len(step_cands) - 1 else 1e9
        significance = min(dist_prev, dist_next)
        scored.append((significance, val, coord))

    # Берём наиболее значимые
    scored.sort(key=lambda x: -x[0])
    selected = [(val, coord) for _, val, coord in scored[:max_count]]
    selected.sort(key=lambda x: x[0])
    return selected


def _cluster_edge_coords(
    lines: List[Tuple],
    axis: str,
    tol: float,
) -> List[float]:
    """Кластеризовать координаты горизонтальных или вертикальных рёбер.

    Args:
        lines: список (pA, pB) отрезков.
        axis: 'horizontal' — собрать Y-координаты горизонтальных рёбер,
              'vertical'   — собрать X-координаты вертикальных рёбер.
        tol: допуск для определения горизонтальности/вертикальности.

    Returns:
        Список уникальных координат (представители кластеров).
    """
    coords: List[float] = []

    for pA, pB in lines:
        ax, ay = float(pA[0]), float(pA[1])
        bx, by = float(pB[0]), float(pB[1])

        if axis == 'horizontal':
            # Горизонтальное ребро: |dy| < tol
            if abs(by - ay) < tol:
                avg_y = (ay + by) / 2.0
                coords.append(avg_y)
        else:
            # Вертикальное ребро: |dx| < tol
            if abs(bx - ax) < tol:
                avg_x = (ax + bx) / 2.0
                coords.append(avg_x)

    if not coords:
        return []

    # Кластеризация: сортируем и объединяем близкие значения
    coords.sort()
    clusters: List[float] = [coords[0]]
    cluster_tol = tol * 3.0  # допуск объединения кластеров

    for c in coords[1:]:
        if c - clusters[-1] > cluster_tol:
            clusters.append(c)
        else:
            # Обновляем представителя как среднее
            clusters[-1] = (clusters[-1] + c) / 2.0

    return clusters
