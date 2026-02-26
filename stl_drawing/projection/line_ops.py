"""
Операции над отрезками в 2D: слияние коллинеарных, приоритет стилей.

Заменяет два дублирующих модуля оригинала:
  - merge_collinear_lines_improved
  - ESKDLineProcessor._merge_collinear_lines
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import norm


# Тип: отрезок = (точка_A, точка_B, стиль)  стиль ∈ {"visible", "hidden"}
Segment = Tuple[np.ndarray, np.ndarray, str]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _project_onto(pt: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """Параметрическая проекция точки на луч (origin, direction)."""
    return float(np.dot(pt - origin, direction))


def _find_collinear_overlap(
    seg_a: Segment,
    seg_b: Segment,
    eps: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Найти объединение двух коллинеарных перекрывающихся отрезков.

    Returns:
        (p_min, p_max) — объединённый отрезок, или None если отрезки
        не коллинеарны / не перекрываются.
    """
    pA, pB, _ = seg_a
    pC, pD, _ = seg_b

    v_ab = pB - pA
    v_cd = pD - pC
    len_ab = norm(v_ab)
    len_cd = norm(v_cd)

    if len_ab < eps or len_cd < eps:
        return None

    dir_a = v_ab / len_ab
    dir_b = v_cd / len_cd

    # Параллельность
    if abs(abs(float(np.dot(dir_a, dir_b))) - 1.0) > 1e-4:
        return None

    # Коллинеарность (расстояние между прямыми ≤ eps)
    v_ac = pC - pA
    cross = float(v_ac[0] * dir_a[1] - v_ac[1] * dir_a[0])
    if abs(cross) > eps:
        return None

    # Параметрическое перекрытие
    tA1 = _project_onto(pA, pA, dir_a)
    tA2 = _project_onto(pB, pA, dir_a)
    tB1 = _project_onto(pC, pA, dir_a)
    tB2 = _project_onto(pD, pA, dir_a)

    if tA2 < tA1:
        tA1, tA2 = tA2, tA1
    if tB2 < tB1:
        tB1, tB2 = tB2, tB1

    # Нет перекрытия
    if min(tA2, tB2) < max(tA1, tB1) - eps:
        return None

    t_min = min(tA1, tB1)
    t_max = max(tA2, tB2)
    return pA + t_min * dir_a, pA + t_max * dir_a


# ---------------------------------------------------------------------------
# Слияние коллинеарных отрезков одного стиля
# ---------------------------------------------------------------------------

def merge_collinear_segments(
    segments: List[Segment],
    model_scale: float,
) -> List[Segment]:
    """Объединить коллинеарные перекрывающиеся отрезки одного стиля.

    Сортирует отрезки по стилю и координатам, затем жадно объединяет
    коллинеарные отрезки одного стиля в более длинные.

    Args:
        segments: список отрезков (pA, pB, style).
        model_scale: характерный размер модели (для масштабирования eps).

    Returns:
        Список объединённых отрезков.
    """
    eps = 1e-6 * model_scale

    # Сортируем для детерминированности
    ordered = sorted(
        segments,
        key=lambda s: (s[2], float(s[0][0]), float(s[0][1]), float(s[1][0]), float(s[1][1])),
    )

    used = [False] * len(ordered)
    result: List[Segment] = []

    for i, base in enumerate(ordered):
        if used[i]:
            continue
        used[i] = True
        current = base

        for j in range(i + 1, len(ordered)):
            if used[j]:
                continue
            cand = ordered[j]
            if cand[2] != current[2]:
                continue  # разные стили — пропускаем

            merged = _find_collinear_overlap(current, cand, eps)
            if merged is not None:
                current = (merged[0], merged[1], current[2])
                used[j] = True

        result.append(current)
    return result


# ---------------------------------------------------------------------------
# Приоритет стилей: видимые перекрывают скрытые
# ---------------------------------------------------------------------------

def _subtract_intervals(
    base: Tuple[float, float],
    subtract_list: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Вычесть список интервалов из базового интервала.

    Args:
        base: базовый интервал (start, end), start ≤ end.
        subtract_list: список интервалов для вычитания.

    Returns:
        Список оставшихся интервалов после вычитания.
    """
    remaining = [base]
    for s_start, s_end in sorted(subtract_list):
        new_remaining = []
        for r_start, r_end in remaining:
            if s_end <= r_start or s_start >= r_end:
                new_remaining.append((r_start, r_end))
            else:
                if s_start > r_start:
                    new_remaining.append((r_start, s_start))
                if s_end < r_end:
                    new_remaining.append((s_end, r_end))
        remaining = new_remaining
    return remaining


def apply_style_priority(
    segments: List[Segment],
    model_scale: float,
) -> List[Segment]:
    """Применить приоритет стилей: видимые отрезки перекрывают скрытые.

    Там, где видимый и скрытый отрезки коллинеарно совпадают,
    скрытый обрезается. Участки скрытого, не перекрытые видимыми,
    сохраняются.

    Args:
        segments: список отрезков (pA, pB, style).
        model_scale: характерный размер для масштабирования eps.

    Returns:
        Список отрезков с применённым приоритетом.
    """
    eps = 1e-6 * model_scale

    visible_segs = [(pA, pB, st) for pA, pB, st in segments if st == "visible"]
    hidden_segs  = [(pA, pB, st) for pA, pB, st in segments if st == "hidden"]

    result: List[Segment] = list(visible_segs)

    for hA, hB, _ in hidden_segs:
        h_vec = hB - hA
        h_len = float(norm(h_vec))
        if h_len < eps:
            continue
        h_dir = h_vec / h_len

        # Находим участки видимых отрезков, коллинеарных со скрытым
        to_subtract: List[Tuple[float, float]] = []
        for vA, vB, _ in visible_segs:
            v_vec = vB - vA
            v_len = float(norm(v_vec))
            if v_len < eps:
                continue
            v_dir = v_vec / v_len

            # Параллельность
            if abs(abs(float(np.dot(h_dir, v_dir))) - 1.0) > 1e-4:
                continue

            # Коллинеарность
            cross = float((vA - hA)[0] * h_dir[1] - (vA - hA)[1] * h_dir[0])
            if abs(cross) > eps:
                continue

            tV1 = float(np.dot(vA - hA, h_dir))
            tV2 = float(np.dot(vB - hA, h_dir))
            if tV1 > tV2:
                tV1, tV2 = tV2, tV1

            # Клипуем к диапазону скрытого отрезка [0, h_len]
            tV1 = max(tV1, 0.0)
            tV2 = min(tV2, h_len)
            if tV2 > tV1 + eps:
                to_subtract.append((tV1, tV2))

        # Остаток скрытого после вычитания видимых
        for r_start, r_end in _subtract_intervals((0.0, h_len), to_subtract):
            if r_end - r_start < eps:
                continue
            result.append((hA + h_dir * r_start, hA + h_dir * r_end, "hidden"))

    return result
