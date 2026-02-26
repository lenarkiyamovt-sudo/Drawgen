"""
Алгоритм выбора минимального набора видов по ГОСТ 2.305-2008.

«Количество изображений должно быть наименьшим, но обеспечивающим
полное представление о предмете» (ГОСТ 2.305-2008 п.5.1).

Модуль изолирует логику фильтрации видов от ESKDDrawingSheet.
"""

import logging
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
from numpy.linalg import norm

from stl_drawing.config import (
    VIEW_HIDDEN_EDGE_WEIGHT,
    VIEW_MIN_INFO_SCORE,
    VIEW_SCORE_RATIO_MIN,
    VIEW_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Противоположные пары видов
OPPOSITE_PAIRS: List[Tuple[str, str]] = [
    ('front', 'back'),
    ('top',   'bottom'),
    ('left',  'right'),
]

# Оси, покрываемые каждым видом
AXIS_MAP: Dict[str, FrozenSet[str]] = {
    'front':  frozenset({'X', 'Y'}),
    'back':   frozenset({'X', 'Y'}),
    'top':    frozenset({'X', 'Z'}),
    'bottom': frozenset({'X', 'Z'}),
    'left':   frozenset({'Y', 'Z'}),
    'right':  frozenset({'Y', 'Z'}),
}

# Финальный порядок видов на чертеже
VIEW_ORDER: List[str] = ['front', 'top', 'bottom', 'right', 'left', 'back']


def select_necessary_views(
    views_data: Dict[str, Dict],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Определить минимальный набор видов по ГОСТ 2.305-2008.

    Алгоритм (5 шагов):
      1. Исключить противоположные виды с меньшей информацией.
      2. Исключить «пустые» виды (только контурный прямоугольник).
      3. Исключить зеркально-симметричные виды.
      4. Гарантировать покрытие всех 3 осей (X, Y, Z).
      5. Убрать информационно-избыточные виды.

    Args:
        views_data: словарь {view_name → {visible: [...], hidden: [...], bbox: {...}}}.

    Returns:
        (selected_views, exclusion_reasons):
          - selected_views: список имён видов в порядке VIEW_ORDER
          - exclusion_reasons: список (view_name, reason) для исключённых видов
    """
    if len(views_data) <= 3:
        return list(views_data.keys()), []

    # Подготовка вспомогательных структур
    edge_sets = _build_edge_sets(views_data)
    scores = {v: _info_score(views_data[v], edge_sets[v]) for v in views_data}
    logger.info("Информационные score видов: %s",
                {v: f"{s:.1f}" for v, s in sorted(scores.items())})

    excluded: Dict[str, str] = {}

    _step1_exclude_opposites(views_data, edge_sets, scores, excluded)
    _step2_exclude_empty(views_data, excluded)
    _step3_exclude_mirror_symmetric(views_data, edge_sets, scores, excluded)
    _step4_ensure_axis_coverage(views_data, scores, excluded)
    _step5_remove_redundant(views_data, scores, excluded)

    selected = [v for v in VIEW_ORDER if v in views_data and v not in excluded]
    reasons = sorted(excluded.items())

    logger.info("Выбрано %d видов: %s", len(selected), selected)
    for v, r in reasons:
        logger.info("  X %s: %s", v, r)

    return selected, reasons


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _normalize_edge(pA, pB, precision: int = 4):
    a = (round(float(pA[0]), precision), round(float(pA[1]), precision))
    b = (round(float(pB[0]), precision), round(float(pB[1]), precision))
    return (a, b) if a <= b else (b, a)


def _edge_set(edges, precision: int = 4) -> FrozenSet:
    return frozenset(_normalize_edge(pA, pB, precision) for pA, pB in edges)


def _build_edge_sets(views_data: Dict) -> Dict[str, Dict]:
    result = {}
    for name, data in views_data.items():
        vis = _edge_set(data['visible'])
        hid = _edge_set(data['hidden'])
        result[name] = {
            'vis': vis, 'hid': hid, 'all': vis | hid,
            'n_vis': len(vis), 'n_hid': len(hid),
        }
    return result


def _is_simple_rectangle(view_data: Dict) -> bool:
    """Проверить, является ли вид простым контурным прямоугольником.

    Простой прямоугольник: все рёбра лежат на контуре bbox, скрытых нет.
    Такой вид не несёт дополнительной геометрической информации.
    """
    bb = view_data['bbox']
    tol = (bb['width'] + bb['height']) * 0.005

    def on_contour(pA, pB) -> bool:
        ax, ay = float(pA[0]), float(pA[1])
        bx, by = float(pB[0]), float(pB[1])
        return (
            (abs(ax - bb['min_x']) < tol and abs(bx - bb['min_x']) < tol) or
            (abs(ax - bb['max_x']) < tol and abs(bx - bb['max_x']) < tol) or
            (abs(ay - bb['min_y']) < tol and abs(by - bb['min_y']) < tol) or
            (abs(ay - bb['max_y']) < tol and abs(by - bb['max_y']) < tol)
        )

    if len(view_data['hidden']) > 0:
        return False
    return all(on_contour(pA, pB) for pA, pB in view_data['visible'])


def _info_score(view_data: Dict, edge_set: Dict) -> float:
    """Информационный score вида (выше → больше уникальной информации)."""
    if _is_simple_rectangle(view_data):
        return 0.0
    return edge_set['n_vis'] * 1.0 + edge_set['n_hid'] * VIEW_HIDDEN_EDGE_WEIGHT


def _jaccard_similarity(set_a: FrozenSet, set_b: FrozenSet) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 1.0


def _mirror_x(edge_set: FrozenSet, bbox: Dict) -> FrozenSet:
    """Зеркалить рёбра относительно центра bbox по оси X."""
    cx = (bbox['min_x'] + bbox['max_x']) / 2
    mirrored = set()
    for (ax, ay), (bx, by) in edge_set:
        ma = (round(2 * cx - ax, 4), ay)
        mb = (round(2 * cx - bx, 4), by)
        mirrored.add((ma, mb) if ma <= mb else (mb, ma))
    return frozenset(mirrored)


def _mirror_y(edge_set: FrozenSet, bbox: Dict) -> FrozenSet:
    """Зеркалить рёбра относительно центра bbox по оси Y."""
    cy = (bbox['min_y'] + bbox['max_y']) / 2
    mirrored = set()
    for (ax, ay), (bx, by) in edge_set:
        ma = (ax, round(2 * cy - ay, 4))
        mb = (bx, round(2 * cy - by, 4))
        mirrored.add((ma, mb) if ma <= mb else (mb, ma))
    return frozenset(mirrored)


# ---------------------------------------------------------------------------
# Шаги алгоритма
# ---------------------------------------------------------------------------

def _step1_exclude_opposites(
    views_data: Dict, edge_sets: Dict, scores: Dict, excluded: Dict
) -> None:
    """Исключить противоположные виды с меньшей информацией."""
    for primary, opposite in OPPOSITE_PAIRS:
        if primary not in edge_sets or opposite not in edge_sets:
            continue
        if primary in excluded or opposite in excluded:
            continue

        # Фронтальный вид всегда обязателен (ГОСТ 2.305-2008 п.5.1)
        if primary == 'front':
            preferred, secondary = primary, opposite
        elif opposite == 'front':
            preferred, secondary = opposite, primary
        elif edge_sets[primary]['n_vis'] >= edge_sets[opposite]['n_vis']:
            preferred, secondary = primary, opposite
        else:
            preferred, secondary = opposite, primary

        score_pref = scores.get(preferred, 0.0)
        score_sec  = scores.get(secondary, 0.0)

        if score_sec == 0.0 and score_pref == 0.0:
            excluded[secondary] = f"пустой прямоугольник (пара {primary}/{opposite})"
            continue

        if score_pref > 0 and score_sec / max(score_pref, 1e-9) < VIEW_SCORE_RATIO_MIN:
            excluded[secondary] = (
                f"мало информации: {score_sec:.0f} vs {score_pref:.0f} ({preferred})"
            )
            continue

        excluded[secondary] = (
            f"противоположный вид к {preferred} "
            f"(visible: {edge_sets[preferred]['n_vis']} vs {edge_sets[secondary]['n_vis']})"
        )


def _step2_exclude_empty(views_data: Dict, excluded: Dict) -> None:
    """Исключить виды, представляющие собой пустой прямоугольник."""
    for name, data in views_data.items():
        if name in excluded or name == 'front':
            continue
        if _is_simple_rectangle(data):
            excluded[name] = "пустой прямоугольник без внутренних рёбер"


def _step3_exclude_mirror_symmetric(
    views_data: Dict, edge_sets: Dict, scores: Dict, excluded: Dict
) -> None:
    """Исключить один из зеркально-симметричных видов."""
    remaining = [v for v in views_data if v not in excluded]
    sym_pairs = [('left', 'right'), ('top', 'bottom')]

    for va, vb in sym_pairs:
        if va not in remaining or vb not in remaining:
            continue

        es_a = edge_sets[va]['all']
        es_b = edge_sets[vb]['all']
        bb_a = views_data[va]['bbox']

        mirrored = _mirror_x(es_a, bb_a) if va in ('left', 'right') else _mirror_y(es_a, bb_a)
        similarity = _jaccard_similarity(mirrored, es_b)

        if similarity > VIEW_SIMILARITY_THRESHOLD:
            loser = va if scores.get(va, 0) <= scores.get(vb, 0) else vb
            winner = vb if loser == va else va
            excluded[loser] = (
                f"зеркально симметричен виду {winner} (similarity={similarity:.0%})"
            )
            remaining = [v for v in remaining if v != loser]


def _step4_ensure_axis_coverage(
    views_data: Dict, scores: Dict, excluded: Dict
) -> None:
    """Восстановить ранее исключённые виды, если осевое покрытие неполное."""
    selected = [v for v in views_data if v not in excluded]
    covered: Set[str] = set()
    for v in selected:
        covered |= AXIS_MAP.get(v, set())

    if len(covered) >= 3:
        return

    missing = {'X', 'Y', 'Z'} - covered
    candidates = sorted(excluded.keys(), key=lambda v: -scores.get(v, 0.0))

    for v in candidates:
        if AXIS_MAP.get(v, set()) & missing:
            logger.info("  Восстановлен вид %s (оси %s)", v, AXIS_MAP.get(v))
            del excluded[v]
            covered |= AXIS_MAP.get(v, set())
            missing = {'X', 'Y', 'Z'} - covered
            if not missing:
                break


def _step5_remove_redundant(
    views_data: Dict, scores: Dict, excluded: Dict
) -> None:
    """Убрать информационно-избыточные виды, если оси всё равно покрыты."""
    selected = [v for v in views_data if v not in excluded]

    for v in sorted(selected, key=lambda x: scores.get(x, 0.0)):
        if v == 'front':
            continue
        if len(selected) <= 2:
            break

        other = [o for o in selected if o != v]
        other_axes: Set[str] = set()
        for o in other:
            other_axes |= AXIS_MAP.get(o, set())

        if len(other_axes) >= 3 and scores.get(v, 0.0) < VIEW_MIN_INFO_SCORE:
            excluded[v] = f"избыточен (score={scores[v]:.1f}, оси покрыты)"
            selected.remove(v)
