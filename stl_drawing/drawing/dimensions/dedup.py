"""
Дедупликация размеров между видами (ГОСТ 2.307-2011).

Каждый размер (габарит, диаметр, ступень) имеет каноническый ключ.
Алгоритм выбирает для каждого ключа наилучший вид по scoring-модели:
  - Фронтальный вид: бонус (основной вид по ГОСТ)
  - Истинная длина (без ракурса): бонус
  - Балансировка нагрузки: бонус для видов с меньшим числом размеров
"""

import logging
from collections import defaultdict
from typing import Dict, List

from stl_drawing.drawing.dimensions.extractor import DimensionCandidate

logger = logging.getLogger(__name__)

# Бонусы scoring-модели
_FRONT_VIEW_BONUS = 2.0     # фронтальный вид — основной (ГОСТ 2.305-2008 п.5.1)
_TRUE_LENGTH_BONUS = 3.0    # размер в истинную величину
_LOW_LOAD_BONUS = 1.0       # вид с меньшим числом размеров
_PRIORITY_PENALTY = 0.5     # штраф за низкий приоритет (ступени)
_GEOMETRIC_SPECIFICITY_BONUS = 3.0  # диаметр/радиус точнее передают геометрию


def deduplicate_dimensions(
    candidates: List[DimensionCandidate],
    selected_views: List[str],
) -> Dict[str, List[DimensionCandidate]]:
    """Выбрать для каждого уникального размера один наилучший вид.

    Args:
        candidates: все кандидаты размеров из всех видов.
        selected_views: список выбранных видов чертежа.

    Returns:
        Словарь {view_name → [DimensionCandidate]} с назначенными размерами.
    """
    if not candidates:
        return {v: [] for v in selected_views}

    # Группировка по каноническому ключу
    groups: Dict[str, List[DimensionCandidate]] = defaultdict(list)
    for c in candidates:
        groups[c.canonical_key].append(c)

    # Счётчик размеров на вид (для балансировки)
    view_load: Dict[str, int] = {v: 0 for v in selected_views}

    # Результат
    result: Dict[str, List[DimensionCandidate]] = {v: [] for v in selected_views}

    # Сортируем группы: сначала габариты (priority=0), потом диаметры, потом ступени
    sorted_keys = sorted(groups.keys(), key=lambda k: min(c.priority for c in groups[k]))

    for key in sorted_keys:
        group = groups[key]
        # Отфильтровать только виды из selected_views
        group = [c for c in group if c.view_name in selected_views]
        if not group:
            continue

        best_candidate = None
        best_score = -1e9

        for c in group:
            score = _score_candidate(c, view_load)
            if score > best_score:
                best_score = score
                best_candidate = c

        if best_candidate is not None:
            vn = best_candidate.view_name
            result[vn].append(best_candidate)
            view_load[vn] = view_load.get(vn, 0) + 1

    total = sum(len(dims) for dims in result.values())
    for vn, dims in result.items():
        if dims:
            logger.info("  Вид %s: %d размеров", vn, len(dims))
    logger.info("Дедупликация: %d уникальных размеров → %d видов",
                total, sum(1 for d in result.values() if d))

    return result


def _score_candidate(
    candidate: DimensionCandidate,
    view_load: Dict[str, int],
) -> float:
    """Вычислить score кандидата для выбора наилучшего вида.

    Args:
        candidate: кандидат размера.
        view_load: текущее количество размеров на каждом виде.

    Returns:
        Score (выше = лучше).
    """
    score = 0.0

    # Бонус за фронтальный вид
    if candidate.view_name == 'front':
        score += _FRONT_VIEW_BONUS

    # Бонус за «истинную величину» — горизонтальные размеры на виде,
    # где горизонтальная ось является целевой 3D-осью без ракурса
    # (все виды ортогональны — размеры всегда в истинную величину на своём виде)
    score += _TRUE_LENGTH_BONUS

    # Бонус за геометрическую специфичность: диаметр/радиус точнее
    # передают геометрию элемента (Ø, R), чем линейная проекция того же размера
    if candidate.dim_type in ('diameter', 'radius'):
        score += _GEOMETRIC_SPECIFICITY_BONUS

    # Штраф за приоритет (ступени менее важны, чем габариты)
    score -= candidate.priority * _PRIORITY_PENALTY

    # Бонус за баланс нагрузки
    load = view_load.get(candidate.view_name, 0)
    if load == 0:
        score += _LOW_LOAD_BONUS
    elif load < 3:
        score += _LOW_LOAD_BONUS * 0.5

    return score
