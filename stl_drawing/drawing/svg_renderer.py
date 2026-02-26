"""
SVG-рендеринг линий чертежа.

Содержит:
- extend_line      — удлинение отрезка на полтолщины линии
- generate_dashes  — разбивка отрезка на ЕСКД-штрихи
- render_view_lines — отрисовка всех линий одного вида в SVG-группу
- render_centerlines — отрисовка осевых линий (штрихпунктирная, красная)
"""

import math
from typing import Dict, List, Optional, Tuple

import svgwrite


# ---------------------------------------------------------------------------
# Геометрические утилиты
# ---------------------------------------------------------------------------

def extend_line(
    pA: Tuple[float, float],
    pB: Tuple[float, float],
    extension: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Удлинить отрезок на `extension` с обоих концов вдоль его направления.

    Используется для видимых линий: торцы продлеваются на S/2,
    чтобы угловые соединения не имели зазоров при butt-концах.

    Args:
        pA, pB: концы отрезка.
        extension: длина удлинения (мм на листе).

    Returns:
        Новые координаты концов (new_A, new_B).
    """
    dx = float(pB[0] - pA[0])
    dy = float(pB[1] - pA[1])
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return pA, pB

    ux, uy = dx / length, dy / length
    new_a = (float(pA[0]) - ux * extension, float(pA[1]) - uy * extension)
    new_b = (float(pB[0]) + ux * extension, float(pB[1]) + uy * extension)
    return new_a, new_b


def generate_dashes(
    pA: Tuple[float, float],
    pB: Tuple[float, float],
    dash_len: float,
    gap_len: float,
) -> Tuple[List[Tuple], bool]:
    """Разбить отрезок на ЕСКД-штрихи.

    Первый и последний сегменты всегда полные дефисы.
    Если отрезок слишком короткий — возвращается как один сплошной
    сегмент с флагом is_thin=True (рисовать тонкой сплошной линией).

    Args:
        pA, pB: концы отрезка (уже в мм на листе).
        dash_len: длина штриха (мм).
        gap_len: длина пробела (мм).

    Returns:
        (segments, is_thin_fallback):
          segments — список ((x1,y1), (x2,y2))
          is_thin_fallback — True если отрезок слишком короткий для штрихов
    """
    dx = float(pB[0] - pA[0])
    dy = float(pB[1] - pA[1])
    length = math.hypot(dx, dy)

    min_length = 2 * dash_len + gap_len
    if length < min_length:
        return [((float(pA[0]), float(pA[1])), (float(pB[0]), float(pB[1])))], True

    n = max(2, round((length + gap_len) / (dash_len + gap_len)))
    actual_dash = (length - (n - 1) * gap_len) / n

    if actual_dash < dash_len * 0.4:
        n -= 1
        if n < 2:
            return [((float(pA[0]), float(pA[1])), (float(pB[0]), float(pB[1])))], True
        actual_dash = (length - (n - 1) * gap_len) / n

    ux, uy = dx / length, dy / length
    ax, ay = float(pA[0]), float(pA[1])
    segments = []
    pos = 0.0
    for _ in range(n):
        s = (ax + ux * pos, ay + uy * pos)
        e = (ax + ux * (pos + actual_dash), ay + uy * (pos + actual_dash))
        segments.append((s, e))
        pos += actual_dash + gap_len

    return segments, False


# ---------------------------------------------------------------------------
# Рендеринг линий вида
# ---------------------------------------------------------------------------

def render_view_lines(
    dwg: svgwrite.Drawing,
    visible_lines: List[Tuple],
    hidden_lines: List[Tuple],
    scale: float,
    translate_x: float,
    translate_y: float,
    eskd_styles: dict,
    centerlines: Optional[List[Dict]] = None,
) -> svgwrite.container.Group:
    """Отрисовать все линии одного вида в SVG-группу.

    Координаты линий (в единицах модели) пересчитываются в мм на листе
    через scale и translate_x/y.

    Args:
        dwg: SVG-документ (для создания элементов).
        visible_lines: список (pA, pB) видимых отрезков (единицы модели).
        hidden_lines: список (pA, pB) скрытых отрезков (единицы модели).
        scale: масштаб (мм на листе / единица модели).
        translate_x, translate_y: смещение вида на листе (мм).
        eskd_styles: словарь стилей из gost_params.calculate_line_parameters.
        centerlines: список осевых линий (из детекции цилиндров).

    Returns:
        SVG-группа с линиями вида.
    """
    view_group = dwg.g()
    params = eskd_styles['_params']
    half_sw = params['S'] / 2
    dash_len = params['dash_length']
    gap_len = params['gap_length']

    hidden_style = eskd_styles['hidden_solid']
    visible_style = eskd_styles['visible']

    # --- Скрытые линии: ручная генерация штрихов (ЕСКД: начало/конец = полный штрих) ---
    for pA, pB in hidden_lines:
        x1 = float(pA[0]) * scale + translate_x
        y1 = float(pA[1]) * scale + translate_y
        x2 = float(pB[0]) * scale + translate_x
        y2 = float(pB[1]) * scale + translate_y

        segs, _ = generate_dashes((x1, y1), (x2, y2), dash_len, gap_len)
        for s, e in segs:
            line = dwg.line(start=s, end=e, **hidden_style)
            line['style'] = "vector-effect: non-scaling-stroke;"
            line['data-line-type'] = 'hidden'
            view_group.add(line)

    # --- Видимые линии: удлиняем на полтолщины (нет зазоров на стыках) ---
    for pA, pB in visible_lines:
        x1 = float(pA[0]) * scale + translate_x
        y1 = float(pA[1]) * scale + translate_y
        x2 = float(pB[0]) * scale + translate_x
        y2 = float(pB[1]) * scale + translate_y

        pA_ext, pB_ext = extend_line((x1, y1), (x2, y2), half_sw)
        line = dwg.line(start=pA_ext, end=pB_ext, **visible_style)
        line['style'] = "vector-effect: non-scaling-stroke;"
        view_group.add(line)

    # --- Осевые линии (штрихпунктирная тонкая, красная) ---
    if centerlines:
        _render_centerlines(dwg, view_group, centerlines, scale,
                            translate_x, translate_y, eskd_styles)

    return view_group


def _render_centerlines(
    dwg: svgwrite.Drawing,
    group: svgwrite.container.Group,
    centerlines: List[Dict],
    scale: float,
    translate_x: float,
    translate_y: float,
    eskd_styles: dict,
) -> None:
    """Отрисовать осевые линии цилиндров (ГОСТ 2.303-68: штрихпунктирная тонкая).

    Args:
        dwg: SVG-документ.
        group: SVG-группа вида для добавления элементов.
        centerlines: список словарей осевых линий.
        scale: масштаб чертежа.
        translate_x, translate_y: смещение вида.
        eskd_styles: словарь стилей.
    """
    params = eskd_styles['_params']
    cl_style = eskd_styles['centerline']
    cl_dash = params['cl_dash']
    cl_gap = params['cl_gap']
    cl_dot = params['cl_dot']
    dasharray = f"{cl_dash},{cl_gap},{cl_dot},{cl_gap}"

    for cl in centerlines:
        if cl['type'] == 'centerline':
            x1 = float(cl['start'][0]) * scale + translate_x
            y1 = float(cl['start'][1]) * scale + translate_y
            x2 = float(cl['end'][0]) * scale + translate_x
            y2 = float(cl['end'][1]) * scale + translate_y

            length = math.hypot(x2 - x1, y2 - y1)
            if length < 1.0:
                continue

            line = dwg.line(start=(x1, y1), end=(x2, y2), **cl_style)
            line['stroke-dasharray'] = dasharray
            line['data-line-type'] = 'centerline'
            group.add(line)

        elif cl['type'] == 'crosshair':
            cx = float(cl['center'][0]) * scale + translate_x
            cy = float(cl['center'][1]) * scale + translate_y
            size_mm = float(cl['size']) * scale

            if size_mm < 0.5:
                continue

            # Горизонтальная осевая
            h_line = dwg.line(
                start=(cx - size_mm, cy), end=(cx + size_mm, cy),
                **cl_style,
            )
            h_line['stroke-dasharray'] = dasharray
            h_line['data-line-type'] = 'centerline'
            group.add(h_line)

            # Вертикальная осевая
            v_line = dwg.line(
                start=(cx, cy - size_mm), end=(cx, cy + size_mm),
                **cl_style,
            )
            v_line['stroke-dasharray'] = dasharray
            v_line['data-line-type'] = 'centerline'
            group.add(v_line)
