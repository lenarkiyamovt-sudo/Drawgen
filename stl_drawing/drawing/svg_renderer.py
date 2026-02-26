"""
SVG-рендеринг линий чертежа.

Содержит:
- extend_line      — удлинение отрезка на полтолщины линии
- generate_dashes  — разбивка отрезка на ЕСКД-штрихи
- render_view_lines — отрисовка всех линий одного вида в SVG-группу
- render_centerlines — отрисовка осевых линий (штрихпунктирная, красная)

Типы линий определены в ESKDLineType (stl_drawing.config).
"""

import math
from typing import Dict, List, Optional, Tuple

import svgwrite

from stl_drawing.config import ESKDLineType


def _filter_svg_attrs(style_dict: dict) -> dict:
    """Filter out internal metadata keys (starting with _) from style dict.

    svgwrite converts underscores to hyphens, making '_gost_type' become
    '-gost-type' which is invalid XML. This function removes such keys.

    Args:
        style_dict: Style dictionary possibly containing metadata keys

    Returns:
        Filtered dictionary safe for SVG attributes
    """
    return {k: v for k, v in style_dict.items() if not k.startswith('_')}


def _get_eskd_type_name(style_dict: dict) -> str:
    """Get the ESKDLineType name from a style dictionary.

    Args:
        style_dict: Style dictionary containing _eskd_type

    Returns:
        ESKDLineType name (e.g., 'SOLID', 'DASHED') or 'unknown'
    """
    eskd_type = style_dict.get('_eskd_type')
    if isinstance(eskd_type, ESKDLineType):
        return eskd_type.name.lower()
    return 'unknown'


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

    Использует ESKDLineType для определения стилей линий.

    Args:
        dwg: SVG-документ (для создания элементов).
        visible_lines: список (pA, pB) видимых отрезков (единицы модели).
        hidden_lines: список (pA, pB) скрытых отрезков (единицы модели).
        scale: масштаб (мм на листе / единица модели).
        translate_x, translate_y: смещение вида на листе (мм).
        eskd_styles: словарь стилей (для параметров штрихов).
        centerlines: список осевых линий (из детекции цилиндров).

    Returns:
        SVG-группа с линиями вида.
    """
    view_group = dwg.g()
    params = eskd_styles['_params']

    # Стили из ESKDLineType (абсолютные значения)
    visible_style = ESKDLineType.SOLID.get_svg_style()
    hidden_style = ESKDLineType.DASHED.get_svg_style()

    # Параметры штрихов для скрытых линий
    dash_pattern = ESKDLineType.DASHED.svg_pattern
    if dash_pattern:
        parts = dash_pattern.split(',')
        dash_len = float(parts[0])
        gap_len = float(parts[1]) if len(parts) > 1 else 1.0
    else:
        dash_len = params['dash_length']
        gap_len = params['gap_length']

    half_sw = ESKDLineType.SOLID.stroke_width / 2

    # --- Скрытые линии: ручная генерация штрихов (ЕСКД: начало/конец = полный штрих) ---
    for pA, pB in hidden_lines:
        x1 = float(pA[0]) * scale + translate_x
        y1 = float(pA[1]) * scale + translate_y
        x2 = float(pB[0]) * scale + translate_x
        y2 = float(pB[1]) * scale + translate_y

        segs, _ = generate_dashes((x1, y1), (x2, y2), dash_len, gap_len)
        # Используем стиль без dasharray (штрихи генерируются вручную)
        seg_style = {k: v for k, v in hidden_style.items() if k != 'stroke-dasharray'}
        for s, e in segs:
            line = dwg.line(start=s, end=e, **seg_style)
            line['style'] = "vector-effect: non-scaling-stroke;"
            line['data-line-type'] = 'hidden'
            line['data-eskd-type'] = 'dashed'
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
        line['data-eskd-type'] = 'solid'
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

    Использует ESKDLineType.CENTER для стиля линий.

    Правила ГОСТ 2.303-68:
      - Вынос осевой за контур тела: 2–5 мм (используем 3 мм).
      - Окружности Ø < 12 мм на бумаге: центровые линии —
        сплошные тонкие (без штрихпунктира).
      - Линия короче двух полных штрихпунктирных циклов —
        рисуется сплошной тонкой.

    Args:
        dwg: SVG-документ.
        group: SVG-группа вида для добавления элементов.
        centerlines: список словарей осевых линий.
        scale: масштаб чертежа.
        translate_x, translate_y: смещение вида.
        eskd_styles: словарь стилей (для обратной совместимости).
    """
    # Стиль из ESKDLineType.CENTER (красный, штрихпунктир)
    cl_style = ESKDLineType.CENTER.get_svg_style(color='red')
    # Убрать dasharray — будет добавлен вручную где нужно
    cl_style_solid = {k: v for k, v in cl_style.items() if k != 'stroke-dasharray'}

    # Паттерн штрихпунктира из ESKDLineType.CENTER
    dasharray = ESKDLineType.CENTER.svg_pattern or "15,3,3,3"
    parts = dasharray.split(',')
    cl_dash = float(parts[0])
    cl_gap = float(parts[1]) if len(parts) > 1 else 3.0
    cl_dot = float(parts[2]) if len(parts) > 2 else 3.0

    # Один полный цикл штрихпунктира: штрих + пробел + точка + пробел
    dash_cycle = cl_dash + cl_gap + cl_dot + cl_gap

    # ГОСТ 2.303-68: вынос осевой за контур — 2–5 мм
    cl_extension_mm = 3.0

    # ГОСТ 2.303-68 п.3.2: предел перехода на сплошную тонкую (мм на бумаге)
    _SMALL_CIRCLE_DIAM_LIMIT = 12.0

    # Минимальное полуплечо перекрестия (мм)
    _MIN_CROSSHAIR_ARM = 5.0

    for cl in centerlines:
        if cl['type'] == 'centerline':
            x1 = float(cl['start'][0]) * scale + translate_x
            y1 = float(cl['start'][1]) * scale + translate_y
            x2 = float(cl['end'][0]) * scale + translate_x
            y2 = float(cl['end'][1]) * scale + translate_y

            body_length = math.hypot(x2 - x1, y2 - y1)
            if body_length < 0.5:
                continue

            # Вынос на 3 мм за габариты тела с каждой стороны
            (x1, y1), (x2, y2) = extend_line((x1, y1), (x2, y2), cl_extension_mm)
            total_length = body_length + 2 * cl_extension_mm

            line = dwg.line(start=(x1, y1), end=(x2, y2), **cl_style_solid)
            # Штрихпунктир только если вмещается ≥ 2 цикла, иначе — сплошная тонкая
            if total_length >= 2 * dash_cycle:
                line['stroke-dasharray'] = dasharray
            line['data-line-type'] = 'centerline'
            line['data-eskd-type'] = 'center'
            group.add(line)

        elif cl['type'] == 'crosshair':
            cx = float(cl['center'][0]) * scale + translate_x
            cy = float(cl['center'][1]) * scale + translate_y
            r_paper = float(cl['radius']) * scale
            diam_paper = 2.0 * r_paper

            # Полуплечо: радиус на бумаге + вынос, но не менее минимума
            arm = max(r_paper + cl_extension_mm, _MIN_CROSSHAIR_ARM)

            if arm < 1.0:
                continue

            # Горизонтальная осевая
            h_line = dwg.line(
                start=(cx - arm, cy), end=(cx + arm, cy),
                **cl_style_solid,
            )
            # ГОСТ 2.303-68 п.3.2: при Ø < 12 мм — сплошная тонкая
            if diam_paper >= _SMALL_CIRCLE_DIAM_LIMIT and 2 * arm >= 2 * dash_cycle:
                h_line['stroke-dasharray'] = dasharray
            h_line['data-line-type'] = 'centerline'
            h_line['data-eskd-type'] = 'center'
            group.add(h_line)

            # Вертикальная осевая
            v_line = dwg.line(
                start=(cx, cy - arm), end=(cx, cy + arm),
                **cl_style_solid,
            )
            if diam_paper >= _SMALL_CIRCLE_DIAM_LIMIT and 2 * arm >= 2 * dash_cycle:
                v_line['stroke-dasharray'] = dasharray
            v_line['data-line-type'] = 'centerline'
            v_line['data-eskd-type'] = 'center'
            group.add(v_line)
