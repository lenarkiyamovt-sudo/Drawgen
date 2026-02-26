"""
Параметры линий по ГОСТ 2.303-68.

Вычисляет толщины и параметры штрихов в зависимости от размера
главного вида на листе. Все ГОСТ-стандарты и форматы — в config.py.
"""

from typing import Dict

from stl_drawing.config import (
    DIM_ARROW_LENGTH,
    DIM_ARROW_WIDTH,
    DIM_FONT_FAMILY,
    DIM_TEXT_HEIGHT,
    GOST_FORMATS,
    GOST_FORMATS_ORDERED,
    GOST_REDUCTION_SCALES,
    S_THIN_DRAWING,
    S_THICK_DRAWING,
)


def calculate_line_parameters(
    front_view_mm: float = 100.0,
    stroke_width: float = None,
) -> Dict[str, Dict]:
    """Вычислить параметры линий по ГОСТ 2.303-68.

    ГОСТ 2.303-68 задаёт:
      - S (основная линия): 0.5–1.4 мм
      - Штриховая: толщина S/3..S/2, штрихи 2–8 мм, промежутки 1–2 мм

    Практическое отображение (размер вида → параметры):
      < 40 мм:   S=0.5, штрих=2, пробел=1
      40–80 мм:  S=0.5, штрих=3, пробел=1
      80–150 мм: S=0.7, штрих=4, пробел=1.5
      > 150 мм:  S=0.7, штрих=5, пробел=2

    Args:
        front_view_mm: размер главного вида на бумаге (мм).
        stroke_width:  переопределить S вручную (None → авто).

    Returns:
        Словарь стилей: 'visible', 'hidden', 'hidden_solid', 'thin', '_params'.
    """
    if stroke_width is None:
        stroke_width = S_THICK_DRAWING if front_view_mm >= 80 else S_THIN_DRAWING

    if front_view_mm < 40:
        dash_length, gap_length = 2.0, 1.0
    elif front_view_mm < 80:
        dash_length, gap_length = 3.0, 1.0
    elif front_view_mm < 150:
        dash_length, gap_length = 4.0, 1.5
    else:
        dash_length, gap_length = 5.0, 2.0

    thin_width = stroke_width / 2.0  # ГОСТ: тонкая линия = S/2

    # ГОСТ 2.303-68: штрихпунктирная тонкая (осевые/центровые линии)
    # Штрих 5–30 мм, промежутки 3–5 мм (включая точку)
    if front_view_mm < 40:
        cl_dash, cl_gap, cl_dot = 5.0, 1.0, 0.5
    elif front_view_mm < 80:
        cl_dash, cl_gap, cl_dot = 8.0, 1.0, 0.5
    elif front_view_mm < 150:
        cl_dash, cl_gap, cl_dot = 12.0, 1.5, 0.5
    else:
        cl_dash, cl_gap, cl_dot = 15.0, 2.0, 0.5

    return {
        'visible': {
            'stroke': 'black',
            'stroke_width': f'{stroke_width}mm',
            'stroke_linecap': 'butt',
        },
        'hidden': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_dasharray': f'{dash_length},{gap_length}',
            'stroke_linecap': 'butt',
        },
        'hidden_solid': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt',
        },
        'thin': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt',
        },
        'centerline': {
            'stroke': 'red',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt',
        },
        'dimension': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt',
        },
        'dimension_text': {
            'font_family': f'{DIM_FONT_FAMILY}, Arial',
            'font_size': f'{DIM_TEXT_HEIGHT}mm',
            'font_style': 'italic',
            'fill': 'black',
            'text_anchor': 'middle',
        },
        'dimension_arrow': {
            'length': DIM_ARROW_LENGTH,
            'width': DIM_ARROW_WIDTH,
        },
        '_params': {
            'S': stroke_width,
            'thin_width': thin_width,
            'dash_length': dash_length,
            'gap_length': gap_length,
            'cl_dash': cl_dash,
            'cl_gap': cl_gap,
            'cl_dot': cl_dot,
        },
    }


def snap_to_gost_scale(working_scale: float) -> float:
    """Округлить масштаб до ближайшего стандартного значения ГОСТ 2.302-68.

    Выбирает наибольший стандартный масштаб, не превышающий working_scale.

    Args:
        working_scale: «сырой» масштаб (отношение мм-на-листе / мм-модели).

    Returns:
        Стандартный масштаб ГОСТ.
    """
    for scale in GOST_REDUCTION_SCALES:
        if scale <= working_scale * (1.0 + 1e-9):
            return scale
    return GOST_REDUCTION_SCALES[-1]
