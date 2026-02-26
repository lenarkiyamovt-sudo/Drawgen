"""
Параметры линий чертежа по ГОСТ 2.303-68.

ГОСТ 2.303-68 «Линии чертежа» определяет:
- 9 типов линий с различными толщинами и начертаниями
- Толщина основной линии S: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4 мм
- Относительные толщины других линий выражены через S

Таблица толщин линий (ГОСТ 2.303-68, таблица 1):
┌────────────────────────────────────────────────────────────────┐
│ Тип линии                │ Толщина    │ Применение             │
├──────────────────────────┼────────────┼────────────────────────┤
│ Сплошная основная       │ S          │ Видимый контур         │
│ Сплошная тонкая         │ S/3 - S/2  │ Размерные, выносные    │
│ Сплошная волнистая      │ S/3 - S/2  │ Линии обрыва           │
│ Штриховая               │ S/3 - S/2  │ Невидимый контур       │
│ Штрихпунктирная тонкая  │ S/3 - S/2  │ Осевые, центровые      │
│ Штрихпунктирная толстая │ S/2 - 2S/3 │ Линии развертки        │
│ Разомкнутая             │ S - 1.5S   │ Линии сечений          │
│ Сплошная тонкая с изл.  │ S/3 - S/2  │ Длинные линии обрыва   │
│ Штрихпунктирная 2 точки │ S/3 - S/2  │ Сгибы на развертках    │
└────────────────────────────────────────────────────────────────┘

Рекомендуемые значения S в зависимости от масштаба и формата:
┌────────────┬───────────────────────────────────────────────────┐
│ Масштаб    │ Рекомендуемая S (мм)                              │
├────────────┼───────────────────────────────────────────────────┤
│ 1:1        │ 0.5 - 0.7 (A4), 0.7 - 1.0 (A3-A0)               │
│ 1:2 - 1:5  │ 0.5 - 0.7                                        │
│ 1:10-1:50  │ 0.5 - 0.6                                        │
│ 1:100-1:1000│ 0.5                                              │
│ 2:1 - 5:1  │ 0.7 - 1.0                                        │
└────────────┴───────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from stl_drawing.config import (
    DIM_ARROW_LENGTH,
    DIM_ARROW_WIDTH,
    DIM_FONT_FAMILY,
    DIM_TEXT_HEIGHT,
    ESKDLineType,
    GOST_FORMATS,
    GOST_FORMATS_ORDERED,
    GOST_REDUCTION_SCALES,
    S_THIN_DRAWING,
    S_THICK_DRAWING,
)


class LineType(Enum):
    """Line types defined in GOST 2.303-68."""
    CONTINUOUS_THICK = "continuous_thick"       # Сплошная основная
    CONTINUOUS_THIN = "continuous_thin"         # Сплошная тонкая
    CONTINUOUS_WAVY = "continuous_wavy"         # Сплошная волнистая
    DASHED = "dashed"                           # Штриховая
    CHAIN_THIN = "chain_thin"                   # Штрихпунктирная тонкая
    CHAIN_THICK = "chain_thick"                 # Штрихпунктирная толстая
    OPEN = "open"                               # Разомкнутая
    CONTINUOUS_THIN_ZIGZAG = "continuous_zigzag"  # Сплошная тонкая с изломами
    CHAIN_DOUBLE_DOT = "chain_double_dot"       # Штрихпунктирная с двумя точками


@dataclass
class LineStyle:
    """Line style parameters according to GOST 2.303-68."""
    line_type: LineType
    thickness_ratio: float      # Ratio relative to S (e.g., 0.5 means S/2)
    dash_length_range: tuple    # (min, max) in mm, None for continuous
    gap_length_range: tuple     # (min, max) in mm, None for continuous
    description_ru: str
    description_en: str


# GOST 2.303-68 Line Specifications
GOST_LINE_SPECS: Dict[LineType, LineStyle] = {
    LineType.CONTINUOUS_THICK: LineStyle(
        line_type=LineType.CONTINUOUS_THICK,
        thickness_ratio=1.0,
        dash_length_range=None,
        gap_length_range=None,
        description_ru="Сплошная основная",
        description_en="Continuous thick (visible outline)",
    ),
    LineType.CONTINUOUS_THIN: LineStyle(
        line_type=LineType.CONTINUOUS_THIN,
        thickness_ratio=0.4,  # S/3 to S/2, using middle value
        dash_length_range=None,
        gap_length_range=None,
        description_ru="Сплошная тонкая",
        description_en="Continuous thin (dimension, extension lines)",
    ),
    LineType.DASHED: LineStyle(
        line_type=LineType.DASHED,
        thickness_ratio=0.4,
        dash_length_range=(2.0, 8.0),
        gap_length_range=(1.0, 2.0),
        description_ru="Штриховая",
        description_en="Dashed (hidden outline)",
    ),
    LineType.CHAIN_THIN: LineStyle(
        line_type=LineType.CHAIN_THIN,
        thickness_ratio=0.4,
        dash_length_range=(5.0, 30.0),
        gap_length_range=(3.0, 5.0),
        description_ru="Штрихпунктирная тонкая",
        description_en="Chain thin (center, axis lines)",
    ),
    LineType.CHAIN_THICK: LineStyle(
        line_type=LineType.CHAIN_THICK,
        thickness_ratio=0.6,  # S/2 to 2S/3
        dash_length_range=(3.0, 8.0),
        gap_length_range=(3.0, 4.0),
        description_ru="Штрихпунктирная толстая",
        description_en="Chain thick (surface treatment lines)",
    ),
    LineType.OPEN: LineStyle(
        line_type=LineType.OPEN,
        thickness_ratio=1.25,  # S to 1.5S
        dash_length_range=(8.0, 20.0),
        gap_length_range=None,  # No gaps, just segments
        description_ru="Разомкнутая",
        description_en="Open (section lines)",
    ),
    LineType.CONTINUOUS_WAVY: LineStyle(
        line_type=LineType.CONTINUOUS_WAVY,
        thickness_ratio=0.4,
        dash_length_range=None,
        gap_length_range=None,
        description_ru="Сплошная волнистая",
        description_en="Continuous wavy (break lines, short)",
    ),
    LineType.CONTINUOUS_THIN_ZIGZAG: LineStyle(
        line_type=LineType.CONTINUOUS_THIN_ZIGZAG,
        thickness_ratio=0.4,
        dash_length_range=None,
        gap_length_range=None,
        description_ru="Сплошная тонкая с изломами",
        description_en="Continuous thin zigzag (break lines, long)",
    ),
    LineType.CHAIN_DOUBLE_DOT: LineStyle(
        line_type=LineType.CHAIN_DOUBLE_DOT,
        thickness_ratio=0.4,
        dash_length_range=(5.0, 30.0),
        gap_length_range=(4.0, 6.0),
        description_ru="Штрихпунктирная с двумя точками",
        description_en="Chain double-dot (bend lines on developments)",
    ),
}


# GOST 2.303-68 Standard S values (mm)
GOST_S_VALUES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4)


# ---------------------------------------------------------------------------
# Маппинг между LineType и ESKDLineType
# ---------------------------------------------------------------------------

LINETYPE_TO_ESKD: Dict[LineType, ESKDLineType] = {
    LineType.CONTINUOUS_THICK: ESKDLineType.SOLID,
    LineType.CONTINUOUS_THIN: ESKDLineType.THIN,
    LineType.DASHED: ESKDLineType.DASHED,
    LineType.CHAIN_THIN: ESKDLineType.CENTER,
    LineType.CHAIN_THICK: ESKDLineType.SOLID,  # Используем SOLID, т.к. нет отдельного
    LineType.OPEN: ESKDLineType.SECTION,
    LineType.CONTINUOUS_WAVY: ESKDLineType.WAVELINE,
    LineType.CONTINUOUS_THIN_ZIGZAG: ESKDLineType.WAVELINE,
    LineType.CHAIN_DOUBLE_DOT: ESKDLineType.DOTTED,
}


def get_eskd_line_type(line_type: LineType) -> ESKDLineType:
    """Преобразовать LineType в ESKDLineType.

    Args:
        line_type: тип линии из GOST_LINE_SPECS

    Returns:
        Соответствующий ESKDLineType
    """
    return LINETYPE_TO_ESKD.get(line_type, ESKDLineType.SOLID)


def get_recommended_s(scale: float, format_name: str = "A4") -> float:
    """Get recommended S value based on scale and format.

    GOST 2.303-68 recommendations for line thickness S:
    - Larger formats (A0-A2): prefer thicker lines (S=0.7-1.0)
    - Smaller formats (A3-A4): prefer thinner lines (S=0.5-0.7)
    - Reduction scales (1:N): prefer thinner lines
    - Enlargement scales (N:1): prefer thicker lines

    Args:
        scale: Drawing scale as decimal (e.g., 0.5 for 1:2, 2.0 for 2:1)
        format_name: Paper format ("A0", "A1", "A2", "A3", "A4")

    Returns:
        Recommended S value in mm
    """
    # Base S based on format
    format_s = {
        "A0": 0.7,
        "A1": 0.7,
        "A2": 0.6,
        "A3": 0.5,
        "A4": 0.5,
    }
    base_s = format_s.get(format_name, 0.5)

    # Adjust for scale
    if scale >= 2.0:
        # Enlargement: use thicker lines
        s = min(base_s + 0.2, 1.0)
    elif scale >= 1.0:
        s = base_s
    elif scale >= 0.5:
        s = base_s
    elif scale >= 0.1:
        # Strong reduction: thin lines
        s = max(base_s - 0.1, 0.5)
    else:
        # Very small scale: minimum thickness
        s = 0.5

    # Snap to nearest GOST value
    return min(GOST_S_VALUES, key=lambda x: abs(x - s))


def calculate_thickness_for_scale(scale: float, line_type: LineType,
                                   format_name: str = "A4") -> float:
    """Calculate line thickness for given scale and line type.

    Args:
        scale: Drawing scale
        line_type: Type of line from GOST 2.303-68
        format_name: Paper format

    Returns:
        Line thickness in mm
    """
    s = get_recommended_s(scale, format_name)
    spec = GOST_LINE_SPECS.get(line_type)
    if spec is None:
        return s
    return s * spec.thickness_ratio


def get_dash_pattern_for_size(view_size_mm: float, line_type: LineType) -> tuple:
    """Get appropriate dash pattern based on view size.

    Larger views use longer dashes for better readability.

    Args:
        view_size_mm: Size of view on paper in mm
        line_type: Type of line

    Returns:
        Tuple (dash_length, gap_length) in mm
    """
    spec = GOST_LINE_SPECS.get(line_type)
    if spec is None or spec.dash_length_range is None:
        return (0, 0)  # Continuous line

    dash_min, dash_max = spec.dash_length_range
    gap_min, gap_max = spec.gap_length_range or (1.0, 2.0)

    # Interpolate based on view size (40-200mm range)
    t = min(1.0, max(0.0, (view_size_mm - 40) / 160))
    dash = dash_min + t * (dash_max - dash_min)
    gap = gap_min + t * (gap_max - gap_min)

    return (dash, gap)


def calculate_line_parameters(
    front_view_mm: float = 100.0,
    stroke_width: Optional[float] = None,
    scale: float = 1.0,
    format_name: str = "A4",
) -> Dict[str, Dict]:
    """Вычислить параметры линий по ГОСТ 2.303-68.

    ГОСТ 2.303-68 «Линии чертежа» определяет:
    - S (основная линия): 0.5–1.4 мм
    - Тонкие линии: S/3 – S/2
    - Штриховая: штрихи 2–8 мм, промежутки 1–2 мм
    - Штрихпунктирная: штрихи 5–30 мм, промежутки 3–5 мм

    Толщина S выбирается на основании:
    - Формата листа (A0-A4)
    - Масштаба чертежа
    - Размера главного вида

    Args:
        front_view_mm: размер главного вида на бумаге (мм)
        stroke_width: переопределить S вручную (None → авто по ГОСТ)
        scale: масштаб чертежа (1.0 для 1:1, 0.5 для 1:2, 2.0 для 2:1)
        format_name: формат листа ("A4", "A3", "A2", "A1", "A0")

    Returns:
        Словарь стилей для всех типов линий + '_params' с числовыми параметрами
    """
    # Determine S value
    if stroke_width is None:
        stroke_width = get_recommended_s(scale, format_name)
        # Adjust based on view size for very small views
        if front_view_mm < 60:
            stroke_width = min(stroke_width, S_THIN_DRAWING)

    # Get dash patterns based on view size
    dash_length, gap_length = get_dash_pattern_for_size(front_view_mm, LineType.DASHED)
    cl_dash, cl_gap = get_dash_pattern_for_size(front_view_mm, LineType.CHAIN_THIN)

    # Chain thin dot size (approximately S/2)
    cl_dot = stroke_width * 0.5

    # Calculate line thicknesses
    thin_width = stroke_width * GOST_LINE_SPECS[LineType.CONTINUOUS_THIN].thickness_ratio
    chain_thick_width = stroke_width * GOST_LINE_SPECS[LineType.CHAIN_THICK].thickness_ratio
    open_width = stroke_width * GOST_LINE_SPECS[LineType.OPEN].thickness_ratio

    return {
        # Сплошная основная — видимый контур (ESKDLineType.SOLID)
        'visible': {
            'stroke': 'black',
            'stroke_width': f'{stroke_width}mm',
            'stroke_linecap': 'butt',
            '_gost_type': LineType.CONTINUOUS_THICK.value,
            '_eskd_type': ESKDLineType.SOLID,
        },
        # Штриховая — невидимый контур (ESKDLineType.DASHED)
        'hidden': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_dasharray': f'{dash_length},{gap_length}',
            'stroke_linecap': 'butt',
            '_gost_type': LineType.DASHED.value,
            '_eskd_type': ESKDLineType.DASHED,
        },
        # Сплошная тонкая — для невидимого контура без штрихов (внутреннее использование)
        'hidden_solid': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt',
            '_gost_type': LineType.CONTINUOUS_THIN.value,
            '_eskd_type': ESKDLineType.THIN,
        },
        # Сплошная тонкая — размерные, выносные линии (ESKDLineType.THIN)
        'thin': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt',
            '_gost_type': LineType.CONTINUOUS_THIN.value,
            '_eskd_type': ESKDLineType.THIN,
        },
        # Штрихпунктирная тонкая — осевые, центровые линии (ESKDLineType.CENTER)
        'centerline': {
            'stroke': 'red',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt',
            '_gost_type': LineType.CHAIN_THIN.value,
            '_eskd_type': ESKDLineType.CENTER,
        },
        # Сплошная тонкая — размерные линии (ESKDLineType.THIN)
        'dimension': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt',
            '_gost_type': LineType.CONTINUOUS_THIN.value,
            '_eskd_type': ESKDLineType.THIN,
        },
        # Разомкнутая — линии сечений (ESKDLineType.SECTION)
        'section': {
            'stroke': 'black',
            'stroke_width': f'{open_width:.2f}mm',
            'stroke_linecap': 'butt',
            '_gost_type': LineType.OPEN.value,
            '_eskd_type': ESKDLineType.SECTION,
        },
        # Штрихпунктирная толстая — линии обозначения поверхностей
        'chain_thick': {
            'stroke': 'black',
            'stroke_width': f'{chain_thick_width:.2f}mm',
            'stroke_linecap': 'butt',
            '_gost_type': LineType.CHAIN_THICK.value,
            '_eskd_type': ESKDLineType.SOLID,
        },
        # Волнистая — линии обрыва (ESKDLineType.WAVELINE)
        'waveline': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'round',
            '_gost_type': LineType.CONTINUOUS_WAVY.value,
            '_eskd_type': ESKDLineType.WAVELINE,
        },
        # Рамка основная (ESKDLineType.FRAME)
        'frame': {
            'stroke': 'black',
            'stroke_width': f'{stroke_width}mm',
            'stroke_linecap': 'butt',
            '_eskd_type': ESKDLineType.FRAME,
        },
        # Рамка тонкая (ESKDLineType.FRAME_THIN)
        'frame_thin': {
            'stroke': 'black',
            'stroke_width': f'{ESKDLineType.FRAME_THIN.stroke_width}mm',
            'stroke_linecap': 'butt',
            '_eskd_type': ESKDLineType.FRAME_THIN,
        },
        # Штамп тонкая (ESKDLineType.STAMP_THIN)
        'stamp_thin': {
            'stroke': 'black',
            'stroke_width': f'{ESKDLineType.STAMP_THIN.stroke_width}mm',
            'stroke_linecap': 'butt',
            '_eskd_type': ESKDLineType.STAMP_THIN,
        },
        # Текст размеров (ГОСТ 2.304-81)
        'dimension_text': {
            'font_family': f'{DIM_FONT_FAMILY}, Arial',
            'font_size': f'{DIM_TEXT_HEIGHT}mm',
            'font_style': 'italic',
            'fill': 'black',
            'text_anchor': 'middle',
        },
        # Стрелки размеров (ГОСТ 2.307-2011)
        'dimension_arrow': {
            'length': DIM_ARROW_LENGTH,
            'width': DIM_ARROW_WIDTH,
        },
        # Параметры для внутреннего использования
        '_params': {
            'S': stroke_width,
            'thin_width': thin_width,
            'dash_length': dash_length,
            'gap_length': gap_length,
            'cl_dash': cl_dash,
            'cl_gap': cl_gap,
            'cl_dot': cl_dot,
            'scale': scale,
            'format': format_name,
        },
    }


def calculate_line_parameters_for_scale(scale: float,
                                         format_name: str = "A4") -> Dict[str, float]:
    """Get all line thicknesses for a given scale and format.

    Convenience function that returns just the numeric thicknesses.

    Args:
        scale: Drawing scale
        format_name: Paper format

    Returns:
        Dict mapping line type names to thickness in mm
    """
    s = get_recommended_s(scale, format_name)
    return {
        'S': s,
        'visible': s,
        'hidden': s * GOST_LINE_SPECS[LineType.DASHED].thickness_ratio,
        'thin': s * GOST_LINE_SPECS[LineType.CONTINUOUS_THIN].thickness_ratio,
        'centerline': s * GOST_LINE_SPECS[LineType.CHAIN_THIN].thickness_ratio,
        'section': s * GOST_LINE_SPECS[LineType.OPEN].thickness_ratio,
        'chain_thick': s * GOST_LINE_SPECS[LineType.CHAIN_THICK].thickness_ratio,
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
