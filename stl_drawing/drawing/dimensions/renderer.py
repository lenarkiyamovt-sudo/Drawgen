"""
SVG-рендеринг размерных линий по ГОСТ 2.307-2011.

Содержит:
- render_dimensions      — отрисовка всех размеров в SVG-группу
- _create_arrow          — стрелка размерной линии
- _create_dim_text       — размерная надпись (ГОСТ 2.304-81 тип Б)
- _create_dim_line       — размерная линия (тонкая сплошная)
- _create_extension_line — выносная линия (тонкая сплошная)
"""

import math
from typing import List, Tuple

import svgwrite

from stl_drawing.config import (
    DIM_ARROW_LENGTH,
    DIM_ARROW_WIDTH,
    DIM_FONT_FAMILY,
    DIM_TEXT_GAP,
    DIM_TEXT_HEIGHT,
)


def _filter_svg_attrs(style_dict: dict) -> dict:
    """Filter out internal metadata keys (starting with _) from style dict."""
    return {k: v for k, v in style_dict.items() if not k.startswith('_')}


# ---------------------------------------------------------------------------
# Типы данных для размещённых размеров
# ---------------------------------------------------------------------------

class PlacedDimension:
    """Размещённый размер с готовыми координатами в мм на листе."""

    __slots__ = (
        'dim_type', 'text_value', 'text_angle',
        'ext_line_a_start', 'ext_line_a_end',
        'ext_line_b_start', 'ext_line_b_end',
        'dim_line_start', 'dim_line_end',
        'arrow_a_pos', 'arrow_a_angle',
        'arrow_b_pos', 'arrow_b_angle',
        'text_pos',
    )

    def __init__(
        self,
        dim_type: str,
        text_value: str,
        text_angle: float,
        ext_line_a_start: Tuple[float, float],
        ext_line_a_end: Tuple[float, float],
        ext_line_b_start: Tuple[float, float],
        ext_line_b_end: Tuple[float, float],
        dim_line_start: Tuple[float, float],
        dim_line_end: Tuple[float, float],
        arrow_a_pos: Tuple[float, float],
        arrow_a_angle: float,
        arrow_b_pos: Tuple[float, float],
        arrow_b_angle: float,
        text_pos: Tuple[float, float],
    ):
        self.dim_type = dim_type
        self.text_value = text_value
        self.text_angle = text_angle
        self.ext_line_a_start = ext_line_a_start
        self.ext_line_a_end = ext_line_a_end
        self.ext_line_b_start = ext_line_b_start
        self.ext_line_b_end = ext_line_b_end
        self.dim_line_start = dim_line_start
        self.dim_line_end = dim_line_end
        self.arrow_a_pos = arrow_a_pos
        self.arrow_a_angle = arrow_a_angle
        self.arrow_b_pos = arrow_b_pos
        self.arrow_b_angle = arrow_b_angle
        self.text_pos = text_pos


# ---------------------------------------------------------------------------
# Публичная функция рендеринга
# ---------------------------------------------------------------------------

def render_dimensions(
    dwg: svgwrite.Drawing,
    placed_dims: List[PlacedDimension],
    eskd_styles: dict,
) -> svgwrite.container.Group:
    """Отрисовать все размещённые размеры в SVG-группу.

    Args:
        dwg: SVG-документ.
        placed_dims: список PlacedDimension с координатами в мм на листе.
        eskd_styles: словарь стилей из calculate_line_parameters.

    Returns:
        SVG-группа со всеми размерами.
    """
    dim_group = dwg.g(id='dimensions')
    line_style = eskd_styles.get('dimension', eskd_styles['thin'])
    text_style = eskd_styles.get('dimension_text', {})
    arrow_params = eskd_styles.get('dimension_arrow', {
        'length': DIM_ARROW_LENGTH, 'width': DIM_ARROW_WIDTH,
    })

    for pd in placed_dims:
        single = dwg.g()
        single['data-dim-type'] = pd.dim_type
        single['data-dim-value'] = pd.text_value

        # Выносные линии
        if pd.dim_type != 'diameter':
            ext_a = _create_extension_line(
                dwg, pd.ext_line_a_start, pd.ext_line_a_end, line_style)
            ext_b = _create_extension_line(
                dwg, pd.ext_line_b_start, pd.ext_line_b_end, line_style)
            single.add(ext_a)
            single.add(ext_b)

        # Размерная линия
        dim_line = _create_dim_line(
            dwg, pd.dim_line_start, pd.dim_line_end, line_style)
        single.add(dim_line)

        # Стрелки
        arrow_a = _create_arrow(
            dwg, pd.arrow_a_pos, pd.arrow_a_angle, arrow_params)
        arrow_b = _create_arrow(
            dwg, pd.arrow_b_pos, pd.arrow_b_angle, arrow_params)
        single.add(arrow_a)
        single.add(arrow_b)

        # Фон текста (белый прямоугольник для читаемости)
        text_bg = _create_text_background(
            dwg, pd.text_value, pd.text_pos, pd.text_angle)
        single.add(text_bg)

        # Текст
        text_el = _create_dim_text(
            dwg, pd.text_value, pd.text_pos, pd.text_angle, text_style)
        single.add(text_el)

        dim_group.add(single)

    return dim_group


# ---------------------------------------------------------------------------
# Примитивы размерных элементов
# ---------------------------------------------------------------------------

def _create_arrow(
    dwg: svgwrite.Drawing,
    position: Tuple[float, float],
    angle_deg: float,
    arrow_params: dict,
) -> svgwrite.path.Path:
    """Создать стрелку размерной линии (заполненный треугольник).

    ГОСТ 2.307-2011: стрелка — равнобедренный треугольник,
    длина ≈ 2.5 мм, угол раствора ≈ 20°.

    Args:
        dwg: SVG-документ.
        position: вершина стрелки (мм на листе).
        angle_deg: угол поворота стрелки (градусы, 0 = →).
        arrow_params: словарь с 'length' и 'width'.

    Returns:
        SVG path-элемент.
    """
    length = arrow_params.get('length', DIM_ARROW_LENGTH)
    width = arrow_params.get('width', DIM_ARROW_WIDTH)
    half_w = width / 2.0

    # Треугольник: вершина в (0,0), основание назад
    d = f"M 0,0 L {-length},{half_w} L {-length},{-half_w} Z"

    x, y = position
    arrow = dwg.path(
        d=d,
        fill='black',
        stroke='none',
        transform=f"translate({x:.4f},{y:.4f}) rotate({angle_deg:.2f})",
    )
    return arrow


def _create_dim_text(
    dwg: svgwrite.Drawing,
    text: str,
    position: Tuple[float, float],
    angle_deg: float,
    text_style: dict,
) -> svgwrite.text.Text:
    """Создать размерную надпись (ГОСТ 2.304-81, тип Б, наклонный).

    Использует явное смещение baseline (cap-height ≈ 0.7em → центр ≈ 0.35em),
    вместо ненадёжного dominant-baseline, как в ESKDDrawingSheet._text() донора.

    Args:
        dwg: SVG-документ.
        text: текст размера (напр. "125", "Ø20").
        position: центр текста (мм на листе).
        angle_deg: угол поворота текста (градусы).
        text_style: словарь стилей шрифта.

    Returns:
        SVG text-элемент.
    """
    x, y = position
    font_family = text_style.get('font_family', f'{DIM_FONT_FAMILY}, Arial, sans-serif')
    font_size = text_style.get('font_size', f'{DIM_TEXT_HEIGHT}mm')

    # Смещение baseline: cap-height ≈ 0.7*em → центр текста при y_bl = y + size*0.35
    # (паттерн из донора stl_to_point_1.py, метод _text())
    y_bl = y + DIM_TEXT_HEIGHT * 0.35

    txt = dwg.text(
        text,
        insert=(x, y_bl),
        font_family=font_family,
        font_size=font_size,
        font_style='italic',
        font_weight='normal',
        text_anchor='middle',
        fill='black',
    )
    if abs(angle_deg) > 0.1:
        txt['transform'] = f"rotate({angle_deg:.2f},{x:.4f},{y:.4f})"

    return txt


def _create_dim_line(
    dwg: svgwrite.Drawing,
    start: Tuple[float, float],
    end: Tuple[float, float],
    line_style: dict,
) -> svgwrite.shapes.Line:
    """Создать размерную линию (тонкая сплошная).

    Args:
        dwg: SVG-документ.
        start, end: концы линии (мм на листе).
        line_style: словарь стилей линии.

    Returns:
        SVG line-элемент.
    """
    filtered_style = _filter_svg_attrs(line_style)
    line = dwg.line(start=start, end=end, **filtered_style)
    line['data-line-type'] = 'dimension'
    return line


def _create_extension_line(
    dwg: svgwrite.Drawing,
    start: Tuple[float, float],
    end: Tuple[float, float],
    line_style: dict,
) -> svgwrite.shapes.Line:
    """Создать выносную линию (тонкая сплошная).

    Args:
        dwg: SVG-документ.
        start, end: концы линии (мм на листе).
        line_style: словарь стилей линии.

    Returns:
        SVG line-элемент.
    """
    filtered_style = _filter_svg_attrs(line_style)
    line = dwg.line(start=start, end=end, **filtered_style)
    line['data-line-type'] = 'extension'
    return line


def _create_text_background(
    dwg: svgwrite.Drawing,
    text: str,
    position: Tuple[float, float],
    angle_deg: float,
) -> svgwrite.shapes.Rect:
    """Создать белый прямоугольник-фон за текстом размера.

    Предотвращает визуальное наложение размерной/выносной линии на текст.

    Args:
        dwg: SVG-документ.
        text: текст размера (для оценки ширины).
        position: центр текста (мм на листе).
        angle_deg: угол поворота текста (градусы).

    Returns:
        SVG rect-элемент (белый, без обводки).
    """
    x, y = position

    # Оценка ширины текста: ~0.6 * высота шрифта на символ (ISOCPEUR italic)
    text_w = len(text) * DIM_TEXT_HEIGHT * 0.6
    text_h = DIM_TEXT_HEIGHT * 1.3
    padding = 0.3  # мм запас

    rect = dwg.rect(
        insert=(x - text_w / 2 - padding, y - text_h + DIM_TEXT_HEIGHT * 0.35),
        size=(text_w + 2 * padding, text_h + padding),
        fill='white',
        stroke='none',
    )
    if abs(angle_deg) > 0.1:
        rect['transform'] = f"rotate({angle_deg:.2f},{x:.4f},{y:.4f})"
    return rect
