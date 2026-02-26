"""
Размещение размеров на чертеже (ГОСТ 2.307-2011).

Правила ГОСТ 2.307-2011:
  - Первая размерная линия: 10 мм от контура
  - Последующие: 7 мм друг от друга
  - Выносные линии: выступ 2 мм за размерную линию
  - Зазор между контуром и началом выносной линии: 1.5 мм
  - Меньшие размеры ближе к контуру (вложенность)
  - Текст над горизонтальными размерными линиями
  - Текст слева от вертикальных размерных линий

Алгоритм:
  1. Группировка размеров по стороне (top/bottom/left/right)
  2. Фильтрация слишком мелких размеров (< DIM_MIN_DISPLAYABLE мм на бумаге)
  3. Сортировка по значению (меньшие ближе к контуру)
  4. Назначение рядов (row 0 = 10 мм, row 1 = 17 мм, ...)
  5. Пересчёт в координаты листа (мм)
  6. Проверка наложений (AABB collision) и сдвиг при конфликтах
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from stl_drawing.config import (
    DIM_ARROW_LENGTH,
    DIM_EXTENSION_GAP,
    DIM_EXTENSION_OVERSHOOT,
    DIM_FIRST_OFFSET,
    DIM_MIN_DISPLAYABLE,
    DIM_NEXT_OFFSET,
    DIM_TEXT_GAP,
    DIM_TEXT_HEIGHT,
)
from stl_drawing.drawing.dimensions.extractor import DimensionCandidate
from stl_drawing.drawing.dimensions.renderer import PlacedDimension

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AABB для проверки наложений
# ---------------------------------------------------------------------------

@dataclass
class _BBox:
    """Ограничивающий прямоугольник для проверки коллизий."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def overlaps(self, other: '_BBox', margin: float = 1.0) -> bool:
        """Проверить перекрытие с другим bbox (с запасом margin мм)."""
        return not (
            self.x_max + margin < other.x_min or
            other.x_max + margin < self.x_min or
            self.y_max + margin < other.y_min or
            other.y_max + margin < self.y_min
        )


def _compute_dim_bbox(pd: PlacedDimension) -> _BBox:
    """Вычислить bounding box размещённого размера (включая текст).

    Args:
        pd: размещённый размер.

    Returns:
        _BBox с координатами в мм на листе.
    """
    # Собрать все точки элементов размера
    points_x = [
        pd.dim_line_start[0], pd.dim_line_end[0],
        pd.arrow_a_pos[0], pd.arrow_b_pos[0],
        pd.text_pos[0],
    ]
    points_y = [
        pd.dim_line_start[1], pd.dim_line_end[1],
        pd.arrow_a_pos[1], pd.arrow_b_pos[1],
        pd.text_pos[1],
    ]

    # Для не-диаметров добавить выносные линии
    if pd.dim_type != 'diameter':
        points_x.extend([
            pd.ext_line_a_start[0], pd.ext_line_a_end[0],
            pd.ext_line_b_start[0], pd.ext_line_b_end[0],
        ])
        points_y.extend([
            pd.ext_line_a_start[1], pd.ext_line_a_end[1],
            pd.ext_line_b_start[1], pd.ext_line_b_end[1],
        ])

    # Оценка ширины и высоты текста
    text_half_w = len(pd.text_value) * DIM_TEXT_HEIGHT * 0.35
    text_h = DIM_TEXT_HEIGHT * 1.0

    # Добавить запас на текст (с учётом его размеров)
    tx, ty = pd.text_pos
    points_x.extend([tx - text_half_w, tx + text_half_w])
    points_y.extend([ty - text_h, ty + text_h * 0.5])

    return _BBox(
        x_min=min(points_x),
        y_min=min(points_y),
        x_max=max(points_x),
        y_max=max(points_y),
    )


# ---------------------------------------------------------------------------
# Главная функция размещения
# ---------------------------------------------------------------------------

def place_dimensions(
    dims_by_view: Dict[str, List[DimensionCandidate]],
    layout: Dict[str, Dict],
    scale: float,
    eskd_styles: dict,
) -> List[PlacedDimension]:
    """Разместить все размеры на листе.

    Args:
        dims_by_view: словарь {view_name → [DimensionCandidate]}.
        layout: словарь компоновки {view_name → layout_entry}.
        scale: масштаб чертежа (мм-на-листе / мм-модели).
        eskd_styles: словарь стилей.

    Returns:
        Список PlacedDimension с готовыми координатами в мм на листе.
    """
    all_placed: List[PlacedDimension] = []
    # Глобальный список bbox для проверки наложений между видами
    global_bboxes: List[_BBox] = []

    # Зоны видов — размеры не должны налезать на другие виды
    view_zones: List[_BBox] = []
    for vn, vl in layout.items():
        view_zones.append(_BBox(
            x_min=vl['x'],
            y_min=vl['y'],
            x_max=vl['x'] + vl['width'],
            y_max=vl['y'] + vl['height'],
        ))

    for view_name, dims in dims_by_view.items():
        if not dims or view_name not in layout:
            continue

        view_layout = layout[view_name]
        tx = view_layout['x'] + view_layout['offset_x']
        ty = view_layout['y'] + view_layout['offset_y']

        # Bbox вида в мм на листе
        vx = view_layout['x']
        vy = view_layout['y']
        vw = view_layout['width']
        vh = view_layout['height']

        placed = _place_view_dims(
            dims, scale, tx, ty, vx, vy, vw, vh,
            global_bboxes, view_zones,
        )
        all_placed.extend(placed)

    logger.info("Размещено %d размеров на листе", len(all_placed))
    return all_placed


def _place_view_dims(
    dims: List[DimensionCandidate],
    scale: float,
    tx: float, ty: float,
    vx: float, vy: float, vw: float, vh: float,
    global_bboxes: List[_BBox],
    view_zones: List[_BBox],
) -> List[PlacedDimension]:
    """Разместить размеры одного вида с проверкой наложений.

    Args:
        dims: список кандидатов размеров.
        scale: масштаб чертежа.
        tx, ty: смещение вида (x + offset_x, y + offset_y).
        vx, vy: позиция вида на листе.
        vw, vh: ширина и высота вида на листе.
        global_bboxes: глобальный список bbox уже размещённых размеров.
        view_zones: зоны видов (для проверки наложений на другие виды).

    Returns:
        Список PlacedDimension.
    """
    # Фильтрация: исключить размеры, слишком мелкие на бумаге
    filtered = []
    for d in dims:
        paper_size = d.value_mm * scale
        if paper_size < DIM_MIN_DISPLAYABLE:
            logger.debug(
                "Размер %.1f мм (%.1f мм на бумаге) отфильтрован как слишком мелкий",
                d.value_mm, paper_size,
            )
            continue
        filtered.append(d)

    if not filtered:
        return []

    # Группировка по стороне
    groups: Dict[str, List[DimensionCandidate]] = {
        'top': [], 'bottom': [], 'left': [], 'right': [],
    }

    for d in filtered:
        side = d.preferred_side
        if d.dim_type == 'diameter':
            side = d.preferred_side if d.preferred_side in groups else 'right'
        elif d.dim_type == 'linear_horizontal':
            side = d.preferred_side if d.preferred_side in ('top', 'bottom') else 'bottom'
        elif d.dim_type == 'linear_vertical':
            side = d.preferred_side if d.preferred_side in ('left', 'right') else 'left'
        groups[side].append(d)

    # Сортировка: меньшие размеры ближе к контуру (ГОСТ 2.307-2011)
    for side in groups:
        groups[side].sort(key=lambda d: d.value_mm)

    placed: List[PlacedDimension] = []

    for side, side_dims in groups.items():
        for row_idx, dim in enumerate(side_dims):
            offset = DIM_FIRST_OFFSET + row_idx * DIM_NEXT_OFFSET

            # Попытка размещения с проверкой наложений (до 5 сдвигов)
            pd = None
            for bump in range(6):
                current_offset = offset + bump * DIM_NEXT_OFFSET

                if dim.dim_type == 'diameter':
                    pd = _place_diameter(dim, scale, tx, ty, vx, vy, vw, vh, current_offset, side)
                elif dim.dim_type == 'linear_horizontal':
                    pd = _place_horizontal(dim, scale, tx, ty, vx, vy, vw, vh, current_offset, side)
                elif dim.dim_type == 'linear_vertical':
                    pd = _place_vertical(dim, scale, tx, ty, vx, vy, vw, vh, current_offset, side)
                else:
                    break

                if pd is None:
                    break

                # Проверка наложений
                new_bbox = _compute_dim_bbox(pd)
                has_overlap = False

                # Проверка с уже размещёнными размерами
                for existing_bbox in global_bboxes:
                    if new_bbox.overlaps(existing_bbox, margin=1.5):
                        has_overlap = True
                        break

                # Проверка с зонами других видов (кроме текущего)
                if not has_overlap:
                    current_view_zone = _BBox(vx, vy, vx + vw, vy + vh)
                    for vz in view_zones:
                        # Пропустить собственный вид
                        if (abs(vz.x_min - current_view_zone.x_min) < 0.1 and
                                abs(vz.y_min - current_view_zone.y_min) < 0.1):
                            continue
                        if new_bbox.overlaps(vz, margin=2.0):
                            has_overlap = True
                            break

                if not has_overlap:
                    break  # Размещение ОК

                # Есть наложение — попробовать следующий ряд
                pd = None

            if pd is not None:
                bbox = _compute_dim_bbox(pd)
                global_bboxes.append(bbox)
                placed.append(pd)

    return placed


# ---------------------------------------------------------------------------
# Размещение горизонтальных размеров
# ---------------------------------------------------------------------------

def _place_horizontal(
    dim: DimensionCandidate,
    scale: float,
    tx: float, ty: float,
    vx: float, vy: float, vw: float, vh: float,
    offset: float,
    side: str,
) -> PlacedDimension:
    """Разместить горизонтальный линейный размер."""
    # Якоря в координатах листа
    ax1 = dim.anchor_a[0] * scale + tx
    ax2 = dim.anchor_b[0] * scale + tx

    # Убедимся, что ax1 < ax2
    if ax1 > ax2:
        ax1, ax2 = ax2, ax1

    if side == 'bottom':
        # Размерная линия ниже вида
        dim_y = vy + vh + offset
        # Начала выносных линий (от нижнего контура вида + зазор)
        ext_start_y = vy + vh + DIM_EXTENSION_GAP
        ext_end_y = dim_y + DIM_EXTENSION_OVERSHOOT
    else:  # top
        dim_y = vy - offset
        ext_start_y = vy - DIM_EXTENSION_GAP
        ext_end_y = dim_y - DIM_EXTENSION_OVERSHOOT

    # Текст над размерной линией
    text_y = dim_y - DIM_TEXT_GAP
    text_x = (ax1 + ax2) / 2.0

    # Форматирование значения
    text_value = _format_dim_value(dim.value_mm)

    return PlacedDimension(
        dim_type=dim.dim_type,
        text_value=text_value,
        text_angle=0.0,
        ext_line_a_start=(ax1, ext_start_y),
        ext_line_a_end=(ax1, ext_end_y),
        ext_line_b_start=(ax2, ext_start_y),
        ext_line_b_end=(ax2, ext_end_y),
        dim_line_start=(ax1, dim_y),
        dim_line_end=(ax2, dim_y),
        arrow_a_pos=(ax1, dim_y),
        arrow_a_angle=0.0,      # → (стрелка указывает вправо)
        arrow_b_pos=(ax2, dim_y),
        arrow_b_angle=180.0,    # ← (стрелка указывает влево)
        text_pos=(text_x, text_y),
    )


# ---------------------------------------------------------------------------
# Размещение вертикальных размеров
# ---------------------------------------------------------------------------

def _place_vertical(
    dim: DimensionCandidate,
    scale: float,
    tx: float, ty: float,
    vx: float, vy: float, vw: float, vh: float,
    offset: float,
    side: str,
) -> PlacedDimension:
    """Разместить вертикальный линейный размер."""
    # Якоря в координатах листа
    ay1 = dim.anchor_a[1] * scale + ty
    ay2 = dim.anchor_b[1] * scale + ty

    if ay1 > ay2:
        ay1, ay2 = ay2, ay1

    if side == 'left':
        dim_x = vx - offset
        ext_start_x = vx - DIM_EXTENSION_GAP
        ext_end_x = dim_x - DIM_EXTENSION_OVERSHOOT
    else:  # right
        dim_x = vx + vw + offset
        ext_start_x = vx + vw + DIM_EXTENSION_GAP
        ext_end_x = dim_x + DIM_EXTENSION_OVERSHOOT

    text_x = dim_x - DIM_TEXT_GAP if side == 'left' else dim_x + DIM_TEXT_GAP
    text_y = (ay1 + ay2) / 2.0

    text_value = _format_dim_value(dim.value_mm)

    return PlacedDimension(
        dim_type=dim.dim_type,
        text_value=text_value,
        text_angle=270.0 if side == 'left' else 90.0,
        ext_line_a_start=(ext_start_x, ay1),
        ext_line_a_end=(ext_end_x, ay1),
        ext_line_b_start=(ext_start_x, ay2),
        ext_line_b_end=(ext_end_x, ay2),
        dim_line_start=(dim_x, ay1),
        dim_line_end=(dim_x, ay2),
        arrow_a_pos=(dim_x, ay1),
        arrow_a_angle=90.0,     # ↓ (стрелка указывает вниз)
        arrow_b_pos=(dim_x, ay2),
        arrow_b_angle=270.0,    # ↑ (стрелка указывает вверх)
        text_pos=(text_x, text_y),
    )


# ---------------------------------------------------------------------------
# Размещение диаметров
# ---------------------------------------------------------------------------

def _place_diameter(
    dim: DimensionCandidate,
    scale: float,
    tx: float, ty: float,
    vx: float, vy: float, vw: float, vh: float,
    offset: float,
    side: str,
) -> Optional[PlacedDimension]:
    """Разместить размер диаметра (Ø).

    Рисует размерную линию через центр окружности,
    горизонтально, с текстом Ø{значение}.
    """
    if dim.center is None or dim.radius is None:
        return None

    cx = dim.center[0] * scale + tx
    cy = dim.center[1] * scale + ty
    r_paper = dim.radius * scale

    # Размерная линия через центр, горизонтально
    dim_start_x = cx - r_paper
    dim_end_x = cx + r_paper

    # Текст справа от окружности с выноской
    text_x = dim_end_x + DIM_ARROW_LENGTH + DIM_TEXT_GAP + 5.0
    text_y = cy - DIM_TEXT_GAP

    text_value = f"\u00d8{_format_dim_value(dim.value_mm)}"

    # Лидер-линия от окружности вправо до текста
    leader_end_x = text_x - 3.0

    return PlacedDimension(
        dim_type='diameter',
        text_value=text_value,
        text_angle=0.0,
        # Выносные линии не используются для диаметра
        ext_line_a_start=(cx, cy),
        ext_line_a_end=(cx, cy),
        ext_line_b_start=(cx, cy),
        ext_line_b_end=(cx, cy),
        # Размерная линия через центр
        dim_line_start=(dim_start_x, cy),
        dim_line_end=(leader_end_x, cy),
        # Стрелки на концах диаметра
        arrow_a_pos=(dim_start_x, cy),
        arrow_a_angle=180.0,   # ← левая стрелка
        arrow_b_pos=(dim_end_x, cy),
        arrow_b_angle=0.0,     # → правая стрелка
        text_pos=(text_x, text_y),
    )


# ---------------------------------------------------------------------------
# Форматирование значений размеров
# ---------------------------------------------------------------------------

def _format_dim_value(value_mm: float) -> str:
    """Форматировать значение размера по ГОСТ 2.307-2011.

    Целые числа без дробной части. Дробные — с одним знаком.
    """
    if abs(value_mm - round(value_mm)) < 0.05:
        return str(int(round(value_mm)))
    else:
        return f"{value_mm:.1f}"
