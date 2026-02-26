"""
Компоновка видов на листе по ГОСТ 2.305-2008 (метод первого угла).

Стандартная 6-видовая схема:
         [bottom]
[right] [front] [left] [back]
         [top]

Метод первого угла (европейская проекция):
  - вид справа → СЛЕВА от фронтального
  - вид сверху → ПОД фронтальным
  - вид снизу → НАД фронтальным
"""

import logging
from typing import Dict, Optional, Tuple

from stl_drawing.config import (
    DIM_MARGIN_RESERVE,
    MARGIN_LEFT,
    MARGIN_OTHER,
    TITLE_BLOCK_H,
    VIEW_SPACING_MM,
    N_GAPS_HORIZONTAL,
    N_GAPS_VERTICAL,
    GOST_FORMATS,
    GOST_FORMATS_ORDERED,
    GOST_REDUCTION_SCALES,
)
from stl_drawing.drawing.gost_params import snap_to_gost_scale

logger = logging.getLogger(__name__)

# Устаревший MARGIN используется только для 3-видового fallback
_LEGACY_MARGIN = 10


def make_layout_entry(view_name: str, x: float, y: float, views_data: Dict, scale: float) -> Dict:
    """Создать запись компоновки для одного вида.

    Args:
        view_name: имя вида.
        x, y: координаты левого-верхнего угла вида (мм).
        views_data: словарь данных видов.
        scale: масштаб чертежа.

    Returns:
        Словарь с ключами: x, y, width, height, center_x, center_y, offset_x, offset_y.
    """
    bbox = views_data[view_name]['bbox']
    w = bbox['width'] * scale
    h = bbox['height'] * scale
    return {
        'x': x, 'y': y,
        'width': w, 'height': h,
        'center_x': x + w / 2,
        'center_y': y + h / 2,
        'offset_x': -bbox['min_x'] * scale,
        'offset_y': -bbox['min_y'] * scale,
    }


def arrange_six_views(views_data: Dict, scale: float, sheet_w: float, sheet_h: float) -> Dict:
    """Расположить 6 видов в схеме первого угла (ГОСТ 2.305-2008).

    Args:
        views_data: данные всех шести видов.
        scale: масштаб чертежа.
        sheet_w, sheet_h: размеры листа (мм).

    Returns:
        Словарь {view_name → layout_entry}.
    """
    s = scale
    sp = VIEW_SPACING_MM
    margin = MARGIN_OTHER

    def w(name): return views_data[name]['bbox']['width'] * s
    def h(name): return views_data[name]['bbox']['height'] * s

    front_w, front_h = w('front'), h('front')
    left_w,  right_w = w('left'), w('right')
    top_h,   bottom_h = h('top'), h('bottom')
    back_w = w('back')

    # Габариты блока
    row_mid_width = right_w + sp + front_w + sp + left_w + sp + back_w
    total_height  = bottom_h + sp + front_h + sp + top_h

    usable_w = sheet_w - MARGIN_LEFT - margin
    usable_h = sheet_h - 2 * margin - TITLE_BLOCK_H

    block_x = MARGIN_LEFT + margin + (usable_w - row_mid_width) / 2
    block_y = margin + (usable_h - total_height) / 2

    mid_row_y = block_y + bottom_h + sp
    front_x   = block_x + right_w + sp

    layout = {}
    layout['front']  = make_layout_entry('front',  front_x, mid_row_y, views_data, scale)
    layout['right']  = make_layout_entry('right',  block_x, mid_row_y + (front_h - h('right')) / 2, views_data, scale)
    layout['left']   = make_layout_entry('left',   front_x + front_w + sp, mid_row_y + (front_h - h('left')) / 2, views_data, scale)
    layout['back']   = make_layout_entry('back',   front_x + front_w + sp + left_w + sp, mid_row_y + (front_h - h('back')) / 2, views_data, scale)
    layout['top']    = make_layout_entry('top',    front_x + (front_w - w('top')) / 2, mid_row_y + front_h + sp, views_data, scale)
    layout['bottom'] = make_layout_entry('bottom', front_x + (front_w - w('bottom')) / 2, block_y, views_data, scale)
    return layout


def arrange_three_views(views_data: Dict, scale: float, sheet_w: float, sheet_h: float) -> Dict:
    """Расположить 3 вида (front + top + right) — минимальный fallback."""
    s = scale
    sp = VIEW_SPACING_MM
    margin = _LEGACY_MARGIN

    front_bbox = views_data['front']['bbox']
    top_bbox   = views_data['top']['bbox']
    right_bbox = views_data['right']['bbox']

    front_w = front_bbox['width'] * s
    front_h = front_bbox['height'] * s
    top_h   = top_bbox['height'] * s
    right_w = right_bbox['width'] * s

    usable_w = sheet_w - 2 * margin
    usable_h = sheet_h - 2 * margin - TITLE_BLOCK_H

    # Общая высота блока: front + sp + top (top ПОД front по ГОСТ 2.305-2008, метод 1-го угла)
    total_h = front_h + sp + top_h
    cx = margin + (usable_w - front_w - sp - right_w) / 2
    cy = margin + (usable_h - total_h) / 2

    layout = {}
    layout['front'] = make_layout_entry('front', cx, cy, views_data, scale)
    # Метод первого угла: вид сверху → ПОД фронтальным
    layout['top']   = make_layout_entry('top',   cx, cy + front_h + sp, views_data, scale)
    layout['right'] = make_layout_entry('right', cx + front_w + sp, cy, views_data, scale)
    return layout


def arrange_grid(views_data: Dict, scale: float, sheet_w: float, sheet_h: float) -> Dict:
    """Расположить произвольный набор видов в сетку 3×N (fallback)."""
    view_names = list(views_data.keys())
    columns = min(len(view_names), 3)
    rows = (len(view_names) + columns - 1) // columns

    max_vw = max(views_data[v]['bbox']['width']  for v in view_names) * scale
    max_vh = max(views_data[v]['bbox']['height'] for v in view_names) * scale

    grid_w = max_vw * columns + 20 * (columns - 1)
    grid_h = max_vh * rows    + 20 * (rows - 1)

    margin = _LEGACY_MARGIN
    sx = margin + (sheet_w - 2 * margin - grid_w) / 2
    sy = margin + (sheet_h - 2 * margin - TITLE_BLOCK_H - grid_h) / 2

    layout = {}
    for i, name in enumerate(view_names):
        col, row = i % columns, i // columns
        vw = views_data[name]['bbox']['width'] * scale
        vh = views_data[name]['bbox']['height'] * scale
        x = sx + col * (max_vw + 20) + (max_vw - vw) / 2
        y = sy + row * (max_vh + 20) + (max_vh - vh) / 2
        layout[name] = make_layout_entry(name, x, y, views_data, scale)
    return layout


def arrange_views(views_data: Dict, scale: float, sheet_w: float, sheet_h: float) -> Dict:
    """Выбрать подходящую схему компоновки и расположить виды.

    Args:
        views_data: словарь данных видов.
        scale: масштаб чертежа.
        sheet_w, sheet_h: размеры листа (мм).

    Returns:
        Словарь {view_name → layout_entry}.
    """
    six = ['front', 'back', 'top', 'bottom', 'left', 'right']

    if all(v in views_data for v in six):
        return arrange_six_views(views_data, scale, sheet_w, sheet_h)

    three = ['front', 'top', 'right']
    if all(v in views_data for v in three):
        return arrange_three_views(views_data, scale, sheet_w, sheet_h)

    return arrange_grid(views_data, scale, sheet_w, sheet_h)


# ---------------------------------------------------------------------------
# Вычисление оптимального масштаба и формата
# ---------------------------------------------------------------------------

def compute_layout_model_dims(views_data: Dict) -> tuple:
    """Вычислить необходимые размеры компоновки в единицах модели.

    Returns:
        (model_width, model_height)
    """
    six = ['front', 'back', 'top', 'bottom', 'left', 'right']
    if all(v in views_data for v in six):
        fw = views_data['front']['bbox']['width']
        fh = views_data['front']['bbox']['height']
        lw = views_data['left']['bbox']['width']
        rw = views_data['right']['bbox']['width']
        bw = views_data['back']['bbox']['width']
        th = views_data['top']['bbox']['height']
        bth = views_data['bottom']['bbox']['height']
        return lw + fw + rw + bw, th + fh + bth

    # Fallback: front + right по X, front + top по Y
    fw = views_data.get('front', {}).get('bbox', {}).get('width', 1.0)
    fh = views_data.get('front', {}).get('bbox', {}).get('height', 1.0)
    rw = views_data.get('right', {}).get('bbox', {}).get('width', 1.0)
    th = views_data.get('top',   {}).get('bbox', {}).get('height', 1.0)
    return fw + rw, fh + th


def compute_layout_dims_mm(views_data: Dict, scale: float) -> Tuple[float, float]:
    """Вычислить габаритные размеры компоновки видов в мм для заданного масштаба.

    Учитывает зазоры между видами и тип компоновки (6/3/grid),
    в точности повторяя логику arrange-функций.

    Args:
        views_data: словарь данных видов.
        scale: масштаб чертежа.

    Returns:
        (total_width_mm, total_height_mm) — габаритные размеры блока видов.
    """
    sp = VIEW_SPACING_MM
    six = ['front', 'back', 'top', 'bottom', 'left', 'right']
    three = ['front', 'top', 'right']

    if all(v in views_data for v in six):
        fw = views_data['front']['bbox']['width'] * scale
        fh = views_data['front']['bbox']['height'] * scale
        lw = views_data['left']['bbox']['width'] * scale
        rw = views_data['right']['bbox']['width'] * scale
        bw = views_data['back']['bbox']['width'] * scale
        th = views_data['top']['bbox']['height'] * scale
        bth = views_data['bottom']['bbox']['height'] * scale
        return (rw + sp + fw + sp + lw + sp + bw,
                bth + sp + fh + sp + th)

    if all(v in views_data for v in three):
        fw = views_data['front']['bbox']['width'] * scale
        fh = views_data['front']['bbox']['height'] * scale
        rw = views_data['right']['bbox']['width'] * scale
        th = views_data['top']['bbox']['height'] * scale
        return (fw + sp + rw, th + sp + fh)

    # Grid fallback (3-column, 20mm spacing — как в arrange_grid)
    view_names = list(views_data.keys())
    if not view_names:
        return (0.0, 0.0)
    columns = min(len(view_names), 3)
    rows = (len(view_names) + columns - 1) // columns
    max_w = max(views_data[v]['bbox']['width'] for v in view_names) * scale
    max_h = max(views_data[v]['bbox']['height'] for v in view_names) * scale
    grid_sp = 20.0
    return (max_w * columns + grid_sp * (columns - 1),
            max_h * rows + grid_sp * (rows - 1))


def front_view_size_mm(views_data: Dict, scale: float) -> float:
    """Размер главного вида на листе в мм (минимальный из ширины и высоты)."""
    if 'front' not in views_data:
        if not views_data:
            return 50.0
        best = max(views_data.values(), key=lambda v: v['bbox']['width'] * v['bbox']['height'])
        return min(best['bbox']['width'] * scale, best['bbox']['height'] * scale)
    fw = views_data['front']['bbox']['width'] * scale
    fh = views_data['front']['bbox']['height'] * scale
    return min(fw, fh)


def select_format_and_scale(views_data: Dict) -> tuple:
    """Выбрать формат и масштаб ГОСТ, максимизирующий масштаб чертежа.

    Стратегия (scale-first, Tekla-style):
      1. Перебрать масштабы ГОСТ от крупного к мелкому (1:1 → 1:1000).
      2. Для каждого масштаба вычислить размеры компоновки в мм
         (с учётом реальных зазоров между видами).
      3. Перебрать форматы от A4 до A0, в обеих ориентациях
         (landscape и portrait).
      4. Вернуть первую подходящую комбинацию =
         максимальный масштаб + минимальный формат + оптимальная ориентация.

    Такой порядок перебора гарантирует наилучшую читаемость (крупный масштаб)
    при минимальном расходе бумаги (наименьший подходящий формат).

    Returns:
        (format_name, scale, sheet_width, sheet_height)
    """
    for gost_scale in GOST_REDUCTION_SCALES:
        layout_w, layout_h = compute_layout_dims_mm(views_data, gost_scale)

        for fmt in GOST_FORMATS_ORDERED:
            short, long_ = GOST_FORMATS[fmt]

            # Попробовать обе ориентации: landscape и portrait
            for sheet_w, sheet_h in [(long_, short), (short, long_)]:
                avail_w = sheet_w - MARGIN_LEFT - MARGIN_OTHER - 2 * DIM_MARGIN_RESERVE
                avail_h = sheet_h - 2 * MARGIN_OTHER - TITLE_BLOCK_H - 2 * DIM_MARGIN_RESERVE

                if avail_w <= 0 or avail_h <= 0:
                    continue

                if layout_w > avail_w + 0.1 or layout_h > avail_h + 0.1:
                    continue

                # Подходит — логируем и возвращаем
                front_mm = front_view_size_mm(views_data, gost_scale)
                fill = (layout_w * layout_h) / (avail_w * avail_h) * 100
                inv = 1.0 / gost_scale if gost_scale > 0 else 9999
                orient = 'landscape' if sheet_w >= sheet_h else 'portrait'

                logger.info(
                    "=> Формат: %s (%s)  масштаб 1:%.4g  "
                    "(фронт %.1f мм, заполнение %.0f%%)",
                    fmt, orient, inv, front_mm, fill,
                )
                return fmt, gost_scale, sheet_w, sheet_h

    # Аварийный fallback: A0 landscape, минимально возможный масштаб
    logger.warning("Ни один масштаб ГОСТ не подошёл — аварийный fallback на A0")
    short, long_ = GOST_FORMATS['A0']
    sheet_w, sheet_h = long_, short
    avail_w = sheet_w - MARGIN_LEFT - MARGIN_OTHER
    avail_h = sheet_h - 2 * MARGIN_OTHER - TITLE_BLOCK_H

    model_w, model_h = compute_layout_model_dims(views_data)
    raw = min(
        (avail_w - N_GAPS_HORIZONTAL * VIEW_SPACING_MM) / max(model_w, 1e-9),
        (avail_h - N_GAPS_VERTICAL * VIEW_SPACING_MM) / max(model_h, 1e-9),
    )
    best_scale = snap_to_gost_scale(raw)
    inv = 1.0 / best_scale if best_scale > 0 else 9999
    logger.info("=> Fallback: A0 landscape  масштаб 1:%.4g", inv)
    return 'A0', best_scale, sheet_w, sheet_h
