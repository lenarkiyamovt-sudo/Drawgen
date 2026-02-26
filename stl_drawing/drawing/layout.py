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
from typing import Dict, Optional

from stl_drawing.config import (
    MARGIN_LEFT,
    MARGIN_OTHER,
    TITLE_BLOCK_H,
    VIEW_SPACING_MM,
    N_GAPS_HORIZONTAL,
    N_GAPS_VERTICAL,
    GOST_FORMATS,
    GOST_FORMATS_ORDERED,
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

    cx = margin + (usable_w - front_w - right_w) / 2
    cy = margin + (usable_h - front_h - top_h) / 2

    layout = {}
    layout['front'] = make_layout_entry('front', cx, cy, views_data, scale)
    layout['top']   = make_layout_entry('top',   cx, cy - top_h - sp, views_data, scale)
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
    """Выбрать формат и масштаб ГОСТ, максимизирующий размер вида на листе.

    Стратегия: перебрать A4→A0, для каждого найти максимальный ГОСТ-масштаб,
    при котором компоновка помещается на лист. Выбрать наибольший масштаб
    (= лучшая читаемость). При равных масштабах — меньший формат.

    Returns:
        (format_name, scale, sheet_width, sheet_height)
    """
    from stl_drawing.config import TITLE_BLOCK_H, MARGIN_LEFT, MARGIN_OTHER

    model_w, model_h = compute_layout_model_dims(views_data)
    sp_mm = VIEW_SPACING_MM
    n_gaps_w = N_GAPS_HORIZONTAL
    n_gaps_h = N_GAPS_VERTICAL

    best_format, best_scale = None, 0.0

    for fmt in GOST_FORMATS_ORDERED:
        short, long_ = GOST_FORMATS[fmt]
        sheet_w, sheet_h = long_, short  # ландшафт

        avail_w = sheet_w - MARGIN_LEFT - MARGIN_OTHER
        avail_h = sheet_h - 2 * MARGIN_OTHER - TITLE_BLOCK_H

        views_w = avail_w - n_gaps_w * sp_mm
        views_h = avail_h - n_gaps_h * sp_mm
        if views_w <= 0 or views_h <= 0:
            continue

        raw_scale = min(views_w / model_w, views_h / model_h)
        gost_scale = snap_to_gost_scale(raw_scale)

        layout_w = model_w * gost_scale + n_gaps_w * sp_mm
        layout_h = model_h * gost_scale + n_gaps_h * sp_mm
        if layout_w > avail_w + 0.1 or layout_h > avail_h + 0.1:
            continue

        front_mm = front_view_size_mm(views_data, gost_scale)
        inv = 1.0 / gost_scale if gost_scale > 0 else 9999
        fill = (layout_w * layout_h) / (avail_w * avail_h) * 100

        logger.info(
            "  %s: avail %dx%d, ГОСТ 1:%.4g, фронт=%.1fмм, заполнение=%.0f%%",
            fmt, int(avail_w), int(avail_h), inv, front_mm, fill,
        )

        if gost_scale > best_scale:
            best_scale = gost_scale
            best_format = fmt

    if best_format is None:
        # Аварийный fallback: A0, минимально возможный масштаб
        from stl_drawing.config import TITLE_BLOCK_H
        best_format = 'A0'
        short, long_ = GOST_FORMATS['A0']
        avail_w = long_ - MARGIN_LEFT - MARGIN_OTHER
        avail_h = short - 2 * MARGIN_OTHER - TITLE_BLOCK_H
        raw = min(
            (avail_w - n_gaps_w * sp_mm) / model_w,
            (avail_h - n_gaps_h * sp_mm) / model_h,
        )
        best_scale = snap_to_gost_scale(raw)

    short, long_ = GOST_FORMATS[best_format]
    sheet_w, sheet_h = long_, short
    inv = 1.0 / best_scale if best_scale > 0 else 9999
    logger.info(
        "=> Выбран формат: %s  масштаб 1:%.4g  (фронт %.1f мм)",
        best_format, inv, front_view_size_mm(views_data, best_scale),
    )
    return best_format, best_scale, sheet_w, sheet_h
