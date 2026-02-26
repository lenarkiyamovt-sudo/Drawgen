"""
Рамка и штамп ЕСКД.

ГОСТ 2.104-2006 Форма 1: основная надпись 185 × 55 мм.
ГОСТ 2.303-68: толщины линий.

Координатная система OpenSCAD (используемая в оригинале):
  x → вправо, y → вверх (y=0 = низ штампа)
Перевод в SVG: svgY = y0 + 55 - openscad_y
"""

import logging
from typing import Dict, Optional

import svgwrite
from svgwrite.masking import ClipPath

from stl_drawing.config import (
    ESKDLineType,
    MARGIN_LEFT,
    MARGIN_OTHER,
    S_FRAME,
    STAMP_FONT_H,
    TITLE_BLOCK_H,
    TITLE_BLOCK_W,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции для работы с ESKDLineType
# ---------------------------------------------------------------------------

def _get_line_style(line_type: ESKDLineType) -> Dict[str, str]:
    """Получить SVG-атрибуты стиля для типа линии ЕСКД.

    Args:
        line_type: тип линии из ESKDLineType

    Returns:
        Словарь SVG-атрибутов для stroke
    """
    return line_type.get_svg_style()


def _apply_line_style(element, line_type: ESKDLineType) -> None:
    """Применить стиль линии ЕСКД к SVG-элементу.

    Args:
        element: SVG-элемент (line, rect, polyline, etc.)
        line_type: тип линии из ESKDLineType
    """
    style = line_type.get_svg_style()
    for attr, value in style.items():
        # SVG атрибуты используют дефис, не подчёркивание
        element[attr.replace('_', '-')] = value


# ---------------------------------------------------------------------------
# Рамка чертежа
# ---------------------------------------------------------------------------

def add_frame(dwg: svgwrite.Drawing, sheet_w: float, sheet_h: float) -> None:
    """Нарисовать рамку чертежа по ГОСТ 2.104-2006.

    Поля: слева 20 мм (корешок), остальные 5 мм.
    Внешняя граница — тонкая линия (FRAME_THIN), внутренняя рамка — основная (FRAME).
    """
    ml = MARGIN_LEFT
    mo = MARGIN_OTHER
    S = S_FRAME

    # Внешняя граница листа — тонкая линия (ESKDLineType.FRAME_THIN)
    outer_rect = dwg.rect(
        insert=(0, 0), size=(sheet_w, sheet_h),
        fill='none',
    )
    _apply_line_style(outer_rect, ESKDLineType.FRAME_THIN)
    dwg.add(outer_rect)

    # Внутренняя рамка — основная линия (ESKDLineType.FRAME)
    inner_rect = dwg.rect(
        insert=(ml, mo), size=(sheet_w - ml - mo, sheet_h - 2 * mo),
        fill='none',
    )
    _apply_line_style(inner_rect, ESKDLineType.FRAME)
    dwg.add(inner_rect)


# ---------------------------------------------------------------------------
# Основная надпись (штамп)
# ---------------------------------------------------------------------------

def add_title_block(
    dwg: svgwrite.Drawing,
    sheet_w: float,
    sheet_h: float,
    scale: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> None:
    """ГОСТ 2.104-2006 Форма 1 — основная надпись 185 × 55 мм."""
    mo = MARGIN_OTHER
    S = S_FRAME

    x0 = sheet_w - mo - TITLE_BLOCK_W
    y0 = sheet_h - mo - TITLE_BLOCK_H

    def X(ox: float) -> float: return x0 + ox
    def Y(oy: float) -> float: return y0 + TITLE_BLOCK_H - oy

    g = dwg.g(id='title-block')

    # Контур штампа (основная линия FRAME)
    stamp_rect = dwg.rect(
        insert=(x0, y0), size=(TITLE_BLOCK_W, TITLE_BLOCK_H),
        fill='white',
    )
    _apply_line_style(stamp_rect, ESKDLineType.FRAME)
    g.add(stamp_rect)

    _draw_title_horizontal_lines(g, dwg, X, Y, S)
    _draw_title_vertical_lines(g, dwg, X, Y, S)
    _draw_title_text(g, dwg, X, Y)
    _draw_title_content(g, dwg, X, Y, metadata or {})

    # Значение масштаба (Графа 6 — центр ячейки y=25, шрифт 1.8 мм)
    if scale and scale > 0:
        inv = 1.0 / scale
        scale_text = f"1:{int(round(inv))}" if abs(inv - round(inv)) < 0.01 else f"1:{inv:.2g}"
        _text(g, dwg, X(174), Y(25), scale_text, size=1.8, anchor='middle')

    dwg.add(g)


def add_additional_stamps(
    dwg: svgwrite.Drawing,
    sheet_w: float,
    sheet_h: float,
    format_name: str = "",
    metadata: Optional[dict] = None,
) -> None:
    """Дополнительные графы по ГОСТ 2.104-2006 (Гр. 19, 21, 22, 26)."""
    mo = MARGIN_OTHER
    ml = MARGIN_LEFT
    S = S_FRAME

    font_kw = dict(
        font_family='ISOCPEUR, Arial, sans-serif',
        font_style='italic', font_weight='normal', fill='black',
    )

    g = dwg.g(id='additional-stamps')

    doc_des = (metadata or {}).get('doc_designation', '')
    _draw_side_stamp(g, dwg, ml, sheet_h, mo, S, font_kw)
    _draw_ref_block(g, dwg, ml, mo, S, doc_des)
    _draw_footer(g, dwg, sheet_w, sheet_h, mo, S, format_name)

    dwg.add(g)


# ---------------------------------------------------------------------------
# Внутренние функции рисования
# ---------------------------------------------------------------------------

def _line(
    g,
    dwg,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    thick: bool = False,
    line_type: Optional[ESKDLineType] = None,
) -> None:
    """Нарисовать линию штампа с указанным типом ЕСКД.

    Args:
        g: SVG-группа
        dwg: svgwrite.Drawing
        x1, y1: начальная точка
        x2, y2: конечная точка
        thick: True для основной линии (FRAME), False для тонкой (STAMP_THIN)
        line_type: явно указанный тип линии (переопределяет thick)
    """
    S = S_FRAME
    if line_type is None:
        line_type = ESKDLineType.FRAME if thick else ESKDLineType.STAMP_THIN

    line = dwg.line(start=(x1, y1), end=(x2, y2))
    _apply_line_style(line, line_type)
    g.add(line)


# ---------------------------------------------------------------------------
# Размеры шрифта для ячеек основного штампа (мм).
# Редактируйте эти константы вручную для подбора.
# ---------------------------------------------------------------------------

# Графа 2 (обозначение документа): ячейка 120×15 мм
FONT_SZ_G2   = 3.5   # мм

# Графа 1 (наименование изделия): ячейка 120×15 мм
FONT_SZ_G1   = 3.5   # мм

# Графа 8 (наименование организации): ячейка 70×25 мм
FONT_SZ_G8   = 3.5   # мм

# Фамилии подписантов: ячейка 23×5 мм (Разраб./Пров./Т.контр./Н.контр./Утв.)
FONT_SZ_SN   = 2.0   # мм

# Графа 26 (боковой шифр документа, повёрнут −90°): доступно 72 мм в длину, 14 мм в высоту
FONT_SZ_G26  = 3.0   # мм


def _text(g, dwg, x, y, text, size=None, anchor='start') -> None:
    """Разместить текст так, чтобы он был визуально центрирован по y.

    y — координата ЦЕНТРА ячейки.
    Смещение baseline: y_bl = y + size * 0.35
    (cap-height ≈ 0.7*em → визуальный центр на 0.35*em ниже top).
    """
    if size is None:
        size = STAMP_FONT_H
    y_bl = y + size * 0.35
    g.add(dwg.text(
        text, insert=(x, y_bl),
        font_family='ISOCPEUR, Arial, sans-serif',
        font_style='italic', font_weight='normal',
        font_size=f'{size}mm', text_anchor=anchor,
        fill='black',
    ))


def _draw_title_horizontal_lines(g, dwg, X, Y, S) -> None:
    """Горизонтальные линии штампа."""
    # Левый блок (x=0..65): каждые 5 мм
    for oy in range(5, 55, 5):
        thick = oy in (30, 35)
        _line(g, dwg, X(0), Y(oy), X(65), Y(oy), thick)

    # Правый блок: y=15, 40 → S, 120 мм (x=65..185)
    for oy in (15, 40):
        _line(g, dwg, X(65), Y(oy), X(185), Y(oy), True)

    # Правый блок: y=20, 35 → S, 50 мм (x=135..185)
    for oy in (20, 35):
        _line(g, dwg, X(135), Y(oy), X(185), Y(oy), True)


def _draw_title_vertical_lines(g, dwg, X, Y, S) -> None:
    """Вертикальные линии штампа."""
    for ox in (17, 40, 55):
        _line(g, dwg, X(ox), Y(0), X(ox), Y(55), True)
    _line(g, dwg, X(65),  Y(0),  X(65),  Y(55), True)
    _line(g, dwg, X(7),   Y(30), X(7),   Y(55), True)
    _line(g, dwg, X(135), Y(0),  X(135), Y(40), True)
    _line(g, dwg, X(140), Y(20), X(140), Y(35), False)
    _line(g, dwg, X(145), Y(20), X(145), Y(35), False)
    _line(g, dwg, X(150), Y(20), X(150), Y(40), True)
    _line(g, dwg, X(167), Y(20), X(167), Y(40), True)
    _line(g, dwg, X(155), Y(15), X(155), Y(20), True)


def _draw_title_text(g, dwg, X, Y) -> None:
    """Текстовые надписи штампа (ГОСТ 2.104-2006, шрифт 0.9 мм)."""
    LABEL_SIZE = 0.9  # мм — мелкий шрифт для подписей граф

    # Заголовок строк изменений (y=31.5 — центр строки высотой 5 мм)
    for ox, label in [(1, 'Изм.'), (8.5, 'Лист'), (21, 'N докум.'), (42, 'Подп.'), (56, 'Дата')]:
        _text(g, dwg, X(ox), Y(31.5), label, size=LABEL_SIZE)

    # Роли подписантов (каждая строка 5 мм, центр на .5)
    for ox, oy, label in [
        (1, 26.5, 'Разраб.'), (1, 21.5, 'Пров.'), (1, 16.5, 'Т.контр.'),
        (1,  6.5, 'Н.контр.'), (1,  1.5, 'Утв.'),
    ]:
        _text(g, dwg, X(ox), Y(oy), label, size=LABEL_SIZE)

    # Правый блок: подписи граф Лит./Масса/Масштаб (y=36.5) и Лист/Листов (y=16.5)
    for ox, oy, label in [
        (139, 36.5, 'Лит.'), (153, 36.5, 'Масса'), (167, 36.5, 'Масштаб'),
        (139, 16.5, 'Лист'), (157, 16.5, 'Листов'),
    ]:
        _text(g, dwg, X(ox), Y(oy), label, size=LABEL_SIZE)


def _clipped_text(
    g, dwg,
    cx: float, cy: float,
    text: str, sz: float,
    clip_x: float, clip_y: float, clip_w: float, clip_h: float,
    clip_id: str,
) -> None:
    """Разместить текст с ограничением по clipPath ячейки.

    cx, cy — координата ЦЕНТРА ячейки (SVG, мм).
    clip_x, clip_y, clip_w, clip_h — прямоугольник ячейки (SVG, мм).
    """
    cp = ClipPath(id=clip_id, clipPathUnits='userSpaceOnUse')
    cp.add(dwg.rect(insert=(clip_x, clip_y), size=(clip_w, clip_h)))
    dwg.defs.add(cp)

    wrapper = dwg.g(**{'clip-path': f'url(#{clip_id})'})
    y_bl = cy + sz * 0.35
    wrapper.add(dwg.text(
        text, insert=(cx, y_bl),
        font_family='ISOCPEUR, Arial, sans-serif',
        font_style='italic', font_weight='normal',
        font_size=f'{sz}mm', text_anchor='middle',
        fill='black',
    ))
    g.add(wrapper)


def _draw_title_content(g, dwg, X, Y, metadata: dict) -> None:
    """Заполнить крупные ячейки и фамилии штампа данными пользователя.

    Графа 2  (x=65..185, y=40..55, 120×15 мм) — Обозначение документа.
    Графа 1  (x=65..185, y=0..15,  120×15 мм) — Наименование изделия.
    Графа 8  (x=65..135, y=15..40,  70×25 мм) — Наименование организации.
    Фамилии (x=17..40, по 5 мм на строку) — Разраб./Пров./Т.контр./Н.контр./Утв.

    Каждая надпись обёрнута в clipPath по границе своей ячейки.
    Выравнивание: по центру и горизонтально и вертикально.
    """
    doc_designation = (metadata or {}).get('doc_designation', '').strip()
    part_name       = (metadata or {}).get('part_name', '').strip()
    org_name        = (metadata or {}).get('org_name', '').strip()
    surname         = (metadata or {}).get('surname', '').strip()

    # Графа 2: x=65..185, y_stamp=40..55 → SVG y: Y(55)..Y(40)
    if doc_designation:
        _clipped_text(g, dwg,
                      cx=X(125), cy=Y(47.5), text=doc_designation, sz=FONT_SZ_G2,
                      clip_x=X(65), clip_y=Y(55), clip_w=120, clip_h=15,
                      clip_id='title-g2-clip')

    # Графа 1: x=65..185, y_stamp=0..15 → SVG y: Y(15)..Y(0)
    if part_name:
        _clipped_text(g, dwg,
                      cx=X(125), cy=Y(7.5), text=part_name, sz=FONT_SZ_G1,
                      clip_x=X(65), clip_y=Y(15), clip_w=120, clip_h=15,
                      clip_id='title-g1-clip')

    # Графа 8: x=65..135, y_stamp=15..40 → SVG y: Y(40)..Y(15)
    if org_name:
        _clipped_text(g, dwg,
                      cx=X(100), cy=Y(27.5), text=org_name, sz=FONT_SZ_G8,
                      clip_x=X(65), clip_y=Y(40), clip_w=70, clip_h=25,
                      clip_id='title-g8-clip')

    # Фамилии (x=17..40 = 23 мм, каждая строка 5 мм)
    # Строки: Разраб.(25..30), Пров.(20..25), Т.контр.(15..20),
    #         Н.контр.(5..10), Утв.(0..5)  — в координатах штампа снизу.
    if surname:
        # (oy_center_stamp, clip_top_stamp) для каждой роли
        surname_rows = [
            (27.5, 30),   # Разраб.
            (22.5, 25),   # Пров.
            (17.5, 20),   # Т.контр.
            ( 7.5, 10),   # Н.контр.
            ( 2.5,  5),   # Утв.
        ]
        for i, (oy_c, oy_top) in enumerate(surname_rows):
            _clipped_text(g, dwg,
                          cx=X(28.5), cy=Y(oy_c), text=surname, sz=FONT_SZ_SN,
                          clip_x=X(17), clip_y=Y(oy_top), clip_w=23, clip_h=5,
                          clip_id=f'title-sn-{i}-clip')


def _draw_side_stamp(g, dwg, ml, sheet_h, mo, S, font_kw) -> None:
    """Дополнительный штамп у левого корешка (ГОСТ 2.104-2006 dop_shtamp).

    Расположение: в поле сшивки (левое поле 20 мм), правый край штампа
    совпадает с левым краем внутренней рамки.
    dg_x = ml - dg_w = 20 - 12 = 8 мм от левого края листа.

    Каждая надпись обёрнута в clipPath по границам своей ячейки, чтобы
    текст не выходил за пределы при render в браузере.
    """
    dg_w, dg_h = 12.0, 150.0
    frame_bottom = sheet_h - mo

    # Штамп в поле сшивки: правый край = левый край внутренней рамки
    dg_x = ml - dg_w   # = 8 мм от края листа
    dg_y = frame_bottom - dg_h

    side_rect = dwg.rect(insert=(dg_x, dg_y), size=(dg_w, dg_h), fill='white')
    _apply_line_style(side_rect, ESKDLineType.FRAME)
    g.add(side_rect)

    # Границы строк (мм от низа штампа): [0, 25, 59, 84, 112, 150]
    row_boundaries = [0, 25, 59, 84, 112, 150]

    # Горизонтальные разделители (тонкие STAMP_THIN) — между строками
    for rb in row_boundaries[1:-1]:   # 25, 59, 84, 112
        ry = frame_bottom - rb
        h_line = dwg.line(start=(dg_x, ry), end=(dg_x + dg_w, ry))
        _apply_line_style(h_line, ESKDLineType.STAMP_THIN)
        g.add(h_line)

    # Вертикальный разделитель (основная FRAME): левая узкая колонка (5 мм) / правая широкая (7 мм)
    split_x = dg_x + 5.0
    v_line = dwg.line(start=(split_x, dg_y), end=(split_x, frame_bottom))
    _apply_line_style(v_line, ESKDLineType.FRAME)
    g.add(v_line)

    # Надписи в правой колонке, повёрнутые на -90°.
    # tx = центр правой колонки: dg_x + 5(split) + 3.5(half of 7mm) = dg_x + 8.5
    # font 1.4 мм, baseline offset = 1.4 * 0.35 ≈ 0.5 мм → insert=(0, 0.5)
    # Каждая надпись обрезается clipPath по границам своей строки (clip в SVG-координатах).
    labels = [
        (12.5,  'Инв. № подл.'),
        (42.0,  'Подп. и дата'),
        (71.5,  'Взам. инв. №'),
        (98.0,  'Инв. № дубл.'),
        (131.0, 'Подп. и дата'),
    ]

    tx = dg_x + 8.5   # центр правой колонки (5 + 3.5)
    clip_x  = dg_x + 5.0   # левый край правой колонки в SVG (= split_x)
    clip_w  = 7.0           # ширина правой колонки

    for i, (from_bottom, label) in enumerate(labels):
        ty = frame_bottom - from_bottom

        # Границы строки в SVG-координатах (y-ось вниз)
        row_top_y = frame_bottom - float(row_boundaries[i + 1])
        row_h     = float(row_boundaries[i + 1] - row_boundaries[i])

        # ClipPath в userSpaceOnUse — координаты в системе чертежа (мм)
        clip_id = f'side-stamp-clip-{i}'
        cp = ClipPath(id=clip_id, clipPathUnits='userSpaceOnUse')
        cp.add(dwg.rect(insert=(clip_x, row_top_y), size=(clip_w, row_h)))
        dwg.defs.add(cp)

        # Группа-обёртка без трансформации — clip применяется в SVG-пространстве
        wrapper = dwg.g(**{'clip-path': f'url(#{clip_id})'})
        txt = dwg.text(label, insert=(0, 0.5), font_size='1.4mm',
                       text_anchor='middle', **font_kw)
        txt['transform'] = f'translate({tx},{ty}) rotate(-90)'
        wrapper.add(txt)
        g.add(wrapper)


def _draw_ref_block(g, dwg, ml, mo, S, doc_designation: str = "") -> None:
    """Графа 26 — обозначение документа (повёрнутое, верхний левый угол).

    Текст повёрнут −90°, читается снизу вверх (как боковой штамп).
    """
    # g26_long = 72 мм — длина вдоль высоты листа (SVG y-направление)
    # g26_cross = 14 мм — ширина поперёк поля сшивки (SVG x-направление)
    g26_long, g26_cross = 72.0, 14.0
    # Rect: ширина (x) = 14 мм, высота (y) = 72 мм — основная линия FRAME
    ref_rect = dwg.rect(insert=(ml, mo), size=(g26_cross, g26_long), fill='white')
    _apply_line_style(ref_rect, ESKDLineType.FRAME)
    g.add(ref_rect)

    if doc_designation.strip():
        sz = FONT_SZ_G26
        # Центр ячейки в SVG-координатах
        tx = ml + g26_cross / 2   # x = 20 + 7 = 27 (центр 14-мм ширины)
        ty = mo + g26_long / 2    # y = 5 + 36 = 41 (центр 72-мм высоты)

        # ClipPath по границам ячейки (14×72 мм)
        clip_id = 'ref-block-clip'
        cp = ClipPath(id=clip_id, clipPathUnits='userSpaceOnUse')
        cp.add(dwg.rect(insert=(ml, mo), size=(g26_cross, g26_long)))
        dwg.defs.add(cp)

        font_kw = dict(
            font_family='ISOCPEUR, Arial, sans-serif',
            font_style='italic', font_weight='normal', fill='black',
        )
        wrapper = dwg.g(**{'clip-path': f'url(#{clip_id})'})
        txt = dwg.text(doc_designation, insert=(0, sz * 0.35),
                       font_size=f'{sz}mm', text_anchor='middle', **font_kw)
        txt['transform'] = f'translate({tx},{ty}) rotate(-90)'
        wrapper.add(txt)
        g.add(wrapper)


def _draw_footer(g, dwg, sheet_w, sheet_h, mo, S, format_name: str = "") -> None:
    """Строка «Копировал» и «Формат XX» ниже штампа."""
    tb_x0 = sheet_w - mo - TITLE_BLOCK_W
    frame_bottom = sheet_h - mo
    below_y = frame_bottom + 3.5
    fmt_label = f'Формат {format_name}'.strip() if format_name else 'Формат'
    _text(g, dwg, tb_x0 + 60,  below_y, 'Копировал', anchor='middle')
    _text(g, dwg, tb_x0 + 150, below_y, fmt_label,   anchor='middle')
