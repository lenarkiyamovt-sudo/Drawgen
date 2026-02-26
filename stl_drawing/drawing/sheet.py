"""
ESKDDrawingSheet — главный класс генерации ЕСКД-чертежа.

Хранит данные видов и координирует:
  - выбор минимального набора видов (view_selector)
  - выбор формата и масштаба (layout)
  - компоновку на листе (layout)
  - SVG-рендеринг (svg_renderer, title_block)

Не содержит геометрической логики — только оркестрация.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import svgwrite

from stl_drawing.config import (
    MARGIN_LEFT,
    MARGIN_OTHER,
    S_THICK_DRAWING,
    S_THIN_DRAWING,
    TITLE_BLOCK_H,
)
from stl_drawing.drawing.gost_params import calculate_line_parameters
from stl_drawing.drawing.layout import (
    arrange_views,
    front_view_size_mm,
    select_format_and_scale,
)
from stl_drawing.drawing.svg_renderer import render_view_lines
from stl_drawing.drawing.title_block import (
    add_additional_stamps,
    add_frame,
    add_title_block,
)
from stl_drawing.drawing.view_selector import select_necessary_views

logger = logging.getLogger(__name__)


class ESKDDrawingSheet:
    """Генератор ЕСКД-чертежа с автоматическим выбором формата и масштаба.

    Использование:
        sheet = ESKDDrawingSheet()
        sheet.add_view_data('front', projected, visible_lines, hidden_lines)
        ...
        sheet.generate_drawing('output.svg')
    """

    def __init__(self) -> None:
        self.views_data: Dict[str, Dict] = {}
        self.scale: float = 1.0
        self.sheet_w: float = 0.0
        self.sheet_h: float = 0.0
        self.format_name: Optional[str] = None
        self.metadata: dict = {}

    def set_metadata(
        self,
        doc_designation: str = "",
        part_name: str = "",
        org_name: str = "",
        surname: str = "",
    ) -> None:
        """Задать метаданные для заполнения штампа.

        Args:
            doc_designation: обозначение документа (Графа 2, 26).
            part_name: наименование изделия (Графа 1).
            org_name: наименование организации (Графа 8).
            surname: фамилия подписанта (Разраб./Пров./Т.контр./Н.контр./Утв.).
        """
        self.metadata = {
            'doc_designation': doc_designation,
            'part_name':       part_name,
            'org_name':        org_name,
            'surname':         surname,
        }

    # ------------------------------------------------------------------
    # Загрузка данных видов
    # ------------------------------------------------------------------

    def add_view_data(
        self,
        view_name: str,
        projected_vertices: np.ndarray,
        visible_lines: List[Tuple],
        hidden_lines: List[Tuple],
    ) -> None:
        """Добавить данные одного вида.

        Args:
            view_name: имя вида ('front', 'top', и т.д.).
            projected_vertices: проекция вершин (N, 3) — XY + Z-глубина.
            visible_lines: список (pA, pB) видимых отрезков.
            hidden_lines:  список (pA, pB) скрытых отрезков.
        """
        bbox = self._compute_bbox(projected_vertices)
        self.views_data[view_name] = {
            'coords':  projected_vertices,
            'visible': visible_lines,
            'hidden':  hidden_lines,
            'bbox':    bbox,
        }

    # ------------------------------------------------------------------
    # Генерация SVG
    # ------------------------------------------------------------------

    def generate_drawing(self, filename: str = "unified_eskd_drawing.svg") -> str:
        """Сгенерировать SVG-чертёж ЕСКД.

        Шаги:
          1. Выбор минимального набора видов (ГОСТ 2.305-2008).
          2. Выбор формата листа и масштаба (ГОСТ 2.301/2.302-68).
          3. Компоновка видов на листе.
          4. Расчёт параметров линий (ГОСТ 2.303-68).
          5. Создание SVG и рендеринг всех элементов.

        Args:
            filename: путь для сохранения SVG.

        Returns:
            Путь к сохранённому файлу.

        Raises:
            ValueError: если нет данных видов.
        """
        if not self.views_data:
            raise ValueError("Нет данных видов для генерации чертежа.")

        # 1. Минимальный набор видов
        selected, excluded = select_necessary_views(self.views_data)
        active_views = {v: self.views_data[v] for v in selected if v in self.views_data}
        logger.info("Используется %d видов: %s", len(active_views), selected)

        # 2. Формат и масштаб
        self.format_name, self.scale, self.sheet_w, self.sheet_h = \
            select_format_and_scale(active_views)

        # 3. Компоновка
        layout = arrange_views(active_views, self.scale, self.sheet_w, self.sheet_h)

        # 4. Параметры линий
        fv_mm = front_view_size_mm(active_views, self.scale)
        stroke_width = S_THICK_DRAWING if fv_mm >= 80 else S_THIN_DRAWING
        eskd_styles = calculate_line_parameters(fv_mm, stroke_width)
        logger.info(
            "Параметры линий: S=%.1f мм, штрих=%.0f мм, пробел=%.1f мм",
            stroke_width,
            eskd_styles['_params']['dash_length'],
            eskd_styles['_params']['gap_length'],
        )

        # 5. SVG
        dwg = svgwrite.Drawing(
            filename,
            size=(f"{self.sheet_w}mm", f"{self.sheet_h}mm"),
            viewBox=f"0 0 {self.sheet_w} {self.sheet_h}",
            debug=False,
        )

        add_frame(dwg, self.sheet_w, self.sheet_h)

        views_group = dwg.g()
        for view_name, view_layout in layout.items():
            view_data = active_views[view_name]
            translate_x = view_layout['x'] + view_layout['offset_x']
            translate_y = view_layout['y'] + view_layout['offset_y']

            view_group = render_view_lines(
                dwg,
                view_data['visible'],
                view_data['hidden'],
                self.scale,
                translate_x,
                translate_y,
                eskd_styles,
            )
            views_group.add(view_group)

        dwg.add(views_group)
        add_title_block(dwg, self.sheet_w, self.sheet_h, self.scale, self.metadata)
        add_additional_stamps(dwg, self.sheet_w, self.sheet_h, self.format_name or "", self.metadata)

        dwg.save()
        logger.info("ЕСКД-чертёж сохранён: %s", filename)
        return filename

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bbox(projected_vertices: np.ndarray) -> Dict:
        """Вычислить ограничивающий прямоугольник проекции."""
        min_x = float(np.min(projected_vertices[:, 0]))
        max_x = float(np.max(projected_vertices[:, 0]))
        min_y = float(np.min(projected_vertices[:, 1]))
        max_y = float(np.max(projected_vertices[:, 1]))
        return {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'width':  max_x - min_x,
            'height': max_y - min_y,
        }
