"""
Строители размерных линий ЕСКД (ГОСТ 2.307-2011).

Реализует паттерн Strategy/Builder: для каждого типа размера свой строитель,
который из DimensionCandidate создаёт DimensionGeometry.

Строители:
  - LinearHorizontalBuilder — горизонтальные линейные размеры
  - LinearVerticalBuilder   — вертикальные линейные размеры
  - DiameterBuilder         — диаметры (Ø)
  - RadiusBuilder           — радиусы (R)

Фабрика:
  - build_dimension()       — создать геометрию через реестр строителей

Будущие строители (фазы 5-6):
  - AngularBuilder          — угловые размеры (°)
  - LinearAlignedBuilder    — наклонные линейные размеры
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import math

from stl_drawing.config import (
    DIM_ARROW_LENGTH,
    DIM_ARROWS_OUTSIDE_THRESHOLD,
    DIM_DOT_RADIUS,
    DIM_DOT_THRESHOLD,
    DIM_EXTENSION_GAP,
    DIM_EXTENSION_OVERSHOOT,
    DIM_LEADER_HORIZONTAL,
    DIM_TEXT_GAP,
    DIM_TICK_LENGTH,
)
from stl_drawing.drawing.dimensions.extractor import DimensionCandidate
from stl_drawing.drawing.dimensions.geometry import (
    ArrowSpec,
    ArrowStyle,
    DimensionGeometry,
    LineSegment,
    TextPlacement,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Форматирование значений размеров
# ---------------------------------------------------------------------------

def _format_dim_value(value_mm: float) -> str:
    """Форматировать числовое значение размера по ГОСТ 2.307-2011.

    Целые числа — без дробной части. Дробные — с одним знаком.

    Args:
        value_mm: значение в мм.

    Returns:
        Строка: "125", "50.5" и т.п.
    """
    if abs(value_mm - round(value_mm)) < 0.05:
        return str(int(round(value_mm)))
    else:
        return f"{value_mm:.1f}"


# ---------------------------------------------------------------------------
# Базовый класс строителя
# ---------------------------------------------------------------------------

class DimensionBuilder(ABC):
    """Абстрактный строитель геометрии размерной линии.

    Каждый конкретный строитель реализует `build()` для своего типа размера.
    Общие методы:
      - _format_value()           — форматирование числового значения
      - _determine_arrow_style()  — выбор стиля стрелок по ГОСТ (фаза 3)
      - _should_arrows_outside()  — нужны ли стрелки наружу (фаза 3)
    """

    @abstractmethod
    def build(
        self,
        candidate: DimensionCandidate,
        scale: float,
        tx: float, ty: float,
        vx: float, vy: float,
        vw: float, vh: float,
        offset: float,
        side: str,
    ) -> Optional[DimensionGeometry]:
        """Построить геометрию размерной линии.

        Args:
            candidate: кандидат размера из extractor.
            scale: масштаб (мм-на-листе / мм-модели).
            tx, ty: смещение вида на листе (view_x + offset_x).
            vx, vy: позиция прямоугольника вида на листе.
            vw, vh: ширина и высота прямоугольника вида.
            offset: отступ от контура до размерной линии (мм).
            side: сторона размещения ('top'|'bottom'|'left'|'right').

        Returns:
            DimensionGeometry или None если построение невозможно.
        """

    def _format_value(self, candidate: DimensionCandidate) -> str:
        """Форматировать значение размера (без префикса).

        Args:
            candidate: кандидат размера.

        Returns:
            Строка числового значения.
        """
        return _format_dim_value(candidate.value_mm)

    def _determine_arrow_style(self, paper_span: float) -> ArrowStyle:
        """Определить стиль стрелок по расстоянию между выносными.

        ГОСТ 2.307-2011 п.5.2:
          - >= 10 мм: стандартные заполненные стрелки
          - 2..10 мм: засечки (TICK) при малом пространстве
          - < 2 мм: точки (DOT)

        Args:
            paper_span: расстояние между точками привязки на бумаге (мм).

        Returns:
            ArrowStyle.
        """
        if paper_span < DIM_DOT_THRESHOLD:
            return ArrowStyle.DOT
        if paper_span < DIM_ARROWS_OUTSIDE_THRESHOLD:
            return ArrowStyle.TICK
        return ArrowStyle.FILLED

    def _should_arrows_outside(self, paper_span: float) -> bool:
        """Нужно ли размещать стрелки снаружи выносных линий.

        ГОСТ 2.307-2011 п.5.2: при расстоянии меньше порога стрелки
        (или засечки) рисуются снаружи размерной линии, а сама линия
        продлевается за выносные.

        Args:
            paper_span: расстояние между точками привязки на бумаге (мм).

        Returns:
            True если стрелки нужно разместить снаружи.
        """
        return paper_span < DIM_ARROWS_OUTSIDE_THRESHOLD


# ---------------------------------------------------------------------------
# Горизонтальный линейный размер
# ---------------------------------------------------------------------------

class LinearHorizontalBuilder(DimensionBuilder):
    """Строитель горизонтальных линейных размеров.

    ГОСТ 2.307-2011:
      - 2 вертикальные выносные линии
      - Горизонтальная размерная линия
      - 2 стрелки (← →), адаптивный стиль по расстоянию
      - Текст над размерной линией, angle=0°
    """

    def build(
        self,
        candidate: DimensionCandidate,
        scale: float,
        tx: float, ty: float,
        vx: float, vy: float,
        vw: float, vh: float,
        offset: float,
        side: str,
    ) -> DimensionGeometry:
        # Якоря в координатах листа
        ax1 = candidate.anchor_a[0] * scale + tx
        ax2 = candidate.anchor_b[0] * scale + tx

        # Обеспечить ax1 < ax2
        if ax1 > ax2:
            ax1, ax2 = ax2, ax1

        paper_span = ax2 - ax1

        if side == 'bottom':
            dim_y = vy + vh + offset
            ext_start_y = vy + vh + DIM_EXTENSION_GAP
            ext_end_y = dim_y + DIM_EXTENSION_OVERSHOOT
        else:  # top
            dim_y = vy - offset
            ext_start_y = vy - DIM_EXTENSION_GAP
            ext_end_y = dim_y - DIM_EXTENSION_OVERSHOOT

        text_value = self._format_value(candidate)

        # Адаптивные стрелки (ГОСТ 2.307-2011 п.5.2)
        arrow_style = self._determine_arrow_style(paper_span)
        arrows_outside = self._should_arrows_outside(paper_span)

        # Размерная линия
        dim_start_x = ax1
        dim_end_x = ax2
        if arrows_outside:
            # Продлить размерную линию за выносные для внешних стрелок
            ext_len = DIM_ARROW_LENGTH + 2.0
            dim_start_x = ax1 - ext_len
            dim_end_x = ax2 + ext_len

        # Стрелки: при внешнем расположении — развёрнуты наружу
        if arrows_outside:
            arrows = [
                ArrowSpec(position=(ax1, dim_y), angle_deg=180.0,
                          style=arrow_style),  # ← (наружу влево)
                ArrowSpec(position=(ax2, dim_y), angle_deg=0.0,
                          style=arrow_style),  # → (наружу вправо)
            ]
        else:
            arrows = [
                ArrowSpec(position=(ax1, dim_y), angle_deg=0.0,
                          style=arrow_style),  # → (внутрь вправо)
                ArrowSpec(position=(ax2, dim_y), angle_deg=180.0,
                          style=arrow_style),  # ← (внутрь влево)
            ]

        # Текст: при DOT-стрелках — на выноске справа; иначе — между/над
        if arrow_style == ArrowStyle.DOT:
            text_x = ax2 + DIM_ARROW_LENGTH + DIM_LEADER_HORIZONTAL
            text_y = dim_y - DIM_TEXT_GAP
            text_placement = TextPlacement.ON_LEADER
            # Лидер-линия от правой выносной до текста
            leader_lines = [
                LineSegment(
                    start=(ax2, dim_y),
                    end=(text_x - 2.0, dim_y),
                ),
            ]
        else:
            text_x = (ax1 + ax2) / 2.0
            text_y = dim_y - DIM_TEXT_GAP
            text_placement = TextPlacement.BETWEEN
            leader_lines = []

        return DimensionGeometry(
            dim_type=candidate.dim_type,
            text_value=text_value,
            extension_lines=[
                LineSegment(start=(ax1, ext_start_y), end=(ax1, ext_end_y)),
                LineSegment(start=(ax2, ext_start_y), end=(ax2, ext_end_y)),
            ],
            dim_line=LineSegment(start=(dim_start_x, dim_y), end=(dim_end_x, dim_y)),
            leader_lines=leader_lines,
            arrows=arrows,
            text_pos=(text_x, text_y),
            text_angle=0.0,
            text_placement=text_placement,
        )


# ---------------------------------------------------------------------------
# Вертикальный линейный размер
# ---------------------------------------------------------------------------

class LinearVerticalBuilder(DimensionBuilder):
    """Строитель вертикальных линейных размеров.

    ГОСТ 2.307-2011:
      - 2 горизонтальные выносные линии
      - Вертикальная размерная линия
      - 2 стрелки (↓ ↑), адаптивный стиль по расстоянию
      - Текст сбоку, angle=270° (left) или 90° (right)
    """

    def build(
        self,
        candidate: DimensionCandidate,
        scale: float,
        tx: float, ty: float,
        vx: float, vy: float,
        vw: float, vh: float,
        offset: float,
        side: str,
    ) -> DimensionGeometry:
        # Якоря в координатах листа
        ay1 = candidate.anchor_a[1] * scale + ty
        ay2 = candidate.anchor_b[1] * scale + ty

        if ay1 > ay2:
            ay1, ay2 = ay2, ay1

        paper_span = ay2 - ay1

        if side == 'left':
            dim_x = vx - offset
            ext_start_x = vx - DIM_EXTENSION_GAP
            ext_end_x = dim_x - DIM_EXTENSION_OVERSHOOT
        else:  # right
            dim_x = vx + vw + offset
            ext_start_x = vx + vw + DIM_EXTENSION_GAP
            ext_end_x = dim_x + DIM_EXTENSION_OVERSHOOT

        text_value = self._format_value(candidate)
        text_angle = 270.0 if side == 'left' else 90.0

        # Адаптивные стрелки (ГОСТ 2.307-2011 п.5.2)
        arrow_style = self._determine_arrow_style(paper_span)
        arrows_outside = self._should_arrows_outside(paper_span)

        # Размерная линия
        dim_start_y = ay1
        dim_end_y = ay2
        if arrows_outside:
            ext_len = DIM_ARROW_LENGTH + 2.0
            dim_start_y = ay1 - ext_len
            dim_end_y = ay2 + ext_len

        # Стрелки: при внешнем расположении — развёрнуты наружу
        if arrows_outside:
            arrows = [
                ArrowSpec(position=(dim_x, ay1), angle_deg=270.0,
                          style=arrow_style),  # ↑ (наружу вверх)
                ArrowSpec(position=(dim_x, ay2), angle_deg=90.0,
                          style=arrow_style),  # ↓ (наружу вниз)
            ]
        else:
            arrows = [
                ArrowSpec(position=(dim_x, ay1), angle_deg=90.0,
                          style=arrow_style),  # ↓ (внутрь вниз)
                ArrowSpec(position=(dim_x, ay2), angle_deg=270.0,
                          style=arrow_style),  # ↑ (внутрь вверх)
            ]

        # Текст: при DOT — на выноске; иначе — между
        if arrow_style == ArrowStyle.DOT:
            # Выноска горизонтальная от верхней точки
            if side == 'left':
                text_x = dim_x - DIM_LEADER_HORIZONTAL - 2.0
            else:
                text_x = dim_x + DIM_LEADER_HORIZONTAL + 2.0
            text_y = ay1
            text_placement = TextPlacement.ON_LEADER
            leader_lines = [
                LineSegment(
                    start=(dim_x, ay1),
                    end=(text_x + (2.0 if side == 'left' else -2.0), ay1),
                ),
            ]
        else:
            text_x = dim_x - DIM_TEXT_GAP if side == 'left' else dim_x + DIM_TEXT_GAP
            text_y = (ay1 + ay2) / 2.0
            text_placement = TextPlacement.BETWEEN
            leader_lines = []

        return DimensionGeometry(
            dim_type=candidate.dim_type,
            text_value=text_value,
            extension_lines=[
                LineSegment(start=(ext_start_x, ay1), end=(ext_end_x, ay1)),
                LineSegment(start=(ext_start_x, ay2), end=(ext_end_x, ay2)),
            ],
            dim_line=LineSegment(start=(dim_x, dim_start_y), end=(dim_x, dim_end_y)),
            leader_lines=leader_lines,
            arrows=arrows,
            text_pos=(text_x, text_y),
            text_angle=text_angle,
            text_placement=text_placement,
        )


# ---------------------------------------------------------------------------
# Диаметр (Ø)
# ---------------------------------------------------------------------------

class DiameterBuilder(DimensionBuilder):
    """Строитель размеров диаметра.

    ГОСТ 2.307-2011:
      - Без выносных линий
      - Размерная линия через центр окружности + лидер-выноска
      - 2 стрелки на окружности, адаптивный стиль
      - Текст "Ø{значение}" на выноске
    """

    def build(
        self,
        candidate: DimensionCandidate,
        scale: float,
        tx: float, ty: float,
        vx: float, vy: float,
        vw: float, vh: float,
        offset: float,
        side: str,
    ) -> Optional[DimensionGeometry]:
        if candidate.center is None or candidate.radius is None:
            return None

        cx = candidate.center[0] * scale + tx
        cy = candidate.center[1] * scale + ty
        r_paper = candidate.radius * scale
        diam_paper = 2.0 * r_paper

        # Адаптивные стрелки
        arrow_style = self._determine_arrow_style(diam_paper)

        # Размерная линия через центр, горизонтально
        dim_start_x = cx - r_paper
        dim_end_x = cx + r_paper

        # Текст справа от окружности с выноской
        text_x = dim_end_x + DIM_ARROW_LENGTH + DIM_TEXT_GAP + 5.0
        text_y = cy - DIM_TEXT_GAP

        text_value = f"\u00d8{self._format_value(candidate)}"

        # Лидер-линия от окружности вправо до текста
        leader_end_x = text_x - 3.0

        # Размерная линия = от левого края окружности до конца лидера
        dim_line = LineSegment(
            start=(dim_start_x, cy),
            end=(leader_end_x, cy),
        )

        return DimensionGeometry(
            dim_type='diameter',
            text_value=text_value,
            extension_lines=[],
            dim_line=dim_line,
            leader_lines=[],
            arrows=[
                ArrowSpec(position=(dim_start_x, cy), angle_deg=180.0,
                          style=arrow_style),  # ←
                ArrowSpec(position=(dim_end_x, cy), angle_deg=0.0,
                          style=arrow_style),  # →
            ],
            text_pos=(text_x, text_y),
            text_angle=0.0,
            text_placement=TextPlacement.ON_LEADER,
        )


# ---------------------------------------------------------------------------
# Радиус (R)
# ---------------------------------------------------------------------------

class RadiusBuilder(DimensionBuilder):
    """Строитель размеров радиуса.

    ГОСТ 2.307-2011:
      - Без выносных линий
      - Размерная линия от центра окружности до дуги + лидер-выноска
      - 1 стрелка на дуге (направлена от центра наружу)
      - Текст "R{значение}" на выноске
    """

    def build(
        self,
        candidate: DimensionCandidate,
        scale: float,
        tx: float, ty: float,
        vx: float, vy: float,
        vw: float, vh: float,
        offset: float,
        side: str,
    ) -> Optional[DimensionGeometry]:
        if candidate.center is None or candidate.radius is None:
            return None

        cx = candidate.center[0] * scale + tx
        cy = candidate.center[1] * scale + ty
        r_paper = candidate.radius * scale

        # Адаптивные стрелки (по диаметру, т.к. span = полный диаметр)
        arrow_style = self._determine_arrow_style(2.0 * r_paper)

        # Текст: "R{значение}" — value_mm уже содержит радиус
        text_value = f"R{self._format_value(candidate)}"

        # Горизонтальная размерная линия от центра до правого края окружности
        edge_x = cx + r_paper

        # Текст справа от окружности с выноской (аналогично DiameterBuilder)
        text_x = edge_x + DIM_ARROW_LENGTH + DIM_TEXT_GAP + 5.0
        text_y = cy - DIM_TEXT_GAP

        # Лидер-линия от окружности вправо до текста
        leader_end_x = text_x - 3.0

        # Размерная линия: от центра до конца лидера
        dim_line = LineSegment(
            start=(cx, cy),
            end=(leader_end_x, cy),
        )

        # Одна стрелка на правом краю окружности, направлена вправо (от центра)
        arrow = ArrowSpec(
            position=(edge_x, cy),
            angle_deg=0.0,  # → от центра наружу
            style=arrow_style,
        )

        return DimensionGeometry(
            dim_type='radius',
            text_value=text_value,
            extension_lines=[],
            dim_line=dim_line,
            leader_lines=[],
            arrows=[arrow],
            text_pos=(text_x, text_y),
            text_angle=0.0,
            text_placement=TextPlacement.ON_LEADER,
        )


# ---------------------------------------------------------------------------
# Реестр строителей
# ---------------------------------------------------------------------------

_REGISTRY = {
    'linear_horizontal': LinearHorizontalBuilder(),
    'linear_vertical': LinearVerticalBuilder(),
    'diameter': DiameterBuilder(),
    'radius': RadiusBuilder(),
}


def build_dimension(
    candidate: DimensionCandidate,
    scale: float,
    tx: float, ty: float,
    vx: float, vy: float,
    vw: float, vh: float,
    offset: float,
    side: str,
) -> Optional[DimensionGeometry]:
    """Построить геометрию размера через реестр строителей.

    Фабричная функция: определяет тип строителя по candidate.dim_type
    и вызывает его build().

    Args:
        candidate: кандидат размера из extractor.
        scale: масштаб чертежа.
        tx, ty: смещение вида на листе.
        vx, vy: позиция вида.
        vw, vh: ширина/высота вида.
        offset: отступ от контура до размерной линии (мм).
        side: сторона размещения.

    Returns:
        DimensionGeometry или None если тип не поддерживается.
    """
    builder = _REGISTRY.get(candidate.dim_type)
    if builder is None:
        logger.debug(
            "Неизвестный тип размера '%s' — нет строителя в реестре",
            candidate.dim_type,
        )
        return None
    return builder.build(candidate, scale, tx, ty, vx, vy, vw, vh, offset, side)
