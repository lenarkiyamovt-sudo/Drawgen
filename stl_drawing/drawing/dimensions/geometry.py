"""
Универсальная геометрия размерных линий ЕСКД (ГОСТ 2.307-2011).

Определяет структуры данных для описания любого типа размера:
линейного, диаметрального, радиусного, углового и др.

Типы:
  - LineSegment     — отрезок (start, end)
  - ArcSpec         — дуга (center, radius, start_angle, end_angle)
  - ArrowStyle      — стиль стрелки (FILLED, TICK, DOT, NONE)
  - ArrowSpec       — стрелка (position, angle, style)
  - TextPlacement   — расположение текста (BETWEEN, OUTSIDE_*, ON_LEADER)
  - DimensionGeometry — полная геометрия одного размера
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from stl_drawing.config import DIM_TEXT_HEIGHT


# ---------------------------------------------------------------------------
# Перечисления
# ---------------------------------------------------------------------------

class ArrowStyle(Enum):
    """Стиль стрелки размерной линии.

    ГОСТ 2.307-2011 п.5.2:
      - FILLED: стандартная стрелка (заполненный треугольник)
      - TICK: засечка 45° (при малом расстоянии между выносными)
      - DOT: точка (при очень малом расстоянии)
      - NONE: без стрелки (стыки цепных размеров)
    """
    FILLED = "filled"
    TICK = "tick"
    DOT = "dot"
    NONE = "none"


class TextPlacement(Enum):
    """Расположение текста размера относительно размерной линии.

    ГОСТ 2.307-2011 п.5.3:
      - BETWEEN: между стрелками (стандартное)
      - OUTSIDE_RIGHT: справа от размерной линии
      - OUTSIDE_LEFT: слева от размерной линии
      - ON_LEADER: на линии-выноске
    """
    BETWEEN = "between"
    OUTSIDE_RIGHT = "outside_right"
    OUTSIDE_LEFT = "outside_left"
    ON_LEADER = "on_leader"


# ---------------------------------------------------------------------------
# Геометрические примитивы
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineSegment:
    """Отрезок в координатах листа (мм).

    Attributes:
        start: начальная точка (x, y).
        end: конечная точка (x, y).
    """
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass(frozen=True)
class ArcSpec:
    """Дуга размерной линии (для угловых размеров).

    Attributes:
        center: центр дуги (мм на листе).
        radius: радиус дуги (мм).
        start_angle_deg: начальный угол (градусы, от оси X, против часовой).
        end_angle_deg: конечный угол (градусы).
    """
    center: Tuple[float, float]
    radius: float
    start_angle_deg: float
    end_angle_deg: float


@dataclass(frozen=True)
class ArrowSpec:
    """Спецификация стрелки размерной линии.

    Attributes:
        position: вершина стрелки (мм на листе).
        angle_deg: угол поворота (градусы, 0 = вправо →).
        style: стиль стрелки (FILLED, TICK, DOT, NONE).
    """
    position: Tuple[float, float]
    angle_deg: float
    style: ArrowStyle = ArrowStyle.FILLED


# ---------------------------------------------------------------------------
# Главная структура: DimensionGeometry
# ---------------------------------------------------------------------------

@dataclass
class DimensionGeometry:
    """Полная геометрия одного размера.

    Универсальная структура, описывающая все элементы размерной линии
    для любого типа размера (линейный, диаметр, радиус, угловой).

    Attributes:
        dim_type: тип размера ('linear_horizontal', 'diameter', и др.)
        text_value: текст размера ("125", "\u00d820", "R10", "45\u00b0")
        extension_lines: выносные линии (0, 1 или 2 шт.)
        dim_line: прямая размерная линия (линейные, диаметр, радиус)
        dim_arc: дуга размерной линии (только угловые размеры)
        leader_lines: линии-выноски (для текста вне контура)
        arrows: стрелки (1 для радиуса, 2 для остальных)
        text_pos: позиция текста (мм на листе)
        text_angle: угол поворота текста (градусы)
        text_placement: способ размещения текста
    """
    dim_type: str
    text_value: str
    extension_lines: List[LineSegment] = field(default_factory=list)
    dim_line: Optional[LineSegment] = None
    dim_arc: Optional[ArcSpec] = None
    leader_lines: List[LineSegment] = field(default_factory=list)
    arrows: List[ArrowSpec] = field(default_factory=list)
    text_pos: Tuple[float, float] = (0.0, 0.0)
    text_angle: float = 0.0
    text_placement: TextPlacement = TextPlacement.BETWEEN

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Вычислить ограничивающий прямоугольник (AABB) в мм на листе.

        Учитывает все геометрические элементы: размерную линию/дугу,
        выносные линии, линии-выноски, стрелки и текст.

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        xs: List[float] = []
        ys: List[float] = []

        # Размерная линия
        if self.dim_line is not None:
            xs.extend([self.dim_line.start[0], self.dim_line.end[0]])
            ys.extend([self.dim_line.start[1], self.dim_line.end[1]])

        # Дуга размерной линии (приближение по bbox дуги)
        if self.dim_arc is not None:
            arc = self.dim_arc
            cx, cy = arc.center
            r = arc.radius
            xs.extend([cx - r, cx + r])
            ys.extend([cy - r, cy + r])

        # Выносные линии
        for ext in self.extension_lines:
            xs.extend([ext.start[0], ext.end[0]])
            ys.extend([ext.start[1], ext.end[1]])

        # Линии-выноски
        for ll in self.leader_lines:
            xs.extend([ll.start[0], ll.end[0]])
            ys.extend([ll.start[1], ll.end[1]])

        # Стрелки
        for arrow in self.arrows:
            xs.append(arrow.position[0])
            ys.append(arrow.position[1])

        # Текст + оценка размеров надписи
        tx, ty = self.text_pos
        xs.append(tx)
        ys.append(ty)

        text_half_w = len(self.text_value) * DIM_TEXT_HEIGHT * 0.35
        text_h = DIM_TEXT_HEIGHT
        xs.extend([tx - text_half_w, tx + text_half_w])
        ys.extend([ty - text_h, ty + text_h * 0.5])

        if not xs or not ys:
            return (0.0, 0.0, 0.0, 0.0)

        return (min(xs), min(ys), max(xs), max(ys))

    # ------------------------------------------------------------------
    # Обратная совместимость
    # ------------------------------------------------------------------

    def to_placed_dimension(self):
        """Преобразовать в PlacedDimension для обратной совместимости.

        Маппинг:
          - extension_lines[0] \u2192 ext_line_a_*
          - extension_lines[1] \u2192 ext_line_b_*
          - dim_line \u2192 dim_line_start/end
          - arrows[0] \u2192 arrow_a_*
          - arrows[1] \u2192 arrow_b_*

        Для типов без выносных (diameter) подставляется точка-заглушка
        (центр между стрелками), как в текущем renderer.

        Returns:
            PlacedDimension с координатами в мм на листе.
        """
        # Отложенный импорт (избежание циклической зависимости)
        from stl_drawing.drawing.dimensions.renderer import PlacedDimension

        # --- Выносные линии ---
        if len(self.extension_lines) >= 2:
            ext_a_start = self.extension_lines[0].start
            ext_a_end = self.extension_lines[0].end
            ext_b_start = self.extension_lines[1].start
            ext_b_end = self.extension_lines[1].end
        elif len(self.extension_lines) == 1:
            ext_a_start = self.extension_lines[0].start
            ext_a_end = self.extension_lines[0].end
            ext_b_start = self.extension_lines[0].start
            ext_b_end = self.extension_lines[0].end
        else:
            # Нет выносных (диаметр, радиус) — заглушка в центре
            dummy = self._compute_center_dummy()
            ext_a_start = dummy
            ext_a_end = dummy
            ext_b_start = dummy
            ext_b_end = dummy

        # --- Размерная линия ---
        if self.dim_line is not None:
            dim_start = self.dim_line.start
            dim_end = self.dim_line.end
        else:
            dim_start = self.text_pos
            dim_end = self.text_pos

        # --- Стрелки ---
        if len(self.arrows) >= 2:
            arrow_a_pos = self.arrows[0].position
            arrow_a_angle = self.arrows[0].angle_deg
            arrow_b_pos = self.arrows[1].position
            arrow_b_angle = self.arrows[1].angle_deg
        elif len(self.arrows) == 1:
            arrow_a_pos = self.arrows[0].position
            arrow_a_angle = self.arrows[0].angle_deg
            arrow_b_pos = self.arrows[0].position
            arrow_b_angle = self.arrows[0].angle_deg
        else:
            arrow_a_pos = dim_start
            arrow_a_angle = 0.0
            arrow_b_pos = dim_end
            arrow_b_angle = 180.0

        # Arrow styles: для типов с 1 стрелкой (radius) второй стрелки нет
        arrow_a_style = self.arrows[0].style.value if len(self.arrows) >= 1 else 'none'
        arrow_b_style = self.arrows[1].style.value if len(self.arrows) >= 2 else 'none'

        # Leader lines → list of (start, end) tuples
        leader_tuples = [(ll.start, ll.end) for ll in self.leader_lines]

        return PlacedDimension(
            dim_type=self.dim_type,
            text_value=self.text_value,
            text_angle=self.text_angle,
            ext_line_a_start=ext_a_start,
            ext_line_a_end=ext_a_end,
            ext_line_b_start=ext_b_start,
            ext_line_b_end=ext_b_end,
            dim_line_start=dim_start,
            dim_line_end=dim_end,
            arrow_a_pos=arrow_a_pos,
            arrow_a_angle=arrow_a_angle,
            arrow_b_pos=arrow_b_pos,
            arrow_b_angle=arrow_b_angle,
            text_pos=self.text_pos,
            arrow_a_style=arrow_a_style,
            arrow_b_style=arrow_b_style,
            leader_lines=leader_tuples,
        )

    def _compute_center_dummy(self) -> Tuple[float, float]:
        """Вычислить точку-заглушку для типов без выносных линий.

        Использует среднюю точку между стрелками (центр окружности
        для диаметра), или среднюю точку размерной линии.
        """
        if len(self.arrows) >= 2:
            cx = (self.arrows[0].position[0] + self.arrows[1].position[0]) / 2
            cy = (self.arrows[0].position[1] + self.arrows[1].position[1]) / 2
            return (cx, cy)
        if self.dim_line is not None:
            mx = (self.dim_line.start[0] + self.dim_line.end[0]) / 2
            my = (self.dim_line.start[1] + self.dim_line.end[1]) / 2
            return (mx, my)
        return self.text_pos
