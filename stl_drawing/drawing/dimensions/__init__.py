"""
Пакет автоматического оразмеривания ЕСКД-чертежей (ГОСТ 2.307-2011).

Модули:
  - extractor:  извлечение размеров из данных видов
  - dedup:      дедупликация размеров между видами
  - placer:     размещение размеров на листе
  - renderer:   SVG-рендеринг размерных элементов
  - geometry:   универсальные структуры геометрии размера
  - builders:   строители размерных линий (стратегия/фабрика)
"""

from stl_drawing.drawing.dimensions.builders import build_dimension
from stl_drawing.drawing.dimensions.dedup import deduplicate_dimensions
from stl_drawing.drawing.dimensions.extractor import (
    DimensionCandidate,
    extract_dimensions,
)
from stl_drawing.drawing.dimensions.geometry import (
    ArrowStyle,
    DimensionGeometry,
    TextPlacement,
)
from stl_drawing.drawing.dimensions.placer import place_dimensions
from stl_drawing.drawing.dimensions.renderer import (
    PlacedDimension,
    render_dimensions,
)

__all__ = [
    'extract_dimensions',
    'deduplicate_dimensions',
    'place_dimensions',
    'render_dimensions',
    'build_dimension',
    'DimensionCandidate',
    'PlacedDimension',
    'DimensionGeometry',
    'ArrowStyle',
    'TextPlacement',
]
