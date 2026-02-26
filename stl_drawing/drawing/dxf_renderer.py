"""
DXF output renderer for ESKD drawings.

Generates DXF files compatible with AutoCAD and other CAD systems.
Uses ezdxf library for DXF creation.

GOST layer naming convention:
- CONTOUR - visible edges (основные линии)
- HIDDEN - hidden edges (штриховые линии)
- CENTER - centerlines (штрих-пунктирные)
- DIMENSION - dimensions and text
- FRAME - drawing frame and title block

Usage:
    from stl_drawing.drawing.dxf_renderer import DxfRenderer

    renderer = DxfRenderer()
    renderer.create_drawing(width_mm=841, height_mm=594)
    renderer.add_line((0, 0), (100, 100), layer='CONTOUR')
    renderer.save('output.dxf')
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment

from stl_drawing.config import ESKDLineType

logger = logging.getLogger(__name__)

# GOST line types (approximate patterns in DXF units)
GOST_LINETYPES = {
    'CONTINUOUS': '',  # Solid line
    'DASHED': 'DASHED',  # Hidden lines (штриховая)
    'DASHDOT': 'DASHDOT',  # Centerlines (штрих-пунктирная)
    'DASHDOT2': 'DASHDOT2',  # Thin centerline
}

# Layer definitions with GOST colors
GOST_LAYERS = {
    'CONTOUR': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 50},  # White/Black, 0.5mm
    'HIDDEN': {'color': 8, 'linetype': 'DASHED', 'lineweight': 25},  # Gray, 0.25mm
    'CENTER': {'color': 1, 'linetype': 'DASHDOT', 'lineweight': 18},  # Red, 0.18mm
    'DIMENSION': {'color': 3, 'linetype': 'CONTINUOUS', 'lineweight': 18},  # Green, 0.18mm
    'FRAME': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 70},  # White/Black, 0.7mm
    'FRAME_THIN': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 18},  # Thin frame
    'STAMP': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 25},  # Stamp lines
    'TEXT': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 25},  # White/Black
    'HATCH': {'color': 8, 'linetype': 'CONTINUOUS', 'lineweight': 13},  # Gray, 0.13mm
    'WAVELINE': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 25},  # Wavy break lines
    'SECTION': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 70},  # Section lines
}

# Mapping ESKDLineType to DXF layers
ESKD_TO_DXF_LAYER: Dict[ESKDLineType, str] = {
    ESKDLineType.SOLID: 'CONTOUR',
    ESKDLineType.THIN: 'DIMENSION',
    ESKDLineType.DASHED: 'HIDDEN',
    ESKDLineType.CENTER: 'CENTER',
    ESKDLineType.DOTTED: 'CENTER',
    ESKDLineType.WAVELINE: 'WAVELINE',
    ESKDLineType.SECTION: 'SECTION',
    ESKDLineType.FRAME: 'FRAME',
    ESKDLineType.FRAME_THIN: 'FRAME_THIN',
    ESKDLineType.STAMP_THIN: 'STAMP',
}


def eskd_type_to_dxf_style(eskd_type: ESKDLineType) -> 'DxfStyle':
    """Convert ESKDLineType to DxfStyle.

    Args:
        eskd_type: ESKD line type

    Returns:
        DxfStyle with appropriate layer and lineweight
    """
    layer = ESKD_TO_DXF_LAYER.get(eskd_type, 'CONTOUR')
    # Lineweight in 0.01mm units (ESKDLineType.stroke_width is in mm)
    lineweight = int(eskd_type.stroke_width * 100)
    return DxfStyle(layer=layer, lineweight=lineweight)


@dataclass
class DxfStyle:
    """Style parameters for DXF entities."""
    layer: str = 'CONTOUR'
    color: Optional[int] = None  # None = ByLayer
    lineweight: Optional[int] = None  # None = ByLayer (in 0.01mm units)
    linetype: Optional[str] = None  # None = ByLayer


class DxfRenderer:
    """DXF drawing renderer using ezdxf."""

    def __init__(self):
        """Initialize renderer."""
        self.doc: Optional[ezdxf.document.Drawing] = None
        self.msp = None  # Modelspace
        self.width_mm: float = 0
        self.height_mm: float = 0

    def create_drawing(
        self,
        width_mm: float,
        height_mm: float,
        dxf_version: str = 'R2010',
    ) -> None:
        """Create new DXF drawing.

        Args:
            width_mm: Drawing width in mm
            height_mm: Drawing height in mm
            dxf_version: DXF version (R12, R2000, R2004, R2007, R2010, R2013, R2018)
        """
        self.width_mm = width_mm
        self.height_mm = height_mm

        # Create document
        self.doc = ezdxf.new(dxf_version, units=units.MM)
        self.msp = self.doc.modelspace()

        # Setup layers
        self._setup_layers()

        # Setup text styles
        self._setup_text_styles()

        # Setup dimension styles
        self._setup_dimension_styles()

        logger.info("Created DXF drawing: %.0f x %.0f mm", width_mm, height_mm)

    def _setup_layers(self) -> None:
        """Create GOST-compliant layers."""
        for name, props in GOST_LAYERS.items():
            self.doc.layers.add(
                name,
                color=props['color'],
                linetype=props.get('linetype', 'CONTINUOUS'),
                lineweight=props.get('lineweight', 25),
            )

    def _setup_text_styles(self) -> None:
        """Create text styles for GOST compliance."""
        # ISOCPEUR is standard for technical drawings
        self.doc.styles.add('GOST', font='isocpeur.shx')
        # Fallback to Arial if ISOCPEUR not available
        self.doc.styles.add('GOST_TTF', font='Arial')

    def _setup_dimension_styles(self) -> None:
        """Create dimension styles per GOST 2.307."""
        dimstyle = self.doc.dimstyles.new('GOST')

        # Text settings
        dimstyle.dxf.dimtxsty = 'GOST'
        dimstyle.dxf.dimtxt = 3.5  # Text height 3.5mm
        dimstyle.dxf.dimgap = 1.0  # Gap from dimension line

        # Arrow settings
        dimstyle.dxf.dimasz = 2.5  # Arrow size
        dimstyle.dxf.dimtsz = 0  # Use arrows, not ticks

        # Extension line settings
        dimstyle.dxf.dimexe = 2.0  # Extension beyond dimension line
        dimstyle.dxf.dimexo = 1.5  # Offset from origin

        # Dimension line settings
        dimstyle.dxf.dimdli = 7.0  # Spacing between baseline dimensions

        # Units
        dimstyle.dxf.dimdec = 1  # Decimal places
        dimstyle.dxf.dimdsep = ord(',')  # Decimal separator

    def add_line(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        style: Optional[DxfStyle] = None,
    ) -> None:
        """Add a line to the drawing.

        Args:
            start: Start point (x, y) in mm
            end: End point (x, y) in mm
            style: Line style (layer, color, etc.)
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        style = style or DxfStyle()
        attribs = {'layer': style.layer}

        if style.color is not None:
            attribs['color'] = style.color
        if style.lineweight is not None:
            attribs['lineweight'] = style.lineweight
        if style.linetype is not None:
            attribs['linetype'] = style.linetype

        self.msp.add_line(start, end, dxfattribs=attribs)

    def add_polyline(
        self,
        points: List[Tuple[float, float]],
        closed: bool = False,
        style: Optional[DxfStyle] = None,
    ) -> None:
        """Add a polyline to the drawing.

        Args:
            points: List of (x, y) points in mm
            closed: Whether to close the polyline
            style: Line style
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        if len(points) < 2:
            return

        style = style or DxfStyle()
        attribs = {'layer': style.layer}

        if style.color is not None:
            attribs['color'] = style.color
        if style.lineweight is not None:
            attribs['lineweight'] = style.lineweight

        self.msp.add_lwpolyline(points, close=closed, dxfattribs=attribs)

    def add_circle(
        self,
        center: Tuple[float, float],
        radius: float,
        style: Optional[DxfStyle] = None,
    ) -> None:
        """Add a circle to the drawing.

        Args:
            center: Center point (x, y) in mm
            radius: Radius in mm
            style: Line style
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        style = style or DxfStyle()
        attribs = {'layer': style.layer}

        if style.color is not None:
            attribs['color'] = style.color

        self.msp.add_circle(center, radius, dxfattribs=attribs)

    def add_arc(
        self,
        center: Tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
        style: Optional[DxfStyle] = None,
    ) -> None:
        """Add an arc to the drawing.

        Args:
            center: Center point (x, y) in mm
            radius: Radius in mm
            start_angle: Start angle in degrees
            end_angle: End angle in degrees
            style: Line style
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        style = style or DxfStyle()
        attribs = {'layer': style.layer}

        if style.color is not None:
            attribs['color'] = style.color

        self.msp.add_arc(center, radius, start_angle, end_angle, dxfattribs=attribs)

    def add_text(
        self,
        text: str,
        position: Tuple[float, float],
        height: float = 3.5,
        rotation: float = 0,
        halign: str = 'LEFT',
        valign: str = 'BASELINE',
        style: Optional[DxfStyle] = None,
    ) -> None:
        """Add text to the drawing.

        Args:
            text: Text string
            position: Position (x, y) in mm
            height: Text height in mm
            rotation: Rotation angle in degrees
            halign: Horizontal alignment (LEFT, CENTER, RIGHT)
            valign: Vertical alignment (BASELINE, BOTTOM, MIDDLE, TOP)
            style: Text style
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        style = style or DxfStyle(layer='TEXT')
        attribs = {
            'layer': style.layer,
            'style': 'GOST',
            'height': height,
            'rotation': rotation,
        }

        if style.color is not None:
            attribs['color'] = style.color

        # Map alignment
        align_map = {
            ('LEFT', 'BASELINE'): TextEntityAlignment.LEFT,
            ('CENTER', 'BASELINE'): TextEntityAlignment.CENTER,
            ('RIGHT', 'BASELINE'): TextEntityAlignment.RIGHT,
            ('LEFT', 'MIDDLE'): TextEntityAlignment.MIDDLE_LEFT,
            ('CENTER', 'MIDDLE'): TextEntityAlignment.MIDDLE_CENTER,
            ('RIGHT', 'MIDDLE'): TextEntityAlignment.MIDDLE_RIGHT,
            ('LEFT', 'TOP'): TextEntityAlignment.TOP_LEFT,
            ('CENTER', 'TOP'): TextEntityAlignment.TOP_CENTER,
            ('RIGHT', 'TOP'): TextEntityAlignment.TOP_RIGHT,
        }
        alignment = align_map.get((halign, valign), TextEntityAlignment.LEFT)

        self.msp.add_text(text, dxfattribs=attribs).set_placement(
            position, align=alignment
        )

    def add_mtext(
        self,
        text: str,
        position: Tuple[float, float],
        width: float = 0,
        height: float = 3.5,
        style: Optional[DxfStyle] = None,
    ) -> None:
        """Add multi-line text to the drawing.

        Args:
            text: Text string (can contain newlines)
            position: Insert position (x, y) in mm
            width: Text box width (0 = no wrapping)
            height: Text height in mm
            style: Text style
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        style = style or DxfStyle(layer='TEXT')
        attribs = {
            'layer': style.layer,
            'style': 'GOST',
            'char_height': height,
        }

        if width > 0:
            attribs['width'] = width

        if style.color is not None:
            attribs['color'] = style.color

        self.msp.add_mtext(text, dxfattribs=attribs).set_location(position)

    def add_linear_dimension(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        distance: float,
        angle: float = 0,
        text: Optional[str] = None,
    ) -> None:
        """Add a linear dimension.

        Args:
            p1: First definition point
            p2: Second definition point
            distance: Distance from geometry to dimension line
            angle: Dimension line angle (0 = horizontal, 90 = vertical)
            text: Override text (None = automatic)
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        # Calculate dimension line position
        import math
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        if angle == 0:  # Horizontal
            dim_pos = (mid_x, p1[1] + distance)
        elif angle == 90:  # Vertical
            dim_pos = (p1[0] + distance, mid_y)
        else:
            rad = math.radians(angle + 90)
            dim_pos = (mid_x + distance * math.cos(rad),
                       mid_y + distance * math.sin(rad))

        dim = self.msp.add_linear_dim(
            base=dim_pos,
            p1=p1,
            p2=p2,
            angle=angle,
            dimstyle='GOST',
            override={'dimtxt': 3.5},
        )

        if text:
            dim.set_text(text)

        dim.render()

    def add_diameter_dimension(
        self,
        center: Tuple[float, float],
        radius: float,
        angle: float = 45,
        text: Optional[str] = None,
    ) -> None:
        """Add a diameter dimension.

        Args:
            center: Circle center
            radius: Circle radius
            angle: Dimension line angle in degrees
            text: Override text (None = automatic with ⌀ prefix)
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        import math
        rad = math.radians(angle)

        # Points on circle
        p1 = (center[0] - radius * math.cos(rad),
              center[1] - radius * math.sin(rad))
        p2 = (center[0] + radius * math.cos(rad),
              center[1] + radius * math.sin(rad))

        dim = self.msp.add_diameter_dim(
            center=center,
            mpoint=p2,
            dimstyle='GOST',
        )

        if text:
            dim.set_text(text)

        dim.render()

    def add_radius_dimension(
        self,
        center: Tuple[float, float],
        radius: float,
        angle: float = 45,
        text: Optional[str] = None,
    ) -> None:
        """Add a radius dimension.

        Args:
            center: Arc/circle center
            radius: Radius
            angle: Dimension line angle in degrees
            text: Override text (None = automatic with R prefix)
        """
        if self.msp is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        import math
        rad = math.radians(angle)

        # Point on arc
        mpoint = (center[0] + radius * math.cos(rad),
                  center[1] + radius * math.sin(rad))

        dim = self.msp.add_radius_dim(
            center=center,
            mpoint=mpoint,
            dimstyle='GOST',
        )

        if text:
            dim.set_text(text)

        dim.render()

    def add_rectangle(
        self,
        corner: Tuple[float, float],
        width: float,
        height: float,
        style: Optional[DxfStyle] = None,
    ) -> None:
        """Add a rectangle to the drawing.

        Args:
            corner: Bottom-left corner (x, y)
            width: Width in mm
            height: Height in mm
            style: Line style
        """
        x, y = corner
        points = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ]
        self.add_polyline(points, closed=True, style=style)

    def add_hatch_lines(
        self,
        lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        style: Optional[DxfStyle] = None,
    ) -> None:
        """Add hatching lines.

        Args:
            lines: List of ((x1, y1), (x2, y2)) line segments
            style: Line style (default: HATCH layer)
        """
        style = style or DxfStyle(layer='HATCH')
        for start, end in lines:
            self.add_line(start, end, style)

    def save(self, path: Union[str, Path]) -> None:
        """Save drawing to DXF file.

        Args:
            path: Output file path
        """
        if self.doc is None:
            raise RuntimeError("Drawing not created. Call create_drawing() first.")

        path = Path(path)
        self.doc.saveas(str(path))
        logger.info("DXF saved: %s", path)


def convert_svg_views_to_dxf(
    views_data: Dict,
    output_path: Union[str, Path],
    width_mm: float,
    height_mm: float,
    scale: float = 1.0,
) -> None:
    """Convert view data to DXF format.

    Args:
        views_data: Dictionary with view line data
        output_path: Output DXF file path
        width_mm: Drawing width
        height_mm: Drawing height
        scale: Drawing scale factor
    """
    renderer = DxfRenderer()
    renderer.create_drawing(width_mm, height_mm)

    for view_name, view in views_data.items():
        # Get view offset
        offset_x = view.get('offset_x', 0)
        offset_y = view.get('offset_y', 0)

        # Add visible lines
        visible_style = DxfStyle(layer='CONTOUR')
        for line in view.get('visible_lines', []):
            start = (line['x1'] + offset_x, line['y1'] + offset_y)
            end = (line['x2'] + offset_x, line['y2'] + offset_y)
            renderer.add_line(start, end, visible_style)

        # Add hidden lines
        hidden_style = DxfStyle(layer='HIDDEN')
        for line in view.get('hidden_lines', []):
            start = (line['x1'] + offset_x, line['y1'] + offset_y)
            end = (line['x2'] + offset_x, line['y2'] + offset_y)
            renderer.add_line(start, end, hidden_style)

        # Add centerlines
        center_style = DxfStyle(layer='CENTER')
        for line in view.get('centerlines', []):
            start = (line['x1'] + offset_x, line['y1'] + offset_y)
            end = (line['x2'] + offset_x, line['y2'] + offset_y)
            renderer.add_line(start, end, center_style)

    renderer.save(output_path)
