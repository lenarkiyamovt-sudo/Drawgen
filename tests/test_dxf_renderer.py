"""
Unit tests for stl_drawing.drawing.dxf_renderer module.

Tests:
- DXF document creation
- Layer setup
- Entity adding (lines, circles, text)
- Dimension adding
- File saving
"""

import tempfile
from pathlib import Path

import pytest

try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False

pytestmark = pytest.mark.skipif(not HAS_EZDXF, reason="ezdxf not installed")

from stl_drawing.drawing.dxf_renderer import (
    DxfRenderer,
    DxfStyle,
    GOST_LAYERS,
    convert_svg_views_to_dxf,
)


class TestDxfRenderer:
    """Tests for DxfRenderer class."""

    def test_create_drawing(self):
        """Test creating a new DXF drawing."""
        renderer = DxfRenderer()
        renderer.create_drawing(841, 594)

        assert renderer.doc is not None
        assert renderer.msp is not None
        assert renderer.width_mm == 841
        assert renderer.height_mm == 594

    def test_gost_layers_created(self):
        """Test that GOST layers are created."""
        renderer = DxfRenderer()
        renderer.create_drawing(210, 297)

        for layer_name in GOST_LAYERS.keys():
            assert layer_name in renderer.doc.layers

    def test_contour_layer_properties(self):
        """Test CONTOUR layer has correct properties."""
        renderer = DxfRenderer()
        renderer.create_drawing(210, 297)

        contour = renderer.doc.layers.get('CONTOUR')
        assert contour.color == 7  # White/Black
        assert contour.dxf.lineweight == 50  # 0.5mm

    def test_hidden_layer_properties(self):
        """Test HIDDEN layer has dashed linetype."""
        renderer = DxfRenderer()
        renderer.create_drawing(210, 297)

        hidden = renderer.doc.layers.get('HIDDEN')
        assert hidden.color == 8  # Gray
        assert 'DASH' in hidden.dxf.linetype.upper()

    def test_dimension_style_created(self):
        """Test GOST dimension style is created."""
        renderer = DxfRenderer()
        renderer.create_drawing(210, 297)

        assert 'GOST' in renderer.doc.dimstyles


class TestAddLine:
    """Tests for add_line method."""

    def test_add_simple_line(self):
        """Test adding a simple line."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_line((0, 0), (50, 50))

        lines = list(renderer.msp.query('LINE'))
        assert len(lines) == 1
        assert lines[0].dxf.start == (0, 0, 0)
        assert lines[0].dxf.end == (50, 50, 0)

    def test_add_line_with_style(self):
        """Test adding line with custom style."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        style = DxfStyle(layer='HIDDEN', color=1)
        renderer.add_line((0, 0), (100, 0), style=style)

        lines = list(renderer.msp.query('LINE'))
        assert len(lines) == 1
        assert lines[0].dxf.layer == 'HIDDEN'
        assert lines[0].dxf.color == 1

    def test_add_line_without_drawing_raises(self):
        """Test that adding line without creating drawing raises error."""
        renderer = DxfRenderer()

        with pytest.raises(RuntimeError, match="Drawing not created"):
            renderer.add_line((0, 0), (10, 10))


class TestAddPolyline:
    """Tests for add_polyline method."""

    def test_add_polyline(self):
        """Test adding a polyline."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        renderer.add_polyline(points)

        polylines = list(renderer.msp.query('LWPOLYLINE'))
        assert len(polylines) == 1
        assert len(list(polylines[0].vertices())) == 4

    def test_add_closed_polyline(self):
        """Test adding a closed polyline."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        points = [(0, 0), (10, 0), (10, 10)]
        renderer.add_polyline(points, closed=True)

        polylines = list(renderer.msp.query('LWPOLYLINE'))
        assert len(polylines) == 1
        assert polylines[0].closed is True

    def test_empty_polyline_ignored(self):
        """Test that empty polyline is ignored."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_polyline([])
        renderer.add_polyline([(0, 0)])  # Single point

        polylines = list(renderer.msp.query('LWPOLYLINE'))
        assert len(polylines) == 0


class TestAddCircle:
    """Tests for add_circle method."""

    def test_add_circle(self):
        """Test adding a circle."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_circle((50, 50), 25)

        circles = list(renderer.msp.query('CIRCLE'))
        assert len(circles) == 1
        assert circles[0].dxf.center == (50, 50, 0)
        assert circles[0].dxf.radius == 25


class TestAddArc:
    """Tests for add_arc method."""

    def test_add_arc(self):
        """Test adding an arc."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_arc((50, 50), 25, 0, 90)

        arcs = list(renderer.msp.query('ARC'))
        assert len(arcs) == 1
        assert arcs[0].dxf.center == (50, 50, 0)
        assert arcs[0].dxf.radius == 25
        assert arcs[0].dxf.start_angle == 0
        assert arcs[0].dxf.end_angle == 90


class TestAddText:
    """Tests for add_text method."""

    def test_add_simple_text(self):
        """Test adding simple text."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_text("Test", (10, 10))

        texts = list(renderer.msp.query('TEXT'))
        assert len(texts) == 1
        assert texts[0].dxf.text == "Test"
        assert texts[0].dxf.height == 3.5  # Default GOST height

    def test_add_text_with_height(self):
        """Test adding text with custom height."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_text("Big", (10, 10), height=7.0)

        texts = list(renderer.msp.query('TEXT'))
        assert len(texts) == 1
        assert texts[0].dxf.height == 7.0

    def test_add_text_with_rotation(self):
        """Test adding rotated text."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_text("Rotated", (10, 10), rotation=45)

        texts = list(renderer.msp.query('TEXT'))
        assert len(texts) == 1
        assert texts[0].dxf.rotation == 45


class TestAddMtext:
    """Tests for add_mtext method."""

    def test_add_mtext(self):
        """Test adding multi-line text."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_mtext("Line1\nLine2", (10, 10))

        mtexts = list(renderer.msp.query('MTEXT'))
        assert len(mtexts) == 1


class TestAddRectangle:
    """Tests for add_rectangle method."""

    def test_add_rectangle(self):
        """Test adding a rectangle."""
        renderer = DxfRenderer()
        renderer.create_drawing(100, 100)

        renderer.add_rectangle((10, 10), 30, 20)

        polylines = list(renderer.msp.query('LWPOLYLINE'))
        assert len(polylines) == 1
        assert polylines[0].closed is True


class TestAddDimensions:
    """Tests for dimension methods."""

    def test_add_linear_dimension(self):
        """Test adding a linear dimension."""
        renderer = DxfRenderer()
        renderer.create_drawing(200, 200)

        renderer.add_linear_dimension((10, 10), (60, 10), distance=15, angle=0)

        dims = list(renderer.msp.query('DIMENSION'))
        assert len(dims) == 1

    def test_add_diameter_dimension(self):
        """Test adding a diameter dimension."""
        renderer = DxfRenderer()
        renderer.create_drawing(200, 200)

        renderer.add_circle((100, 100), 30)
        renderer.add_diameter_dimension((100, 100), 30)

        dims = list(renderer.msp.query('DIMENSION'))
        assert len(dims) == 1

    def test_add_radius_dimension(self):
        """Test adding a radius dimension."""
        renderer = DxfRenderer()
        renderer.create_drawing(200, 200)

        renderer.add_arc((100, 100), 25, 0, 90)
        renderer.add_radius_dimension((100, 100), 25)

        dims = list(renderer.msp.query('DIMENSION'))
        assert len(dims) == 1


class TestSave:
    """Tests for save method."""

    def test_save_dxf(self):
        """Test saving DXF file."""
        renderer = DxfRenderer()
        renderer.create_drawing(210, 297)
        renderer.add_line((0, 0), (100, 100))

        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
            temp_path = Path(f.name)

        try:
            renderer.save(temp_path)
            assert temp_path.exists()
            assert temp_path.stat().st_size > 0

            # Verify it can be read back
            doc = ezdxf.readfile(str(temp_path))
            assert doc is not None
        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_without_drawing_raises(self):
        """Test that saving without creating drawing raises error."""
        renderer = DxfRenderer()

        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(RuntimeError, match="Drawing not created"):
                renderer.save(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


class TestConvertSvgViewsToDxf:
    """Tests for convert_svg_views_to_dxf function."""

    def test_convert_simple_views(self):
        """Test converting simple view data to DXF."""
        views_data = {
            'front': {
                'offset_x': 100,
                'offset_y': 100,
                'visible_lines': [
                    {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 0},
                    {'x1': 50, 'y1': 0, 'x2': 50, 'y2': 30},
                ],
                'hidden_lines': [
                    {'x1': 10, 'y1': 10, 'x2': 40, 'y2': 10},
                ],
                'centerlines': [],
            },
        }

        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
            temp_path = Path(f.name)

        try:
            convert_svg_views_to_dxf(views_data, temp_path, 210, 297)

            assert temp_path.exists()

            # Read and verify
            doc = ezdxf.readfile(str(temp_path))
            msp = doc.modelspace()

            # Should have 3 lines total
            lines = list(msp.query('LINE'))
            assert len(lines) == 3

            # Check layers
            contour_lines = [l for l in lines if l.dxf.layer == 'CONTOUR']
            hidden_lines = [l for l in lines if l.dxf.layer == 'HIDDEN']
            assert len(contour_lines) == 2
            assert len(hidden_lines) == 1

        finally:
            temp_path.unlink(missing_ok=True)

    def test_convert_with_centerlines(self):
        """Test converting views with centerlines."""
        views_data = {
            'top': {
                'offset_x': 50,
                'offset_y': 50,
                'visible_lines': [],
                'hidden_lines': [],
                'centerlines': [
                    {'x1': 0, 'y1': 25, 'x2': 50, 'y2': 25},
                ],
            },
        }

        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
            temp_path = Path(f.name)

        try:
            convert_svg_views_to_dxf(views_data, temp_path, 100, 100)

            doc = ezdxf.readfile(str(temp_path))
            msp = doc.modelspace()

            lines = list(msp.query('LINE'))
            assert len(lines) == 1
            assert lines[0].dxf.layer == 'CENTER'

        finally:
            temp_path.unlink(missing_ok=True)


class TestDxfStyle:
    """Tests for DxfStyle dataclass."""

    def test_default_style(self):
        """Test default style values."""
        style = DxfStyle()
        assert style.layer == 'CONTOUR'
        assert style.color is None
        assert style.lineweight is None

    def test_custom_style(self):
        """Test custom style values."""
        style = DxfStyle(layer='HIDDEN', color=5, lineweight=35)
        assert style.layer == 'HIDDEN'
        assert style.color == 5
        assert style.lineweight == 35


class TestDxfIntegration:
    """Integration tests for DXF renderer."""

    def test_complete_drawing(self):
        """Test creating a complete drawing with all elements."""
        renderer = DxfRenderer()
        renderer.create_drawing(297, 210)  # A4 landscape

        # Frame
        frame_style = DxfStyle(layer='FRAME')
        renderer.add_rectangle((5, 5), 287, 200, frame_style)

        # Title block area
        renderer.add_rectangle((5, 5), 185, 55, frame_style)

        # Some geometry
        contour = DxfStyle(layer='CONTOUR')
        renderer.add_rectangle((100, 100), 80, 60, contour)
        renderer.add_circle((140, 130), 15, contour)

        # Hidden lines
        hidden = DxfStyle(layer='HIDDEN')
        renderer.add_line((100, 120), (180, 120), hidden)

        # Centerlines
        center = DxfStyle(layer='CENTER')
        renderer.add_line((130, 130), (150, 130), center)
        renderer.add_line((140, 120), (140, 140), center)

        # Dimensions
        renderer.add_linear_dimension((100, 100), (180, 100), 15, angle=0)

        # Text
        renderer.add_text("Деталь", (140, 180), height=5, halign='CENTER')

        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
            temp_path = Path(f.name)

        try:
            renderer.save(temp_path)

            # Verify file
            doc = ezdxf.readfile(str(temp_path))
            msp = doc.modelspace()

            assert len(list(msp.query('LWPOLYLINE'))) >= 3
            assert len(list(msp.query('CIRCLE'))) == 1
            assert len(list(msp.query('LINE'))) >= 2
            assert len(list(msp.query('TEXT'))) >= 1
            assert len(list(msp.query('DIMENSION'))) >= 1

        finally:
            temp_path.unlink(missing_ok=True)
