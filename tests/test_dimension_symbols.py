"""
Unit tests for stl_drawing.drawing.dimensions.symbols module.

Tests:
- GOST 2.307-2011 dimension symbol formatting
- Tolerance formatting
- Chamfer and thread notation
"""

import pytest

from stl_drawing.drawing.dimensions.symbols import (
    DimensionType,
    ToleranceValue,
    SYMBOLS,
    ASCII_FALLBACKS,
    format_dimension,
    format_tolerance,
    format_chamfer,
    format_thread,
    get_dimension_prefix,
    detect_dimension_type,
)


class TestSymbols:
    """Tests for symbol definitions."""

    def test_unicode_diameter_symbol(self):
        """Test Unicode diameter symbol is correct."""
        assert SYMBOLS['diameter'] == '\u2300'  # ⌀

    def test_unicode_plus_minus(self):
        """Test Unicode plus-minus symbol is correct."""
        assert SYMBOLS['plus_minus'] == '\u00B1'  # ±

    def test_unicode_degree(self):
        """Test Unicode degree symbol is correct."""
        assert SYMBOLS['degree'] == '\u00B0'  # °

    def test_all_symbols_have_ascii_fallback(self):
        """Test that all Unicode symbols have ASCII fallbacks."""
        for key in SYMBOLS:
            assert key in ASCII_FALLBACKS


class TestToleranceValue:
    """Tests for ToleranceValue dataclass."""

    def test_symmetric_tolerance(self):
        """Test symmetric tolerance detection."""
        tol = ToleranceValue(upper=0.1, lower=-0.1)
        assert tol.is_symmetric

    def test_asymmetric_tolerance(self):
        """Test asymmetric tolerance detection."""
        tol = ToleranceValue(upper=0.02, lower=-0.01)
        assert not tol.is_symmetric

    def test_single_sided_upper(self):
        """Test single-sided upper tolerance."""
        tol = ToleranceValue(upper=0.02, lower=None)
        assert tol.is_single_sided

    def test_single_sided_lower(self):
        """Test single-sided lower tolerance."""
        tol = ToleranceValue(upper=None, lower=-0.01)
        assert tol.is_single_sided


class TestFormatDimension:
    """Tests for format_dimension function."""

    def test_linear_dimension_integer(self):
        """Test linear dimension with integer value."""
        result = format_dimension(100, DimensionType.LINEAR)
        assert result == "100"

    def test_linear_dimension_decimal(self):
        """Test linear dimension with decimal value."""
        result = format_dimension(50.5, DimensionType.LINEAR, decimals=1)
        assert result == "50.5"

    def test_diameter_dimension(self):
        """Test diameter dimension with ⌀ prefix."""
        result = format_dimension(25, DimensionType.DIAMETER)
        assert result.startswith(SYMBOLS['diameter'])
        assert "25" in result

    def test_diameter_ascii_fallback(self):
        """Test diameter with ASCII fallback."""
        result = format_dimension(25, DimensionType.DIAMETER, use_unicode=False)
        assert result.startswith(ASCII_FALLBACKS['diameter'])

    def test_radius_dimension(self):
        """Test radius dimension with R prefix."""
        result = format_dimension(10, DimensionType.RADIUS)
        assert result.startswith("R")
        assert "10" in result

    def test_square_dimension(self):
        """Test square dimension with □ prefix."""
        result = format_dimension(30, DimensionType.SQUARE)
        assert SYMBOLS['square'] in result
        assert "30" in result

    def test_sphere_diameter(self):
        """Test sphere diameter with S⌀ prefix."""
        result = format_dimension(20, DimensionType.SPHERE_DIAMETER)
        assert "S" in result
        assert SYMBOLS['diameter'] in result
        assert "20" in result

    def test_sphere_radius(self):
        """Test sphere radius with SR prefix."""
        result = format_dimension(10, DimensionType.SPHERE_RADIUS)
        assert result.startswith("SR")
        assert "10" in result

    def test_angle_dimension(self):
        """Test angle dimension with ° suffix."""
        result = format_dimension(45, DimensionType.ANGLE)
        assert "45" in result
        assert SYMBOLS['degree'] in result

    def test_dimension_with_tolerance(self):
        """Test dimension with symmetric tolerance."""
        tol = ToleranceValue(upper=0.1, lower=-0.1)
        result = format_dimension(50, DimensionType.LINEAR, tolerance=tol)
        assert "50" in result
        assert SYMBOLS['plus_minus'] in result


class TestFormatTolerance:
    """Tests for format_tolerance function."""

    def test_symmetric_tolerance(self):
        """Test symmetric tolerance formatting."""
        tol = ToleranceValue(upper=0.1, lower=-0.1)
        result = format_tolerance(tol)
        assert SYMBOLS['plus_minus'] in result
        assert "0.1" in result

    def test_asymmetric_tolerance(self):
        """Test asymmetric tolerance formatting."""
        tol = ToleranceValue(upper=0.02, lower=-0.01)
        result = format_tolerance(tol)
        assert "+0.02" in result
        assert "-0.01" in result

    def test_upper_only_tolerance(self):
        """Test upper-only tolerance formatting."""
        tol = ToleranceValue(upper=0.05, lower=None)
        result = format_tolerance(tol)
        assert "+0.05" in result

    def test_lower_only_tolerance(self):
        """Test lower-only tolerance formatting."""
        tol = ToleranceValue(upper=None, lower=-0.03)
        result = format_tolerance(tol)
        assert "-0.03" in result

    def test_empty_tolerance(self):
        """Test empty tolerance returns empty string."""
        tol = ToleranceValue(upper=None, lower=None)
        result = format_tolerance(tol)
        assert result == ""


class TestFormatChamfer:
    """Tests for format_chamfer function."""

    def test_standard_chamfer_45deg(self):
        """Test standard 45° chamfer formatting."""
        result = format_chamfer(2)
        assert "2" in result
        assert SYMBOLS['multiply'] in result
        assert "45" in result
        assert SYMBOLS['degree'] in result

    def test_chamfer_custom_angle(self):
        """Test chamfer with custom angle."""
        result = format_chamfer(1.5, angle_deg=30)
        assert "1.5" in result
        assert "30" in result

    def test_chamfer_ascii(self):
        """Test chamfer with ASCII fallback."""
        result = format_chamfer(2, use_unicode=False)
        assert "2" in result
        assert ASCII_FALLBACKS['multiply'] in result


class TestFormatThread:
    """Tests for format_thread function."""

    def test_metric_thread_standard_pitch(self):
        """Test metric thread with standard pitch."""
        result = format_thread(12, thread_type="M")
        assert result == "M12"

    def test_metric_thread_custom_pitch(self):
        """Test metric thread with custom pitch."""
        result = format_thread(12, pitch_mm=1.5, thread_type="M")
        assert "M12" in result
        assert SYMBOLS['multiply'] in result
        assert "1.5" in result

    def test_pipe_thread(self):
        """Test pipe thread notation."""
        result = format_thread(0.5, thread_type="G")
        assert result.startswith("G")

    def test_trapezoidal_thread(self):
        """Test trapezoidal thread notation."""
        result = format_thread(20, pitch_mm=4, thread_type="Tr")
        assert "Tr20" in result
        assert "4" in result


class TestGetDimensionPrefix:
    """Tests for get_dimension_prefix function."""

    def test_linear_no_prefix(self):
        """Test linear dimension has no prefix."""
        prefix = get_dimension_prefix(DimensionType.LINEAR)
        assert prefix == ""

    def test_diameter_prefix(self):
        """Test diameter prefix."""
        prefix = get_dimension_prefix(DimensionType.DIAMETER)
        assert prefix == SYMBOLS['diameter']

    def test_radius_prefix(self):
        """Test radius prefix."""
        prefix = get_dimension_prefix(DimensionType.RADIUS)
        assert prefix == "R"


class TestDetectDimensionType:
    """Tests for detect_dimension_type function."""

    def test_detect_diameter_from_context(self):
        """Test detection of diameter from context."""
        result = detect_dimension_type(25, "cylinder diameter")
        assert result == DimensionType.DIAMETER

    def test_detect_hole(self):
        """Test detection of diameter from hole context."""
        result = detect_dimension_type(10, "hole")
        assert result == DimensionType.DIAMETER

    def test_detect_radius(self):
        """Test detection of radius from context."""
        result = detect_dimension_type(5, "fillet radius")
        assert result == DimensionType.RADIUS

    def test_detect_chamfer(self):
        """Test detection of chamfer from context."""
        result = detect_dimension_type(2, "chamfer")
        assert result == DimensionType.CHAMFER

    def test_detect_linear_default(self):
        """Test default detection is linear."""
        result = detect_dimension_type(100)
        assert result == DimensionType.LINEAR
