"""
Unit tests for stl_drawing.drawing.hatching module.

Tests:
- Hatching parameter calculation
- Hatch line generation
- Polygon clipping
- Material patterns
"""

import math
import numpy as np
import pytest

from stl_drawing.drawing.hatching import (
    MaterialType,
    HatchPattern,
    HatchLine,
    STANDARD_PATTERNS,
    calculate_hatch_spacing,
    calculate_hatch_angle,
    generate_hatch_lines,
    get_hatch_pattern,
)


class TestMaterialPatterns:
    """Tests for standard material patterns."""

    def test_metal_pattern_exists(self):
        """Test that metal pattern is defined."""
        assert MaterialType.METAL in STANDARD_PATTERNS
        pattern = STANDARD_PATTERNS[MaterialType.METAL]
        assert pattern.angle_deg == 45.0

    def test_all_materials_have_patterns(self):
        """Test that key materials have patterns defined."""
        key_materials = [
            MaterialType.METAL,
            MaterialType.PLASTIC,
            MaterialType.GLASS,
        ]
        for mat in key_materials:
            assert mat in STANDARD_PATTERNS

    def test_plastic_has_double_line(self):
        """Test that plastic pattern uses double lines."""
        pattern = STANDARD_PATTERNS[MaterialType.PLASTIC]
        assert pattern.double_line is True

    def test_concrete_has_crosshatch(self):
        """Test that concrete pattern has secondary angle."""
        pattern = STANDARD_PATTERNS[MaterialType.CONCRETE]
        assert pattern.secondary_angle is not None


class TestCalculateHatchSpacing:
    """Tests for calculate_hatch_spacing function."""

    def test_small_area_narrow_spacing(self):
        """Test that small areas get narrow spacing."""
        spacing = calculate_hatch_spacing(50)  # 50 mm²
        assert spacing <= 2.0

    def test_large_area_wide_spacing(self):
        """Test that large areas get wider spacing."""
        spacing = calculate_hatch_spacing(10000)  # 10000 mm²
        assert spacing >= 3.0

    def test_respects_min_spacing(self):
        """Test minimum spacing constraint."""
        spacing = calculate_hatch_spacing(1, min_spacing=2.0)
        assert spacing >= 2.0

    def test_respects_max_spacing(self):
        """Test maximum spacing constraint."""
        spacing = calculate_hatch_spacing(100000, max_spacing=8.0)
        assert spacing <= 8.0


class TestCalculateHatchAngle:
    """Tests for calculate_hatch_angle function."""

    def test_first_part_base_angle(self):
        """Test first part gets base angle."""
        angle = calculate_hatch_angle(0, base_angle=45.0)
        assert angle == 45.0

    def test_second_part_rotated(self):
        """Test second part gets different angle."""
        angle0 = calculate_hatch_angle(0)
        angle1 = calculate_hatch_angle(1)
        assert angle0 != angle1

    def test_angles_cycle(self):
        """Test that angles cycle for many parts."""
        angles = [calculate_hatch_angle(i) for i in range(10)]
        # Should have some repetition after 6 parts
        assert len(set(angles)) <= 6


class TestHatchPattern:
    """Tests for HatchPattern dataclass."""

    def test_create_pattern(self):
        """Test creating a custom pattern."""
        pattern = HatchPattern(
            material=MaterialType.METAL,
            angle_deg=30.0,
            spacing_mm=3.0,
        )
        assert pattern.angle_deg == 30.0
        assert pattern.spacing_mm == 3.0
        assert pattern.double_line is False

    def test_default_values(self):
        """Test default pattern values."""
        pattern = HatchPattern(material=MaterialType.METAL)
        assert pattern.angle_deg == 45.0
        assert pattern.spacing_mm == 2.0


class TestGenerateHatchLines:
    """Tests for generate_hatch_lines function."""

    def test_square_polygon(self):
        """Test hatching a simple square."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        lines = generate_hatch_lines(square, angle_deg=0, spacing_mm=2.0)

        assert len(lines) > 0
        # Horizontal lines at 0° should span full width
        for line in lines:
            assert isinstance(line, HatchLine)

    def test_45_degree_angle(self):
        """Test 45 degree hatching."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        lines = generate_hatch_lines(square, angle_deg=45, spacing_mm=2.0)

        assert len(lines) > 0

    def test_empty_polygon(self):
        """Test with less than 3 vertices."""
        lines = generate_hatch_lines([(0, 0), (1, 1)], spacing_mm=2.0)
        assert lines == []

    def test_spacing_affects_line_count(self):
        """Test that spacing affects number of lines."""
        square = [(0, 0), (20, 0), (20, 20), (0, 20)]

        lines_narrow = generate_hatch_lines(square, spacing_mm=2.0)
        lines_wide = generate_hatch_lines(square, spacing_mm=5.0)

        assert len(lines_narrow) > len(lines_wide)

    def test_triangle_polygon(self):
        """Test hatching a triangle."""
        triangle = [(0, 0), (10, 0), (5, 10)]
        lines = generate_hatch_lines(triangle, angle_deg=45, spacing_mm=2.0)

        assert len(lines) > 0

    def test_offset_shifts_pattern(self):
        """Test that offset shifts the hatch pattern."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]

        lines1 = generate_hatch_lines(square, spacing_mm=2.0, offset=0)
        lines2 = generate_hatch_lines(square, spacing_mm=2.0, offset=1.0)

        # Lines should be different with offset
        if len(lines1) > 0 and len(lines2) > 0:
            assert lines1[0].start != lines2[0].start


class TestGetHatchPattern:
    """Tests for get_hatch_pattern function."""

    def test_returns_pattern(self):
        """Test that function returns a HatchPattern."""
        pattern = get_hatch_pattern(MaterialType.METAL)
        assert isinstance(pattern, HatchPattern)
        assert pattern.material == MaterialType.METAL

    def test_scale_affects_spacing(self):
        """Test that larger scale increases spacing."""
        pattern_normal = get_hatch_pattern(MaterialType.METAL, scale=1.0)
        pattern_large = get_hatch_pattern(MaterialType.METAL, scale=2.0)

        assert pattern_large.spacing_mm >= pattern_normal.spacing_mm

    def test_line_width_from_stroke(self):
        """Test that line width is based on stroke width."""
        pattern = get_hatch_pattern(MaterialType.METAL, stroke_width=1.0)

        # Should be S/3 to S/2
        assert 0.3 <= pattern.line_width_mm <= 0.5

    def test_unknown_material_defaults_to_metal(self):
        """Test fallback for undefined materials."""
        # Access a material not in STANDARD_PATTERNS
        pattern = get_hatch_pattern(MaterialType.INSULATION)
        # Should return metal pattern as fallback
        assert pattern.material == MaterialType.METAL


class TestHatchLine:
    """Tests for HatchLine dataclass."""

    def test_create_hatch_line(self):
        """Test creating a hatch line."""
        line = HatchLine(start=(0, 0), end=(10, 10))
        assert line.start == (0, 0)
        assert line.end == (10, 10)

    def test_hatch_line_length(self):
        """Test calculating line length."""
        line = HatchLine(start=(0, 0), end=(3, 4))
        length = math.hypot(
            line.end[0] - line.start[0],
            line.end[1] - line.start[1]
        )
        assert length == 5.0


class TestComplexPolygons:
    """Tests for hatching complex polygon shapes."""

    def test_l_shaped_polygon(self):
        """Test hatching an L-shaped polygon."""
        l_shape = [
            (0, 0), (10, 0), (10, 5), (5, 5),
            (5, 10), (0, 10)
        ]
        lines = generate_hatch_lines(l_shape, angle_deg=45, spacing_mm=2.0)
        assert len(lines) > 0

    def test_concave_polygon(self):
        """Test hatching a concave polygon (arrow shape)."""
        arrow = [
            (0, 5), (10, 0), (7, 5), (10, 10)
        ]
        lines = generate_hatch_lines(arrow, angle_deg=45, spacing_mm=2.0)
        # Should still produce some lines
        assert isinstance(lines, list)
