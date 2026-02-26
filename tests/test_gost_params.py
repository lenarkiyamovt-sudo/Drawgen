"""
Unit tests for stl_drawing.drawing.gost_params module.

Tests:
- GOST 2.303-68 line specifications
- Line thickness calculations
- Dash pattern calculations
- Scale-based parameter selection
"""

import pytest

from stl_drawing.drawing.gost_params import (
    LineType,
    LineStyle,
    GOST_LINE_SPECS,
    GOST_S_VALUES,
    get_recommended_s,
    calculate_thickness_for_scale,
    get_dash_pattern_for_size,
    calculate_line_parameters,
    calculate_line_parameters_for_scale,
    snap_to_gost_scale,
)


class TestGOSTLineSpecs:
    """Tests for GOST 2.303-68 line specifications."""

    def test_all_line_types_defined(self):
        """Test that all line types have specifications."""
        for line_type in LineType:
            assert line_type in GOST_LINE_SPECS

    def test_continuous_thick_ratio(self):
        """Test that continuous thick line has ratio 1.0."""
        spec = GOST_LINE_SPECS[LineType.CONTINUOUS_THICK]
        assert spec.thickness_ratio == 1.0

    def test_thin_lines_ratio_range(self):
        """Test that thin lines have ratio in S/3 to S/2 range."""
        thin_types = [
            LineType.CONTINUOUS_THIN,
            LineType.DASHED,
            LineType.CHAIN_THIN,
        ]
        for lt in thin_types:
            spec = GOST_LINE_SPECS[lt]
            # S/3 = 0.33, S/2 = 0.5
            assert 0.3 <= spec.thickness_ratio <= 0.5

    def test_dashed_has_dash_range(self):
        """Test that dashed line has dash length range."""
        spec = GOST_LINE_SPECS[LineType.DASHED]
        assert spec.dash_length_range is not None
        assert spec.dash_length_range[0] >= 2.0
        assert spec.dash_length_range[1] <= 8.0

    def test_chain_thin_has_longer_dashes(self):
        """Test that chain thin has longer dashes than dashed."""
        dashed = GOST_LINE_SPECS[LineType.DASHED]
        chain = GOST_LINE_SPECS[LineType.CHAIN_THIN]
        assert chain.dash_length_range[0] > dashed.dash_length_range[0]


class TestGOSTSValues:
    """Tests for standard S values."""

    def test_s_values_ordered(self):
        """Test that S values are in ascending order."""
        for i in range(len(GOST_S_VALUES) - 1):
            assert GOST_S_VALUES[i] < GOST_S_VALUES[i + 1]

    def test_s_values_range(self):
        """Test that S values are in GOST range (0.5-1.4)."""
        assert GOST_S_VALUES[0] >= 0.5
        assert GOST_S_VALUES[-1] <= 1.4


class TestGetRecommendedS:
    """Tests for get_recommended_s function."""

    def test_small_format_thin(self):
        """Test that A4 gets thinner lines."""
        s = get_recommended_s(1.0, "A4")
        assert s <= 0.6

    def test_large_format_thick(self):
        """Test that A0/A1 get thicker lines."""
        s = get_recommended_s(1.0, "A0")
        assert s >= 0.6

    def test_enlargement_scale(self):
        """Test that enlargement scales get thicker lines."""
        s_normal = get_recommended_s(1.0, "A4")
        s_enlarge = get_recommended_s(2.0, "A4")
        assert s_enlarge >= s_normal

    def test_returns_gost_value(self):
        """Test that returned value is a valid GOST S value."""
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for fmt in ["A4", "A3", "A2", "A1", "A0"]:
                s = get_recommended_s(scale, fmt)
                assert s in GOST_S_VALUES


class TestCalculateThicknessForScale:
    """Tests for calculate_thickness_for_scale function."""

    def test_visible_line_thickness(self):
        """Test visible line equals S."""
        thickness = calculate_thickness_for_scale(1.0, LineType.CONTINUOUS_THICK)
        s = get_recommended_s(1.0, "A4")
        assert thickness == s

    def test_thin_line_thickness(self):
        """Test thin line is less than S."""
        thick = calculate_thickness_for_scale(1.0, LineType.CONTINUOUS_THICK)
        thin = calculate_thickness_for_scale(1.0, LineType.CONTINUOUS_THIN)
        assert thin < thick


class TestGetDashPatternForSize:
    """Tests for get_dash_pattern_for_size function."""

    def test_small_view_short_dashes(self):
        """Test that small views get shorter dashes."""
        dash_small, gap_small = get_dash_pattern_for_size(30, LineType.DASHED)
        dash_large, gap_large = get_dash_pattern_for_size(200, LineType.DASHED)
        assert dash_small < dash_large

    def test_continuous_returns_zero(self):
        """Test that continuous lines return zero dash."""
        dash, gap = get_dash_pattern_for_size(100, LineType.CONTINUOUS_THICK)
        assert dash == 0
        assert gap == 0


class TestCalculateLineParameters:
    """Tests for calculate_line_parameters function."""

    def test_returns_all_styles(self):
        """Test that all required styles are returned."""
        params = calculate_line_parameters()
        required_keys = ['visible', 'hidden', 'thin', 'centerline',
                        'dimension', 'dimension_text', 'dimension_arrow', '_params']
        for key in required_keys:
            assert key in params

    def test_visible_has_stroke_width(self):
        """Test that visible style has stroke_width."""
        params = calculate_line_parameters()
        assert 'stroke_width' in params['visible']
        assert 'mm' in params['visible']['stroke_width']

    def test_hidden_has_dasharray(self):
        """Test that hidden style has stroke_dasharray."""
        params = calculate_line_parameters()
        assert 'stroke_dasharray' in params['hidden']

    def test_params_section_exists(self):
        """Test that _params contains all numeric values."""
        params = calculate_line_parameters()
        required = ['S', 'thin_width', 'dash_length', 'gap_length']
        for key in required:
            assert key in params['_params']

    def test_scale_affects_thickness(self):
        """Test that scale parameter affects line thickness."""
        params_small = calculate_line_parameters(scale=0.1)
        params_large = calculate_line_parameters(scale=2.0)
        s_small = params_small['_params']['S']
        s_large = params_large['_params']['S']
        # Large scale should have equal or larger S
        assert s_large >= s_small

    def test_custom_stroke_width(self):
        """Test that custom stroke_width overrides auto."""
        params = calculate_line_parameters(stroke_width=1.0)
        assert params['_params']['S'] == 1.0

    def test_gost_type_annotation(self):
        """Test that styles have GOST type annotation."""
        params = calculate_line_parameters()
        assert params['visible'].get('_gost_type') == 'continuous_thick'
        assert params['hidden'].get('_gost_type') == 'dashed'


class TestCalculateLineParametersForScale:
    """Tests for calculate_line_parameters_for_scale function."""

    def test_returns_all_line_types(self):
        """Test that all line types are returned."""
        params = calculate_line_parameters_for_scale(1.0)
        assert 'S' in params
        assert 'visible' in params
        assert 'hidden' in params
        assert 'thin' in params
        assert 'centerline' in params

    def test_visible_equals_s(self):
        """Test that visible thickness equals S."""
        params = calculate_line_parameters_for_scale(1.0)
        assert params['visible'] == params['S']

    def test_thin_less_than_s(self):
        """Test that thin lines are thinner than S."""
        params = calculate_line_parameters_for_scale(1.0)
        assert params['thin'] < params['S']


class TestSnapToGostScale:
    """Tests for snap_to_gost_scale function."""

    def test_exact_scale_unchanged(self):
        """Test that exact GOST scales are unchanged."""
        assert snap_to_gost_scale(1.0) == 1.0
        assert snap_to_gost_scale(0.5) == 0.5

    def test_intermediate_snaps_down(self):
        """Test that intermediate values snap to smaller scale."""
        # 0.7 should snap to 0.5 (next smaller standard scale)
        result = snap_to_gost_scale(0.7)
        assert result == 0.5

    def test_very_small_returns_minimum(self):
        """Test that very small scales return minimum."""
        result = snap_to_gost_scale(0.0001)
        assert result > 0
