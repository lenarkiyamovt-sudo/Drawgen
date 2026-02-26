"""
Unit tests for stl_drawing.project_config module.

Tests:
- Configuration dataclasses
- JSON serialization/deserialization
- Config file loading
- Config merging
"""

import json
import tempfile
from pathlib import Path

import pytest

from stl_drawing.project_config import (
    DrawingConfig,
    LinesConfig,
    DimensionsConfig,
    TitleBlockConfig,
    OutputConfig,
    HatchingConfig,
    ProjectConfig,
    find_config_file,
    load_config,
    merge_configs,
    create_sample_config,
    CONFIG_FILENAME,
)


class TestDrawingConfig:
    """Tests for DrawingConfig dataclass."""

    def test_default_values(self):
        """Test default drawing config values."""
        config = DrawingConfig()
        assert config.format == "auto"
        assert config.scale is None
        assert config.orientation == "auto"

    def test_custom_values(self):
        """Test custom drawing config values."""
        config = DrawingConfig(format="A3", scale=0.5)
        assert config.format == "A3"
        assert config.scale == 0.5


class TestLinesConfig:
    """Tests for LinesConfig dataclass."""

    def test_default_sharp_angle(self):
        """Test default sharp angle is 10 degrees."""
        config = LinesConfig()
        assert config.sharp_angle_deg == 10.0

    def test_enable_flags_default_true(self):
        """Test enable flags are True by default."""
        config = LinesConfig()
        assert config.enable_merge is True
        assert config.enable_priority is True
        assert config.enable_sharp_edges is True


class TestDimensionsConfig:
    """Tests for DimensionsConfig dataclass."""

    def test_gost_defaults(self):
        """Test GOST-compliant default values."""
        config = DimensionsConfig()
        assert config.arrow_length == 2.5
        assert config.text_height == 3.5
        assert config.font_family == "ISOCPEUR"


class TestTitleBlockConfig:
    """Tests for TitleBlockConfig dataclass."""

    def test_empty_defaults(self):
        """Test that string fields default to empty."""
        config = TitleBlockConfig()
        assert config.organization == ""
        assert config.designer == ""
        assert config.document_name == ""

    def test_sheet_defaults(self):
        """Test default sheet numbering."""
        config = TitleBlockConfig()
        assert config.sheet_number == 1
        assert config.total_sheets == 1


class TestProjectConfig:
    """Tests for ProjectConfig dataclass."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = ProjectConfig()
        assert isinstance(config.drawing, DrawingConfig)
        assert isinstance(config.lines, LinesConfig)
        assert isinstance(config.dimensions, DimensionsConfig)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ProjectConfig()
        d = config.to_dict()

        assert 'drawing' in d
        assert 'lines' in d
        assert 'dimensions' in d
        assert d['drawing']['format'] == 'auto'

    def test_to_json(self):
        """Test converting config to JSON string."""
        config = ProjectConfig()
        json_str = config.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert 'drawing' in data

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            'drawing': {
                'format': 'A3',
                'scale': 0.5,
            },
            'lines': {
                'sharp_angle_deg': 15.0,
            },
        }
        config = ProjectConfig.from_dict(data)

        assert config.drawing.format == 'A3'
        assert config.drawing.scale == 0.5
        assert config.lines.sharp_angle_deg == 15.0

    def test_from_json(self):
        """Test creating config from JSON string."""
        json_str = '''
        {
            "drawing": {"format": "A2"},
            "dimensions": {"arrow_length": 3.0}
        }
        '''
        config = ProjectConfig.from_json(json_str)

        assert config.drawing.format == 'A2'
        assert config.dimensions.arrow_length == 3.0

    def test_save_and_load(self):
        """Test saving and loading config file."""
        config = ProjectConfig()
        config.drawing.format = 'A1'
        config.title_block.designer = 'Test User'

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)

            loaded = ProjectConfig.load(temp_path)

            assert loaded.drawing.format == 'A1'
            assert loaded.title_block.designer == 'Test User'
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_partial_dict_doesnt_break(self):
        """Test that partial config dict doesn't raise errors."""
        data = {'drawing': {'format': 'A4'}}  # Only drawing section
        config = ProjectConfig.from_dict(data)

        assert config.drawing.format == 'A4'
        # Other sections should have defaults
        assert config.lines.sharp_angle_deg == 10.0


class TestFindConfigFile:
    """Tests for find_config_file function."""

    def test_explicit_config_found(self):
        """Test finding explicit config path."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
            f.write(b'{}')

        try:
            found = find_config_file(explicit_config=temp_path)
            assert found == Path(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_explicit_config_not_found(self):
        """Test that missing explicit config returns None."""
        found = find_config_file(explicit_config='/nonexistent/path.json')
        assert found is None

    def test_no_config_returns_none(self):
        """Test that no config file returns None."""
        found = find_config_file()
        # May or may not find one depending on environment
        assert found is None or isinstance(found, Path)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_returns_defaults_when_no_file(self):
        """Test that load_config returns defaults when no file found."""
        config = load_config()
        assert isinstance(config, ProjectConfig)
        assert config.drawing.format == 'auto'

    def test_load_from_explicit_file(self):
        """Test loading from explicit config file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump({'drawing': {'format': 'A0'}}, f)
            temp_path = f.name

        try:
            config = load_config(explicit_config=temp_path)
            assert config.drawing.format == 'A0'
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_invalid_json_returns_defaults(self):
        """Test that invalid JSON returns defaults with warning."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            f.write('not valid json {{{')
            temp_path = f.name

        try:
            config = load_config(explicit_config=temp_path)
            # Should return defaults, not crash
            assert config.drawing.format == 'auto'
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_override_non_default_values(self):
        """Test that non-default values override base."""
        base = ProjectConfig()
        override = ProjectConfig()
        override.drawing.format = 'A3'

        merged = merge_configs(base, override)

        assert merged.drawing.format == 'A3'

    def test_default_values_not_overridden(self):
        """Test that default values don't override base."""
        base = ProjectConfig()
        base.drawing.format = 'A2'
        override = ProjectConfig()  # All defaults

        merged = merge_configs(base, override)

        # Base value should be preserved
        assert merged.drawing.format == 'A2'

    def test_title_block_override(self):
        """Test that non-empty title block values override."""
        base = ProjectConfig()
        base.title_block.designer = 'Original'

        override = ProjectConfig()
        override.title_block.designer = 'New Designer'

        merged = merge_configs(base, override)

        assert merged.title_block.designer == 'New Designer'


class TestCreateSampleConfig:
    """Tests for create_sample_config function."""

    def test_creates_valid_json(self):
        """Test that sample config is valid JSON."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            create_sample_config(temp_path)

            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert 'drawing' in data
            assert 'lines' in data
            assert 'dimensions' in data
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_sample_has_comments(self):
        """Test that sample config has documentation comments."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            create_sample_config(temp_path)

            with open(temp_path, 'r') as f:
                data = json.load(f)

            # Should have _comment fields for documentation
            assert '_comment' in data
            assert '_comment' in data['drawing']
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigRoundTrip:
    """Integration tests for config serialization round-trip."""

    def test_full_roundtrip(self):
        """Test complete save/load cycle preserves all values."""
        original = ProjectConfig()
        original.drawing.format = 'A1'
        original.drawing.scale = 0.25
        original.lines.sharp_angle_deg = 12.5
        original.dimensions.arrow_length = 3.5
        original.title_block.designer = 'Test Engineer'
        original.output.formats = ['svg', 'dxf']

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            original.save(temp_path)
            loaded = ProjectConfig.load(temp_path)

            assert loaded.drawing.format == 'A1'
            assert loaded.drawing.scale == 0.25
            assert loaded.lines.sharp_angle_deg == 12.5
            assert loaded.dimensions.arrow_length == 3.5
            assert loaded.title_block.designer == 'Test Engineer'
            assert loaded.output.formats == ['svg', 'dxf']
        finally:
            Path(temp_path).unlink(missing_ok=True)
