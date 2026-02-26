"""
JSON-based project configuration for stl_drawing.

Allows overriding default configuration values through:
1. .eskd.json file in the current directory
2. .eskd.json file in the STL file's directory
3. Explicit config file path via CLI

Configuration hierarchy (later overrides earlier):
1. Built-in defaults (config.py)
2. User config (~/.eskd.json)
3. Project config (./.eskd.json)
4. CLI arguments

Example .eskd.json:
{
    "drawing": {
        "format": "A3",
        "scale": 0.5,
        "orientation": "landscape"
    },
    "lines": {
        "stroke_width": 0.7,
        "sharp_angle_deg": 15.0
    },
    "dimensions": {
        "arrow_length": 3.0,
        "text_height": 4.0,
        "font_family": "Arial"
    },
    "title_block": {
        "organization": "My Company",
        "designer": "John Doe"
    },
    "output": {
        "formats": ["svg", "dxf"],
        "prefix": "drawing_"
    }
}
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Default config filename
CONFIG_FILENAME = ".eskd.json"


@dataclass
class DrawingConfig:
    """Drawing format and scale configuration."""
    format: str = "auto"  # A4, A3, A2, A1, A0, or "auto"
    scale: Optional[float] = None  # None = auto-select
    orientation: str = "auto"  # "landscape", "portrait", or "auto"
    view_spacing_mm: float = 30.0
    min_main_view_mm: float = 40.0


@dataclass
class LinesConfig:
    """Line styles configuration."""
    stroke_width: Optional[float] = None  # None = auto based on scale
    sharp_angle_deg: float = 10.0
    enable_merge: bool = True
    enable_priority: bool = True
    enable_sharp_edges: bool = True


@dataclass
class DimensionsConfig:
    """Dimension annotation configuration."""
    arrow_length: float = 2.5
    arrow_width: float = 0.8
    text_height: float = 3.5
    text_gap: float = 1.0
    first_offset: float = 10.0
    next_offset: float = 7.0
    extension_overshoot: float = 2.0
    extension_gap: float = 1.5
    min_displayable: float = 4.0
    font_family: str = "ISOCPEUR"


@dataclass
class TitleBlockConfig:
    """Title block (stamp) configuration per GOST 2.104."""
    organization: str = ""
    department: str = ""
    designer: str = ""
    checker: str = ""
    tech_controller: str = ""
    norm_controller: str = ""
    approver: str = ""
    document_name: str = ""
    document_number: str = ""
    material: str = ""
    scale_text: str = ""
    sheet_number: int = 1
    total_sheets: int = 1
    mass: Optional[float] = None  # Auto-calculated if None


@dataclass
class OutputConfig:
    """Output file configuration."""
    formats: List[str] = field(default_factory=lambda: ["svg"])
    prefix: str = ""
    suffix: str = ""
    output_dir: str = ""


@dataclass
class HatchingConfig:
    """Hatching configuration for sections."""
    default_material: str = "metal"
    spacing_mm: float = 2.0
    angle_deg: float = 45.0


@dataclass
class ProjectConfig:
    """Complete project configuration."""
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    lines: LinesConfig = field(default_factory=LinesConfig)
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    title_block: TitleBlockConfig = field(default_factory=TitleBlockConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hatching: HatchingConfig = field(default_factory=HatchingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        logger.info("Configuration saved to %s", path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            ProjectConfig instance
        """
        config = cls()

        if 'drawing' in data:
            for key, value in data['drawing'].items():
                if hasattr(config.drawing, key):
                    setattr(config.drawing, key, value)

        if 'lines' in data:
            for key, value in data['lines'].items():
                if hasattr(config.lines, key):
                    setattr(config.lines, key, value)

        if 'dimensions' in data:
            for key, value in data['dimensions'].items():
                if hasattr(config.dimensions, key):
                    setattr(config.dimensions, key, value)

        if 'title_block' in data:
            for key, value in data['title_block'].items():
                if hasattr(config.title_block, key):
                    setattr(config.title_block, key, value)

        if 'output' in data:
            for key, value in data['output'].items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)

        if 'hatching' in data:
            for key, value in data['hatching'].items():
                if hasattr(config.hatching, key):
                    setattr(config.hatching, key, value)

        return config

    @classmethod
    def from_json(cls, json_str: str) -> 'ProjectConfig':
        """Create configuration from JSON string.

        Args:
            json_str: JSON configuration string

        Returns:
            ProjectConfig instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ProjectConfig':
        """Load configuration from JSON file.

        Args:
            path: Input file path

        Returns:
            ProjectConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Configuration loaded from %s", path)
        return cls.from_dict(data)


def find_config_file(
    stl_path: Optional[Union[str, Path]] = None,
    explicit_config: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Find configuration file using search hierarchy.

    Search order:
    1. Explicit config path (if provided)
    2. .eskd.json in STL file's directory
    3. .eskd.json in current working directory
    4. ~/.eskd.json in user's home directory

    Args:
        stl_path: Path to STL file being processed
        explicit_config: Explicitly specified config path

    Returns:
        Path to config file if found, None otherwise
    """
    # 1. Explicit config
    if explicit_config:
        explicit = Path(explicit_config)
        if explicit.exists():
            return explicit
        logger.warning("Explicit config not found: %s", explicit)

    # 2. STL file's directory
    if stl_path:
        stl_dir = Path(stl_path).parent
        stl_config = stl_dir / CONFIG_FILENAME
        if stl_config.exists():
            return stl_config

    # 3. Current working directory
    cwd_config = Path.cwd() / CONFIG_FILENAME
    if cwd_config.exists():
        return cwd_config

    # 4. User's home directory
    home_config = Path.home() / CONFIG_FILENAME
    if home_config.exists():
        return home_config

    return None


def load_config(
    stl_path: Optional[Union[str, Path]] = None,
    explicit_config: Optional[Union[str, Path]] = None,
) -> ProjectConfig:
    """Load configuration with fallback to defaults.

    Args:
        stl_path: Path to STL file being processed
        explicit_config: Explicitly specified config path

    Returns:
        ProjectConfig instance (defaults if no config file found)
    """
    config_path = find_config_file(stl_path, explicit_config)

    if config_path:
        try:
            return ProjectConfig.load(config_path)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load config %s: %s", config_path, e)

    return ProjectConfig()


def merge_configs(base: ProjectConfig, override: ProjectConfig) -> ProjectConfig:
    """Merge two configurations, with override taking precedence.

    Only non-default values from override are applied.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged ProjectConfig
    """
    merged = ProjectConfig.from_dict(base.to_dict())

    # Merge drawing config
    for key, value in asdict(override.drawing).items():
        default_value = getattr(DrawingConfig(), key)
        if value != default_value:
            setattr(merged.drawing, key, value)

    # Merge lines config
    for key, value in asdict(override.lines).items():
        default_value = getattr(LinesConfig(), key)
        if value != default_value:
            setattr(merged.lines, key, value)

    # Merge dimensions config
    for key, value in asdict(override.dimensions).items():
        default_value = getattr(DimensionsConfig(), key)
        if value != default_value:
            setattr(merged.dimensions, key, value)

    # Merge title_block config - always override non-empty strings
    for key, value in asdict(override.title_block).items():
        if value:  # Non-empty/non-None
            setattr(merged.title_block, key, value)

    # Merge output config
    for key, value in asdict(override.output).items():
        default_value = getattr(OutputConfig(), key)
        if value != default_value:
            setattr(merged.output, key, value)

    return merged


def apply_config_to_globals(config: ProjectConfig) -> None:
    """Apply configuration to global constants in config.py.

    This modifies the stl_drawing.config module in-place.

    Args:
        config: Configuration to apply
    """
    from stl_drawing import config as cfg

    # Lines
    if config.lines.sharp_angle_deg != 10.0:
        cfg.SHARP_ANGLE_DEGREES = config.lines.sharp_angle_deg
        import numpy as np
        cfg.SHARP_ANGLE_COS = np.cos(np.radians(config.lines.sharp_angle_deg))

    cfg.ENABLE_MERGE = config.lines.enable_merge
    cfg.ENABLE_PRIORITY = config.lines.enable_priority
    cfg.ENABLE_SHARP_EDGES = config.lines.enable_sharp_edges

    # Drawing layout
    cfg.VIEW_SPACING_MM = config.drawing.view_spacing_mm
    cfg.MIN_MAIN_VIEW_MM = config.drawing.min_main_view_mm

    # Dimensions
    cfg.DIM_ARROW_LENGTH = config.dimensions.arrow_length
    cfg.DIM_ARROW_WIDTH = config.dimensions.arrow_width
    cfg.DIM_TEXT_HEIGHT = config.dimensions.text_height
    cfg.DIM_TEXT_GAP = config.dimensions.text_gap
    cfg.DIM_FIRST_OFFSET = config.dimensions.first_offset
    cfg.DIM_NEXT_OFFSET = config.dimensions.next_offset
    cfg.DIM_EXTENSION_OVERSHOOT = config.dimensions.extension_overshoot
    cfg.DIM_EXTENSION_GAP = config.dimensions.extension_gap
    cfg.DIM_MIN_DISPLAYABLE = config.dimensions.min_displayable
    cfg.DIM_FONT_FAMILY = config.dimensions.font_family

    logger.debug("Applied project config to global constants")


def create_sample_config(path: Union[str, Path] = CONFIG_FILENAME) -> None:
    """Create a sample configuration file with documentation.

    Args:
        path: Output file path (default: .eskd.json)
    """
    sample = {
        "_comment": "STL to ESKD Drawing Generator configuration",
        "_version": "1.0",
        "drawing": {
            "_comment": "Drawing format and scale settings",
            "format": "auto",
            "scale": None,
            "orientation": "auto",
            "view_spacing_mm": 30.0,
            "min_main_view_mm": 40.0,
        },
        "lines": {
            "_comment": "Line style settings per GOST 2.303-68",
            "stroke_width": None,
            "sharp_angle_deg": 10.0,
            "enable_merge": True,
            "enable_priority": True,
            "enable_sharp_edges": True,
        },
        "dimensions": {
            "_comment": "Dimension settings per GOST 2.307-2011",
            "arrow_length": 2.5,
            "arrow_width": 0.8,
            "text_height": 3.5,
            "font_family": "ISOCPEUR",
        },
        "title_block": {
            "_comment": "Title block (stamp) fields per GOST 2.104",
            "organization": "",
            "designer": "",
            "document_name": "",
            "document_number": "",
            "material": "",
        },
        "output": {
            "_comment": "Output file settings",
            "formats": ["svg"],
            "prefix": "",
            "output_dir": "",
        },
        "hatching": {
            "_comment": "Section hatching per GOST 2.306-68",
            "default_material": "metal",
            "spacing_mm": 2.0,
            "angle_deg": 45.0,
        },
    }

    path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    logger.info("Sample configuration created: %s", path)
