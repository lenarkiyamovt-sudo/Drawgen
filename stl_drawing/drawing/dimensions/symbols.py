"""
Dimension symbols and formatting according to GOST 2.307-2011.

GOST 2.307-2011 "Dimension and tolerance notation" defines:
- Prefix symbols: diameter (⌀), radius (R), square (□), sphere (S⌀, SR)
- Tolerance symbols: ± for symmetric, +/- for asymmetric
- Special notations: chamfer (C×), thread (M×, G×), taper

Symbols are defined both as Unicode characters and as ASCII fallbacks
for systems that don't support Unicode rendering.

Unicode dimension symbols:
    ⌀ (U+2300) - diameter
    ∅ (U+2205) - alternative diameter (set emptiness)
    R - radius (plain letter)
    □ (U+25A1) - square dimension
    ± (U+00B1) - plus-minus tolerance
    × (U+00D7) - multiplication sign (for chamfers)
    ° (U+00B0) - degree sign
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class DimensionType(Enum):
    """Types of dimensions as defined in GOST 2.307-2011."""
    LINEAR = "linear"               # Linear distance
    DIAMETER = "diameter"           # Diameter (⌀)
    RADIUS = "radius"               # Radius (R)
    SQUARE = "square"               # Square dimension (□)
    SPHERE_DIAMETER = "sphere_dia"  # Sphere diameter (S⌀)
    SPHERE_RADIUS = "sphere_rad"    # Sphere radius (SR)
    CHAMFER = "chamfer"             # Chamfer (C×45°)
    THREAD = "thread"               # Thread (M, G)
    ANGLE = "angle"                 # Angular dimension (°)
    ARC_LENGTH = "arc_length"       # Arc length (⌒)


# Unicode symbols for dimension prefixes
SYMBOLS = {
    'diameter': '\u2300',       # ⌀
    'diameter_alt': '\u2205',   # ∅ (sometimes used as alternative)
    'radius': 'R',              # R (plain letter per GOST)
    'square': '\u25A1',         # □
    'sphere': 'S',              # S prefix for spheres
    'plus_minus': '\u00B1',     # ±
    'multiply': '\u00D7',       # × (for chamfers)
    'degree': '\u00B0',         # °
    'arc': '\u2312',            # ⌒ (arc)
}

# ASCII fallbacks for systems without Unicode support
ASCII_FALLBACKS = {
    'diameter': 'dia',
    'diameter_alt': 'D',
    'radius': 'R',
    'square': 'SQ',
    'sphere': 'S',
    'plus_minus': '+/-',
    'multiply': 'x',
    'degree': 'deg',
    'arc': 'arc',
}


@dataclass
class ToleranceValue:
    """Tolerance specification for a dimension.

    GOST 2.307-2011 defines tolerance notation:
    - Symmetric: 50±0.1 or 50 +/-0.1
    - Asymmetric: 50 +0.02/-0.01 (upper first, then lower)
    - Single-sided: 50 +0.02 or 50 -0.01
    """
    upper: Optional[float] = None   # Upper deviation (positive = +)
    lower: Optional[float] = None   # Lower deviation (negative = -)

    @property
    def is_symmetric(self) -> bool:
        """Check if tolerance is symmetric (±)."""
        if self.upper is None or self.lower is None:
            return False
        return abs(self.upper + self.lower) < 1e-9

    @property
    def is_single_sided(self) -> bool:
        """Check if only one deviation is specified."""
        return (self.upper is not None) != (self.lower is not None)


def format_dimension(
    value_mm: float,
    dim_type: DimensionType = DimensionType.LINEAR,
    tolerance: Optional[ToleranceValue] = None,
    decimals: int = 0,
    use_unicode: bool = True,
) -> str:
    """Format dimension value with appropriate prefix and tolerance.

    GOST 2.307-2011 formatting rules:
    - Diameter: ⌀50, ⌀25.5
    - Radius: R10, R5.5
    - Square: □30
    - Sphere: S⌀20, SR10
    - Chamfer: 2×45°
    - Linear: 100, 50.5

    Args:
        value_mm: Dimension value in millimeters
        dim_type: Type of dimension
        tolerance: Optional tolerance specification
        decimals: Number of decimal places (0 = integer)
        use_unicode: Use Unicode symbols (True) or ASCII fallbacks (False)

    Returns:
        Formatted dimension string
    """
    symbols = SYMBOLS if use_unicode else ASCII_FALLBACKS

    # Format numeric value
    if decimals == 0 and value_mm == int(value_mm):
        num_str = str(int(value_mm))
    else:
        num_str = f"{value_mm:.{decimals}f}"

    # Add prefix based on dimension type
    prefix = ""
    suffix = ""

    if dim_type == DimensionType.DIAMETER:
        prefix = symbols['diameter']
    elif dim_type == DimensionType.RADIUS:
        prefix = symbols['radius']
    elif dim_type == DimensionType.SQUARE:
        prefix = symbols['square']
    elif dim_type == DimensionType.SPHERE_DIAMETER:
        prefix = symbols['sphere'] + symbols['diameter']
    elif dim_type == DimensionType.SPHERE_RADIUS:
        prefix = symbols['sphere'] + symbols['radius']
    elif dim_type == DimensionType.ANGLE:
        suffix = symbols['degree']
    elif dim_type == DimensionType.CHAMFER:
        # Chamfer format: 2×45° (value is chamfer size, 45° is typical)
        suffix = symbols['multiply'] + "45" + symbols['degree']

    # Format tolerance if specified
    tol_str = ""
    if tolerance is not None:
        tol_str = format_tolerance(tolerance, use_unicode)

    return f"{prefix}{num_str}{tol_str}{suffix}"


def format_tolerance(
    tolerance: ToleranceValue,
    use_unicode: bool = True,
) -> str:
    """Format tolerance value according to GOST 2.307-2011.

    Examples:
        - Symmetric: ±0.1
        - Asymmetric: +0.02/-0.01
        - Upper only: +0.02
        - Lower only: -0.01

    Args:
        tolerance: Tolerance specification
        use_unicode: Use Unicode symbols

    Returns:
        Formatted tolerance string
    """
    symbols = SYMBOLS if use_unicode else ASCII_FALLBACKS

    if tolerance.upper is None and tolerance.lower is None:
        return ""

    if tolerance.is_symmetric and tolerance.upper is not None:
        # Symmetric tolerance: ±0.1
        return f"{symbols['plus_minus']}{abs(tolerance.upper):.2g}"

    # Asymmetric tolerance
    parts = []
    if tolerance.upper is not None:
        sign = "+" if tolerance.upper >= 0 else ""
        parts.append(f"{sign}{tolerance.upper:.2g}")
    if tolerance.lower is not None:
        sign = "+" if tolerance.lower >= 0 else ""
        parts.append(f"{sign}{tolerance.lower:.2g}")

    return "/".join(parts)


def format_chamfer(size_mm: float, angle_deg: float = 45.0,
                   use_unicode: bool = True) -> str:
    """Format chamfer dimension per GOST 2.307-2011.

    Standard chamfer notation: C×angle° (e.g., 2×45°)
    For 45° chamfers, often shown as just the size.

    Args:
        size_mm: Chamfer size in mm
        angle_deg: Chamfer angle in degrees (default 45°)
        use_unicode: Use Unicode symbols

    Returns:
        Formatted chamfer string
    """
    symbols = SYMBOLS if use_unicode else ASCII_FALLBACKS

    # Format size (integer if whole number)
    if size_mm == int(size_mm):
        size_str = str(int(size_mm))
    else:
        size_str = f"{size_mm:.1f}"

    # Format angle
    if angle_deg == int(angle_deg):
        angle_str = str(int(angle_deg))
    else:
        angle_str = f"{angle_deg:.1f}"

    return f"{size_str}{symbols['multiply']}{angle_str}{symbols['degree']}"


def format_thread(
    diameter_mm: float,
    pitch_mm: Optional[float] = None,
    thread_type: str = "M",
    use_unicode: bool = True,
) -> str:
    """Format thread dimension per GOST standards.

    Thread notation:
    - Metric: M12, M12×1.5 (with non-standard pitch)
    - Pipe: G1/2, G3/4
    - Trapezoidal: Tr20×4

    Args:
        diameter_mm: Nominal diameter in mm
        pitch_mm: Thread pitch (None for standard pitch)
        thread_type: Type prefix ("M", "G", "Tr", etc.)
        use_unicode: Use Unicode symbols

    Returns:
        Formatted thread string
    """
    symbols = SYMBOLS if use_unicode else ASCII_FALLBACKS

    # Format diameter (integer if whole number)
    if diameter_mm == int(diameter_mm):
        dia_str = str(int(diameter_mm))
    else:
        dia_str = f"{diameter_mm:.1f}"

    result = f"{thread_type}{dia_str}"

    # Add pitch if non-standard
    if pitch_mm is not None:
        if pitch_mm == int(pitch_mm):
            pitch_str = str(int(pitch_mm))
        else:
            pitch_str = f"{pitch_mm:.2g}"
        result += f"{symbols['multiply']}{pitch_str}"

    return result


def get_dimension_prefix(dim_type: DimensionType,
                         use_unicode: bool = True) -> str:
    """Get the prefix symbol for a dimension type.

    Args:
        dim_type: Type of dimension
        use_unicode: Use Unicode symbols

    Returns:
        Prefix string (may be empty for linear dimensions)
    """
    symbols = SYMBOLS if use_unicode else ASCII_FALLBACKS

    prefixes = {
        DimensionType.LINEAR: "",
        DimensionType.DIAMETER: symbols['diameter'],
        DimensionType.RADIUS: symbols['radius'],
        DimensionType.SQUARE: symbols['square'],
        DimensionType.SPHERE_DIAMETER: symbols['sphere'] + symbols['diameter'],
        DimensionType.SPHERE_RADIUS: symbols['sphere'] + symbols['radius'],
        DimensionType.CHAMFER: "",
        DimensionType.THREAD: "",
        DimensionType.ANGLE: "",
        DimensionType.ARC_LENGTH: symbols['arc'],
    }

    return prefixes.get(dim_type, "")


def detect_dimension_type(value_mm: float, context: str = "") -> DimensionType:
    """Attempt to detect dimension type from value and context.

    This is a heuristic function that tries to determine the appropriate
    dimension type based on the numeric value and optional context hints.

    Args:
        value_mm: Dimension value in mm
        context: Optional context string (e.g., "cylinder", "hole", "chamfer")

    Returns:
        Best guess for dimension type
    """
    context_lower = context.lower()

    if "diameter" in context_lower or "hole" in context_lower:
        return DimensionType.DIAMETER
    if "radius" in context_lower:
        return DimensionType.RADIUS
    if "chamfer" in context_lower:
        return DimensionType.CHAMFER
    if "thread" in context_lower:
        return DimensionType.THREAD
    if "angle" in context_lower:
        return DimensionType.ANGLE
    if "sphere" in context_lower:
        if value_mm < 20:  # Small spheres often shown as radius
            return DimensionType.SPHERE_RADIUS
        return DimensionType.SPHERE_DIAMETER

    return DimensionType.LINEAR
