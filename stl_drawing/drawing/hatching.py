"""
Hatching parameters and rendering according to GOST 2.306-68.

GOST 2.306-68 "Designations of graphic materials and rules for their
indication in drawings" defines:
- Hatching patterns for different materials (metal, plastic, wood, etc.)
- Hatching line spacing: 1-10mm depending on area size
- Hatching angle: 45° for general materials, variations for adjacent parts
- Line thickness: thin continuous line (S/3 to S/2)

Standard hatching types:
- Metal: 45° parallel lines
- Plastic/rubber: 45° with double lines at edges
- Glass: 45° with diagonal groups
- Wood: along grain pattern
- Concrete: dots with lines
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MaterialType(Enum):
    """Material types for hatching per GOST 2.306-68."""
    METAL = "metal"                    # Metals and hard alloys
    PLASTIC = "plastic"                # Plastics, rubber, leather
    GLASS = "glass"                    # Glass, ceramics
    WOOD_ALONG = "wood_along"          # Wood (along grain)
    WOOD_CROSS = "wood_cross"          # Wood (cross grain)
    CONCRETE = "concrete"              # Concrete, reinforced concrete
    BRICK = "brick"                    # Brick, ceramic blocks
    SOIL = "soil"                      # Natural soil
    LIQUID = "liquid"                  # Liquids
    INSULATION = "insulation"          # Thermal/sound insulation


@dataclass
class HatchPattern:
    """Hatching pattern definition.

    Attributes:
        material: Material type
        angle_deg: Primary hatching angle in degrees
        spacing_mm: Distance between hatch lines in mm
        line_width_mm: Hatch line width in mm
        double_line: Use double lines at edges (plastics)
        secondary_angle: Secondary angle for cross-hatching
    """
    material: MaterialType
    angle_deg: float = 45.0
    spacing_mm: float = 2.0
    line_width_mm: float = 0.25
    double_line: bool = False
    secondary_angle: Optional[float] = None


# Standard patterns per GOST 2.306-68
STANDARD_PATTERNS = {
    MaterialType.METAL: HatchPattern(
        material=MaterialType.METAL,
        angle_deg=45.0,
        spacing_mm=2.0,
    ),
    MaterialType.PLASTIC: HatchPattern(
        material=MaterialType.PLASTIC,
        angle_deg=45.0,
        spacing_mm=2.0,
        double_line=True,
    ),
    MaterialType.GLASS: HatchPattern(
        material=MaterialType.GLASS,
        angle_deg=45.0,
        spacing_mm=3.0,
    ),
    MaterialType.WOOD_ALONG: HatchPattern(
        material=MaterialType.WOOD_ALONG,
        angle_deg=0.0,  # Along grain
        spacing_mm=2.0,
    ),
    MaterialType.CONCRETE: HatchPattern(
        material=MaterialType.CONCRETE,
        angle_deg=45.0,
        spacing_mm=3.0,
        secondary_angle=135.0,  # Cross-hatch
    ),
}


def calculate_hatch_spacing(
    section_area_mm2: float,
    min_spacing: float = 1.0,
    max_spacing: float = 10.0,
) -> float:
    """Calculate optimal hatching spacing based on section area.

    GOST 2.306-68 recommends:
    - Small sections: 1-2mm spacing
    - Medium sections: 2-5mm spacing
    - Large sections: 5-10mm spacing

    Args:
        section_area_mm2: Section area in mm² (on paper)
        min_spacing: Minimum spacing (default 1mm)
        max_spacing: Maximum spacing (default 10mm)

    Returns:
        Recommended spacing in mm
    """
    # Heuristic: spacing proportional to sqrt(area)
    # Small area (<100mm²): ~1-2mm
    # Medium area (100-1000mm²): ~2-5mm
    # Large area (>1000mm²): ~5-10mm

    if section_area_mm2 < 100:
        spacing = 1.5
    elif section_area_mm2 < 500:
        spacing = 2.0
    elif section_area_mm2 < 2000:
        spacing = 3.0
    elif section_area_mm2 < 5000:
        spacing = 5.0
    else:
        spacing = min(max_spacing, 2.0 + math.sqrt(section_area_mm2) / 50)

    return max(min_spacing, min(max_spacing, spacing))


def calculate_hatch_angle(
    part_index: int = 0,
    base_angle: float = 45.0,
) -> float:
    """Calculate hatching angle to differentiate adjacent parts.

    GOST 2.306-68 recommends varying angle for adjacent parts in section:
    - Part 0: 45°
    - Part 1: 135° (or -45°)
    - Part 2: 30°
    - Part 3: 60°

    Args:
        part_index: Index of part in assembly (0-based)
        base_angle: Base angle for first part

    Returns:
        Hatching angle in degrees
    """
    angle_variations = [0, 90, -15, 15, -30, 30]
    variation_idx = part_index % len(angle_variations)
    return base_angle + angle_variations[variation_idx]


@dataclass
class HatchLine:
    """Single hatch line segment."""
    start: Tuple[float, float]
    end: Tuple[float, float]


def generate_hatch_lines(
    polygon: List[Tuple[float, float]],
    angle_deg: float = 45.0,
    spacing_mm: float = 2.0,
    offset: float = 0.0,
) -> List[HatchLine]:
    """Generate hatching lines for a polygon.

    Creates parallel lines at specified angle that fill the polygon area.

    Args:
        polygon: List of (x, y) vertices defining closed polygon
        angle_deg: Hatching angle in degrees (0=horizontal)
        spacing_mm: Distance between lines in mm
        offset: Offset from first line (for pattern variation)

    Returns:
        List of HatchLine segments clipped to polygon
    """
    if len(polygon) < 3:
        return []

    # Convert to numpy for easier manipulation
    pts = np.array(polygon, dtype=np.float64)

    # Get bounding box
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    bbox_diag = np.linalg.norm(max_pt - min_pt)

    if bbox_diag < 1e-6:
        return []

    # Angle in radians
    angle_rad = math.radians(angle_deg)

    # Direction along hatch lines and perpendicular
    hatch_dir = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    perp_dir = np.array([-math.sin(angle_rad), math.cos(angle_rad)])

    # Project polygon to perpendicular direction to find extent
    perp_proj = pts @ perp_dir
    perp_min = perp_proj.min()
    perp_max = perp_proj.max()

    # Generate lines
    lines = []
    d = perp_min + offset
    while d <= perp_max:
        # Line: point + t * hatch_dir where (point - origin) · perp_dir = d
        # Create a long line segment
        line_origin = d * perp_dir
        p1 = line_origin - bbox_diag * hatch_dir
        p2 = line_origin + bbox_diag * hatch_dir

        # Clip line to polygon
        clipped = _clip_line_to_polygon(p1, p2, polygon)
        for seg in clipped:
            lines.append(HatchLine(start=seg[0], end=seg[1]))

        d += spacing_mm

    logger.debug(
        "Generated %d hatch lines at %.1f° with %.1fmm spacing",
        len(lines), angle_deg, spacing_mm
    )

    return lines


def _clip_line_to_polygon(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    polygon: List[Tuple[float, float]],
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Clip a line segment to a polygon boundary.

    Uses ray casting to find intersection points, then returns
    segments that are inside the polygon.

    Args:
        p1, p2: Line endpoints
        polygon: Closed polygon vertices

    Returns:
        List of (start, end) tuples for segments inside polygon
    """
    # Find all intersections with polygon edges
    n = len(polygon)
    intersections = []

    line_dir = p2 - p1
    line_len = np.linalg.norm(line_dir)
    if line_len < 1e-10:
        return []
    line_dir = line_dir / line_len

    for i in range(n):
        e1 = np.array(polygon[i])
        e2 = np.array(polygon[(i + 1) % n])

        # Find intersection using parametric form
        edge_dir = e2 - e1
        denom = line_dir[0] * edge_dir[1] - line_dir[1] * edge_dir[0]

        if abs(denom) < 1e-10:
            continue  # Parallel

        diff = e1 - p1
        t = (diff[0] * edge_dir[1] - diff[1] * edge_dir[0]) / denom
        s = (diff[0] * line_dir[1] - diff[1] * line_dir[0]) / denom

        # Check if intersection is within both segments
        if 0 <= t <= line_len and 0 <= s <= 1:
            intersections.append(t)

    if len(intersections) < 2:
        return []

    # Sort intersections
    intersections.sort()

    # Create segments (every pair of intersections)
    segments = []
    for i in range(0, len(intersections) - 1, 2):
        t1, t2 = intersections[i], intersections[i + 1]
        start = p1 + t1 * line_dir
        end = p1 + t2 * line_dir
        segments.append(
            ((float(start[0]), float(start[1])),
             (float(end[0]), float(end[1])))
        )

    return segments


def get_hatch_pattern(
    material: MaterialType = MaterialType.METAL,
    scale: float = 1.0,
    stroke_width: float = 0.5,
) -> HatchPattern:
    """Get hatching pattern for material type, adjusted for scale.

    Args:
        material: Material type
        scale: Drawing scale (e.g., 0.5 for 1:2)
        stroke_width: Base line thickness S in mm

    Returns:
        HatchPattern with adjusted parameters
    """
    base = STANDARD_PATTERNS.get(material, STANDARD_PATTERNS[MaterialType.METAL])

    # Adjust spacing for scale (larger scale = wider spacing)
    spacing = base.spacing_mm
    if scale > 1:
        spacing = base.spacing_mm * min(2.0, scale)
    elif scale < 0.5:
        spacing = max(1.0, base.spacing_mm * 0.75)

    # Line width: S/3 to S/2 per GOST 2.303-68
    line_width = stroke_width * 0.4

    return HatchPattern(
        material=base.material,
        angle_deg=base.angle_deg,
        spacing_mm=spacing,
        line_width_mm=line_width,
        double_line=base.double_line,
        secondary_angle=base.secondary_angle,
    )


def render_hatch_to_svg(
    dwg,  # svgwrite.Drawing
    polygon: List[Tuple[float, float]],
    pattern: HatchPattern,
    transform_x: float = 0.0,
    transform_y: float = 0.0,
) -> 'svgwrite.container.Group':
    """Render hatching pattern to SVG group.

    Args:
        dwg: svgwrite.Drawing instance
        polygon: Polygon vertices in mm (paper coordinates)
        pattern: Hatching pattern to use
        transform_x, transform_y: Translation offset

    Returns:
        SVG group containing hatch lines
    """
    group = dwg.g()
    group['data-hatch-material'] = pattern.material.value

    # Apply transform to polygon
    transformed_poly = [
        (x + transform_x, y + transform_y)
        for x, y in polygon
    ]

    # Generate primary hatch lines
    lines = generate_hatch_lines(
        transformed_poly,
        angle_deg=pattern.angle_deg,
        spacing_mm=pattern.spacing_mm,
    )

    style = {
        'stroke': 'black',
        'stroke_width': f'{pattern.line_width_mm}mm',
        'stroke_linecap': 'butt',
    }

    for hline in lines:
        line = dwg.line(start=hline.start, end=hline.end, **style)
        group.add(line)

    # Add secondary hatching for cross-hatch patterns
    if pattern.secondary_angle is not None:
        secondary_lines = generate_hatch_lines(
            transformed_poly,
            angle_deg=pattern.secondary_angle,
            spacing_mm=pattern.spacing_mm,
            offset=pattern.spacing_mm / 2,  # Offset for visual clarity
        )
        for hline in secondary_lines:
            line = dwg.line(start=hline.start, end=hline.end, **style)
            group.add(line)

    logger.debug(
        "Rendered %d hatch lines for %s material",
        len(lines), pattern.material.value
    )

    return group
