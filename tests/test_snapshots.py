"""
Snapshot tests for STL-to-ESKD Drawing Generator.

Tests that generated SVG drawings match expected "golden" snapshots.
These tests detect regressions in the rendering pipeline.

Usage:
    # Run normally (compare against snapshots)
    pytest tests/test_snapshots.py -v

    # Update snapshots after intentional changes
    pytest tests/test_snapshots.py -v --update-snapshots

    # Skip snapshot tests in CI without reference files
    pytest tests/test_snapshots.py -v --ignore-missing-snapshots
"""

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import pytest

# Directory containing snapshot files
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


# ============================================================================
# SVG Analysis Utilities
# ============================================================================

def parse_svg_structure(svg_content: str) -> Dict:
    """Parse SVG and extract structural information for comparison.

    Extracts:
    - Element counts by type
    - View positions and sizes
    - Line counts and types
    - Text elements
    """
    # Remove XML declaration to avoid parsing issues
    svg_content = re.sub(r'^<\?xml[^?]*\?>\s*', '', svg_content)

    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError:
        return {'error': 'Failed to parse SVG'}

    # Namespace handling
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    def find_all(elem, tag):
        """Find all elements with or without namespace."""
        # Try with namespace
        found = elem.findall(f'.//svg:{tag}', ns)
        if not found:
            # Try without namespace
            found = elem.findall(f'.//{tag}')
        return found

    structure = {
        'groups': len(find_all(root, 'g')),
        'paths': len(find_all(root, 'path')),
        'lines': len(find_all(root, 'line')),
        'polylines': len(find_all(root, 'polyline')),
        'circles': len(find_all(root, 'circle')),
        'texts': len(find_all(root, 'text')),
        'rects': len(find_all(root, 'rect')),
    }

    # Get viewBox dimensions
    viewbox = root.get('viewBox', '')
    if viewbox:
        parts = viewbox.split()
        if len(parts) == 4:
            structure['viewbox_width'] = float(parts[2])
            structure['viewbox_height'] = float(parts[3])

    # Count elements with specific classes (if present)
    for elem in find_all(root, 'g'):
        class_attr = elem.get('class', '')
        if 'view-' in class_attr:
            structure.setdefault('views', []).append(class_attr)

    return structure


def extract_svg_metrics(svg_content: str) -> Dict:
    """Extract quantitative metrics from SVG for regression testing.

    These metrics help detect significant changes:
    - Total element count
    - Approximate line length (path d attribute length)
    - Text content hash
    """
    # Parse structure
    structure = parse_svg_structure(svg_content)

    # Calculate total elements
    total = sum(v for k, v in structure.items() if isinstance(v, int))

    # Approximate path complexity (d attribute lengths)
    path_complexity = 0
    for match in re.finditer(r'd="([^"]*)"', svg_content):
        path_complexity += len(match.group(1))

    # Hash of text content
    text_content = ''.join(re.findall(r'>([^<]+)</', svg_content))
    text_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]

    return {
        'total_elements': total,
        'path_complexity': path_complexity,
        'text_hash': text_hash,
        'structure': structure,
    }


def compare_svg_metrics(
    actual: Dict,
    expected: Dict,
    tolerance: float = 0.1,
) -> Tuple[bool, List[str]]:
    """Compare two SVG metric dictionaries.

    Args:
        actual: Metrics from generated SVG
        expected: Metrics from snapshot
        tolerance: Allowed relative difference (0.1 = 10%)

    Returns:
        (passed, list of difference descriptions)
    """
    differences = []

    # Compare total elements
    a_total = actual.get('total_elements', 0)
    e_total = expected.get('total_elements', 0)
    if e_total > 0:
        diff = abs(a_total - e_total) / e_total
        if diff > tolerance:
            differences.append(
                f"Total elements: {a_total} vs expected {e_total} "
                f"(diff: {diff*100:.1f}%)"
            )

    # Compare path complexity
    a_path = actual.get('path_complexity', 0)
    e_path = expected.get('path_complexity', 0)
    if e_path > 0:
        diff = abs(a_path - e_path) / e_path
        if diff > tolerance:
            differences.append(
                f"Path complexity: {a_path} vs expected {e_path} "
                f"(diff: {diff*100:.1f}%)"
            )

    # Compare structure counts
    a_struct = actual.get('structure', {})
    e_struct = expected.get('structure', {})
    for key in ['groups', 'paths', 'lines', 'texts', 'rects']:
        a_val = a_struct.get(key, 0)
        e_val = e_struct.get(key, 0)
        if e_val > 0:
            diff = abs(a_val - e_val) / max(e_val, 1)
            if diff > tolerance:
                differences.append(
                    f"{key}: {a_val} vs expected {e_val}"
                )

    return len(differences) == 0, differences


# ============================================================================
# Snapshot Management
# ============================================================================

def get_snapshot_path(name: str, ext: str = 'json') -> Path:
    """Get path to snapshot file."""
    return SNAPSHOTS_DIR / f"{name}.{ext}"


def load_snapshot(name: str) -> Optional[Dict]:
    """Load snapshot metrics from JSON file."""
    import json

    path = get_snapshot_path(name)
    if not path.exists():
        return None

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_snapshot(name: str, metrics: Dict) -> None:
    """Save snapshot metrics to JSON file."""
    import json

    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    path = get_snapshot_path(name)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_svg_snapshot(name: str, svg_content: str) -> None:
    """Save full SVG content as snapshot."""
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    path = get_snapshot_path(name, 'svg')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(svg_content)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def run_pipeline_and_read_svg():
    """Fixture that runs pipeline and returns SVG content."""
    def _run(stl_path: Path, **kwargs) -> str:
        from main import run_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.svg"

            run_pipeline(
                stl_path=str(stl_path),
                output_svg=str(output_path),
                **kwargs
            )

            assert output_path.exists(), f"SVG not generated: {output_path}"
            return output_path.read_text(encoding='utf-8')

    return _run


# ============================================================================
# Snapshot Tests
# ============================================================================

class TestFuelStlSnapshot:
    """Snapshot tests for fuel.stl model."""

    def test_fuel_svg_structure(
        self,
        fuel_stl_path: Path,
        run_pipeline_and_read_svg,
        update_snapshots: bool,
        ignore_missing_snapshots: bool,
    ):
        """Test fuel.stl SVG output matches snapshot."""
        snapshot_name = "fuel_stl"

        # Generate SVG
        svg_content = run_pipeline_and_read_svg(
            fuel_stl_path,
            designation="АБВГ.123456.001",
            part_name="Бак топливный",
            org_name="Test Org",
        )

        # Extract metrics
        metrics = extract_svg_metrics(svg_content)

        if update_snapshots:
            # Save new snapshot
            save_snapshot(snapshot_name, metrics)
            save_svg_snapshot(snapshot_name, svg_content)
            pytest.skip("Snapshot updated")
            return

        # Load expected
        expected = load_snapshot(snapshot_name)

        if expected is None:
            if ignore_missing_snapshots:
                pytest.skip(f"Snapshot not found: {snapshot_name}")
            # First run - create initial snapshot
            save_snapshot(snapshot_name, metrics)
            save_svg_snapshot(snapshot_name, svg_content)
            pytest.skip("Initial snapshot created")
            return

        # Compare
        passed, differences = compare_svg_metrics(metrics, expected)

        if not passed:
            msg = "SVG structure changed:\n" + "\n".join(f"  - {d}" for d in differences)
            pytest.fail(msg)

    def test_fuel_has_dimensions(
        self,
        fuel_stl_path: Path,
        run_pipeline_and_read_svg,
    ):
        """Test that fuel.stl drawing has dimensions."""
        svg_content = run_pipeline_and_read_svg(fuel_stl_path)
        structure = parse_svg_structure(svg_content)

        # Should have text elements (dimensions + title block)
        assert structure['texts'] > 5, "Expected dimension text elements"

    def test_fuel_has_multiple_views(
        self,
        fuel_stl_path: Path,
        run_pipeline_and_read_svg,
    ):
        """Test that fuel.stl drawing has multiple views."""
        svg_content = run_pipeline_and_read_svg(fuel_stl_path)
        structure = parse_svg_structure(svg_content)

        # Should have multiple groups (views)
        assert structure['groups'] >= 3, "Expected at least 3 view groups"


class TestShvellerStlSnapshot:
    """Snapshot tests for Shveller_16P.stl model."""

    def test_shveller_svg_structure(
        self,
        shveller_stl_path: Path,
        run_pipeline_and_read_svg,
        update_snapshots: bool,
        ignore_missing_snapshots: bool,
    ):
        """Test Shveller_16P.stl SVG output matches snapshot."""
        snapshot_name = "shveller_16p_stl"

        # Generate SVG
        svg_content = run_pipeline_and_read_svg(
            shveller_stl_path,
            designation="АБВГ.789012.002",
            part_name="Швеллер 16П",
        )

        # Extract metrics
        metrics = extract_svg_metrics(svg_content)

        if update_snapshots:
            save_snapshot(snapshot_name, metrics)
            save_svg_snapshot(snapshot_name, svg_content)
            pytest.skip("Snapshot updated")
            return

        expected = load_snapshot(snapshot_name)

        if expected is None:
            if ignore_missing_snapshots:
                pytest.skip(f"Snapshot not found: {snapshot_name}")
            save_snapshot(snapshot_name, metrics)
            save_svg_snapshot(snapshot_name, svg_content)
            pytest.skip("Initial snapshot created")
            return

        passed, differences = compare_svg_metrics(metrics, expected)

        if not passed:
            msg = "SVG structure changed:\n" + "\n".join(f"  - {d}" for d in differences)
            pytest.fail(msg)

    def test_shveller_valid_svg(
        self,
        shveller_stl_path: Path,
        run_pipeline_and_read_svg,
    ):
        """Test that Shveller SVG is valid XML."""
        svg_content = run_pipeline_and_read_svg(shveller_stl_path)

        # Should not have XML parsing errors
        structure = parse_svg_structure(svg_content)
        assert 'error' not in structure, f"SVG parsing error: {structure.get('error')}"

    def test_shveller_has_title_block(
        self,
        shveller_stl_path: Path,
        run_pipeline_and_read_svg,
    ):
        """Test that Shveller drawing has title block."""
        svg_content = run_pipeline_and_read_svg(shveller_stl_path)
        structure = parse_svg_structure(svg_content)

        # Title block should have rectangles
        assert structure['rects'] >= 10, "Expected title block rectangles"

        # Should have text elements
        assert structure['texts'] > 0, "Expected text in title block"


class TestSvgValidation:
    """General SVG validation tests."""

    @pytest.mark.parametrize("stl_fixture", ["fuel_stl_path", "shveller_stl_path"])
    def test_svg_no_empty_paths(
        self,
        stl_fixture: str,
        run_pipeline_and_read_svg,
        request,
    ):
        """Test that generated SVG has no empty path elements."""
        stl_path = request.getfixturevalue(stl_fixture)
        svg_content = run_pipeline_and_read_svg(stl_path)

        # Check for empty d="" attributes
        empty_paths = re.findall(r'd="\s*"', svg_content)
        assert len(empty_paths) == 0, f"Found {len(empty_paths)} empty paths"

    @pytest.mark.parametrize("stl_fixture", ["fuel_stl_path", "shveller_stl_path"])
    def test_svg_valid_coordinates(
        self,
        stl_fixture: str,
        run_pipeline_and_read_svg,
        request,
    ):
        """Test that SVG has no NaN or Inf coordinates."""
        stl_path = request.getfixturevalue(stl_fixture)
        svg_content = run_pipeline_and_read_svg(stl_path)

        # Check for invalid values
        assert 'NaN' not in svg_content, "Found NaN in SVG"
        assert 'Infinity' not in svg_content, "Found Infinity in SVG"
        assert 'inf' not in svg_content.lower(), "Found inf in SVG"


class TestDeterminism:
    """Tests for deterministic output."""

    def test_fuel_deterministic(
        self,
        fuel_stl_path: Path,
        run_pipeline_and_read_svg,
    ):
        """Test that running pipeline twice produces identical output."""
        svg1 = run_pipeline_and_read_svg(fuel_stl_path)
        svg2 = run_pipeline_and_read_svg(fuel_stl_path)

        metrics1 = extract_svg_metrics(svg1)
        metrics2 = extract_svg_metrics(svg2)

        # Structure should be identical
        assert metrics1['structure'] == metrics2['structure'], \
            "Non-deterministic output detected"
