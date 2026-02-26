"""
Batch processing for STL to ESKD drawing conversion.

Provides:
- Folder-based batch conversion (STL → SVG)
- Progress tracking and reporting
- Parallel processing support
- Error handling and logging

Usage:
    from stl_drawing.batch import batch_convert, BatchResult

    results = batch_convert(
        input_dir="./models",
        output_dir="./drawings",
        pattern="*.stl",
        parallel=True,
    )
    print(results.summary())
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from stl_drawing.project_config import ProjectConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of a single file conversion."""
    input_path: Path
    output_path: Optional[Path] = None
    success: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def status(self) -> str:
        """Get status string."""
        return "OK" if self.success else "FAILED"


@dataclass
class BatchResult:
    """Result of batch conversion."""
    results: List[ConversionResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def total(self) -> int:
        """Total number of files processed."""
        return len(self.results)

    @property
    def successful(self) -> int:
        """Number of successful conversions."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        """Number of failed conversions."""
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.successful / self.total

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Batch Conversion Summary",
            "=" * 40,
            f"Total files:     {self.total}",
            f"Successful:      {self.successful}",
            f"Failed:          {self.failed}",
            f"Success rate:    {self.success_rate:.1f}%",
            f"Total time:      {self.total_duration_seconds:.1f}s",
            "",
        ]

        if self.failed > 0:
            lines.append("Failed files:")
            for r in self.results:
                if not r.success:
                    lines.append(f"  - {r.input_path.name}: {r.error}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'total': self.total,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': self.success_rate,
            'total_duration_seconds': self.total_duration_seconds,
            'results': [
                {
                    'input': str(r.input_path),
                    'output': str(r.output_path) if r.output_path else None,
                    'success': r.success,
                    'error': r.error,
                    'duration': r.duration_seconds,
                }
                for r in self.results
            ],
        }


def find_stl_files(
    input_dir: Union[str, Path],
    pattern: str = "*.stl",
    recursive: bool = False,
) -> List[Path]:
    """Find STL files in directory.

    Args:
        input_dir: Directory to search
        pattern: Glob pattern for STL files
        recursive: Search subdirectories if True

    Returns:
        List of STL file paths
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    if recursive:
        files = list(input_dir.rglob(pattern))
    else:
        files = list(input_dir.glob(pattern))

    # Also check for uppercase extension
    pattern_upper = pattern.replace('.stl', '.STL')
    if recursive:
        files.extend(input_dir.rglob(pattern_upper))
    else:
        files.extend(input_dir.glob(pattern_upper))

    # Remove duplicates and sort
    files = sorted(set(files))

    logger.info("Found %d STL files in %s", len(files), input_dir)
    return files


def convert_single_file(
    input_path: Path,
    output_dir: Path,
    config: Optional[ProjectConfig] = None,
    output_prefix: str = "",
    output_suffix: str = "",
) -> ConversionResult:
    """Convert a single STL file to SVG.

    Args:
        input_path: Path to STL file
        output_dir: Output directory for SVG
        config: Project configuration
        output_prefix: Prefix for output filename
        output_suffix: Suffix for output filename

    Returns:
        ConversionResult with status and details
    """
    start_time = time.perf_counter()

    # Determine output path
    stem = input_path.stem
    output_name = f"{output_prefix}{stem}{output_suffix}.svg"
    output_path = output_dir / output_name

    result = ConversionResult(input_path=input_path)

    try:
        # Import here to avoid circular imports and allow multiprocessing
        from main import run_pipeline

        # Get title block info from config
        designation = ""
        part_name = stem
        org_name = ""
        surname = ""

        if config:
            designation = config.title_block.document_number
            if config.title_block.document_name:
                part_name = config.title_block.document_name
            org_name = config.title_block.organization
            surname = config.title_block.designer

        # Run conversion
        run_pipeline(
            stl_path=str(input_path),
            output_svg=str(output_path),
            designation=designation,
            part_name=part_name,
            org_name=org_name,
            surname=surname,
        )

        result.success = True
        result.output_path = output_path

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error("Failed to convert %s: %s", input_path.name, e)

    result.duration_seconds = time.perf_counter() - start_time
    return result


def _worker_convert(args: tuple) -> ConversionResult:
    """Worker function for parallel conversion."""
    input_path, output_dir, config_dict, prefix, suffix = args

    # Reconstruct config from dict (for pickling in multiprocessing)
    config = None
    if config_dict:
        config = ProjectConfig.from_dict(config_dict)

    return convert_single_file(
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        config=config,
        output_prefix=prefix,
        output_suffix=suffix,
    )


def batch_convert(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.stl",
    recursive: bool = False,
    config: Optional[ProjectConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    output_prefix: str = "",
    output_suffix: str = "",
    progress_callback: Optional[Callable[[int, int, ConversionResult], None]] = None,
) -> BatchResult:
    """Batch convert STL files to SVG drawings.

    Args:
        input_dir: Directory containing STL files
        output_dir: Output directory (default: same as input)
        pattern: Glob pattern for STL files
        recursive: Search subdirectories
        config: Project configuration
        config_path: Path to .eskd.json config file
        parallel: Use parallel processing
        max_workers: Maximum parallel workers (None = CPU count)
        output_prefix: Prefix for output filenames
        output_suffix: Suffix for output filenames
        progress_callback: Called after each file: (current, total, result)

    Returns:
        BatchResult with conversion statistics

    Example:
        results = batch_convert(
            input_dir="./models",
            output_dir="./drawings",
            parallel=True,
        )
        print(f"Converted {results.successful}/{results.total} files")
    """
    start_time = time.perf_counter()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config is None and config_path:
        config = load_config(explicit_config=config_path)
    elif config is None:
        # Try to find config in input directory
        config = load_config(stl_path=input_dir / "dummy.stl")

    # Find STL files
    stl_files = find_stl_files(input_dir, pattern, recursive)

    if not stl_files:
        logger.warning("No STL files found in %s", input_dir)
        return BatchResult(total_duration_seconds=time.perf_counter() - start_time)

    logger.info(
        "Starting batch conversion: %d files, parallel=%s",
        len(stl_files), parallel
    )

    results: List[ConversionResult] = []
    config_dict = config.to_dict() if config else None

    if parallel:
        # Parallel processing with ThreadPoolExecutor
        # (ProcessPoolExecutor has issues with imports)
        work_items = [
            (str(f), str(output_dir), config_dict, output_prefix, output_suffix)
            for f in stl_files
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker_convert, item): item[0]
                for item in work_items
            }

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)

                if progress_callback:
                    progress_callback(i, len(stl_files), result)

                status = "OK" if result.success else "FAILED"
                logger.info(
                    "[%d/%d] %s: %s (%.1fs)",
                    i, len(stl_files), result.input_path.name,
                    status, result.duration_seconds
                )
    else:
        # Sequential processing
        for i, stl_file in enumerate(stl_files, 1):
            result = convert_single_file(
                input_path=stl_file,
                output_dir=output_dir,
                config=config,
                output_prefix=output_prefix,
                output_suffix=output_suffix,
            )
            results.append(result)

            if progress_callback:
                progress_callback(i, len(stl_files), result)

            status = "OK" if result.success else "FAILED"
            logger.info(
                "[%d/%d] %s: %s (%.1fs)",
                i, len(stl_files), result.input_path.name,
                status, result.duration_seconds
            )

    batch_result = BatchResult(
        results=results,
        total_duration_seconds=time.perf_counter() - start_time,
    )

    logger.info(
        "Batch conversion complete: %d/%d successful (%.1f%%) in %.1fs",
        batch_result.successful, batch_result.total,
        batch_result.success_rate, batch_result.total_duration_seconds
    )

    return batch_result


def batch_convert_cli():
    """CLI entry point for batch conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch convert STL files to ESKD SVG drawings"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing STL files"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*.stl",
        help="File pattern (default: *.stl)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search subdirectories"
    )
    parser.add_argument(
        "-c", "--config",
        dest="config_path",
        help="Path to .eskd.json config file"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        dest="max_workers",
        help="Maximum parallel jobs"
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Output filename prefix"
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Output filename suffix"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        result = batch_convert(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            config_path=args.config_path,
            parallel=args.parallel,
            max_workers=args.max_workers,
            output_prefix=args.prefix,
            output_suffix=args.suffix,
        )

        print("\n" + result.summary())

        return 0 if result.failed == 0 else 1

    except Exception as e:
        logger.error("Batch conversion failed: %s", e)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(batch_convert_cli())
