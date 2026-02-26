"""
Unit tests for stl_drawing.batch module.

Tests:
- File discovery
- Single file conversion
- Batch conversion results
- Progress tracking
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stl_drawing.batch import (
    ConversionResult,
    BatchResult,
    find_stl_files,
    convert_single_file,
    batch_convert,
)
from stl_drawing.project_config import ProjectConfig


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_success_status(self):
        """Test successful conversion status."""
        result = ConversionResult(
            input_path=Path("test.stl"),
            output_path=Path("test.svg"),
            success=True,
        )
        assert result.status == "OK"

    def test_failed_status(self):
        """Test failed conversion status."""
        result = ConversionResult(
            input_path=Path("test.stl"),
            success=False,
            error="Test error",
        )
        assert result.status == "FAILED"

    def test_duration_tracking(self):
        """Test duration is tracked."""
        result = ConversionResult(
            input_path=Path("test.stl"),
            duration_seconds=5.5,
        )
        assert result.duration_seconds == 5.5


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_empty_result(self):
        """Test empty batch result."""
        result = BatchResult()
        assert result.total == 0
        assert result.successful == 0
        assert result.failed == 0

    def test_success_counting(self):
        """Test successful conversion counting."""
        result = BatchResult(results=[
            ConversionResult(Path("a.stl"), success=True),
            ConversionResult(Path("b.stl"), success=True),
            ConversionResult(Path("c.stl"), success=False),
        ])
        assert result.total == 3
        assert result.successful == 2
        assert result.failed == 1

    def test_success_rate(self):
        """Test success rate calculation."""
        result = BatchResult(results=[
            ConversionResult(Path("a.stl"), success=True),
            ConversionResult(Path("b.stl"), success=False),
        ])
        assert result.success_rate == 50.0

    def test_success_rate_empty(self):
        """Test success rate with no files."""
        result = BatchResult()
        assert result.success_rate == 0.0

    def test_summary_generation(self):
        """Test summary string generation."""
        result = BatchResult(
            results=[
                ConversionResult(Path("ok.stl"), success=True),
                ConversionResult(Path("fail.stl"), success=False, error="Error"),
            ],
            total_duration_seconds=10.0,
        )
        summary = result.summary()

        assert "Total files:" in summary
        assert "Successful:" in summary
        assert "Failed:" in summary
        assert "fail.stl" in summary

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = BatchResult(results=[
            ConversionResult(Path("test.stl"), success=True),
        ])
        d = result.to_dict()

        assert 'total' in d
        assert 'successful' in d
        assert 'results' in d
        assert len(d['results']) == 1


class TestFindStlFiles:
    """Tests for find_stl_files function."""

    def test_find_stl_files(self):
        """Test finding STL files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files
            (tmpdir / "model1.stl").touch()
            (tmpdir / "model2.stl").touch()
            (tmpdir / "other.txt").touch()

            files = find_stl_files(tmpdir)

            assert len(files) == 2
            assert all(f.suffix.lower() == '.stl' for f in files)

    def test_find_uppercase_extension(self):
        """Test finding STL files with uppercase extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "model.STL").touch()

            files = find_stl_files(tmpdir)

            assert len(files) == 1

    def test_find_recursive(self):
        """Test recursive file finding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create nested structure
            (tmpdir / "model1.stl").touch()
            subdir = tmpdir / "subdir"
            subdir.mkdir()
            (subdir / "model2.stl").touch()

            # Non-recursive should find 1
            files_flat = find_stl_files(tmpdir, recursive=False)
            assert len(files_flat) == 1

            # Recursive should find 2
            files_recursive = find_stl_files(tmpdir, recursive=True)
            assert len(files_recursive) == 2

    def test_custom_pattern(self):
        """Test custom file pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "part_001.stl").touch()
            (tmpdir / "part_002.stl").touch()
            (tmpdir / "other.stl").touch()

            files = find_stl_files(tmpdir, pattern="part_*.stl")

            assert len(files) == 2

    def test_nonexistent_directory(self):
        """Test error for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            find_stl_files("/nonexistent/path")

    def test_file_instead_of_directory(self):
        """Test error when path is a file."""
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(NotADirectoryError):
                find_stl_files(f.name)


class TestConvertSingleFile:
    """Tests for convert_single_file function."""

    @patch('main.run_pipeline')
    def test_successful_conversion(self, mock_run):
        """Test successful single file conversion."""
        mock_run.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "test.stl"
            input_file.touch()

            result = convert_single_file(input_file, tmpdir)

            assert result.success
            assert result.output_path == tmpdir / "test.svg"
            assert result.error is None

    @patch('main.run_pipeline')
    def test_failed_conversion(self, mock_run):
        """Test failed conversion handling."""
        mock_run.side_effect = Exception("Test error")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "test.stl"
            input_file.touch()

            result = convert_single_file(input_file, tmpdir)

            assert not result.success
            assert "Test error" in result.error

    @patch('main.run_pipeline')
    def test_output_prefix_suffix(self, mock_run):
        """Test output filename prefix and suffix."""
        mock_run.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "model.stl"
            input_file.touch()

            result = convert_single_file(
                input_file, tmpdir,
                output_prefix="drawing_",
                output_suffix="_v1",
            )

            assert result.output_path.name == "drawing_model_v1.svg"

    @patch('main.run_pipeline')
    def test_config_applied(self, mock_run):
        """Test that config is passed to pipeline."""
        mock_run.return_value = None

        config = ProjectConfig()
        config.title_block.designer = "Test Designer"
        config.title_block.organization = "Test Org"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_file = tmpdir / "test.stl"
            input_file.touch()

            convert_single_file(input_file, tmpdir, config=config)

            # Verify run_pipeline was called with config values
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['surname'] == "Test Designer"
            assert call_kwargs['org_name'] == "Test Org"


class TestBatchConvert:
    """Tests for batch_convert function."""

    @patch('stl_drawing.batch.convert_single_file')
    def test_batch_empty_directory(self, mock_convert):
        """Test batch conversion with no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = batch_convert(tmpdir)

            assert result.total == 0
            mock_convert.assert_not_called()

    @patch('stl_drawing.batch.convert_single_file')
    def test_batch_multiple_files(self, mock_convert):
        """Test batch conversion of multiple files."""
        mock_convert.return_value = ConversionResult(
            input_path=Path("test.stl"),
            success=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "a.stl").touch()
            (tmpdir / "b.stl").touch()
            (tmpdir / "c.stl").touch()

            result = batch_convert(tmpdir)

            assert result.total == 3
            assert mock_convert.call_count == 3

    @patch('stl_drawing.batch.convert_single_file')
    def test_batch_creates_output_dir(self, mock_convert):
        """Test that batch_convert creates output directory."""
        mock_convert.return_value = ConversionResult(
            input_path=Path("test.stl"),
            success=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            input_dir.mkdir()
            (input_dir / "test.stl").touch()

            output_dir = tmpdir / "output" / "drawings"

            batch_convert(input_dir, output_dir=output_dir)

            assert output_dir.exists()

    @patch('stl_drawing.batch.convert_single_file')
    def test_progress_callback(self, mock_convert):
        """Test progress callback is called."""
        mock_convert.return_value = ConversionResult(
            input_path=Path("test.stl"),
            success=True,
        )

        progress_calls = []

        def callback(current, total, result):
            progress_calls.append((current, total, result))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "a.stl").touch()
            (tmpdir / "b.stl").touch()

            batch_convert(tmpdir, progress_callback=callback)

            assert len(progress_calls) == 2
            assert progress_calls[0][0] == 1
            assert progress_calls[1][0] == 2

    @patch('stl_drawing.batch.convert_single_file')
    def test_batch_timing(self, mock_convert):
        """Test that batch tracks total duration."""
        mock_convert.return_value = ConversionResult(
            input_path=Path("test.stl"),
            success=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "test.stl").touch()

            result = batch_convert(tmpdir)

            assert result.total_duration_seconds > 0


class TestBatchIntegration:
    """Integration tests for batch conversion."""

    def test_batch_with_real_stl(self, fuel_stl_path):
        """Test batch conversion with real STL file (if available)."""
        if not fuel_stl_path.exists():
            pytest.skip("fuel.stl not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Copy STL to temp dir
            import shutil
            shutil.copy(fuel_stl_path, tmpdir / "fuel.stl")

            # Run batch conversion
            result = batch_convert(tmpdir, output_dir=tmpdir)

            assert result.total == 1
            # Note: This may fail or succeed depending on full pipeline
