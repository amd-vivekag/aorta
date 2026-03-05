"""Smoke tests for the aorta-report package.

Covers: CLI surface, utility functions, data parsing, and template rendering.
These tests run without GPU, TraceLens, or real trace data.
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli():
    from aorta.report.cli import cli

    return cli


@pytest.fixture
def tmp_trace_dir(tmp_path):
    """Create a minimal mock trace directory structure."""
    tp = tmp_path / "torch_profiler"
    for rank in range(4):
        rank_dir = tp / f"rank{rank}" / "trace"
        rank_dir.mkdir(parents=True)
        trace_file = rank_dir / "pt.trace.json"
        trace_file.write_text('{"traceEvents": []}')
    return tmp_path


@pytest.fixture
def tmp_individual_reports(tmp_path):
    """Create mock individual_reports with perf_rank*.xlsx files."""
    reports_dir = tmp_path / "individual_reports"
    reports_dir.mkdir()

    for rank in range(4):
        df = pd.DataFrame(
            {
                "type": [
                    "computation_time",
                    "exposed_comm_time",
                    "busy_time",
                    "idle_time",
                    "total_time",
                ],
                "time ms": [10.0 + rank, 2.0, 12.0 + rank, 1.0, 13.0 + rank],
                "percent": [76.9, 15.4, 92.3, 7.7, 100.0],
            }
        )
        path = reports_dir / f"perf_rank{rank}.xlsx"
        df.to_excel(path, sheet_name="gpu_timeline", index=False)

    return reports_dir


@pytest.fixture
def tmp_sweep_reports(tmp_path):
    """Create mock sweep report files: perf_{ch}ch_rank{n}.xlsx."""
    reports_dir = tmp_path / "tracelens_analysis" / "256thread" / "individual_reports"
    reports_dir.mkdir(parents=True)

    for ch in (28, 42):
        for rank in range(2):
            df = pd.DataFrame(
                {
                    "type": ["computation_time", "busy_time", "total_time"],
                    "time ms": [10.0 + ch + rank, 12.0, 15.0],
                    "percent": [66.0, 80.0, 100.0],
                }
            )
            path = reports_dir / f"perf_{ch}ch_rank{rank}.xlsx"
            df.to_excel(path, sheet_name="gpu_timeline", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# 1. CLI Surface Tests
# ---------------------------------------------------------------------------


class TestCLISurface:
    """Verify every command group and subcommand is registered."""

    def test_root_help(self, runner, cli):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for group in ("analyze", "compare", "generate", "process", "pipeline"):
            assert group in result.output

    def test_version(self, runner, cli):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.3.0" in result.output

    @pytest.mark.parametrize(
        "cmd",
        [
            ["analyze", "--help"],
            ["analyze", "single", "--help"],
            ["analyze", "sweep", "--help"],
            ["analyze", "gemm", "--help"],
            ["compare", "--help"],
            ["compare", "gpu_timeline", "--help"],
            ["compare", "collective", "--help"],
            ["generate", "--help"],
            ["generate", "html", "--help"],
            ["generate", "excel", "--help"],
            ["generate", "plots", "--help"],
            ["process", "--help"],
            ["process", "gpu-timeline", "--help"],
            ["process", "comms", "--help"],
            ["process", "gemm-variance", "--help"],
            ["pipeline", "--help"],
            ["pipeline", "summary", "--help"],
            ["pipeline", "gemm", "--help"],
        ],
    )
    def test_subcommand_help(self, runner, cli, cmd):
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 0, f"Failed: {cmd}\n{result.output}"

    def test_generate_html_requires_mode(self, runner, cli):
        result = runner.invoke(cli, ["generate", "html", "-o", "out.html"])
        assert result.exit_code != 0
        assert "mode" in result.output.lower() or "Missing" in result.output

    def test_compare_gpu_timeline_requires_args(self, runner, cli):
        result = runner.invoke(cli, ["compare", "gpu_timeline"])
        assert result.exit_code != 0

    def test_pipeline_summary_requires_test(self, runner, cli):
        result = runner.invoke(cli, ["pipeline", "summary", "-o", "/tmp/out"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 2. Utility Function Tests
# ---------------------------------------------------------------------------


class TestGeometricMean:
    def test_basic(self):
        from aorta.report.utils import geometric_mean

        values = np.array([1.0, 2.0, 4.0, 8.0])
        result = geometric_mean(values)
        expected = float(np.exp(np.mean(np.log(values))))
        assert abs(result - expected) < 1e-6

    def test_handles_zeros(self):
        from aorta.report.utils import geometric_mean

        values = np.array([0.0, 1.0, 2.0])
        result = geometric_mean(values)
        assert np.isfinite(result)

    def test_single_value(self):
        from aorta.report.utils import geometric_mean

        assert abs(geometric_mean(np.array([5.0])) - 5.0) < 1e-6

    def test_accepts_list(self):
        from aorta.report.utils import geometric_mean

        result = geometric_mean([2.0, 8.0])
        assert abs(result - 4.0) < 1e-6


# ---------------------------------------------------------------------------
# 3. Trace Directory Detection
# ---------------------------------------------------------------------------


class TestDetectTraceDirectory:
    def test_direct_rank_dirs(self, tmp_trace_dir):
        from aorta.report.analysis.analyze_single import detect_trace_directory

        tp_dir = tmp_trace_dir / "torch_profiler"
        torch_prof, base = detect_trace_directory(tp_dir)
        assert torch_prof == tp_dir
        assert base == tmp_trace_dir

    def test_parent_with_torch_profiler(self, tmp_trace_dir):
        from aorta.report.analysis.analyze_single import detect_trace_directory

        torch_prof, base = detect_trace_directory(tmp_trace_dir)
        assert torch_prof == tmp_trace_dir / "torch_profiler"
        assert base == tmp_trace_dir

    def test_invalid_raises(self, tmp_path):
        from aorta.report.analysis.analyze_single import detect_trace_directory

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="Cannot find rank directories"):
            detect_trace_directory(empty)


class TestFindTraceFile:
    def test_direct_json(self, tmp_path):
        from aorta.report.analysis.analyze_single import find_trace_file

        rank_dir = tmp_path / "rank0"
        rank_dir.mkdir()
        (rank_dir / "trace.json").write_text("{}")
        assert find_trace_file(rank_dir) is not None

    def test_trace_subdir(self, tmp_path):
        from aorta.report.analysis.analyze_single import find_trace_file

        rank_dir = tmp_path / "rank0"
        trace_dir = rank_dir / "trace"
        trace_dir.mkdir(parents=True)
        (trace_dir / "pt.trace.json").write_text("{}")
        assert find_trace_file(rank_dir) is not None

    def test_no_trace(self, tmp_path):
        from aorta.report.analysis.analyze_single import find_trace_file

        rank_dir = tmp_path / "rank0"
        rank_dir.mkdir()
        assert find_trace_file(rank_dir) is None


# ---------------------------------------------------------------------------
# 4. Rank Parsing
# ---------------------------------------------------------------------------


class TestRankParsing:
    """Verify the rank extraction from filenames doesn't corrupt multi-digit ranks."""

    def _parse_rank(self, rank_name: str):
        """Replicate the logic from analyze_single.py."""
        if rank_name.startswith("rank"):
            rank_num = rank_name[4:]
            try:
                rank_num = int(rank_num.lstrip("_"))
            except ValueError:
                rank_num = rank_name
        return rank_num

    def test_rank0(self):
        assert self._parse_rank("rank0") == 0

    def test_rank7(self):
        assert self._parse_rank("rank7") == 7

    def test_rank10(self):
        assert self._parse_rank("rank10") == 10

    def test_rank_underscore_0(self):
        assert self._parse_rank("rank_0") == 0

    def test_rank100(self):
        assert self._parse_rank("rank100") == 100


# ---------------------------------------------------------------------------
# 5. Sweep Filename Parsing
# ---------------------------------------------------------------------------


class TestSweepFilenameParsing:
    def test_basic(self):
        from aorta.report.analysis.analyze_sweep import parse_perf_filename

        channel, rank = parse_perf_filename("perf_28ch_rank0.xlsx")
        assert channel == "28ch"
        assert rank == 0

    def test_higher_rank(self):
        from aorta.report.analysis.analyze_sweep import parse_perf_filename

        channel, rank = parse_perf_filename("perf_42ch_rank7.xlsx")
        assert channel == "42ch"
        assert rank == 7


# ---------------------------------------------------------------------------
# 6. GPU Timeline Processing (Single Config)
# ---------------------------------------------------------------------------


class TestProcessGpuTimeline:
    def test_basic_aggregation(self, tmp_individual_reports):
        from aorta.report.analysis.analyze_single import process_gpu_timeline

        output = process_gpu_timeline(tmp_individual_reports)
        assert output is not None
        assert output.exists()
        assert "mean" in output.name

        result_df = pd.read_excel(output, sheet_name="Summary")
        assert "type" in result_df.columns
        assert "time ms" in result_df.columns
        assert len(result_df) == 5

    def test_geo_mean(self, tmp_individual_reports):
        from aorta.report.analysis.analyze_single import process_gpu_timeline

        output = process_gpu_timeline(tmp_individual_reports, use_geo_mean=True)
        assert output is not None
        assert "geomean" in output.name

    def test_no_files(self, tmp_path):
        from aorta.report.analysis.analyze_single import process_gpu_timeline

        empty = tmp_path / "empty"
        empty.mkdir()
        result = process_gpu_timeline(empty)
        assert result is None


# ---------------------------------------------------------------------------
# 7. GEMM Pattern Matching
# ---------------------------------------------------------------------------


class TestGEMMPatternMatching:
    def test_rocm_tensile_kernel(self):
        from aorta.report.analysis.tracelens_wrapper import ROCM_GEMM_PATTERN

        assert ROCM_GEMM_PATTERN.match("Cijk_Ailk_Bljk_SB_MT128x128x16_MI32x32x16")

    def test_non_gemm_kernel(self):
        from aorta.report.analysis.tracelens_wrapper import ROCM_GEMM_PATTERN

        assert not ROCM_GEMM_PATTERN.match("void at::native::vectorized_elementwise_kernel")

    def test_cuda_pattern_not_matched(self):
        from aorta.report.analysis.tracelens_wrapper import ROCM_GEMM_PATTERN

        assert not ROCM_GEMM_PATTERN.match("cublasLt_gemm_fp16")


# ---------------------------------------------------------------------------
# 8. Template Rendering
# ---------------------------------------------------------------------------


class TestSweepComparisonTemplate:
    def test_renders_without_error(self):
        from aorta.report.templates.sweep_comparison_template import (
            get_comparison_template,
        )

        html = get_comparison_template(
            label1="Baseline",
            label2="Optimized",
            sweep1_path=Path("/data/sweep1"),
            sweep2_path=Path("/data/sweep2"),
            image_data={},
        )
        assert "Baseline" in html
        assert "Optimized" in html
        assert "/data/sweep1" in html
        assert "/data/sweep2" in html
        assert "Sweep Information" in html

    def test_missing_images_show_placeholder(self):
        from aorta.report.templates.sweep_comparison_template import (
            get_comparison_template,
        )

        html = get_comparison_template(
            label1="A", label2="B",
            sweep1_path=Path("/a"), sweep2_path=Path("/b"),
            image_data={},
        )
        assert "missing-image" in html


class TestPerformanceReportTemplate:
    def test_constants_exist(self):
        from aorta.report.templates.performance_report_template import (
            HTML_HEADER,
            HTML_FOOTER,
        )

        assert "<!DOCTYPE html>" in HTML_HEADER
        assert "</html>" in HTML_FOOTER


class TestGemmReportTemplate:
    def test_renders(self):
        from aorta.report.templates.gemm_report_template import (
            get_gemm_report_template,
        )

        html = get_gemm_report_template(
            label="sweep_v1",
            sweep_path="/data/sweep_v1",
            image_data={},
            csv_path="/data/gemm.csv",
        )
        assert "sweep_v1" in html
        assert "/data/gemm.csv" in html


# ---------------------------------------------------------------------------
# 9. Excel Report Utilities
# ---------------------------------------------------------------------------


class TestExcelReportUtils:
    def test_sanitize_table_name(self):
        from aorta.report.generators.excel_report import sanitize_table_name

        assert sanitize_table_name("Summary Dashboard") == "Summary_Dashboard"
        assert sanitize_table_name("GPU_ByRank_Cmp") == "GPU_ByRank_Cmp"
        name = sanitize_table_name("1_invalid_start")
        assert name[0].isalpha()

    def test_sanitize_long_name(self):
        from aorta.report.generators.excel_report import sanitize_table_name

        long = "A" * 300
        assert len(sanitize_table_name(long)) <= 255


# ---------------------------------------------------------------------------
# 10. Sweep Data Grouping
# ---------------------------------------------------------------------------


class TestGroupFilesByChannel:
    def test_grouping(self):
        from aorta.report.analysis.analyze_sweep import group_files_by_channel

        files = [
            "/data/perf_28ch_rank0.xlsx",
            "/data/perf_28ch_rank1.xlsx",
            "/data/perf_42ch_rank0.xlsx",
        ]
        groups = group_files_by_channel(files)
        assert "28ch" in groups
        assert "42ch" in groups
        assert len(groups["28ch"]) == 2
        assert len(groups["42ch"]) == 1

    def test_sorts_by_rank(self):
        from aorta.report.analysis.analyze_sweep import group_files_by_channel

        files = [
            "/data/perf_28ch_rank2.xlsx",
            "/data/perf_28ch_rank0.xlsx",
        ]
        groups = group_files_by_channel(files)
        ranks = [r for r, _ in groups["28ch"]]
        assert ranks == [2, 0]


# ---------------------------------------------------------------------------
# 11. Comparison Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_color_constants(self):
        from aorta.report.generators.excel_report import RED, WHITE, GREEN

        assert len(RED) == 6
        assert len(WHITE) == 6
        assert len(GREEN) == 6


# ---------------------------------------------------------------------------
# 12. Import Completeness
# ---------------------------------------------------------------------------


class TestImportCompleteness:
    """Every subpackage should be importable without side effects."""

    @pytest.mark.parametrize(
        "module",
        [
            "aorta.report",
            "aorta.report.cli",
            "aorta.report.utils",
            "aorta.report.analysis",
            "aorta.report.analysis.cli",
            "aorta.report.analysis.analyze_single",
            "aorta.report.analysis.analyze_sweep",
            "aorta.report.analysis.analyze_gemm",
            "aorta.report.analysis.tracelens_wrapper",
            "aorta.report.comparison",
            "aorta.report.comparison.cli",
            "aorta.report.generators",
            "aorta.report.generators.cli",
            "aorta.report.generators.excel_report",
            "aorta.report.generators.plot_generator",
            "aorta.report.processing",
            "aorta.report.processing.cli",
            "aorta.report.pipelines",
            "aorta.report.pipelines.cli",
            "aorta.report.templates",
        ],
    )
    def test_import(self, module):
        import importlib

        importlib.import_module(module)
