"""Smoke tests to verify all packages are importable after workspace restructuring."""

import importlib
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")


class TestCoreImports:
    def test_utils_init(self):
        from aorta.utils import detect_accelerator, get_device, setup_logging

    def test_utils_config(self):
        from aorta.utils.config import load_config, merge_cli_overrides

    def test_utils_device(self):
        from aorta.utils.device import IS_ROCM, BACKEND_NAME, DeviceProperties

    def test_utils_timing(self):
        from aorta.utils.timing import CPUTimer, EventTiming, StreamTimer

    def test_utils_streams(self):
        from aorta.utils.streams import create_streams, get_available_devices

    def test_utils_logging(self):
        from aorta.utils.logging import setup_logging

    def test_utils_distributed(self):
        from aorta.utils.distributed import get_rank, get_world_size


class TestReportImports:
    def test_report_init(self):
        from aorta.report import cli, main

    def test_report_version(self):
        from aorta.report import __version__
        assert __version__

    def test_report_cli(self):
        from aorta.report.cli import cli

    def test_report_analysis(self):
        from aorta.report.analysis import cli as analysis_cli

    def test_report_generators(self):
        from aorta.report.generators import cli as generators_cli

    def test_report_pipelines(self):
        from aorta.report.pipelines import cli as pipelines_cli


class TestRaceImports:
    def test_race_init(self):
        from aorta.race import RaceConfig, ReproducerConfig, ReproducerResult

    def test_race_config(self):
        from aorta.race.config import RaceConfig, ReproducerConfig

    def test_race_base(self):
        from aorta.race.base import BaseReproducer

    def test_race_modes(self):
        from aorta.race.modes import AVAILABLE_MODES


class TestHwQueueImports:
    def test_hw_queue_init(self):
        from aorta.hw_queue_eval import __version__
        assert __version__

    def test_hw_queue_cli(self):
        from aorta.hw_queue_eval.cli import main

    def test_hw_queue_harness(self):
        from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness

    def test_hw_queue_metrics(self):
        from aorta.hw_queue_eval.core.metrics import LatencyMetrics, MetricsCollector

    def test_hw_queue_workload_base(self):
        from aorta.hw_queue_eval.workloads.base import BaseWorkload

    def test_hw_queue_registry(self):
        from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class TestTrainingImports:
    def test_training_init(self):
        import aorta.training

    def test_training_data(self):
        from aorta.training.data import SyntheticDatasetConfig

    def test_training_models(self):
        from aorta.training.models import ModelConfig

    def test_training_profiling(self):
        from aorta.training.profiling import stream_profiler


class TestNamespacePackages:
    """Verify namespace packages work -- multiple packages share the aorta namespace."""

    def test_multiple_packages_coexist(self):
        import aorta.utils
        import aorta.report
        import aorta.race
        import aorta.hw_queue_eval

    def test_no_root_init(self):
        """The aorta namespace should NOT have a single __init__.py."""
        import aorta
        assert not hasattr(aorta, "__version__"), (
            "aorta root should be a namespace package without __init__.py"
        )
