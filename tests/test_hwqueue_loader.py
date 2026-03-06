"""Tests for aorta.report.processing.hwqueue_loader."""

import json
import tempfile
from pathlib import Path

import pytest

from aorta.report.processing.hwqueue_loader import (
    HWQueueLoader,
    HWQueueLoaderError,
    SingleRunData,
    SweepData,
)


def _make_single_run_dict(**kwargs):
    base = {
        "throughput": 10.0,
        "stream_count": 2,
        "latency_ms": {},
        "total_time_ms": 100.0,
        "throughput_unit": "ops/s",
    }
    base.update(kwargs)
    return base


class TestSweepDataFromDict:
    """Tests for SweepData.from_dict() schema validation."""

    def test_valid_results_list(self):
        """Valid list of dicts should parse without error."""
        data = {
            "workload": "test_wl",
            "results": [_make_single_run_dict(stream_count=1), _make_single_run_dict(stream_count=2)],
        }
        sweep = SweepData.from_dict(data)
        assert len(sweep.results) == 2
        assert sweep.workload_name == "test_wl"

    def test_empty_results_list(self):
        """Empty results list is valid."""
        sweep = SweepData.from_dict({"workload": "wl", "results": []})
        assert sweep.results == []

    def test_missing_results_key(self):
        """Missing 'results' key defaults to empty list."""
        sweep = SweepData.from_dict({"workload": "wl"})
        assert sweep.results == []

    def test_non_dict_string_entry_raises(self):
        """A string entry in results must raise HWQueueLoaderError."""
        data = {"workload": "wl", "results": ["not_a_dict"]}
        with pytest.raises(HWQueueLoaderError, match=r"results\[0\].*dict.*str"):
            SweepData.from_dict(data)

    def test_non_dict_integer_entry_raises(self):
        """An integer entry in results must raise HWQueueLoaderError."""
        data = {"workload": "wl", "results": [42]}
        with pytest.raises(HWQueueLoaderError, match=r"results\[0\].*dict.*int"):
            SweepData.from_dict(data)

    def test_non_dict_list_entry_raises(self):
        """A nested list entry in results must raise HWQueueLoaderError."""
        data = {"workload": "wl", "results": [[1, 2, 3]]}
        with pytest.raises(HWQueueLoaderError, match=r"results\[0\].*dict.*list"):
            SweepData.from_dict(data)

    def test_non_dict_none_entry_raises(self):
        """A None entry in results must raise HWQueueLoaderError."""
        data = {"workload": "wl", "results": [None]}
        with pytest.raises(HWQueueLoaderError, match=r"results\[0\].*dict.*NoneType"):
            SweepData.from_dict(data)

    def test_second_entry_invalid_reports_correct_index(self):
        """Error message should include the index of the invalid entry."""
        data = {
            "workload": "wl",
            "results": [
                _make_single_run_dict(stream_count=1),
                "bad_entry",
            ],
        }
        with pytest.raises(HWQueueLoaderError, match=r"results\[1\]"):
            SweepData.from_dict(data)

    def test_error_is_not_attribute_error(self):
        """Non-dict entries must not cause AttributeError to escape."""
        data = {"workload": "wl", "results": ["bad"]}
        try:
            SweepData.from_dict(data)
        except HWQueueLoaderError:
            pass
        except AttributeError:
            pytest.fail("AttributeError escaped instead of HWQueueLoaderError")


class TestHWQueueLoaderLoadSweep:
    """Tests for HWQueueLoader.load_sweep() with invalid results entries."""

    def test_load_sweep_with_non_dict_results_raises(self, tmp_path):
        """load_sweep should propagate HWQueueLoaderError for invalid results entries."""
        bad_file = tmp_path / "bad_results.json"
        bad_file.write_text(json.dumps({"workload": "wl", "results": ["not_a_dict"]}))
        with pytest.raises(HWQueueLoaderError, match=r"results\[0\]"):
            HWQueueLoader.load_sweep(bad_file)

    def test_load_auto_sweep_with_non_dict_results_raises(self, tmp_path):
        """load_auto should propagate HWQueueLoaderError for invalid results entries."""
        bad_file = tmp_path / "bad_results.json"
        bad_file.write_text(json.dumps({"workload": "wl", "results": [42]}))
        with pytest.raises(HWQueueLoaderError, match=r"results\[0\]"):
            HWQueueLoader.load_auto(bad_file)
