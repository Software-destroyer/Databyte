"""Tests for main.py CLI behavior."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Import main module functions.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import build_parser, main, cmd_mock


# ── Parser tests ─────────────────────────────────────────────────────────

class TestParser:
    def test_no_command_shows_help(self, capsys):
        """Running without a command should print help and exit 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_mock_command_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["mock", "--seed", "99"])
        assert args.command == "mock"
        assert args.seed == 99

    def test_generate_mock_command_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["generate-mock", "--rows", "500"])
        assert args.command == "generate-mock"
        assert args.rows == 500

    def test_preprocess_requires_args(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["preprocess"])

    def test_preprocess_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["preprocess", "--cicids", "a.csv", "--unsw", "b.csv"])
        assert args.cicids == "a.csv"
        assert args.unsw == "b.csv"

    def test_dashboard_port(self):
        parser = build_parser()
        args = parser.parse_args(["dashboard", "--port", "8502"])
        assert args.port == 8502

    def test_evaluate_default_no_csv(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate"])
        assert args.results_csv is None

    def test_run_requires_args(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])


# ── Mock command ─────────────────────────────────────────────────────────

class TestMockCommand:
    def test_mock_produces_csv(self, tmp_path: Path, monkeypatch):
        """The 'mock' command should write a consolidated CSV."""
        monkeypatch.setattr("main.RESULTS_DIR", tmp_path)
        parser = build_parser()
        args = parser.parse_args(["mock"])
        args.func(args)
        output = tmp_path / "consolidated_results.csv"
        assert output.is_file()
        df = pd.read_csv(output)
        assert len(df) > 0
        assert "accuracy" in df.columns
        assert "evaluation_mode" in df.columns

    def test_mock_deterministic(self, tmp_path: Path, monkeypatch):
        """Two runs with the same seed produce identical results."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        dir1.mkdir()
        dir2.mkdir()

        parser = build_parser()

        monkeypatch.setattr("main.RESULTS_DIR", dir1)
        args = parser.parse_args(["mock", "--seed", "42"])
        args.func(args)

        monkeypatch.setattr("main.RESULTS_DIR", dir2)
        args = parser.parse_args(["mock", "--seed", "42"])
        args.func(args)

        df1 = pd.read_csv(dir1 / "consolidated_results.csv")
        df2 = pd.read_csv(dir2 / "consolidated_results.csv")
        pd.testing.assert_frame_equal(df1, df2)


# ── Error handling ────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_preprocess_missing_file(self, monkeypatch):
        """Preprocessing should fail clearly when CSV is missing."""
        parser = build_parser()
        args = parser.parse_args([
            "preprocess",
            "--cicids", "nonexistent_cic.csv",
            "--unsw", "nonexistent_unsw.csv",
        ])
        with pytest.raises(SystemExit) as exc_info:
            args.func(args)
        assert exc_info.value.code == 1

    def test_baselines_missing_parquet(self, tmp_path: Path, monkeypatch):
        """Baselines should fail when preprocessed data is missing."""
        monkeypatch.setattr("main.PROCESSED_DATA_DIR", tmp_path)
        parser = build_parser()
        args = parser.parse_args(["baselines"])
        with pytest.raises(SystemExit) as exc_info:
            args.func(args)
        assert exc_info.value.code == 1

    def test_evaluate_no_results(self, tmp_path: Path, monkeypatch):
        """Evaluate should fail clearly when no result files exist."""
        monkeypatch.setattr("main.RESULTS_DIR", tmp_path)
        parser = build_parser()
        args = parser.parse_args(["evaluate"])
        with pytest.raises(SystemExit) as exc_info:
            args.func(args)
        assert exc_info.value.code == 1
