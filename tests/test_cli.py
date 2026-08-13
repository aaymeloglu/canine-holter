# tests/test_cli.py
import os
import sys
import tempfile
import pytest
from canine_holter.cli import main

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
INPUT_PATH = os.path.join(FIXTURES_DIR, "mitdb_119", "119")


def test_main_runs_end_to_end_and_prints_report_path(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as out_dir:
        monkeypatch.setattr(sys, "argv", ["canine-holter", INPUT_PATH, "--out", out_dir])
        main()

        report_path = os.path.join(out_dir, "report.md")
        assert os.path.exists(report_path)

        captured = capsys.readouterr()
        assert captured.out.strip() == f"Report written to {report_path}"


def test_main_accepts_each_dog_weight_class_choice(monkeypatch, capsys):
    for weight_class in ("small", "medium", "large"):
        with tempfile.TemporaryDirectory() as out_dir:
            monkeypatch.setattr(
                sys,
                "argv",
                ["canine-holter", INPUT_PATH, "--out", out_dir, "--dog-weight-class", weight_class],
            )
            main()
            assert os.path.exists(os.path.join(out_dir, "report.md"))


def test_main_rejects_invalid_dog_weight_class(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as out_dir:
        monkeypatch.setattr(
            sys,
            "argv",
            ["canine-holter", INPUT_PATH, "--out", out_dir, "--dog-weight-class", "huge"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err
        # arg parsing should fail before the pipeline ever runs
        assert not os.path.exists(os.path.join(out_dir, "report.md"))


def test_main_requires_out_argument(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["canine-holter", INPUT_PATH])
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--out" in captured.err
    assert "required" in captured.err
