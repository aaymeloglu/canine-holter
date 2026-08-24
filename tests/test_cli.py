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

        report_path = os.path.join(out_dir, "report.pdf")
        assert os.path.exists(report_path)
        assert os.listdir(out_dir) == ["report.pdf"]

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
            assert os.path.exists(os.path.join(out_dir, "report.pdf"))


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
        assert not os.path.exists(os.path.join(out_dir, "report.pdf"))


def test_main_requires_out_argument(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["canine-holter", INPUT_PATH])
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--out" in captured.err
    assert "required" in captured.err


def test_main_start_time_override_reaches_report(monkeypatch, report_text):
    with tempfile.TemporaryDirectory() as out_dir:
        monkeypatch.setattr(
            sys,
            "argv",
            ["canine-holter", INPUT_PATH, "--out", out_dir, "--start-time", "2026-08-23 15:36"],
        )
        main()
        assert "- Recording start: 2026-08-23 15:36:00" in report_text()


def test_main_rejects_unparseable_start_time(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as out_dir:
        monkeypatch.setattr(
            sys,
            "argv",
            ["canine-holter", INPUT_PATH, "--out", out_dir, "--start-time", "teatime"],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert "start-time" in capsys.readouterr().err


def test_main_unsupported_input_error_is_not_blamed_on_start_time(monkeypatch, capsys):
    """Only a start-time parse failure may be reported as a --start-time
    problem; an unrelated load error must surface as itself."""
    with tempfile.TemporaryDirectory() as out_dir:
        bogus = os.path.join(out_dir, "notes.txt")
        open(bogus, "w").write("not a recording")
        monkeypatch.setattr(
            sys,
            "argv",
            ["canine-holter", bogus, "--out", out_dir, "--start-time", "15:36"],
        )
        with pytest.raises(ValueError, match="Unsupported recording input"):
            main()
        assert "start-time" not in capsys.readouterr().err
