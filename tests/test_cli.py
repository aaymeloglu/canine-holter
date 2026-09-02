# tests/test_cli.py
import os
import sys
import tempfile

import pytest

from canine_holter import cli
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


def test_main_passes_the_options_to_the_pipeline(monkeypatch, capsys):
    captured = {}

    def fake_run_analysis(input_path, out_dir, dog_weight_class, start_time):
        captured.update(
            input_path=input_path,
            out_dir=out_dir,
            dog_weight_class=dog_weight_class,
            start_time=start_time,
        )
        return os.path.join(out_dir, "report.pdf")

    monkeypatch.setattr(cli, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(
        sys,
        "argv",
        ["canine-holter", INPUT_PATH, "--out", "/out", "--dog-weight-class", "large"],
    )
    main()

    assert captured == {
        "input_path": INPUT_PATH,
        "out_dir": "/out",
        "dog_weight_class": "large",
        "start_time": None,
    }
    assert capsys.readouterr().out.strip() == "Report written to /out/report.pdf"


def test_main_start_time_override_reaches_report(monkeypatch, report_text):
    with tempfile.TemporaryDirectory() as out_dir:
        monkeypatch.setattr(
            sys,
            "argv",
            ["canine-holter", INPUT_PATH, "--out", out_dir, "--start-time", "2026-08-23 15:36"],
        )
        main()
        assert "Start: 2026-08-23 15:36:00" in report_text()


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
