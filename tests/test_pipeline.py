# tests/test_pipeline.py
import os
import re
import tempfile
from datetime import date, datetime

import numpy as np
import pytest
from canine_holter.arrhythmia.burden import summarize
from canine_holter.classify.rules import classify_beats
from canine_holter.detection.detect import detect_beats
from canine_holter.ingest.loader import load_recording
from canine_holter.pipeline import parse_start_time, run_analysis
from canine_holter.report.generate import build_content
from canine_holter.types import Recording
from tests.native_flash_factory import data_block, metadata_block

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# mitdb_119 has 19 annotated PVCs among ~65 beats; the bounds are loose so
# retuning does not break them, but empty or garbage output does.
MIN_EXPECTED_TOTAL_BEATS = 40
MAX_EXPECTED_TOTAL_BEATS = 90
MIN_EXPECTED_PVC_COUNT = 10
MAX_EXPECTED_PVC_COUNT = 19


def test_report_contains_plausible_stats_for_known_fixture():
    """run_analysis excludes the first and last minute and this fixture is
    60 s, so the check runs one stage below it."""
    rec = load_recording(os.path.join(FIXTURES_DIR, "mitdb_119", "119"))
    labeled = classify_beats(detect_beats(rec.samples, rec.sample_rate))
    content = build_content(labeled, summarize(labeled), rec.start_time)
    rows = {r.label: r.value for g in content.summary_groups for r in g.rows}
    total_beats = int(rows["Total beats"])
    pvc_count = int(rows["PVCs"].split()[0])

    assert MIN_EXPECTED_TOTAL_BEATS <= total_beats <= MAX_EXPECTED_TOTAL_BEATS, (
        f"total beats {total_beats} outside plausible range for this fixture"
    )
    assert MIN_EXPECTED_PVC_COUNT <= pvc_count <= MAX_EXPECTED_PVC_COUNT, (
        f"PVC count {pvc_count} outside plausible range for this fixture"
    )


# The fixture tests above use WFDB; this exercises the native DR200 path.

_FLASH_SPIKE_PERIOD = 90  # 180 Hz / 90 samples = 2 Hz = 120 bpm


def _spike_train_block_encoding():
    # channels 0 and 1: +70,+70 up then -70,-70 back to baseline each period;
    # a sharp, detectable QRS-like spike. Codes 7=+70, 9=-70 in the DR200
    # delta table. Two leads carry it because a beat needs two leads to
    # agree; channel 2 stays flat.
    encoded = np.zeros((304, 3), dtype=np.uint8)
    # +1/-1 count jitter between spikes: a real baseline is never exactly
    # flat, and a flat one is what quality gating excludes.
    encoded[0::2, :2] = 1
    encoded[1::2, :2] = 15
    phase = np.arange(304) % _FLASH_SPIKE_PERIOD
    encoded[phase == 0, :2] = 7
    encoded[phase == 1, :2] = 7
    encoded[phase == 2, :2] = 9
    encoded[phase == 3, :2] = 9
    return encoded


def _write_synthetic_flash(path, block_count: int = 15) -> None:
    encoded = _spike_train_block_encoding()
    blocks = (
        data_block(encoded, source_position=5900 + index * 304 * 4)
        for index in range(block_count)
    )
    path.write_bytes(metadata_block() + b"".join(blocks) + bytes(511))


def test_run_analysis_end_to_end_on_native_flash(tmp_path, report_text):
    flash_path = tmp_path / "flash.dat"
    _write_synthetic_flash(flash_path, block_count=110)  # ~186 s: a minute survives the edge rule
    out_dir = tmp_path / "out"

    report_path = run_analysis(str(flash_path), str(out_dir))

    assert os.path.exists(report_path)
    content = report_text()
    total_beats_match = re.search(r"Total beats:\s*(\d+)", content)
    assert total_beats_match, f"report missing 'Total beats' line:\n{content}"
    # ~66 s analyzed of a 120 bpm spike train is ~130 beats; assert a
    # non-trivial count so a wiring bug that silently drops the signal fails.
    assert int(total_beats_match.group(1)) >= 60
    assert re.search(r"Analyzed:\s*0h 1m \(3[0-9]%\)", content), content


def test_run_analysis_excludes_a_recording_shorter_than_the_edge_minutes(tmp_path, report_text):
    """The default 25 s synthetic recording lies inside the first minute, so
    it is excluded whole: the report says so instead of counting beats."""
    flash_path = tmp_path / "flash.dat"
    _write_synthetic_flash(flash_path)
    run_analysis(str(flash_path), str(tmp_path / "out"))
    content = report_text()
    assert re.search(r"Analyzed:\s*0h 0m \(0%\)", content), content
    assert re.search(r"Total beats:\s*0\b", content), content


def test_run_analysis_trims_an_off_body_tail_before_detection(tmp_path, report_text, monkeypatch):
    fs = 100.0
    t = np.arange(0, 3600, 1 / fs)
    ecg = np.sin(2 * np.pi * 1.5 * t)
    tone = 0.5 * np.where(np.arange(int(7200 * fs)) % 2 == 0, 1.0, -1.0)
    rec = Recording(samples=np.concatenate([ecg, tone]), sample_rate=fs, start_time=None, source="synthetic")
    seen = {}
    monkeypatch.setattr("canine_holter.pipeline.load_recording", lambda _: rec)
    monkeypatch.setattr(
        "canine_holter.pipeline.detect_beats",
        lambda samples, rate: seen.setdefault("n", len(samples)) and detect_beats(samples, rate),
    )
    run_analysis("ignored", str(tmp_path / "out"))
    assert seen["n"] == 360000
    content = report_text()
    assert re.search(r"Duration:\s*1h 0m", content), content
    assert re.search(r"Recorder ran:\s*3h 0m \(off-body tail trimmed\)", content), content


# --- start-time override -----------------------------------------------------
HEADER = datetime(2026, 8, 23, 15, 33, 8)


def test_parse_start_time_hh_mm_borrows_header_date():
    assert parse_start_time("15:36", HEADER) == datetime(2026, 8, 23, 15, 36)


def test_parse_start_time_hh_mm_ss():
    assert parse_start_time("15:36:10", HEADER) == datetime(2026, 8, 23, 15, 36, 10)


def test_parse_start_time_full_datetime_ignores_header():
    assert parse_start_time("2026-08-22 09:00", HEADER) == datetime(2026, 8, 22, 9, 0)


def test_parse_start_time_time_only_without_header_uses_today():
    result = parse_start_time("15:36", None)
    assert result.date() == date.today()
    assert (result.hour, result.minute) == (15, 36)


def test_parse_start_time_rejects_garbage():
    with pytest.raises(ValueError):
        parse_start_time("half past three", HEADER)


def test_run_analysis_start_time_override_appears_in_report(report_text):
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        run_analysis(input_path, out_dir, start_time=datetime(2026, 8, 23, 15, 36))
        assert "Start: 2026-08-23 15:36:00" in report_text()
