# tests/test_pipeline.py
import os
import re
import tempfile
from canine_holter.pipeline import run_analysis

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Ground truth for tests/fixtures/mitdb_119/119 (from wfdb .atr annotations):
# 19 known PVC ('V') beats out of ~65 total detected beats. See
# test_mitbih_validation.py, which independently requires >= 50% sensitivity
# (i.e. >= 10 of 19) on this same detect -> classify path. These bounds are
# intentionally looser than the exact 18/65 currently observed, so pipeline
# tests don't fail on classifier retuning, while still catching a pipeline
# that runs "successfully" but produces empty/garbage output (e.g. 0 beats,
# 0 PVCs, or a report with no numbers at all).
MIN_EXPECTED_TOTAL_BEATS = 40
MAX_EXPECTED_TOTAL_BEATS = 90
MIN_EXPECTED_PVC_COUNT = 10  # matches the >= 50% sensitivity floor elsewhere
MAX_EXPECTED_PVC_COUNT = 19  # can't exceed ground truth PVC count


def test_run_analysis_produces_report_from_fixture():
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        report_path = run_analysis(input_path, out_dir)
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 0


def test_run_analysis_report_contains_plausible_stats_for_known_fixture():
    """Guards against a regression where the pipeline runs end-to-end without
    error but silently produces garbage or empty stats (e.g. an ingest bug
    that yields zero beats, or a wiring bug that passes the wrong beats into
    summarize/write_report)."""
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        report_path = run_analysis(input_path, out_dir)
        with open(report_path) as f:
            content = f.read()

    total_beats_match = re.search(r"Total beats:\s*(\d+)", content)
    pvc_count_match = re.search(r"PVC count:\s*(\d+)", content)
    assert total_beats_match, f"report missing 'Total beats' line:\n{content}"
    assert pvc_count_match, f"report missing 'PVC count' line:\n{content}"

    total_beats = int(total_beats_match.group(1))
    pvc_count = int(pvc_count_match.group(1))

    assert MIN_EXPECTED_TOTAL_BEATS <= total_beats <= MAX_EXPECTED_TOTAL_BEATS, (
        f"total beats {total_beats} outside plausible range for this fixture"
    )
    assert MIN_EXPECTED_PVC_COUNT <= pvc_count <= MAX_EXPECTED_PVC_COUNT, (
        f"PVC count {pvc_count} outside plausible range for this fixture"
    )
