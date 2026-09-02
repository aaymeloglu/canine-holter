"""Hand-checked windows of Teeny's recordings with the PVCs a careful eye
finds on three-lead strips. See scripts/extract_teeny_fixtures.py for
provenance and docs/superpowers/specs/2026-09-01-lead-agreement-qrs-width-design.md
for why each window exists."""
import os
import numpy as np
import pytest
from canine_holter.classify.rules import classify_beats
from canine_holter.detection.detect import detect_beats

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MATCH_TOLERANCE_SEC = 0.15


def _load(date, name):
    z = np.load(os.path.join(FIXTURES_DIR, f"teeny_{date}", f"{name}.npz"))
    return z["channels"].astype(float), float(z["sample_rate"]), z["pvc_times"]


@pytest.mark.parametrize("date, name", [
    # asleep with sinus arrhythmia; on channel 1 alone the detector takes
    # every T wave for a beat, early and wide on every lead (36 "PVCs" in
    # 23:55-00:13 of the channel-1 report)
    ("2026-08-27", "midnight"),
    # one PVC, wide and differently shaped on all three leads; it peaks
    # 50 ms later on channels 1 and 2 than on channel 0
    ("2026-08-27", "pvc"),
    # one PVC after a run of small fast beats
    ("2026-08-25", "pvc_run_end"),
])
def test_pvcs_on_teeny_window(date, name):
    channels, sample_rate, truth = _load(date, name)
    beats = classify_beats(detect_beats(channels, sample_rate))
    flagged = [b.time for b in beats if b.label == "V"]
    assert len(flagged) == len(truth), f"{name}: flagged {flagged}, expected {list(truth)}"
    assert all(min(abs(f - t) for t in truth) <= MATCH_TOLERANCE_SEC for f in flagged), f"{name}: flagged {flagged}, expected {list(truth)}"
