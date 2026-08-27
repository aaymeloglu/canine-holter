"""Hand-counted windows of Teeny's 2026-08-25 DR200 recording - the first
canine ground truth in the suite. See scripts/extract_teeny_fixtures.py for
provenance and docs/superpowers/specs/2026-08-26-detector-tachycardia-and-t-wave-design.md
for why each window exists."""
import os
import numpy as np
import pytest
from canine_holter.detection.detect import detect_beats

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "teeny_2026-08-25")
MATCH_TOLERANCE_SEC = 0.15


def _load(name):
    z = np.load(os.path.join(FIXTURES_DIR, f"{name}.npz"))
    return z["channels"][0].astype(float), float(z["sample_rate"]), z["beat_times"]


def _sensitivity_precision(detected, truth):
    detected = np.asarray(detected)
    hits = sum(1 for t in truth if len(detected) and np.min(np.abs(detected - t)) <= MATCH_TOLERANCE_SEC)
    return hits / len(truth), hits / max(1, len(detected))


@pytest.mark.parametrize("name, min_sensitivity, min_precision", [
    ("tachy", 0.90, 0.95),   # sinus tachycardia ~150 bpm: the search-back's reason to exist
    ("lying", 0.90, 0.95),   # small QRS spike + large T trough
    ("lying_t", 0.90, 0.95), # same posture, longer QT: NeuroKit detects the T waves - the T-wave rule's reason to exist
    ("quiet", 1.00, 1.00),   # 7 beats and a real 4.67 s pause; nine phantom candidates on flat baseline
])
def test_detect_beats_on_teeny_window(name, min_sensitivity, min_precision):
    samples, sample_rate, truth = _load(name)
    detected = [b.time for b in detect_beats(samples, sample_rate)]
    sensitivity, precision = _sensitivity_precision(detected, truth)
    assert sensitivity >= min_sensitivity, f"{name}: found {len(detected)} of {len(truth)} beats; sensitivity {sensitivity:.2f}"
    assert precision >= min_precision, f"{name}: {len(detected)} detections for {len(truth)} beats; precision {precision:.2f}"
