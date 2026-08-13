import os
import numpy as np
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.detection.detect import detect_beats

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_detects_beats_in_mitbih_fixture():
    rec = load_local_record(
        os.path.join(FIXTURES_DIR, "mitdb_119", "119"), source="mitdb_119"
    )
    beats = detect_beats(rec.samples, rec.sample_rate)
    # 60s at a resting ~75bpm should be roughly 60-90 beats; a wide sanity
    # range avoids a brittle exact-count assertion
    assert 50 <= len(beats) <= 100
    assert beats[0].rr_interval is None  # first beat has no prior beat
    assert all(b.rr_interval is not None for b in beats[1:])
    assert all(b.label is None for b in beats)  # not classified yet
    # QRS delineation should succeed for at least some beats; a regression
    # that silently makes qrs_duration always None would otherwise pass
    assert any(b.qrs_duration is not None for b in beats)


def test_detects_beats_in_canine_fixture():
    rec = load_local_record(
        os.path.join(FIXTURES_DIR, "physiozoo_dog1", "Dog_01"), source="physiozoo_dog1"
    )
    beats = detect_beats(rec.samples, rec.sample_rate)
    assert len(beats) > 0
    assert beats[0].rr_interval is None  # first beat has no prior beat
    assert all(b.rr_interval is not None for b in beats[1:])
    assert all(b.label is None for b in beats)  # not classified yet


def test_returns_empty_list_for_flat_signal():
    # A flat/constant signal has no meaningful gradient for NeuroKit2's
    # R-peak detector to key off of, so it reliably yields zero R-peaks -
    # confirmed via a scratch run (0 peaks for 1000 zero samples at 360Hz),
    # exercising the len(r_peaks) < 2 early-return branch.
    flat_signal = np.zeros(1000)
    beats = detect_beats(flat_signal, sample_rate=360.0)
    assert beats == []
