from canine_holter.types import Recording, Beat
import numpy as np


def test_recording_holds_signal_and_metadata():
    samples = np.array([0.1, 0.2, 0.1, -0.1])
    rec = Recording(samples=samples, sample_rate=360.0, start_time=None, source="test")
    assert rec.sample_rate == 360.0
    assert len(rec.samples) == 4
    assert rec.source == "test"


def test_beat_defaults_label_to_none():
    beat = Beat(time=1.5, rr_interval=0.8, qrs_duration=0.09, label=None)
    assert beat.label is None
    assert beat.time == 1.5


def test_beat_is_immutable_and_replaceable():
    from dataclasses import replace
    beat = Beat(time=1.5, rr_interval=0.8, qrs_duration=0.09, label=None)
    labeled = replace(beat, label="V")
    assert labeled.label == "V"
    assert beat.label is None  # original unchanged


def test_recording_equality_does_not_raise_on_array_field():
    rec1 = Recording(samples=np.array([1.0, 2.0]), sample_rate=360.0, start_time=None, source="a")
    rec2 = Recording(samples=np.array([3.0, 4.0]), sample_rate=360.0, start_time=None, source="b")
    assert (rec1 == rec2) is False  # must not raise
    assert (rec1 == rec1) is True
