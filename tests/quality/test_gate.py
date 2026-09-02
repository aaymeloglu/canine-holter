"""Quality gating against literal expected spans. The synthetic signal is a
1.5 Hz sine at 100 Hz (peak-to-peak 2.0 in every 5 s window); artifact is
injected by scaling, flattening, or shrinking stretches of it."""
import numpy as np
import pytest
from canine_holter.quality.gate import SignalQuality, assess_quality, exclude_beats
from canine_holter.types import Beat

FS = 100.0


def _sine(seconds):
    t = np.arange(0, seconds, 1 / FS)
    return np.sin(2 * np.pi * 1.5 * t)


def _at(start_sec, end_sec):
    return slice(int(start_sec * FS), int(end_sec * FS))


def _tone(seconds, amplitude=0.5):
    """The DR400's open-electrode lead-off excitation: alternating samples."""
    return amplitude * np.where(np.arange(int(seconds * FS)) % 2 == 0, 1.0, -1.0)


def test_clean_recording_excludes_only_the_first_and_last_minute():
    q = assess_quality(_sine(600), FS)
    assert q.duration_sec == 600.0
    assert q.excluded == ((0.0, 60.0), (540.0, 600.0))
    assert q.analyzed_sec == 480.0


def test_high_amplitude_burst_is_excluded_with_two_second_padding():
    x = _sine(600)
    x[_at(200, 210)] *= 10  # 20 peak-to-peak vs a median of 2
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (198.0, 212.0), (540.0, 600.0))


def test_flat_stretch_is_excluded():
    x = _sine(600)
    x[_at(300, 320)] = 0.7
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (298.0, 322.0), (540.0, 600.0))


def test_low_amplitude_stretch_is_excluded():
    x = _sine(600)
    x[_at(400, 410)] *= 0.01  # 0.02 peak-to-peak, under 0.1x the median
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (398.0, 412.0), (540.0, 600.0))


def test_bursts_within_bridge_sec_form_one_span():
    x = _sine(600)
    x[_at(200, 205)] *= 10
    x[_at(230, 235)] *= 10  # 25 s gap: bridged
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (198.0, 237.0), (540.0, 600.0))


def test_bursts_further_apart_than_bridge_sec_stay_separate():
    x = _sine(600)
    x[_at(200, 205)] *= 10
    x[_at(245, 250)] *= 10  # 40 s gap: not bridged
    assert assess_quality(x, FS).excluded == (
        (0.0, 60.0), (198.0, 207.0), (243.0, 252.0), (540.0, 600.0)
    )


def test_burst_near_the_edge_merges_into_the_edge_span():
    x = _sine(600)
    x[_at(80, 85)] *= 10  # 20 s after the first minute: bridged into it
    assert assess_quality(x, FS).excluded == ((0.0, 87.0), (540.0, 600.0))


def test_all_zero_recording_is_fully_excluded():
    q = assess_quality(np.zeros(60000), FS)
    assert q.excluded == ((0.0, 600.0),)
    assert q.analyzed_sec == 0.0


def test_recording_shorter_than_two_edge_minutes_is_fully_excluded():
    assert assess_quality(_sine(30), FS).excluded == ((0.0, 30.0),)


def test_empty_recording():
    q = assess_quality(np.array([]), FS)
    assert (q.duration_sec, q.excluded, q.analyzed_sec) == (0.0, (), 0.0)


def test_analyzed_within_subtracts_overlap_with_excluded_spans():
    q = SignalQuality(600.0, ((0.0, 60.0), (198.0, 212.0), (540.0, 600.0)))
    assert q.analyzed_within(0.0, 60.0) == 0.0
    assert q.analyzed_within(100.0, 150.0) == 50.0
    assert q.analyzed_within(180.0, 240.0) == pytest.approx(46.0)
    assert q.analyzed_within(500.0, 600.0) == 40.0


def test_contains_is_inclusive_at_span_edges():
    q = SignalQuality(100.0, ((10.0, 20.0),))
    assert q.contains(10.0) and q.contains(20.0) and q.contains(15.0)
    assert not q.contains(9.99) and not q.contains(20.01)


def _beats(times, rr=0.5):
    return [
        Beat(time=t, rr_interval=None if i == 0 else rr, qrs_duration=0.06, label=None)
        for i, t in enumerate(times)
    ]


def test_exclude_beats_drops_beats_inside_spans_and_resets_the_next_rr():
    beats = _beats([9.0, 9.5, 10.0, 12.0, 20.0, 20.5, 21.0])
    kept = exclude_beats(beats, SignalQuality(30.0, ((10.0, 20.0),)))
    assert [b.time for b in kept] == [9.0, 9.5, 20.5, 21.0]
    assert [b.rr_interval for b in kept] == [None, 0.5, None, 0.5]


def test_exclude_beats_keeps_everything_with_no_spans():
    beats = _beats([1.0, 1.5, 2.0])
    assert exclude_beats(beats, SignalQuality(30.0, ())) == beats


def test_exclude_beats_handles_consecutive_spans():
    beats = _beats([5.0, 5.5, 12.0, 25.0, 25.5, 40.0, 40.5])
    kept = exclude_beats(beats, SignalQuality(50.0, ((6.0, 20.0), (30.0, 35.0))))
    assert [(b.time, b.rr_interval) for b in kept] == [
        (5.0, None), (5.5, 0.5), (25.0, None), (25.5, 0.5), (40.0, None), (40.5, 0.5)
    ]


def test_lead_off_tone_is_excluded_even_at_ecg_like_amplitude():
    x = _sine(600)
    x[_at(200, 210)] = _tone(10)  # 1.0 peak-to-peak: inside the amplitude band
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (198.0, 212.0), (540.0, 600.0))


def test_amplitude_reference_ignores_lead_off_windows():
    # 400 s of faint tone would drag the median to 0.3 and make the 2.0 ECG
    # look like a high-amplitude burst.
    x = np.concatenate([_sine(200), _tone(400, amplitude=0.15)])
    q = assess_quality(x, FS)
    assert q.excluded == ((0.0, 60.0), (198.0, 600.0))
    assert q.duration_sec == 600.0
    assert q.trimmed_sec == 0.0
