import os
import numpy as np
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.detection.detect import detect_beats, _qrs_width

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
    # The energy-envelope QRS width measurement (replacing NeuroKit2's wave
    # delineation, which returned NaN for 19/19 ground-truth PVC beats in
    # this same fixture) should succeed for nearly every beat, not just
    # "at least one" - confirmed 65/65 (100%) on this fixture; 90% leaves
    # slack for incidental edge-of-recording beats without masking a
    # regression back toward the old near-total-failure behavior.
    non_none = sum(1 for b in beats if b.qrs_duration is not None)
    assert non_none / len(beats) >= 0.9, (
        f"only {non_none}/{len(beats)} beats had a non-None qrs_duration"
    )


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


def _gaussian_pulse(t, center, sigma, amplitude=2.0):
    return amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def test_wide_synthetic_beat_measures_wider_than_narrow_synthetic_beat():
    # Build a deterministic synthetic signal: a narrow Gaussian "QRS" pulse
    # (sigma=8ms, roughly a normal QRS) followed by a wide one (sigma=40ms,
    # roughly a PVC-like wide complex), at known sample positions. This
    # exercises the envelope-based _qrs_width measurement directly through
    # the public detect_beats() entry point without depending on any real
    # ECG fixture.
    sample_rate = 500.0
    duration_sec = 4.0
    n = int(duration_sec * sample_rate)
    t = np.arange(n) / sample_rate
    signal = np.zeros(n)
    signal += _gaussian_pulse(t, center=1.0, sigma=0.008)  # narrow beat
    signal += _gaussian_pulse(t, center=2.0, sigma=0.040)  # wide beat

    beats = detect_beats(signal, sample_rate)

    assert len(beats) == 2
    narrow_beat, wide_beat = beats
    assert narrow_beat.qrs_duration is not None
    assert wide_beat.qrs_duration is not None
    assert wide_beat.qrs_duration > narrow_beat.qrs_duration


def test_qrs_width_finds_onset_at_clamped_search_window_start():
    # Regression test for an off-by-one where the backward (onset) scan used
    # range(r_peak, lo, -1), which never evaluates envelope[lo] - invisible
    # unless the true threshold crossing sits exactly at the clamped window
    # boundary (lo=0, i.e. an R-peak within the search window of sample 0).
    # envelope[0] is the only sample below threshold on the onset side;
    # a scan that skips index 0 finds no onset and wrongly returns None.
    sample_rate = 500.0
    envelope = np.array([0.5, 5.0, 10.0, 0.5, 0.5])
    r_peak = 2
    search_half = 75  # larger than r_peak, so lo clamps to 0

    duration = _qrs_width(envelope, r_peak, search_half, sample_rate)

    assert duration is not None
    onset_index, offset_index = 0, 3
    assert duration == (offset_index - onset_index) / sample_rate


def test_qrs_width_returns_none_for_zero_energy_peak():
    # An R-peak sitting on a flat (zero-energy) stretch of the envelope has no
    # measurable width, so the threshold is meaningless and width is undefined.
    envelope = np.zeros(50)
    assert _qrs_width(envelope, r_peak=25, search_half=20, sample_rate=180.0) is None


def test_qrs_width_returns_none_when_envelope_never_drops_below_threshold():
    # A monotonically rising envelope that stays above the threshold on both
    # sides of the peak within the search window yields no onset/offset
    # crossing, so no width can be measured.
    envelope = np.arange(1, 51, dtype=float)
    r_peak = 49  # the peak is the maximum; nothing to its right, left stays high
    assert _qrs_width(envelope, r_peak=r_peak, search_half=3, sample_rate=180.0) is None


# --- phantom-beat rejection ---------------------------------------------------
# Teeny's first recording (2026-08-23) had 54 of 59 "PVCs" sitting on flat
# baseline: 0.03-0.11 mV peak-to-peak at the flagged instant vs ~2.9 mV for
# a real R-wave. The R-peak detector invents beats inside long RR gaps at
# slow resting rates, and those phantoms cascade into false PVCs. Peaks
# far below the recording's own median R amplitude are dropped.
from canine_holter.detection.detect import _reject_low_amplitude_peaks


def _pulse_signal(sample_rate, peak_samples, amplitudes, n):
    """Zeros with a one-sample spike of the given amplitude at each peak."""
    sig = np.zeros(n)
    for p, a in zip(peak_samples, amplitudes):
        sig[p] = a
    return sig


def test_reject_drops_peak_far_below_median_amplitude():
    sr = 100.0
    peaks = np.array([100, 200, 250, 300, 400])
    sig = _pulse_signal(sr, peaks, [2.0, 2.0, 0.05, 2.0, 2.0], 500)
    kept = _reject_low_amplitude_peaks(sig, peaks, sr)
    assert kept.tolist() == [100, 200, 300, 400]


def test_reject_keeps_every_peak_when_amplitudes_are_uniform():
    sr = 100.0
    peaks = np.array([100, 200, 300, 400])
    sig = _pulse_signal(sr, peaks, [2.0, 2.0, 2.0, 2.0], 500)
    kept = _reject_low_amplitude_peaks(sig, peaks, sr)
    assert kept.tolist() == [100, 200, 300, 400]


def test_reject_keeps_a_genuinely_smaller_but_real_beat():
    # A beat at half the median amplitude is a plausible small PVC and must
    # survive; only peaks under MIN_R_AMPLITUDE_FRACTION (0.2) are phantoms.
    sr = 100.0
    peaks = np.array([100, 200, 300, 400])
    sig = _pulse_signal(sr, peaks, [2.0, 1.0, 2.0, 2.0], 500)
    kept = _reject_low_amplitude_peaks(sig, peaks, sr)
    assert kept.tolist() == [100, 200, 300, 400]


def test_reject_keeps_a_negative_qrs_beat_whose_peak_sits_on_its_small_r_wave():
    # Teeny's 2026-08-25 recording: lying down, the analysis lead's QRS is a
    # small positive r then a deep S ~80 ms later. NeuroKit puts the peak on
    # the r, and a +/-60 ms window never reaches the S, so 552 of the
    # report's 1209 "pauses" (13.04 s longest) were real beats thrown away.
    sr = 180.0
    peaks = np.array([200, 400, 600, 800])
    sig = _pulse_signal(sr, peaks, [2.0, 2.0, 0.25, 2.0], 1000)
    sig[600 + 15] = -1.0  # the S wave, 83 ms after the r the detector chose
    kept = _reject_low_amplitude_peaks(sig, peaks, sr)
    assert kept.tolist() == [200, 400, 600, 800]


def test_reject_returns_no_peaks_when_median_amplitude_is_zero():
    # Every "peak" on flat signal: there is nothing to vouch for, fail closed.
    sr = 100.0
    peaks = np.array([100, 200, 300])
    kept = _reject_low_amplitude_peaks(np.zeros(500), peaks, sr)
    assert kept.tolist() == []


def test_detect_beats_reports_gap_as_one_long_rr_after_dropping_phantom(monkeypatch):
    # NeuroKit's peak set is pinned so the test is about what detect_beats
    # does with a phantom, not about whether NeuroKit produces one.
    import neurokit2 as nk
    sr = 100.0
    real = [100, 200, 300, 500, 600]
    phantom = 400  # inside the 300->500 gap, on baseline
    sig = _pulse_signal(sr, real, [2.0] * len(real), 700)
    monkeypatch.setattr(nk, "ecg_clean", lambda s, sampling_rate: s)
    monkeypatch.setattr(
        nk, "ecg_peaks",
        lambda s, sampling_rate: (None, {"ECG_R_Peaks": np.array(sorted(real + [phantom]))}),
    )
    beats = detect_beats(sig, sr)
    assert [b.time for b in beats] == [1.0, 2.0, 3.0, 5.0, 6.0]
    assert beats[3].rr_interval == 2.0


# --- rate-gated search-back ---------------------------------------------------
# NeuroKit's threshold rises with beat density until it sits on the QRS at
# ~150 bpm and misses beats in fragments; the classifier then reads normal
# beats against an inflated RR baseline as a "VT run". A gap of more than
# 1.5x the local RR in fast rhythm is a missed beat, never sinus arrhythmia.
from canine_holter.detection.detect import fill_fast_gaps


def _spike_signal(sample_rate, times_sec, amplitudes, duration_sec):
    """Zeros with a one-sample spike of the given amplitude at each time."""
    sig = np.zeros(int(duration_sec * sample_rate))
    for t, a in zip(times_sec, amplitudes):
        sig[int(round(t * sample_rate))] = a
    return sig


def test_fill_fast_gaps_adds_the_missed_beat_in_a_fast_rhythm():
    sr = 200.0
    times = [t * 0.4 for t in range(12)]  # 150 bpm
    sig = _spike_signal(sr, times, [2.0] * 12, 5.0)
    peaks = np.array([int(round(t * sr)) for t in times if t != 2.0])  # beat 5 missed by the detector
    filled = fill_fast_gaps(sig, peaks, sr)
    assert filled.tolist() == [int(round(t * sr)) for t in times]


def test_fill_fast_gaps_leaves_a_gap_in_slow_rhythm_alone():
    # 1.2 s -> 2.4 s is sinus arrhythmia territory, not a missed beat, even
    # with a candidate spike sitting inside the gap.
    sr = 200.0
    times = [t * 1.2 for t in range(8)]
    sig = _spike_signal(sr, times, [2.0] * 8, 10.0)
    peaks = np.array([int(round(t * sr)) for t in times if t != 6.0])
    assert fill_fast_gaps(sig, peaks, sr).tolist() == peaks.tolist()


def test_fill_fast_gaps_ignores_a_candidate_far_below_the_neighbours():
    sr = 200.0
    times = [t * 0.4 for t in range(12)]
    amps = [2.0] * 12
    amps[5] = 0.1  # 5% of the neighbours: noise, not a beat
    sig = _spike_signal(sr, times, amps, 5.0)
    peaks = np.array([int(round(t * sr)) for t in times if t != 2.0])
    assert fill_fast_gaps(sig, peaks, sr).tolist() == peaks.tolist()


def test_fill_fast_gaps_keeps_one_candidate_per_refractory_period():
    # Two spikes 130 ms apart inside the gap: only one can be a beat, and it
    # is the larger one.
    sr = 200.0
    times = [t * 0.4 for t in range(12)]
    sig = _spike_signal(sr, times, [2.0] * 12, 5.0)
    sig[int(round(2.0 * sr))] = 0.0
    sig[int(round(1.95 * sr))] = 2.0
    sig[int(round(2.08 * sr))] = 1.5
    peaks = np.array([int(round(t * sr)) for t in times if t != 2.0])
    filled = fill_fast_gaps(sig, peaks, sr)
    assert filled.tolist() == sorted(peaks.tolist() + [int(round(1.95 * sr))])


# --- interpolated T-wave rejection --------------------------------------------
# Lying down, Teeny's analysis lead shows the QRS as a ~0.3 mV spike and the
# T wave as a ~0.7 mV trough 0.2-0.35 s later; past NeuroKit's 300 ms minimum
# spacing the T wave is detected as a second beat, early and "wide", and
# labelled V (38 in one hour of the 2026-08-25 report). The gradient feature
# cannot tell them apart - the broad T scores higher than the tiny spike -
# but timing can: removing a T-wave detection leaves the rhythm undisturbed,
# while a PVC that early resets it or is followed by a compensatory pause.
from canine_holter.detection.detect import drop_interpolated_t_waves


def _peaks(sr, times_sec):
    return np.array([int(round(t * sr)) for t in times_sec])


def test_drop_interpolated_t_waves_removes_an_interpolated_candidate_in_slow_rhythm():
    sr = 100.0
    beats = [t * 1.2 for t in range(10)]
    with_t = sorted(beats + [6.0 + 0.35])  # a "beat" 350 ms after beat 5, rhythm unchanged
    assert drop_interpolated_t_waves(_peaks(sr, with_t), sr).tolist() == _peaks(sr, beats).tolist()


def test_drop_interpolated_t_waves_keeps_a_pvc_followed_by_a_compensatory_pause():
    sr = 100.0
    beats = [t * 1.2 for t in range(6)] + [6.0 + 0.35] + [t * 1.2 for t in range(7, 10)]  # 6.0 -> 6.35 -> 8.4
    peaks = _peaks(sr, beats)
    assert drop_interpolated_t_waves(peaks, sr).tolist() == peaks.tolist()


def test_drop_interpolated_t_waves_keeps_an_early_beat_that_resets_the_rhythm_by_more_than_the_tolerance():
    # 6.0 -> 6.4 -> 8.0: the next beat is 2.0 s after A, 1.67x the local RR - not one RR.
    sr = 100.0
    beats = [t * 1.2 for t in range(6)] + [6.4, 8.0, 9.2, 10.4]
    peaks = _peaks(sr, beats)
    assert drop_interpolated_t_waves(peaks, sr).tolist() == peaks.tolist()


def test_drop_interpolated_t_waves_is_off_in_fast_rhythm():
    # At 0.4 s RR a candidate 0.35 s after a beat is simply the next beat;
    # no neighbouring interval is slow enough to be a sinus reference.
    sr = 100.0
    beats = [0.0, 0.4, 0.8, 1.2, 1.55, 1.95, 2.35, 2.75]
    peaks = _peaks(sr, beats)
    assert drop_interpolated_t_waves(peaks, sr).tolist() == peaks.tolist()


def test_drop_interpolated_t_waves_follows_sinus_arrhythmia_beat_to_beat():
    # RR lengthening 1.2 -> 1.5 -> 1.8: the T wave after the 1.5 s beat is
    # judged against its neighbours, not a running average.
    sr = 100.0
    beats = [0.0, 1.2, 2.4, 3.6, 5.1, 6.9, 8.7]
    with_t = sorted(beats + [5.1 + 0.35])
    assert drop_interpolated_t_waves(_peaks(sr, with_t), sr).tolist() == _peaks(sr, beats).tolist()


def test_drop_interpolated_t_waves_skips_the_next_beats_own_t_wave_when_looking_ahead():
    # Both beats carry a detected T wave; the second T must not hide the
    # sinus interval the first is compared with.
    sr = 100.0
    beats = [0.0, 1.4, 2.8, 4.2, 5.6, 7.0, 8.4]
    with_t = sorted(beats + [4.2 + 0.35, 5.6 + 0.35])
    assert drop_interpolated_t_waves(_peaks(sr, with_t), sr).tolist() == _peaks(sr, beats).tolist()


def test_drop_interpolated_t_waves_needs_a_rhythm_history_first():
    # Three accepted intervals before it can call the rhythm slow; until
    # then even a perfect interpolation pattern is kept.
    sr = 100.0
    peaks = _peaks(sr, [0.0, 1.2, 1.55, 2.4, 3.6])
    assert drop_interpolated_t_waves(peaks, sr).tolist() == peaks.tolist()


# --- QRS width above the noise floor ------------------------------------------
# Hash noise keeps the derivative-energy envelope above 10% of its peak, so
# the width crossing lands on the noise, not the QRS edge: 62 of 93 "PVCs"
# on Teeny's 2026-08-25 report were normal beats 3-5 samples "wider" than
# baseline. The threshold now also clears the local noise floor.
from canine_holter.detection.detect import _qrs_energy_envelope


def _triangle_beat(sample_rate, center_sec, width_sec, amplitude, duration_sec):
    sig = np.zeros(int(duration_sec * sample_rate))
    half = int(width_sec * sample_rate / 2)
    center = int(center_sec * sample_rate)
    ramp = np.linspace(0, amplitude, half + 1)
    sig[center - half: center + 1] = ramp
    sig[center: center + half + 1] = ramp[::-1]
    return sig


def test_qrs_width_is_unchanged_by_noise_that_would_otherwise_widen_it():
    sr = 180.0
    clean = _triangle_beat(sr, 2.0, 0.06, 2.0, 4.0)
    noise = np.random.default_rng(0).normal(0, 0.08, clean.shape)
    search_half = int(0.15 * sr)
    peak = int(2.0 * sr)
    width_clean = _qrs_width(_qrs_energy_envelope(clean, sr), peak, search_half, sr)
    width_noisy = _qrs_width(_qrs_energy_envelope(clean + noise, sr), peak, search_half, sr)
    assert abs(width_noisy - width_clean) <= 2.5 / sr  # within two samples either way


def test_qrs_width_is_none_for_a_beat_buried_in_noise():
    sr = 180.0
    clean = _triangle_beat(sr, 2.0, 0.06, 0.05, 4.0)
    noise = np.random.default_rng(1).normal(0, 0.5, clean.shape)
    assert _qrs_width(_qrs_energy_envelope(clean + noise, sr), int(2.0 * sr), int(0.15 * sr), sr) is None


def test_drop_interpolated_t_waves_acts_in_a_moderate_rhythm_under_100_bpm():
    # 0.7 s rhythm (86 bpm): the T wave still lands 0.3 s after the QRS, past
    # NeuroKit's 300 ms minimum spacing. 18 of the 93 "PVCs" on Teeny's
    # 2026-08-25 report were T waves in a 0.75-0.85 s rhythm.
    sr = 100.0
    beats = [t * 0.7 for t in range(10)]
    with_t = sorted(beats + [3.5 + 0.3])
    assert drop_interpolated_t_waves(_peaks(sr, with_t), sr).tolist() == _peaks(sr, beats).tolist()
