# Beat detection: tachycardia search-back and T-wave rejection - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `detect_beats` missing beats at tachycardia and detecting T waves as beats in the lying-down morphology, with the first canine ground-truth fixtures in CI.

**Architecture:** NeuroKit stays the primary detector. Two pure post-passes in `detection/detect.py` - `fill_fast_gaps` (rate-gated search-back) and `drop_interpolated_t_waves` (timing rule) - run between the phantom filter and width measurement. Three ~10-20 s slices of Teeny's 2026-08-25 recording with hand-counted beat times become fixtures.

**Tech Stack:** numpy, scipy (`uniform_filter1d`, `find_peaks`), neurokit2, pytest. Spec: `docs/superpowers/specs/2026-08-26-detector-tachycardia-and-t-wave-design.md`.

---

## File structure

- `scripts/extract_teeny_fixtures.py` - one-off, documents provenance: reads a `flash.dat`, writes the three `.npz` fixtures with their hand-counted beat times.
- `tests/fixtures/teeny_2026-08-25/{tachy,lying,quiet}.npz` - `channels` (3, n) float32 mV, `sample_rate`, `beat_times` (s, hand-counted), `offset_sec` (position in the recording).
- `tests/detection/test_teeny_fixtures.py` - sensitivity/precision of `detect_beats` on each fixture (acceptance tests).
- `src/canine_holter/detection/detect.py` - the two new functions and their constants; `detect_beats` wired.
- `tests/detection/test_detect.py` - synthetic unit tests for each rule.
- `CLAUDE.md` - detection paragraph and the "Known limits" item.

---

### Task 1: Fixtures and acceptance tests

**Files:**
- Create: `scripts/extract_teeny_fixtures.py`
- Create: `tests/fixtures/teeny_2026-08-25/tachy.npz`, `lying.npz`, `quiet.npz`
- Create: `tests/detection/test_teeny_fixtures.py`

- [ ] **Step 1: Write the extraction script**

```python
"""One-off: cut the hand-counted windows of Teeny's 2026-08-25 recording
into test fixtures. Run manually with the flash.dat path; the fixtures are
committed so the suite never needs the recording.

    .venv/bin/python scripts/extract_teeny_fixtures.py ~/Downloads/teeny-holter-2026-08-26/flash.dat

Beat times were read by eye from zoomed three-channel plots (see the
2026-08-26 detector spec) and snapped to the steepest sample of the
cleaned analysis lead within +/-80 ms.
"""
import os
import sys
import numpy as np
from canine_holter.ingest.loader import load_recording

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "teeny_2026-08-25")

# name: (offset from recording start in s, duration s, beat times within the window)
WINDOWS = {
    # 08:47:52.45, sinus tachycardia ~150 bpm; NeuroKit finds ~16 of these 23
    "tachy": (44560.45, 9.55, [0.222, 0.556, 0.900, 1.267, 1.633, 2.006, 2.394, 2.794, 3.194, 3.594, 3.994, 4.389, 4.800, 5.206, 5.639, 6.111, 6.617, 7.139, 7.589, 7.978, 8.344, 8.844, 9.361]),
    # 15:22:53, lying down: a 0.3 mV QRS spike then a 0.7 mV T trough 0.2-0.35 s later
    "lying": (68257.0, 20.0, [1.006, 2.900, 4.167, 4.950, 7.078, 9.122, 10.406, 11.172, 13.028, 15.317, 16.578, 17.372, 19.606]),
    # 17:06:18, upright and still, with a real 4.67 s sinus pause
    "quiet": (74462.0, 16.0, [2.456, 4.483, 5.961, 7.367, 12.033, 13.839, 15.672]),
}


def main(flash_path: str) -> None:
    rec = load_recording(flash_path)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (offset, duration, beat_times) in WINDOWS.items():
        i0, i1 = int(offset * rec.sample_rate), int((offset + duration) * rec.sample_rate)
        np.savez(
            os.path.join(OUT_DIR, f"{name}.npz"),
            channels=rec.channels[:, i0:i1].astype(np.float32),
            sample_rate=rec.sample_rate,
            beat_times=np.array(beat_times),
            offset_sec=offset,
        )
        print(f"{name}: {duration} s, {len(beat_times)} beats")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/extract_teeny_fixtures.py ~/Downloads/teeny-holter-2026-08-26/flash.dat && ls -la tests/fixtures/teeny_2026-08-25/`
Expected: three files, ~20-45 KB each.

- [ ] **Step 3: Write the acceptance tests**

```python
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
    ("lying", 0.90, 0.95),   # small QRS spike + large T trough: the T-wave rule's reason to exist
    ("quiet", 1.00, 1.00),   # 7 beats and a real 4.67 s pause; nine phantom candidates on flat baseline
])
def test_detect_beats_on_teeny_window(name, min_sensitivity, min_precision):
    samples, sample_rate, truth = _load(name)
    detected = [b.time for b in detect_beats(samples, sample_rate)]
    sensitivity, precision = _sensitivity_precision(detected, truth)
    assert sensitivity >= min_sensitivity, f"{name}: found {len(detected)} of {len(truth)} beats; sensitivity {sensitivity:.2f}"
    assert precision >= min_precision, f"{name}: {len(detected)} detections for {len(truth)} beats; precision {precision:.2f}"
```

- [ ] **Step 4: Run them to see which fail today**

Run: `.venv/bin/python -m pytest tests/detection/test_teeny_fixtures.py -v`
Expected: `tachy` FAILS on sensitivity (NeuroKit misses ~7 of 23), `lying` FAILS on precision (T waves detected), `quiet` PASSES. Record the actual numbers in the commit message.

- [ ] **Step 5: Commit the fixtures and the red tests**

```bash
git add scripts/extract_teeny_fixtures.py tests/fixtures/teeny_2026-08-25 tests/detection/test_teeny_fixtures.py
git commit -m "Tests: hand-counted Teeny 2026-08-25 windows as canine ground truth (tachy and lying red)"
```

---

### Task 2: `fill_fast_gaps`

**Files:**
- Modify: `src/canine_holter/detection/detect.py`
- Test: `tests/detection/test_detect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/detection/test_detect.py`:

```python
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
    # Two full-size spikes 100 ms apart inside the gap: only one can be a beat.
    sr = 200.0
    times = [t * 0.4 for t in range(12)]
    sig = _spike_signal(sr, times, [2.0] * 12, 5.0)
    sig[int(round(2.1 * sr))] = 2.0
    peaks = np.array([int(round(t * sr)) for t in times if t != 2.0])
    filled = fill_fast_gaps(sig, peaks, sr)
    assert len(filled) == len(peaks) + 1
    assert int(round(2.0 * sr)) in filled.tolist()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/detection/test_detect.py -k fill_fast_gaps -v`
Expected: 4 FAIL / ERROR with `ImportError: cannot import name 'fill_fast_gaps'`.

- [ ] **Step 3: Implement**

In `src/canine_holter/detection/detect.py`, add the imports and constants:

```python
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
```

```python
# Search-back for beats the detector missed at fast rates. NeuroKit's
# threshold is 1.5x a 0.75 s mean of the gradient; at ~150 bpm that mean
# rises until the threshold sits on the QRS and beats are lost in
# fragments. In fast rhythm a gap over GAP_FACTOR x the local RR is a missed
# beat; in slow rhythm it is sinus arrhythmia, so the pass is off there
# (and T waves, which sit inside the refractory at fast rates, cannot be
# filled in). LOCAL_RR_BEATS is the rhythm memory shared with the T-wave rule.
LOCAL_RR_BEATS = 8
FAST_RR_SEC = 0.8  # local median RR under this (>= 75 bpm) is "fast rhythm"
GAP_FACTOR = 1.5
FILL_FEATURE_FRACTION = 0.35  # candidate gradient feature vs the neighbouring beats' median
FILL_REFRACTORY_SEC = 0.25  # enforced after the fiducial is placed, against both neighbours
GRADIENT_SMOOTH_SEC = 0.1  # NeuroKit's own feature: |gradient| boxcar-smoothed
FIDUCIAL_HALF_SEC = 0.05  # the beat is the largest deflection from baseline this close to the steepest point
```

and the function:

```python
def _gradient_feature(cleaned: np.ndarray, sample_rate: float) -> np.ndarray:
    return uniform_filter1d(np.abs(np.gradient(cleaned)), max(1, int(GRADIENT_SMOOTH_SEC * sample_rate)))


def _fiducial(cleaned: np.ndarray, index: int, sample_rate: float) -> int:
    """The largest |deflection| from the preceding 200 ms baseline within
    FIDUCIAL_HALF_SEC of index - polarity-agnostic, so a negative QRS lands
    on its trough rather than its small r wave."""
    half = int(FIDUCIAL_HALF_SEC * sample_rate)
    lo, hi = max(0, index - half), min(len(cleaned), index + half + 1)
    baseline = np.median(cleaned[max(0, index - int(0.2 * sample_rate)): index]) if index > 0 else 0.0
    return lo + int(np.argmax(np.abs(cleaned[lo:hi] - baseline)))


def fill_fast_gaps(cleaned: np.ndarray, peaks: np.ndarray, sample_rate: float) -> np.ndarray:
    """Add beats inside gaps that are implausible for a fast local rhythm.

    Only acts when the median of the previous LOCAL_RR_BEATS RRs is under
    FAST_RR_SEC and the gap exceeds GAP_FACTOR times it. Candidates are
    peaks of the gradient feature at least FILL_FEATURE_FRACTION of the
    surrounding beats' feature, placed at their fiducial, at least
    FILL_REFRACTORY_SEC from the previous accepted peak and from the gap's end.
    """
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return peaks
    feature = _gradient_feature(cleaned, sample_rate)
    half = int(GRADIENT_SMOOTH_SEC * sample_rate)
    peak_feature = np.array([feature[max(0, p - half): p + half + 1].max() for p in peaks])
    refractory = int(FILL_REFRACTORY_SEC * sample_rate)
    added = []
    for i in range(1, len(peaks)):
        a, b = peaks[i - 1], peaks[i]
        previous_rr = np.diff(peaks[max(0, i - 1 - LOCAL_RR_BEATS): i]) / sample_rate
        if len(previous_rr) < 3:
            continue
        local_rr = float(np.median(previous_rr))
        if local_rr >= FAST_RR_SEC or (b - a) / sample_rate <= GAP_FACTOR * local_rr:
            continue
        reference = float(np.median(peak_feature[max(0, i - 1 - LOCAL_RR_BEATS): i + 1]))
        lo, hi = a + refractory, b - refractory
        if hi <= lo:
            continue
        candidates, _ = find_peaks(feature[lo:hi], height=FILL_FEATURE_FRACTION * reference, distance=refractory)
        last = a
        for candidate in candidates:
            fiducial = _fiducial(cleaned, lo + candidate, sample_rate)
            if fiducial - last >= refractory and b - fiducial >= refractory:
                added.append(fiducial)
                last = fiducial
    return np.array(sorted(set(peaks.tolist()) | set(added)), dtype=int)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/detection/test_detect.py -k fill_fast_gaps -v`
Expected: 4 PASS. If the candidate lands one sample off the spike, the fiducial search is the culprit - a one-sample spike's gradient peaks on the sample before it; the `_fiducial` argmax of |x - baseline| corrects that, so a failure here means `FIDUCIAL_HALF_SEC` did not reach the spike.

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/detection/detect.py tests/detection/test_detect.py
git commit -m "Detection: fill_fast_gaps - rate-gated search-back for beats missed at tachycardia"
```

---

### Task 3: `drop_interpolated_t_waves`

**Files:**
- Modify: `src/canine_holter/detection/detect.py`
- Test: `tests/detection/test_detect.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/detection/test_detect.py`:

```python
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
    # At 0.4 s RR a candidate 0.35 s after a beat is simply the next beat.
    sr = 100.0
    beats = [t * 0.4 for t in range(10)] + [3.6 + 0.35]
    peaks = _peaks(sr, sorted(beats))
    assert drop_interpolated_t_waves(peaks, sr).tolist() == peaks.tolist()


def test_drop_interpolated_t_waves_needs_a_rhythm_history_first():
    sr = 100.0
    peaks = _peaks(sr, [0.0, 1.2, 1.55, 2.4])  # too few RRs to know the rhythm
    assert drop_interpolated_t_waves(peaks, sr).tolist() == peaks.tolist()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/detection/test_detect.py -k drop_interpolated -v`
Expected: 5 FAIL with `ImportError: cannot import name 'drop_interpolated_t_waves'`.

- [ ] **Step 3: Implement**

Constants, beside the search-back ones:

```python
# T-wave rejection in slow rhythm. Lying down, Teeny's analysis lead shows
# the QRS as a small spike and the T wave as a large trough 0.2-0.35 s
# later; past NeuroKit's 300 ms minimum spacing the T is detected as a
# beat. The gradient feature cannot separate them (the broad T scores ~2x
# the spike), so the rule is timing: a candidate this soon after a beat,
# whose removal leaves the beat-to-beat interval equal to the local RR, is
# interpolated - a T wave. A PVC that early resets the rhythm or is
# followed by a compensatory pause. Known cost: a genuinely interpolated
# R-on-T PVC at rest is dropped too.
SLOW_RR_SEC = 0.8  # local median RR over this (< 75 bpm) is "slow rhythm"
T_WAVE_MAX_COUPLING_SEC = 0.45
T_WAVE_RHYTHM_TOLERANCE = 0.25  # |A->C - local RR| within this fraction means B was interpolated
```

Function:

```python
def drop_interpolated_t_waves(peaks: np.ndarray, sample_rate: float) -> np.ndarray:
    """Drop a peak B that follows A within T_WAVE_MAX_COUPLING_SEC in slow
    rhythm when the next peak C sits one local RR after A - B is a T wave
    interpolated into an undisturbed rhythm. Sequential and causal: the
    local RR is the median of the last LOCAL_RR_BEATS accepted intervals."""
    times = np.asarray(peaks, dtype=int) / sample_rate
    keep = np.ones(len(times), dtype=bool)
    rr_history: list[float] = []
    i = 1
    while i < len(times) - 1:
        a, b, c = times[i - 1], times[i], times[i + 1]
        local_rr = float(np.median(rr_history[-LOCAL_RR_BEATS:])) if len(rr_history) >= 3 else None
        if (
            local_rr is not None
            and local_rr > SLOW_RR_SEC
            and (b - a) < T_WAVE_MAX_COUPLING_SEC
            and abs((c - a) - local_rr) < T_WAVE_RHYTHM_TOLERANCE * local_rr
        ):
            keep[i] = False
            rr_history.append(c - a)
            i += 2
            continue
        rr_history.append(b - a)
        i += 1
    return np.asarray(peaks, dtype=int)[keep]
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/detection/test_detect.py -k drop_interpolated -v`
Expected: 5 PASS. Check the "resets" test by hand: local RR 1.2, C - A = 2.0, |2.0 - 1.2| = 0.8 > 0.25 x 1.2 = 0.3, kept. Compensatory: C - A = 2.4, kept.

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/detection/detect.py tests/detection/test_detect.py
git commit -m "Detection: drop_interpolated_t_waves - timing rule for T waves detected as beats in slow rhythm"
```

---

### Task 4: Wire the passes into `detect_beats`

**Files:**
- Modify: `src/canine_holter/detection/detect.py` (`detect_beats`)
- Test: `tests/detection/test_teeny_fixtures.py` (from Task 1, currently red)

- [ ] **Step 1: Confirm the acceptance tests are still red**

Run: `.venv/bin/python -m pytest tests/detection/test_teeny_fixtures.py -v`
Expected: `tachy` and `lying` FAIL.

- [ ] **Step 2: Wire**

In `detect_beats`, replace

```python
    r_peaks = _reject_low_amplitude_peaks(cleaned, r_info["ECG_R_Peaks"], sample_rate)
```

with

```python
    r_peaks = _reject_low_amplitude_peaks(cleaned, r_info["ECG_R_Peaks"], sample_rate)
    r_peaks = fill_fast_gaps(cleaned, r_peaks, sample_rate)
    r_peaks = drop_interpolated_t_waves(r_peaks, sample_rate)
```

and extend the docstring with one paragraph:

```
    Two post-passes correct NeuroKit's two known failure modes on Teeny's
    recordings: fill_fast_gaps recovers beats missed at tachycardia, and
    drop_interpolated_t_waves removes T waves detected as beats in slow
    rhythm. See docs/superpowers/specs/2026-08-26-detector-tachycardia-and-t-wave-design.md.
```

- [ ] **Step 3: Run the acceptance tests and the full suite**

Run: `.venv/bin/python -m pytest tests/detection/test_teeny_fixtures.py -v && .venv/bin/python -m pytest -q`
Expected: all three fixtures PASS; whole suite green (MIT-BIH 119 sensitivity unchanged). If `tachy` sensitivity is below 0.90, print the detected vs truth times and check whether the misses are at the window start (no rhythm history yet - acceptable, lower the window's threshold to what the data supports and say so in the commit) or in the middle (a real gap the search-back did not fill - inspect `FILL_FEATURE_FRACTION` against the printed feature values before touching it). If `lying` precision is below 0.95, print the surviving extra detections: a T wave surviving means `(c - a)` fell outside the tolerance for that beat - report the ratio; do not widen the tolerance beyond 0.3 (the spec's ceiling).

- [ ] **Step 4: Commit**

```bash
git add src/canine_holter/detection/detect.py
git commit -m "Detection: run the search-back and T-wave passes in detect_beats"
```

---

### Task 5: Validate on both recordings, document, PR

**Files:**
- Modify: `CLAUDE.md` (detection paragraph; "Known limits" item on tachycardia misses)
- Modify: `docs/superpowers/specs/2026-08-26-detector-tachycardia-and-t-wave-design.md` (fill in the measured "after" column)

- [ ] **Step 1: Regenerate both reports and tabulate**

Run:
```bash
.venv/bin/canine-holter ~/Downloads/teeny-holter-2026-08-26/flash.dat --out ~/Downloads/teeny-holter-2026-08-26/
.venv/bin/canine-holter samples/teeny-2026-08-23/flash.dat --out ~/Downloads/teeny-holter-2026-08-23/
```
Read page 1 of each PDF and record: total beats, PVCs, couplets, triplets, VT runs, pauses, longest pause, HR min/mean/max. Compare with the spec's "before" column (2026-08-25: 85863 beats, 96 PVCs, 2 couplets, 1 triplet, 1 VT run, 895 pauses, 9.69 s, 23/59/193; 2026-08-23: 7 PVCs, 1 couplet, 31 pauses, 2.97 s).

- [ ] **Step 2: Update the spec's results table with the measured numbers and CLAUDE.md**

CLAUDE.md `detection/detect.py` bullet: append "Two post-passes then correct NeuroKit's known failure modes: `fill_fast_gaps` (rate-gated search-back; at ~150 bpm NeuroKit's threshold sits on the QRS and misses beats, which the classifier then reads as a VT run) and `drop_interpolated_t_waves` (lying down, the T wave is detected as a beat 0.2-0.35 s after a small QRS spike; a candidate that early in slow rhythm whose removal leaves the rhythm undisturbed is a T wave). Both are gated on the local rhythm so each can only act where its failure mode exists; the 2026-08-26 detector spec has the bake-off of every alternative. `tests/fixtures/teeny_2026-08-25/` holds three hand-counted windows of a real recording - the canine ground truth these are measured against."

CLAUDE.md "Known limits": replace the "Beat detection misses beats during tachycardia" item with "Beat detection: remaining PVC false positives are motion noise in active hours (the 08:25-09:25 hour of the 2026-08-25 recording); per-beat noise rejection (template-correlation SQI) is the next detection item and depends on this one. Lying-down morphology is a lead-axis problem - Ch 3 shows a 3.5 mV QRS where Ch 1 shows 0.3 mV - and multi-lead detection is the structural fix."

- [ ] **Step 3: Commit, push, open the PR**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-08-26-detector-tachycardia-and-t-wave-design.md
git commit -m "Docs: detector post-passes and remaining limits"
git push -u origin detector-tachycardia-t-wave
gh pr create --title "Detection: search-back for tachycardia misses and T-wave rejection, with canine ground-truth fixtures" --body-file -
```
PR body = the spec's Problem and Design summaries plus the measured before/after table; end with the generated-with footer. Watch CI: `gh pr checks --watch`.
