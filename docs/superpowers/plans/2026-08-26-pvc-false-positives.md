# PVC false positives - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the 93 PVCs on Teeny's 2026-08-25 report to the ~20 the strips support, by measuring QRS width above the local noise floor, requiring an absolute width margin, and lowering the T-wave rule's rhythm floor.

**Architecture:** Three constants-and-conditions changes in existing functions - `_qrs_width` (detection), the wide test in `classify_beats` (classify), `SLOW_RR_SEC` (detection). No new modules, no contract changes. Spec: `docs/superpowers/specs/2026-08-26-pvc-false-positives-design.md`.

**Tech Stack:** numpy, pytest.

---

### Task 1: Noise-floor QRS width

**Files:**
- Modify: `src/canine_holter/detection/detect.py` (`_qrs_width`, new constant)
- Test: `tests/detection/test_detect.py`

- [ ] **Step 1: Failing tests**

Append to `tests/detection/test_detect.py`:

```python
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
    assert abs(width_noisy - width_clean) <= 2 / sr


def test_qrs_width_is_none_for_a_beat_buried_in_noise():
    sr = 180.0
    clean = _triangle_beat(sr, 2.0, 0.06, 0.05, 4.0)
    noise = np.random.default_rng(1).normal(0, 0.5, clean.shape)
    assert _qrs_width(_qrs_energy_envelope(clean + noise, sr), int(2.0 * sr), int(0.15 * sr), sr) is None
```

- [ ] **Step 2: Run, expect the first to FAIL** (the noisy width is several samples wider; the second may already pass or fail - record which)

Run: `.venv/bin/python -m pytest tests/detection/test_detect.py -k "noise_that_would or buried" -v`

- [ ] **Step 3: Implement**

Constant beside `QRS_WIDTH_THRESHOLD_FRACTION`:

```python
# The crossing threshold must also clear the local noise floor: the median
# of the envelope over the surrounding +/-1 s (QRS complexes occupy well
# under half of any second). Hash noise otherwise holds the envelope above
# 10% of the peak and the width lands on the noise, not the QRS edge.
QRS_NOISE_FLOOR_FACTOR = 4.0
QRS_NOISE_FLOOR_CONTEXT_SEC = 1.0
```

In `_qrs_width`, replace `threshold = QRS_WIDTH_THRESHOLD_FRACTION * local_peak` with:

```python
    context = envelope[max(0, r_peak - int(QRS_NOISE_FLOOR_CONTEXT_SEC * sample_rate)): r_peak + int(QRS_NOISE_FLOOR_CONTEXT_SEC * sample_rate)]
    threshold = max(QRS_WIDTH_THRESHOLD_FRACTION * local_peak, QRS_NOISE_FLOOR_FACTOR * float(np.median(context)))
    if threshold >= local_peak:
        return None  # buried in noise: no measurable width
```

and extend the docstring: "or if the local noise floor reaches the peak itself (the beat is buried in noise)".

- [ ] **Step 4: Run all detection tests** - `.venv/bin/python -m pytest tests/detection -q` - all pass.
- [ ] **Step 5: Commit** - `git commit -m "Detection: QRS width threshold clears the local noise floor"`

---

### Task 2: Absolute width margin in the classifier

**Files:**
- Modify: `src/canine_holter/classify/rules.py`
- Test: `tests/classify/test_rules.py`

- [ ] **Step 1: Failing tests**

```python
def test_a_beat_only_a_few_samples_wider_than_baseline_is_not_a_pvc():
    # 1.3x but 15 ms wider: three samples at 180 Hz, jitter, not a wide QRS.
    beats = [Beat(time=i * 0.8, rr_interval=0.8 if i else None, qrs_duration=0.061, label=None) for i in range(10)]
    beats[6] = Beat(time=beats[6].time - 0.3, rr_interval=0.5, qrs_duration=0.078, label=None)
    assert classify_beats(beats)[6].label == "N"


def test_a_beat_both_proportionally_and_absolutely_wider_is_a_pvc():
    beats = [Beat(time=i * 0.8, rr_interval=0.8 if i else None, qrs_duration=0.061, label=None) for i in range(10)]
    beats[6] = Beat(time=beats[6].time - 0.3, rr_interval=0.5, qrs_duration=0.100, label=None)
    assert classify_beats(beats)[6].label == "V"
```

- [ ] **Step 2: Run, expect the first to FAIL** (`V` today).
- [ ] **Step 3: Implement**

```python
QRS_WIDTH_MARGIN_SEC = 0.030  # and at least this much wider: 1.25x is three samples at 180 Hz, one small square is 40 ms
```

```python
                is_wide = (
                    beat.qrs_duration > QRS_WIDTH_RATIO * qrs_base
                    and beat.qrs_duration - qrs_base >= QRS_WIDTH_MARGIN_SEC
                )
```

Docstring: add "and by at least QRS_WIDTH_MARGIN_SEC - a ratio alone is sample jitter at 180 Hz".

- [ ] **Step 4: Run the classify tests and the MIT-BIH validation** - `.venv/bin/python -m pytest tests/classify tests/test_mitbih_validation.py -q`. If an existing test used a width between 1.25x and baseline + 30 ms to mean "wide", update that test's width to satisfy the margin and say so in the commit.
- [ ] **Step 5: Commit** - `git commit -m "Classify: a wide QRS must also be 30 ms wider than baseline"`

---

### Task 3: T-rule floor 0.6 s

**Files:**
- Modify: `src/canine_holter/detection/detect.py` (`SLOW_RR_SEC`)
- Test: `tests/detection/test_detect.py`

- [ ] **Step 1: Failing test**

```python
def test_drop_interpolated_t_waves_acts_in_a_moderate_rhythm_under_100_bpm():
    # 0.7 s rhythm (86 bpm): the T wave still lands 0.3 s after the QRS.
    sr = 100.0
    beats = [t * 0.7 for t in range(10)]
    with_t = sorted(beats + [3.5 + 0.3])
    assert drop_interpolated_t_waves(_peaks(sr, with_t), sr).tolist() == _peaks(sr, beats).tolist()
```

- [ ] **Step 2: Run, expect FAIL** (kept today: 0.7 < 0.8).
- [ ] **Step 3: Implement** - `SLOW_RR_SEC = 0.6` with the comment "only intervals over this (< 100 bpm) serve as the sinus reference; at 75-100 bpm the T wave still clears NeuroKit's 300 ms spacing, and tachycardia is excluded by the median gate".
- [ ] **Step 4: Run the full suite** - `.venv/bin/python -m pytest -q`.
- [ ] **Step 5: Commit** - `git commit -m "Detection: T-wave rule acts below 100 bpm"`

---

### Task 4: Validate, document, PR

- [ ] **Step 1:** Regenerate both recordings; list every V with its clock time and compare against the contact-sheet classes (script in the session scratchpad: `pvc_sheet.py` / `combo.py` logic - the classes are recorded in the spec). Expected 2026-08-25 ~20 (5 real), 2026-08-23 2 (both real).
- [ ] **Step 2:** Fill the spec's results with the measured numbers; update `CLAUDE.md` (`classify/rules.py` bullet: the margin; `detection/detect.py`: the noise floor; Known limits: residual T waves in sinus-arrhythmia swings, multi-lead detection).
- [ ] **Step 3:** Commit, push, `gh pr create`, watch CI.
