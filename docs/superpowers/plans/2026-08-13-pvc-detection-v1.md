# Canine Holter PVC Detection v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that reads an ECG recording, detects heartbeats, flags premature ventricular complexes (PVCs) and their patterns (couplets/triplets/VT runs), flags secondary arrhythmias (brady/tachycardia, pauses), and produces a report — wrapped in a Tkinter GUI and shipped as a signed, notarized macOS `.app` via GitHub Releases.

**Architecture:** A linear pipeline (`ingest` -> `detection` -> `classify` -> `arrhythmia` -> `report`) where each stage communicates through plain dataclasses (`Recording`, `Beat`, `ArrhythmiaSummary`). `cli.py` and `gui/app.py` both call the same `run_analysis()` entry point. See `docs/superpowers/specs/2026-08-13-pvc-detection-design.md` for full rationale.

**Tech Stack:** Python 3.11+, NeuroKit2 (R-peak detection/QRS delineation), wfdb-python (test fixture data from PhysioNet), NumPy, Matplotlib, Tkinter (GUI), pytest, PyInstaller (packaging).

**Out of scope for this plan:** the DR200 raw-format parser (`ingest/dr200.py`) — blocked on confirming the device's export format (pending ALBA response). Everything in this plan is built and tested against MIT-BIH and PhysioZoo fixture data instead, with `ingest` designed so the DR200 parser slots in later without changing anything downstream.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/canine_holter/__init__.py`
- Create: `src/canine_holter/ingest/__init__.py`
- Create: `src/canine_holter/detection/__init__.py`
- Create: `src/canine_holter/classify/__init__.py`
- Create: `src/canine_holter/arrhythmia/__init__.py`
- Create: `src/canine_holter/report/__init__.py`
- Create: `src/canine_holter/gui/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create the package layout and empty `__init__.py` files**

```bash
mkdir -p src/canine_holter/{ingest,detection,classify,arrhythmia,report,gui}
mkdir -p tests/fixtures
touch src/canine_holter/__init__.py
touch src/canine_holter/ingest/__init__.py
touch src/canine_holter/detection/__init__.py
touch src/canine_holter/classify/__init__.py
touch src/canine_holter/arrhythmia/__init__.py
touch src/canine_holter/report/__init__.py
touch src/canine_holter/gui/__init__.py
touch tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "canine-holter"
version = "0.1.0"
description = "Holter monitor ECG interpretation for canine PVC/arrhythmia screening"
requires-python = ">=3.11"
dependencies = [
    "neurokit2>=0.2.9",
    "wfdb>=4.1",
    "numpy>=1.26,<2",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pyinstaller>=6.0"]

[project.scripts]
canine-holter = "canine_holter.cli:main"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Create a venv and install the package in editable/dev mode**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: install completes with no errors. NeuroKit2 pulls in scipy/pandas/matplotlib transitively.

- [ ] **Step 4: Verify pytest runs with zero tests collected**

Run: `pytest -v`
Expected: `no tests ran` (exit code 0 or 5, not an error) — confirms the package installed and pytest can find the project.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src tests .gitignore
git commit -m "Scaffold project structure and dependencies"
```

---

### Task 2: Core data types

**Files:**
- Create: `src/canine_holter/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'canine_holter.types'`

- [ ] **Step 3: Write the implementation**

```python
# src/canine_holter/types.py
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass(frozen=True)
class Recording:
    """A single-lead ECG recording, in millivolts, with metadata."""
    samples: np.ndarray
    sample_rate: float
    start_time: datetime | None
    source: str


@dataclass(frozen=True)
class Beat:
    """A single detected heartbeat.

    time: seconds from the start of the recording
    rr_interval: seconds since the previous beat; None for the first beat
    qrs_duration: seconds; None if QRS delineation failed for this beat
    label: "N" (normal), "V" (PVC), "U" (undetermined), or None (not yet classified)
    """
    time: float
    rr_interval: float | None
    qrs_duration: float | None
    label: str | None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/types.py tests/test_types.py
git commit -m "Add Recording and Beat core types"
```

---

### Task 3: Fetch local test fixtures from PhysioNet

This is a one-off script you run manually (not part of the pytest suite) to download small ECG samples and commit them to the repo, so all later tests run offline.

**Files:**
- Create: `scripts/fetch_fixtures.py`

- [ ] **Step 1: Write the fixture-fetching script**

```python
# scripts/fetch_fixtures.py
"""One-off script to fetch small local test fixtures from PhysioNet.

Run manually: python scripts/fetch_fixtures.py

Requires network access. Fixtures are committed to the repo afterward so
the test suite never needs network access.
"""
import os
import wfdb

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")


def fetch_mitbih_sample():
    """First 60 seconds of MIT-BIH record 119 (360Hz), with beat annotations.
    Record 119 has a ventricular bigeminy pattern with 19 PVC ('V') beats in
    the first 60 seconds alone - necessary for Task 7's validation, which
    needs real PVC-labeled beats within the fixture window. (Record 100, a
    more commonly-cited "standard" record, has only 1 PVC in its entire
    30-minute recording, occurring outside any 60s slice - confirmed by
    reading its annotation file - so it was rejected as a fixture choice.)"""
    out_dir = os.path.join(FIXTURES_DIR, "mitdb_119")
    os.makedirs(out_dir, exist_ok=True)
    record = wfdb.rdrecord("119", pn_dir="mitdb", sampto=21600)
    ann = wfdb.rdann("119", "atr", pn_dir="mitdb", sampto=21600)
    wfdb.wrsamp(
        record_name="119",
        fs=record.fs,
        units=record.units,
        sig_name=record.sig_name,
        p_signal=record.p_signal,
        write_dir=out_dir,
    )
    ann.record_name = "119"
    ann.wrann(write_dir=out_dir)
    print(f"Wrote MIT-BIH fixture to {out_dir}")


def fetch_physiozoo_dog_sample():
    """A canine ECG sample from the PhysioZoo Mammalian NSR Database.

    Confirmed via the PhysioNet file browser at
    physionet.org/files/physiozoo/1.0.0/wfdb_format/dog/ - 17 subdirectories
    Dog_01 through Dog_17, each with Dog_NN.{dat,hea,qrs}. Normal sinus
    rhythm only (no PVC labels) - used for canine-morphology R-peak
    validation, not PVC ground truth.
    """
    RECORD_NAME = "Dog_01"
    PN_DIR = "physiozoo/1.0.0/wfdb_format/dog/Dog_01"
    out_dir = os.path.join(FIXTURES_DIR, "physiozoo_dog1")
    os.makedirs(out_dir, exist_ok=True)
    record = wfdb.rdrecord(RECORD_NAME, pn_dir=PN_DIR)
    ann = wfdb.rdann(RECORD_NAME, "qrs", pn_dir=PN_DIR)
    wfdb.wrsamp(
        record_name=RECORD_NAME,
        fs=record.fs,
        units=record.units,
        sig_name=record.sig_name,
        p_signal=record.p_signal,
        write_dir=out_dir,
    )
    ann.record_name = RECORD_NAME
    ann.wrann(write_dir=out_dir)
    print(f"Wrote PhysioZoo fixture to {out_dir}")


if __name__ == "__main__":
    fetch_mitbih_sample()
    fetch_physiozoo_dog_sample()
```

- [ ] **Step 2: Run the script and verify fixture files were created**

```bash
python scripts/fetch_fixtures.py
ls tests/fixtures/mitdb_119/
ls tests/fixtures/physiozoo_dog1/
```

Expected: `mitdb_119` contains `.dat`/`.hea`/`.atr` files; `physiozoo_dog1` contains `.dat`/`.hea`/`.qrs` files.

- [ ] **Step 3: Commit the fixtures**

```bash
git add scripts/fetch_fixtures.py tests/fixtures/
git commit -m "Add fixture-fetching script and committed test fixtures"
```

---

### Task 4: Ingest — load a local WFDB record into a Recording

**Files:**
- Create: `src/canine_holter/ingest/wfdb_loader.py`
- Test: `tests/ingest/test_wfdb_loader.py`
- Create: `tests/ingest/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_wfdb_loader.py
import os
from canine_holter.ingest.wfdb_loader import load_local_record

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_loads_mitbih_fixture_as_recording():
    path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    rec = load_local_record(path, source="mitdb_119")
    assert rec.sample_rate == 360.0
    assert len(rec.samples) > 0
    assert rec.source == "mitdb_119"


def test_loads_physiozoo_fixture_as_recording():
    path = os.path.join(FIXTURES_DIR, "physiozoo_dog1", "Dog_01")
    rec = load_local_record(path, source="physiozoo_dog1")
    assert rec.sample_rate > 0
    assert len(rec.samples) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_wfdb_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'canine_holter.ingest.wfdb_loader'`

- [ ] **Step 3: Write the implementation**

```python
# src/canine_holter/ingest/wfdb_loader.py
import wfdb
from canine_holter.types import Recording


def load_local_record(record_path: str, source: str, channel: int = 0) -> Recording:
    """Load a local WFDB record (a .dat/.hea pair sharing `record_path` as
    their base path, e.g. 'tests/fixtures/mitdb_119/119') into a Recording.

    Uses the first signal channel by default (channel=0) since this
    pipeline is single-lead.
    """
    record = wfdb.rdrecord(record_path)
    samples = record.p_signal[:, channel]
    return Recording(
        samples=samples,
        sample_rate=float(record.fs),
        start_time=None,
        source=source,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_wfdb_loader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/ingest/wfdb_loader.py tests/ingest/
git commit -m "Add WFDB local-record loader"
```

---

### Task 5: Detection — R-peak and QRS delineation wrapper

**Files:**
- Create: `src/canine_holter/detection/detect.py`
- Test: `tests/detection/test_detect.py`
- Create: `tests/detection/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/detection/test_detect.py
import os
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


def test_detects_beats_in_canine_fixture():
    rec = load_local_record(
        os.path.join(FIXTURES_DIR, "physiozoo_dog1", "Dog_01"), source="physiozoo_dog1"
    )
    beats = detect_beats(rec.samples, rec.sample_rate)
    assert len(beats) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/detection/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'canine_holter.detection.detect'`

- [ ] **Step 3: Write the implementation**

```python
# src/canine_holter/detection/detect.py
import math
import neurokit2 as nk
import numpy as np
from canine_holter.types import Beat


def detect_beats(samples: np.ndarray, sample_rate: float) -> list[Beat]:
    """Detect R-peaks and delineate QRS onset/offset, returning unlabeled Beats."""
    cleaned = nk.ecg_clean(samples, sampling_rate=sample_rate)
    _, r_info = nk.ecg_peaks(cleaned, sampling_rate=sample_rate)
    r_peaks = r_info["ECG_R_Peaks"]
    if len(r_peaks) < 2:
        return []

    _, waves = nk.ecg_delineate(
        cleaned, r_peaks, sampling_rate=sample_rate, method="dwt"
    )
    onsets = waves.get("ECG_R_Onsets", [None] * len(r_peaks))
    offsets = waves.get("ECG_R_Offsets", [None] * len(r_peaks))

    beats = []
    for i, r in enumerate(r_peaks):
        time = r / sample_rate
        rr = (r - r_peaks[i - 1]) / sample_rate if i > 0 else None

        qrs_duration = None
        if i < len(onsets) and i < len(offsets):
            onset, offset = onsets[i], offsets[i]
            if onset is not None and offset is not None:
                if not (math.isnan(onset) or math.isnan(offset)):
                    qrs_duration = (offset - onset) / sample_rate

        beats.append(
            Beat(time=time, rr_interval=rr, qrs_duration=qrs_duration, label=None)
        )
    return beats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/detection/test_detect.py -v`
Expected: PASS (2 passed). If the beat-count sanity range fails, print `len(beats)` and adjust the range rather than the detection logic — the goal is confirming detection runs correctly end-to-end, not hitting an exact literature number.

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/detection/detect.py tests/detection/
git commit -m "Add NeuroKit2-based R-peak and QRS delineation wrapper"
```

---

### Task 6: Classify — rules-based PVC detector

**Files:**
- Create: `src/canine_holter/classify/rules.py`
- Test: `tests/classify/test_rules.py`
- Create: `tests/classify/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/classify/test_rules.py
from canine_holter.types import Beat
from canine_holter.classify.rules import classify_beats


def _beat(time, rr, qrs):
    return Beat(time=time, rr_interval=rr, qrs_duration=qrs, label=None)


def test_first_beats_are_undetermined_until_baseline_established():
    beats = [_beat(0.0, None, 0.08)]
    labeled = classify_beats(beats)
    assert labeled[0].label == "U"


def test_normal_regular_beats_are_labeled_normal():
    # 8 beats at a steady 0.8s RR / 0.08s QRS establish baseline, 9th matches
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(9)]
    labeled = classify_beats(beats)
    assert all(b.label in ("N", "U") for b in labeled)
    assert labeled[-1].label == "N"


def test_premature_and_wide_beat_is_labeled_pvc():
    # 8 steady normal beats to establish baseline (0.8s RR, 0.08s QRS)
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    # 9th beat: premature (RR well below 0.85 * 0.8 = 0.68) and wide
    # (QRS well above 1.25 * 0.08 = 0.10)
    beats.append(_beat(8 * 0.8 - 0.3, 0.5, 0.14))
    labeled = classify_beats(beats)
    assert labeled[-1].label == "V"


def test_premature_but_normal_width_beat_is_not_pvc():
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    beats.append(_beat(8 * 0.8 - 0.3, 0.5, 0.08))  # premature, normal-width
    labeled = classify_beats(beats)
    assert labeled[-1].label == "N"


def test_missing_qrs_duration_is_undetermined_not_pvc():
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    beats.append(_beat(8 * 0.8 - 0.3, 0.5, None))  # premature, no QRS reading
    labeled = classify_beats(beats)
    assert labeled[-1].label == "U"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/classify/test_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'canine_holter.classify.rules'`

- [ ] **Step 3: Write the implementation**

```python
# src/canine_holter/classify/rules.py
import statistics
from collections import deque
from dataclasses import replace
from canine_holter.types import Beat

PREMATURITY_RATIO = 0.85  # RR < 85% of local baseline -> premature
QRS_WIDTH_RATIO = 1.25  # QRS > 125% of local baseline -> wide
BASELINE_WINDOW = 8  # number of recent "N" beats used to compute baseline


def classify_beats(beats: list[Beat]) -> list[Beat]:
    """Label each beat "N" (normal), "V" (PVC), or "U" (undetermined).

    A beat is "V" only when it is BOTH premature (RR well below the local
    baseline) AND wide (QRS well above the local baseline) - this matches
    the standard clinical heuristic for identifying ventricular ectopy.
    Baseline is computed causally from the most recent beats labeled "N",
    so thresholds adapt to each recording's (and each dog's) own rhythm
    rather than relying on a fixed literature value.

    Bootstrapping: before any baseline exists, there is nothing to compare
    a beat's RR/QRS against, so a beat with complete measurements is
    provisionally labeled "N" (this is what seeds the baseline). A beat is
    only "U" (undetermined) when its own RR interval or QRS duration is
    missing - never as a substitute for "we haven't decided yet".
    """
    baseline_rr: deque[float] = deque(maxlen=BASELINE_WINDOW)
    baseline_qrs: deque[float] = deque(maxlen=BASELINE_WINDOW)
    labeled: list[Beat] = []

    for beat in beats:
        label = "U"

        have_baseline = len(baseline_rr) > 0 and len(baseline_qrs) > 0
        have_measurements = beat.rr_interval is not None and beat.qrs_duration is not None

        if have_measurements:
            if have_baseline:
                rr_base = statistics.median(baseline_rr)
                qrs_base = statistics.median(baseline_qrs)
                is_premature = beat.rr_interval < PREMATURITY_RATIO * rr_base
                is_wide = beat.qrs_duration > QRS_WIDTH_RATIO * qrs_base
                label = "V" if (is_premature and is_wide) else "N"
            else:
                # No baseline yet: nothing to measure prematurity/width
                # against, so this beat can't be flagged as a PVC. Treat it
                # as provisionally normal so the baseline can seed itself.
                label = "N"

        labeled_beat = replace(beat, label=label)
        labeled.append(labeled_beat)

        if label == "N":
            if beat.rr_interval is not None:
                baseline_rr.append(beat.rr_interval)
            if beat.qrs_duration is not None:
                baseline_qrs.append(beat.qrs_duration)

    return labeled
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/classify/test_rules.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/classify/rules.py tests/classify/
git commit -m "Add rules-based PVC classifier"
```

---

### Task 7: Validate the classifier against MIT-BIH ground truth

This is the "pipeline mechanics" validation from the design spec: confirms the detect+classify pipeline correctly identifies known PVC beats in real (human) annotated data. It is explicitly a code-correctness check, not a claim of canine accuracy.

**Files:**
- Create: `tests/test_mitbih_validation.py`

- [ ] **Step 1: Write the validation test**

```python
# tests/test_mitbih_validation.py
"""Pipeline mechanics validation against MIT-BIH ground truth.

This confirms the detect -> classify pipeline runs correctly end-to-end and
finds a reasonable fraction of real PVC beats. It is NOT a claim of canine
clinical accuracy - MIT-BIH is human data. See
docs/superpowers/specs/2026-08-13-pvc-detection-design.md for why canine
validation requires real Teeny recordings instead.
"""
import os
import wfdb
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.detection.detect import detect_beats
from canine_holter.classify.rules import classify_beats

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_classifier_finds_most_known_pvc_beats_in_mitbih_record_119():
    fixture_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    rec = load_local_record(fixture_path, source="mitdb_119")
    ann = wfdb.rdann(fixture_path, "atr")

    # Ground-truth PVC beat times (annotation symbol 'V'), in seconds
    ground_truth_pvc_times = {
        sample / rec.sample_rate
        for sample, symbol in zip(ann.sample, ann.symbol)
        if symbol == "V"
    }
    assert len(ground_truth_pvc_times) > 0, "fixture should contain at least one PVC"

    beats = detect_beats(rec.samples, rec.sample_rate)
    labeled = classify_beats(beats)
    detected_pvc_times = [b.time for b in labeled if b.label == "V"]

    # A detected PVC "matches" a ground-truth PVC if within 50ms of it
    TOLERANCE_SEC = 0.05
    matched = sum(
        1
        for gt_time in ground_truth_pvc_times
        if any(abs(gt_time - d_time) <= TOLERANCE_SEC for d_time in detected_pvc_times)
    )
    sensitivity = matched / len(ground_truth_pvc_times)

    # Rules-based v1 on human data is a coarse mechanics check, not a tuned
    # classifier - 50% catches "the pipeline is fundamentally working"
    # without demanding human-tuned accuracy from a canine-first design.
    assert sensitivity >= 0.5, (
        f"Only matched {matched}/{len(ground_truth_pvc_times)} "
        f"({sensitivity:.0%}) known PVC beats"
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_mitbih_validation.py -v`
Expected: PASS. If it fails, print the actual sensitivity and inspect whether detection or classification is the weak link before changing thresholds - don't tune `PREMATURITY_RATIO`/`QRS_WIDTH_RATIO` purely to make this pass, since record 119 is human data and the real tuning target is canine recordings later.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mitbih_validation.py
git commit -m "Add MIT-BIH pipeline mechanics validation test"
```

---

### Task 8: Arrhythmia burden aggregation

**Files:**
- Create: `src/canine_holter/arrhythmia/burden.py`
- Test: `tests/arrhythmia/test_burden.py`
- Create: `tests/arrhythmia/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/arrhythmia/test_burden.py
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize


def _beat(time, rr, label):
    return Beat(time=time, rr_interval=rr, qrs_duration=0.08, label=label)


def test_counts_total_and_pvc_beats():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(1.6, 0.8, "V"),
        _beat(2.4, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.total_beats == 4
    assert summary.pvc_count == 1
    assert summary.pvc_burden_pct == 25.0


def test_detects_couplet_and_triplet():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "V"),
        _beat(1.6, 0.8, "V"),
        _beat(2.4, 0.8, "N"),
        _beat(3.2, 0.8, "V"),
        _beat(4.0, 0.8, "V"),
        _beat(4.8, 0.8, "V"),
        _beat(5.6, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.couplets == 1
    assert summary.triplets == 1
    assert summary.vtach_runs == 0


def test_detects_vtach_run_of_four_or_more():
    beats = [_beat(0.0, None, "N")] + [
        _beat(i * 0.8, 0.8, "V") for i in range(1, 5)
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.vtach_runs == 1
    assert summary.triplets == 0  # a run of 4 is not double-counted as a triplet


def test_flags_pause_above_threshold():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(3.5, 2.7, "N"),  # 2.7s gap - a real pause for a dog
        _beat(4.3, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert len(summary.pauses) == 1
    assert summary.pauses[0] == 3.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/arrhythmia/test_burden.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'canine_holter.arrhythmia.burden'`

- [ ] **Step 3: Write the implementation**

```python
# src/canine_holter/arrhythmia/burden.py
from dataclasses import dataclass
from canine_holter.types import Beat

# Provisional defaults - not yet calibrated against real canine recordings.
# See docs/superpowers/specs/2026-08-13-pvc-detection-design.md, "Open items".
PAUSE_THRESHOLD_SEC = 2.5
BRADYCARDIA_HR_THRESHOLD = {"small": 60, "medium": 50, "large": 45}
TACHYCARDIA_HR_THRESHOLD = {"small": 180, "medium": 160, "large": 150}
SUSTAINED_EVENT_MIN_BEATS = 3  # consecutive beats needed to call it "sustained"


@dataclass(frozen=True)
class ArrhythmiaSummary:
    total_beats: int
    pvc_count: int
    pvc_burden_pct: float
    couplets: int
    triplets: int
    vtach_runs: int
    bradycardia_events: list[tuple[float, float]]
    tachycardia_events: list[tuple[float, float]]
    pauses: list[float]


def pvc_runs(beats: list[Beat]) -> list[list[Beat]]:
    """Group consecutive PVC-labeled beats into runs."""
    runs: list[list[Beat]] = []
    current: list[Beat] = []
    for beat in beats:
        if beat.label == "V":
            current.append(beat)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _sustained_hr_events(
    beats: list[Beat], threshold_bpm: float, direction: str
) -> list[tuple[float, float]]:
    """Find stretches of >= SUSTAINED_EVENT_MIN_BEATS consecutive beats whose
    instantaneous HR is below (direction="brady") or above (direction="tachy")
    threshold_bpm. Returns (start_time, end_time) for each stretch."""
    events: list[tuple[float, float]] = []
    run_start: float | None = None
    run_len = 0
    prev_time = None

    for beat in beats:
        if beat.rr_interval is None or beat.rr_interval <= 0:
            if run_len >= SUSTAINED_EVENT_MIN_BEATS and run_start is not None:
                events.append((run_start, prev_time))
            run_start, run_len = None, 0
            prev_time = beat.time
            continue

        hr = 60.0 / beat.rr_interval
        is_match = hr < threshold_bpm if direction == "brady" else hr > threshold_bpm

        if is_match:
            if run_len == 0:
                run_start = beat.time - beat.rr_interval
            run_len += 1
        else:
            if run_len >= SUSTAINED_EVENT_MIN_BEATS and run_start is not None:
                events.append((run_start, prev_time))
            run_start, run_len = None, 0

        prev_time = beat.time

    if run_len >= SUSTAINED_EVENT_MIN_BEATS and run_start is not None:
        events.append((run_start, prev_time))

    return events


def summarize(beats: list[Beat], dog_weight_class: str = "medium") -> ArrhythmiaSummary:
    """Aggregate a labeled Beat sequence into an ArrhythmiaSummary.

    dog_weight_class: "small", "medium", or "large" - selects brady/tachy
    thresholds. These are provisional defaults; real calibration happens
    against Teeny's own recordings over time (see design spec).
    """
    total_beats = len(beats)
    pvc_beats = [b for b in beats if b.label == "V"]
    pvc_count = len(pvc_beats)
    pvc_burden_pct = (pvc_count / total_beats * 100) if total_beats else 0.0

    couplets = triplets = vtach_runs = 0
    for run in pvc_runs(beats):
        n = len(run)
        if n == 2:
            couplets += 1
        elif n == 3:
            triplets += 1
        elif n >= 4:
            vtach_runs += 1

    pauses = [b.time for b in beats if b.rr_interval and b.rr_interval >= PAUSE_THRESHOLD_SEC]

    brady_threshold = BRADYCARDIA_HR_THRESHOLD[dog_weight_class]
    tachy_threshold = TACHYCARDIA_HR_THRESHOLD[dog_weight_class]
    bradycardia_events = _sustained_hr_events(beats, brady_threshold, "brady")
    tachycardia_events = _sustained_hr_events(beats, tachy_threshold, "tachy")

    return ArrhythmiaSummary(
        total_beats=total_beats,
        pvc_count=pvc_count,
        pvc_burden_pct=pvc_burden_pct,
        couplets=couplets,
        triplets=triplets,
        vtach_runs=vtach_runs,
        bradycardia_events=bradycardia_events,
        tachycardia_events=tachycardia_events,
        pauses=pauses,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/arrhythmia/test_burden.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/arrhythmia/burden.py tests/arrhythmia/
git commit -m "Add arrhythmia burden aggregation (PVC runs, brady/tachy, pauses)"
```

---

### Task 9: Report generation

**Files:**
- Create: `src/canine_holter/report/generate.py`
- Test: `tests/report/test_generate.py`
- Create: `tests/report/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/report/test_generate.py
import os
import tempfile
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.generate import write_report


def _beat(time, rr, label, qrs=0.08):
    return Beat(time=time, rr_interval=rr, qrs_duration=qrs, label=label)


def test_writes_markdown_report_with_summary_stats():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(1.6, 0.8, "V"),
        _beat(2.4, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")

    with tempfile.TemporaryDirectory() as out_dir:
        report_path = write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        assert os.path.exists(report_path)
        content = open(report_path).read()
        assert "PVC" in content
        assert "1" in content  # pvc_count appears somewhere in the stats


def test_generates_strip_plot_for_each_pvc_run():
    beats = [_beat(0.0, None, "N")] + [
        _beat(i * 0.8, 0.8, "V") for i in range(1, 4)
    ]  # a triplet
    summary = summarize(beats, dog_weight_class="medium")
    import numpy as np
    samples = np.sin(np.linspace(0, 20, 2000))  # dummy waveform, just needs a shape

    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summary, out_dir, samples=samples, sample_rate=100.0)
        plot_files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(plot_files) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/report/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'canine_holter.report.generate'`

- [ ] **Step 3: Write the implementation**

```python
# src/canine_holter/report/generate.py
import os
import matplotlib
matplotlib.use("Agg")  # no display needed - this runs headless in CLI/CI
import matplotlib.pyplot as plt
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import ArrhythmiaSummary, pvc_runs

STRIP_WINDOW_SEC = 6.0  # seconds of context shown around each flagged run


def _flagged_runs(beats: list[Beat]) -> list[list[Beat]]:
    """PVC runs of 2+ beats (couplets, triplets, VT runs) - the events worth
    a rhythm-strip plot. Isolated single PVCs are counted in the summary
    stats but not individually plotted, to avoid an unbounded number of
    plots on a high-burden recording."""
    return [run for run in pvc_runs(beats) if len(run) >= 2]


def _plot_strip(samples: np.ndarray, sample_rate: float, center_time: float, out_path: str) -> None:
    half_window = STRIP_WINDOW_SEC / 2
    start_sample = max(0, int((center_time - half_window) * sample_rate))
    end_sample = min(len(samples), int((center_time + half_window) * sample_rate))
    segment = samples[start_sample:end_sample]
    t = np.arange(len(segment)) / sample_rate

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, segment, linewidth=0.8)
    ax.set_title(f"Rhythm strip around t={center_time:.1f}s")
    ax.set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    out_dir: str,
    samples: np.ndarray | None,
    sample_rate: float | None,
) -> str:
    """Write a markdown summary report plus (if waveform data is provided)
    rhythm-strip PNGs for each flagged multi-beat PVC run. Returns the path
    to the markdown report."""
    os.makedirs(out_dir, exist_ok=True)

    lines = [
        "# Holter Analysis Report",
        "",
        "**This is a screening aid, not a diagnosis. Review with a veterinary cardiologist.**",
        "",
        "## Summary",
        f"- Total beats: {summary.total_beats}",
        f"- PVC count: {summary.pvc_count}",
        f"- PVC burden: {summary.pvc_burden_pct:.2f}%",
        f"- Couplets: {summary.couplets}",
        f"- Triplets: {summary.triplets}",
        f"- VT runs (4+ consecutive PVCs): {summary.vtach_runs}",
        f"- Pauses (>= threshold): {len(summary.pauses)}",
        f"- Sustained bradycardia events: {len(summary.bradycardia_events)}",
        f"- Sustained tachycardia events: {len(summary.tachycardia_events)}",
        "",
    ]

    flagged = _flagged_runs(beats)
    if flagged:
        lines.append("## Flagged events (couplets, triplets, VT runs)")
        for i, run in enumerate(flagged):
            center_time = (run[0].time + run[-1].time) / 2
            lines.append(f"- Event {i + 1}: {len(run)} consecutive PVCs at ~t={center_time:.1f}s")
            if samples is not None and sample_rate is not None:
                plot_path = os.path.join(out_dir, f"event_{i + 1}_strip.png")
                _plot_strip(samples, sample_rate, center_time, plot_path)
                lines.append(f"  ![event {i + 1}]({os.path.basename(plot_path)})")
        lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/report/test_generate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/report/generate.py tests/report/
git commit -m "Add markdown report and rhythm-strip plot generation"
```

---

### Task 10: Wire it together — `run_analysis()` and the CLI

**Files:**
- Create: `src/canine_holter/pipeline.py`
- Create: `src/canine_holter/cli.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import os
import tempfile
from canine_holter.pipeline import run_analysis

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_run_analysis_produces_report_from_fixture():
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        report_path = run_analysis(input_path, out_dir)
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'canine_holter.pipeline'`

- [ ] **Step 3: Write `pipeline.py`**

```python
# src/canine_holter/pipeline.py
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.detection.detect import detect_beats
from canine_holter.classify.rules import classify_beats
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.generate import write_report


def run_analysis(input_path: str, out_dir: str, dog_weight_class: str = "medium") -> str:
    """Run the full ingest -> detect -> classify -> summarize -> report
    pipeline against a local WFDB record. Returns the path to the written
    markdown report.

    NOTE: input_path currently must be a local WFDB record (see
    ingest/wfdb_loader.py). DR200-native files aren't supported yet - see
    docs/superpowers/specs/2026-08-13-pvc-detection-design.md, "Open items".
    """
    rec = load_local_record(input_path, source=input_path)
    beats = detect_beats(rec.samples, rec.sample_rate)
    labeled = classify_beats(beats)
    summary = summarize(labeled, dog_weight_class=dog_weight_class)
    return write_report(
        labeled, summary, out_dir, samples=rec.samples, sample_rate=rec.sample_rate
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write the CLI**

```python
# src/canine_holter/cli.py
import argparse
import sys
from canine_holter.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="canine-holter",
        description="Analyze a Holter ECG recording for PVC burden and arrhythmias.",
    )
    parser.add_argument("input", help="Path to a local WFDB record (base path, no extension)")
    parser.add_argument("--out", required=True, help="Output directory for the report")
    parser.add_argument(
        "--dog-weight-class",
        choices=["small", "medium", "large"],
        default="medium",
        help="Selects brady/tachycardia thresholds (default: medium)",
    )
    args = parser.parse_args()

    report_path = run_analysis(args.input, args.out, dog_weight_class=args.dog_weight_class)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Manually verify the CLI runs end-to-end**

```bash
canine-holter tests/fixtures/mitdb_119/119 --out /tmp/canine-holter-report
cat /tmp/canine-holter-report/report.md
```

Expected: a report.md with summary stats prints without error.

- [ ] **Step 7: Commit**

```bash
git add src/canine_holter/pipeline.py src/canine_holter/cli.py tests/test_pipeline.py
git commit -m "Wire pipeline stages together and add CLI entry point"
```

---

### Task 11: Tkinter GUI wrapper

**Files:**
- Create: `src/canine_holter/gui/app.py`
- Test: `tests/gui/test_app.py`
- Create: `tests/gui/__init__.py`

The GUI itself (file dialog, window) isn't meaningfully unit-testable — verify it manually. What IS testable is that the GUI module's core "run the pipeline and report success/failure" function works independently of Tkinter being on screen.

- [ ] **Step 1: Write the failing test**

```python
# tests/gui/test_app.py
import os
import tempfile
from canine_holter.gui.app import analyze_and_report

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_analyze_and_report_returns_report_path_on_success():
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        result = analyze_and_report(input_path, out_dir)
        assert result.success is True
        assert os.path.exists(result.report_path)


def test_analyze_and_report_reports_failure_on_bad_input():
    with tempfile.TemporaryDirectory() as out_dir:
        result = analyze_and_report("/nonexistent/path/nope", out_dir)
        assert result.success is False
        assert result.error_message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gui/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'canine_holter.gui.app'`

- [ ] **Step 3: Write the implementation**

```python
# src/canine_holter/gui/app.py
import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from canine_holter.pipeline import run_analysis


@dataclass
class AnalysisResult:
    success: bool
    report_path: str | None
    error_message: str | None


def analyze_and_report(input_path: str, out_dir: str) -> AnalysisResult:
    """Runs the pipeline and captures success/failure as data, so this is
    testable without a display and reusable by both the GUI and CLI."""
    try:
        report_path = run_analysis(input_path, out_dir)
        return AnalysisResult(success=True, report_path=report_path, error_message=None)
    except Exception as exc:  # noqa: BLE001 - surfacing any failure to the GUI is the point
        return AnalysisResult(success=False, report_path=None, error_message=str(exc))


def _open_in_default_app(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def _on_pick_file() -> None:
    input_path = filedialog.askopenfilename(title="Select a Holter recording (.hea file)")
    if not input_path:
        return
    # WFDB records are referenced by their base path (no extension)
    base_path = os.path.splitext(input_path)[0]
    out_dir = filedialog.askdirectory(title="Select an output folder for the report")
    if not out_dir:
        return

    result = analyze_and_report(base_path, out_dir)
    if result.success:
        messagebox.showinfo("Done", f"Report written to {result.report_path}")
        _open_in_default_app(result.report_path)
    else:
        messagebox.showerror("Analysis failed", result.error_message)


def main() -> None:
    root = tk.Tk()
    root.title("Canine Holter Analyzer")
    root.geometry("360x160")

    label = tk.Label(root, text="Select a Holter recording to analyze", pady=20)
    label.pack()

    button = tk.Button(root, text="Choose Recording...", command=_on_pick_file, padx=20, pady=10)
    button.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gui/test_app.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Manually verify the GUI launches**

```bash
python -m canine_holter.gui.app
```

Expected: a small window opens with a "Choose Recording..." button. Click it, pick `tests/fixtures/mitdb_119/119.hea`, pick an output folder, confirm a success dialog appears and the report opens.

- [ ] **Step 6: Commit**

```bash
git add src/canine_holter/gui/app.py tests/gui/
git commit -m "Add Tkinter GUI wrapper around the analysis pipeline"
```

---

### Task 12: PyInstaller packaging spec

**Files:**
- Create: `canine-holter.spec`

- [ ] **Step 1: Generate a starting spec file**

```bash
pyi-makespec --windowed --name canine-holter --onedir src/canine_holter/gui/app.py
```

This creates `canine-holter.spec` in the repo root.

- [ ] **Step 2: Edit the generated spec to include NeuroKit2's data files**

NeuroKit2 and its dependencies (scipy, pandas) sometimes need hidden-import hints for PyInstaller. Open `canine-holter.spec` and update the `Analysis(...)` block's `hiddenimports`:

```python
a = Analysis(
    ['src/canine_holter/gui/app.py'],
    pathex=['src'],
    hiddenimports=['neurokit2', 'scipy.special.cython_special'],
    # ... leave other Analysis(...) args as generated
)
```

- [ ] **Step 3: Build locally and verify the `.app` launches**

```bash
pyinstaller canine-holter.spec
open dist/canine-holter.app
```

Expected: the app opens (unsigned, so on your own Mac you may need to right-click -> Open the first time). Click "Choose Recording...", pick the MIT-BIH fixture, confirm the report generates. If PyInstaller errors on a missing module, add it to `hiddenimports` and rebuild - this is expected iteration, not a design problem.

- [ ] **Step 4: Commit**

```bash
git add canine-holter.spec
git commit -m "Add PyInstaller packaging spec"
```

---

### Task 13: GitHub Actions signed release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Write the release workflow, adapted from hedgehog's build-installers.yml**

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-macos:
    runs-on: macos-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Import Apple certificate
        env:
          APPLE_CERTIFICATE_P12: ${{ secrets.APPLE_CERTIFICATE_P12 }}
          APPLE_CERTIFICATE_PASSWORD: ${{ secrets.APPLE_CERTIFICATE_PASSWORD }}
        run: |
          KEYCHAIN_PATH="$RUNNER_TEMP/build.keychain"
          KEYCHAIN_PASSWORD=$(openssl rand -base64 32)

          echo "$APPLE_CERTIFICATE_P12" | base64 --decode > "$RUNNER_TEMP/certificate.p12"

          security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
          security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
          security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"

          security import "$RUNNER_TEMP/certificate.p12" \
            -P "$APPLE_CERTIFICATE_PASSWORD" \
            -A -t cert -f pkcs12 -k "$KEYCHAIN_PATH"

          security set-key-partition-list -S apple-tool:,apple:,codesign: \
            -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"

          security list-keychains -d user -s "$KEYCHAIN_PATH" $(security list-keychains -d user | tr -d '"')

          echo "KEYCHAIN_PATH=$KEYCHAIN_PATH" >> "$GITHUB_ENV"

      - name: Build .app with PyInstaller
        run: pyinstaller canine-holter.spec

      - name: Codesign the app
        env:
          APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}
        run: |
          codesign --deep --force --options runtime \
            --sign "$APPLE_SIGNING_IDENTITY" \
            --keychain "$KEYCHAIN_PATH" \
            dist/canine-holter.app

      - name: Package into a DMG
        run: |
          hdiutil create -volname "Canine Holter" -srcfolder dist/canine-holter.app \
            -ov -format UDZO dist/canine-holter.dmg

      - name: Notarize and staple
        env:
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_ID_PASSWORD: ${{ secrets.APPLE_ID_PASSWORD }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
        run: |
          xcrun notarytool submit dist/canine-holter.dmg \
            --apple-id "$APPLE_ID" \
            --password "$APPLE_ID_PASSWORD" \
            --team-id "$APPLE_TEAM_ID" \
            --wait --timeout 30m
          xcrun stapler staple dist/canine-holter.dmg

      - name: Clean up keychain
        if: always()
        run: |
          security delete-keychain "$KEYCHAIN_PATH" || true
          rm -f "$RUNNER_TEMP/certificate.p12" || true

      - name: Upload to GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh release create "$GITHUB_REF_NAME" dist/canine-holter.dmg --title "$GITHUB_REF_NAME" --generate-notes
```

- [ ] **Step 2: Copy the Apple signing secrets from the hedgehog repo**

```bash
for name in APPLE_CERTIFICATE_P12 APPLE_CERTIFICATE_PASSWORD APPLE_SIGNING_IDENTITY APPLE_ID APPLE_ID_PASSWORD APPLE_TEAM_ID; do
  echo "Set $name on aaymeloglu/canine-holter (copy the value from hedgehog's repo settings or your credential store - gh secret values can't be read back, only re-set)"
done
```

GitHub doesn't allow reading a secret's value back out, so these have to be re-entered from wherever the original values are stored (not copied programmatically from the hedgehog repo). Use `gh secret set <NAME> --repo aaymeloglu/canine-holter` for each, or set them via the GitHub web UI under Settings -> Secrets and variables -> Actions.

- [ ] **Step 3: Commit the workflow**

```bash
git add .github/workflows/release.yml
git commit -m "Add signed/notarized macOS release workflow"
```

- [ ] **Step 4: Tag and verify a release builds (only after secrets are set)**

```bash
git tag v0.1.0
git push origin main --tags
gh run watch  # follow the workflow run
```

Expected: workflow completes, `gh release view v0.1.0` shows a `canine-holter.dmg` asset.

---

### Task 14: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the placeholder README with real usage docs**

```markdown
# canine-holter

Tools for recording and interpreting ambulatory ECG (Holter monitor) data from a dog, starting with an ALBA Medical DR200 unit.

## Background

Started after our Doberman (Teeny) had a seizure-like episode with no clear diagnosis and a month+ wait for a cardiology referral. Rather than wait on the clinic's loaner Holter program every time, we bought our own DR200 recorder + canine vest. This repo reads and interprets the raw recordings ourselves - screening for PVC burden (the standard Doberman occult DCM metric) and other arrhythmias.

**This is a screening/triage aid, not a diagnostic tool.** It doesn't replace a cardiologist's read.

## Status

v1: rules-based PVC detection, tested against MIT-BIH (human) and PhysioZoo (canine, no PVC labels) data. DR200-native file support isn't built yet - see `docs/superpowers/specs/2026-08-13-pvc-detection-design.md` for why and what's blocking it.

## Usage

```bash
pip install -e ".[dev]"
canine-holter path/to/recording --out report/
```

Or launch the GUI: `python -m canine_holter.gui.app` (or download the signed `.app` from [Releases](https://github.com/aaymeloglu/canine-holter/releases)).

## Development

```bash
pip install -e ".[dev]"
pytest
```

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for design history.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Update README with usage and status"
```

---

## Explicitly deferred (not in this plan)

- `ingest/dr200.py` — blocked on confirming the DR200's raw export format
- ML-based beat classification (v2, once real canine ground-truth data exists)
- AFib detection (needs reliable P-wave delineation on noisy ambulatory data)
- Outreach to Wess (LMU Munich) / PulseNet authors for potential real canine PVC-labeled data
- Windows/Linux packaging
