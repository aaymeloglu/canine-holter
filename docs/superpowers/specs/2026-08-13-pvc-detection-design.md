# Canine Holter PVC Detection & Interpretation — Design

Date: 2026-08-13

## Motivation

Our Doberman (Teeny) had a seizure-like episode with no clear diagnosis and a month-plus wait for a cardiology referral. Dobermans are predisposed to occult dilated cardiomyopathy (DCM), which is screened for via Holter monitoring: cardiologists count premature ventricular complexes (PVCs) over a 24hr recording and flag concerning patterns (couplets, triplets, runs of ventricular tachycardia). We bought our own DR200 Holter recorder (ALBA Medical) + canine vest so we aren't dependent on clinic wait times whenever a future incident happens. This project reads and interprets those recordings ourselves.

This is a screening/triage tool, not a diagnostic one. It doesn't replace the cardiologist's read — it tells us whether to treat a given recording as urgent or as something that can wait for the next scheduled read.

## Goals

- Detect and count PVC burden per recording (primary goal — matches the actual clinical screening metric for Doberman occult DCM)
- Flag PVC patterns: couplets, triplets, runs of ventricular tachycardia
- Flag secondary arrhythmia concerns: bradycardia/tachycardia (dog-adjusted thresholds), pauses
- Produce a report (summary stats + rhythm-strip plots around flagged events) that's usable without a cardiologist in the loop
- Ship a signed, notarized macOS `.app` via GitHub Releases so a non-technical user (Jaime) can download and run it without installing Python

## Non-goals (v1)

- Not diagnostic-grade, not a replacement for cardiologist review
- No real-time/live monitoring — DR200 is a batch recorder, this is post-hoc analysis of a completed recording
- No AFib detection requiring P-wave analysis (P-wave delineation on noisy ambulatory canine recordings is unreliable; deferred)
- No ML-based beat classification in v1 (see "PVC classification approach" below)
- No Windows/Linux packaging (Jaime is the only non-technical consumer, and she's on a Mac)

## PVC classification approach

**v1: rules-based**, not machine learning. A beat is flagged as a PVC when it is:
- Significantly premature (RR interval well below the local running average), **and**
- Abnormally wide QRS relative to the local baseline (ventricular beats bypass the normal conduction pathway and are wider)

Thresholds are calibrated against each recording's own local baseline rather than fixed literature values, since "abnormal" is relative to the individual dog. Sinus arrhythmia (RR variation tied to respiration) is physiologically normal in dogs — pause/bradycardia detection must not flag normal respiratory-driven variation as pathological.

**Why not ML now:** research turned up no public canine-labeled ECG dataset (a deeper search is running in parallel to confirm this — see "Open items"). Human data (e.g. MIT-BIH Arrhythmia Database) can't safely stand in for canine ground truth: dogs have faster heart rates and different QRS morphology than humans, so a human-trained classifier's accuracy on canine recordings is unvalidated and unvalidatable without real labeled canine data anyway.

**Path to v2:** the `classify` module is isolated behind a plain interface (`Beat` in, label out) specifically so a learned model can be swapped in later without touching ingestion, aggregation, or reporting. The on-ramp to that: accumulate real Teeny recordings, ideally cross-checked against her cardiologist's own read of the same recording, building a small real canine ground-truth set over time.

## Architecture

Data flow: raw recorder file → standardized signal → R-peak/QRS detection → beat classification → burden aggregation → report.

```
canine-holter/
  ingest/       # raw file -> (samples, sample_rate, channel info, recording start time)
                # - dr200.py: DR200/Northeast-Monitoring format parser (blocked on
                #   confirming the raw export format - see Open items)
                # - wfdb_loader.py: loads MIT-BIH records, used only for pipeline testing
  detection/    # NeuroKit2 wrapper: R-peak detection + QRS onset/offset delineation
  classify/     # beat -> Normal | PVC | Other (rules-based, see above)
  arrhythmia/   # burden aggregation: PVC count/24h, couplet/triplet/VT-run detection,
                # brady/tachy flags, pause detection
  report/       # markdown/HTML summary + matplotlib rhythm-strip images around
                # each flagged event
  gui/          # Tkinter file-picker wrapper around the same pipeline
  cli.py        # `canine-holter analyze <file> --out report/`
  tests/
```

Modules communicate only through plain data structures at their boundary (e.g. a list of `Beat(time, rr_interval, qrs_width, label)` passed from `classify` to `arrhythmia`). No module reaches into another's internals. This is what lets the classifier be swapped later, and lets `ingest` grow a real DR200 parser without any downstream module changing.

Stack: Python. Libraries: NeuroKit2 (R-peak detection + QRS delineation, actively maintained, richest algorithm choice), wfdb-python (MIT-BIH access for pipeline testing), matplotlib (rhythm-strip plots), Tkinter (GUI, ships with Python — no extra dependency), PyInstaller (packaging).

## Packaging & release

GitHub Actions workflow triggered on version tags (`v*`), adapting the existing proven pattern from the `hedgehog` repo's `build-installers.yml`:

1. `macos-latest` runner, install deps, `pyinstaller` builds `canine-holter.app` (onedir mode — more reliable than onefile for Tkinter apps, at the cost of a slightly larger download)
2. Import Apple Developer cert into a temp keychain (`security create-keychain` / `security import`)
3. `codesign` the `.app` with the Developer ID
4. Package into a `.dmg`
5. `xcrun notarytool submit --wait` + `xcrun stapler staple`
6. Clean up the keychain
7. `gh release upload` attaches the signed, notarized `.dmg` to the GitHub Release

Requires copying 6 secrets from the `hedgehog` repo into this one (repo-scoped, not shared): `APPLE_CERTIFICATE_P12`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_ID_PASSWORD`, `APPLE_TEAM_ID`.

Result: tagging a release builds a signed, notarized `.dmg` with no Gatekeeper friction for Jaime.

## Testing & validation strategy

- **Unit tests** (pytest) per module in isolation — e.g. `classify` tested against synthetic `Beat` sequences with hand-constructed PVC/couplet/triplet cases.
- **Pipeline mechanics test** against MIT-BIH (via wfdb-python): confirms R-peak detection finds known beats and the rules-based classifier correctly flags known-`V`-annotated beats in human data. This validates that the code runs correctly end-to-end — explicitly **not** a claim of canine clinical accuracy, and documented as such. MIT-BIH remains the only source of PVC-labeled beats for testing the classification *logic* itself.
- **Canine-morphology R-peak validation** against the [PhysioZoo Mammalian NSR Database](https://physionet.org/content/physiozoo/1.0.0/) (17 dog ECG recordings, R-peak annotated, on PhysioNet). Normal sinus rhythm only — no PVC labels — but confirms R-peak detection works correctly on actual canine QRS morphology and heart rate, not just human. Complements the MIT-BIH mechanics test rather than replacing it.
- **Real validation**: once real Teeny recordings exist, ideally cross-checked against her cardiologist's own read of the same recording. This builds the real canine ground-truth set that both validates v1's rules-based thresholds and eventually enables a learned classifier (v2).
- CI gates on pipeline-mechanics regressions only; there's no canine accuracy data to gate on yet.

## Open items

1. **DR200 raw export format unconfirmed.** Pending response from ALBA Medical (emailed 2026-08-13) on whether the DR200 exports a documented/open format (suspected to be Northeast Monitoring-family `flash.dat`, which has a public-domain format and an existing open-source reader, `ishneholterlib`) or something proprietary. Blocks `ingest/dr200.py` specifically; nothing else in the pipeline depends on this.
2. **Canine ECG dataset search — completed.** No public dataset with canine beats labeled normal-vs-PVC exists (checked Zenodo, Figshare, Dryad, Mendeley, IEEE DataPort, Harvard Dataverse, Kaggle, UCI ML Repository, GitHub, VetCompass, and the major vet cardiology schools directly). The PhysioZoo Mammalian NSR Database (17 dog recordings, PhysioNet) has R-peak annotations but normal sinus rhythm only — folded into the testing strategy above. Two direct-contact leads for real PVC-labeled canine data, both currently unconfirmed: Wess et al. (LMU Munich, the group that defined the Doberman PVC-burden screening thresholds this project targets) and Dourson/Santilli et al.'s "PulseNet" paper (arXiv:2305.15424, 1,462 canine ECG recordings labeled normal/abnormal by 3 board-certified veterinary cardiologists). Neither has a confirmed public release; outreach emails to both are a candidate next step, independent of and not blocking v1 implementation.
3. **Canine HR/pause thresholds are provisional.** Bradycardia/tachycardia/pause flags will use reasonable default ranges initially but are expected to need calibration per-dog rather than relying on fixed literature values, similar to the PVC prematurity threshold.
