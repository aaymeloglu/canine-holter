# DR400 recordings: reused cards and long off-body tails

**Date:** 2026-09-01
**Status:** approved design.

## Problem

Teeny's third recording (2026-08-27 10:18:49, a NorthEast DR400 with the
three-wire patch cable, card copied 2026-09-01 as `flash2.dat`, 408 MB)
fails to load and, once loaded, is gated backwards.

The DR400 writes the same container as the DR200: 512-byte checksummed
blocks, `SampleRate=180`, `SampleStorageFormat=1`, three channels of
four-bit deltas. The existing decoder reads every block. Three things
differ from the two DR200 recordings analyzed so far:

1. **The card was reused.** The file holds 716,602 valid ECG blocks (336 h
   at 180 Hz) but the recorder ran at most 129 h. Two bytes at offset 466
   of every ECG block hold a recording sequence number and four bytes at
   offset 468 hold the recorder serial number (`Serial_number=127226` in the
   header; the block field reads 27226, the last five digits; on the DR200
   file `052837` and 52837). The sequence number is 8 for the first 266,289 ECG blocks,
   then 6, then 5 (older recordings by the same recorder), then 40 with
   serial 13515 (another recorder). Source positions skip by ±4 bytes at
   the first two boundaries and happen to be contiguous at the third. The
   parser today raises "Non-contiguous DR200 ECG blocks" at 124.9 h.
2. **The recording is 124.9 h long but Teeny wore it for ~24.5 h.** The
   recorder was mailed back running. Off the body the DR400's front end
   records its AC lead-off excitation: a square wave at exactly half the
   sample rate (raw counts alternate 1281, 1302, 1281, 1302; 0.26 mV
   peak-to-peak). 68,398 of the 89,946 five-second windows are this tone.
3. **The quality gate keeps the tone and drops the ECG.** Its amplitude
   reference is the recording-wide median peak-to-peak, which the tone
   pulls to 0.26 mV; real QRS windows (1.5 mV median) then exceed the 4x
   ceiling. The gate reports 100 h analyzed and 25 h excluded, the reverse
   of the truth.

The file was named `flash2.dat`, which the loader does not recognize.

## Goals

1. Load a reused-card `flash.dat` from either recorder, keeping only the
   blocks of the recording named in the header, and fail closed on
   anything that does not match the documented layout.
2. Exclude open-electrode signal wherever it occurs, and keep the
   amplitude reference honest when off-body time exceeds on-body time.
3. Trim a long off-body tail so detection, the hourly table, and the
   timeline cover the worn part of the recording, and say in the report
   how long the recorder actually ran.
4. Accept a native file under any name.

## Non-goals

- A DR200 mailed back running. Off-body it records rail-to-rail swings
  then flat line, not a lead-off tone; only one such tail (8 min) has been
  seen and the existing rules handled it. No rule is designed for a case
  without evidence.
- Off-body signal that passes the amplitude rules and is not the tone. A
  22-minute stretch at hour 112 in transit has ECG-like amplitude; it is
  inside the trimmed tail here, but a similar stretch in a short tail
  would be analyzed. Same as before this change.
- Multi-lead quality or detection.

## 1. Parser (`ingest/dr200.py`)

Every ECG block's bytes 466..472 are its **session key**: `u16` recording
sequence number then `u32` recorder serial number. The first ECG block's
key is the recording's key. Scanning stops at the first ECG block whose
key differs; that block and everything after it are stale data from an
earlier recording on the card and are neither decoded nor validated.
Blocks before it must still be contiguous in source position and pass
every existing check.

The header's `Serial_number` is not cross-checked: on the DR400 the block
field holds only the last five digits (27226 for 127226), and one sample
is not enough to fix the rule. The key is compared only with itself.

`_inspect_native_flash` returns the count of session ECG blocks; the
decode loop stops after that many. Non-ECG blocks after the boundary are
ignored too.

The module keeps its name; the format is NorthEast's, shared by both
recorders. User-facing text says "DR200/DR400".

## 2. Loader (`ingest/loader.py`)

A `.dat` file that is not a decoded channel is native when its first
block starts with the little-endian length 512 and contains
`SampleRate=`. Sniffing reads 512 bytes. The name `flash.dat` still works
without sniffing.

## 3. Quality gate (`quality/gate.py`)

New rule, per 5 s window:

| Rule | Constant | Fires when | What it is for |
|---|---|---|---|
| Lead-off tone | `MAX_DIFFERENCE_POWER_RATIO = 3.0` | variance of the sample-to-sample differences over variance of the samples exceeds 3 | open electrodes: the front end's AC lead-off excitation at half the sample rate |

The ratio is bounded by 4 (a pure alternating signal) and is 2 for white
noise. Evidence: Teeny's on-body 24 h has a 99th percentile of 1.77 at
180 Hz; MIT-BIH 119 at 360 Hz reads 0.05 at most; the PhysioZoo dog at
500 Hz 0.06; the DR400 tail reads 3.5-4.0. A 5 Hz ventricular flutter at
180 Hz would read 0.03, so the rule cannot exclude the events the tool
exists to find (the reason kurtosis was rejected). The rule is not the
"high-frequency power ratio" rejected in the 2026-08-26 spec: that was
tuned to catch hash noise at a 0.3 threshold; this fires only when
almost all power sits at the Nyquist frequency.

The amplitude reference becomes the median peak-to-peak of the windows
not flagged by the two absolute rules (lead-off tone, flat line). The
relative rules (4x, 0.1x) are unchanged. With no window left, the
recording is excluded whole, as today.

### Tail trimming

After the per-window rules and before the edge rules. The tail starts at
the earliest lead-off run of at least `TAIL_MIN_RUN_SEC = 1800` after
which at least `TAIL_LEAD_OFF_FRACTION = 0.9` of the windows to the end of
the recording are lead-off. The recording then ends at that run's start.
The 30-minute run keeps a loose electrode, and a vest slipping off in the
last minutes, as excluded time inside the duration, as today. The 90
percent test keeps an off-body hour followed by a re-attached recorder
inside the duration unless the tail after it is more than nine times
longer; the report always states the trimmed time, so that case is
visible rather than silent.

The tail is judged whole rather than gap by gap because transit is not
pure tone: on Teeny's file the package being handled produces
amplitude-plausible, non-tone signal for up to 44 minutes at a stretch
(hour 112) and for the final 32 minutes before the card came out, so any
rule that bridges gaps by length or by analyzable time fails to reach the
end of the recording. Rejected for that reason.

On Teeny's DR400 file the tail starts at 24.76 h, a 24.6 h run, and
94.8 percent of the remaining windows are lead-off. The on-body ECG ends
at ~24.5 h; the removal (rail-to-rail) minutes between stay excluded.

`SignalQuality` gains `trimmed_sec: float = 0.0`. `duration_sec` is the
kept length; the recorder ran `duration_sec + trimmed_sec`. The edge rule
for the last minute applies at the kept end.

## 4. Pipeline and report

`run_analysis` slices `samples` and `channels` to `duration_sec` before
detection when `trimmed_sec > 0`, so detection never sees the tail.

`ArrhythmiaSummary` gains `trimmed_sec` from the quality result. When a
tail was trimmed the Recording panel gains a row after Duration:
`Recorder ran 124h 55m` with the reference text `off-body tail trimmed`
(the reference column is too narrow for a sentence). Untrimmed reports
are unchanged.

## 5. Docs and text

- `docs/dr200-format.md`: the session-key fields with the evidence above,
  the reused-card behavior, the lead-off tone, and the DR400 fixture
  fingerprint (size and SHA-256).
- CLI help, GUI file-type label, README, CLAUDE.md: DR200/DR400.

## Testing

- Parser: the block factory gains `sequence` and `serial` arguments and a
  `Serial_number` header line. Tests with literal bytes: stale blocks
  after the boundary are dropped (including a non-contiguous one and a
  zero-padding-then-data one, which today raise); a header serial that
  disagrees with the block serial raises; a non-contiguous block inside
  the session still raises.
- Loader: a renamed native file loads; a `.dat` with neither the
  recognized name nor the signature is rejected.
- Gate: a synthetic alternating tone stretch is excluded; the median is
  unaffected by a majority-tone recording; the trimming rule on 60 min of
  ECG followed by 2 h of tone trims to 60 min, a 10 min tone tail is not
  trimmed, a 60 min tone gap followed by 60 min of ECG is not trimmed, a
  10 min signal stretch inside the tail is bridged, 15 min of signal at
  the very end of a 3 h tail is still trimmed, and a 10 s tone blip
  during wear neither starts the tail nor moves it.
- Pipeline and report: a trimmed recording's Duration row.
- Acceptance: `flash2.dat` loads (266,289 blocks, 124.93 h, start
  2026-08-27 10:18:49), trims to ~24.8 h, and the DR200 2026-08-25 file's
  gating is unchanged (24.32 h analyzed, two spans).
