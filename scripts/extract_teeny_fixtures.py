"""One-off: cut the hand-checked windows of Teeny's recordings into test
fixtures. Run manually with the flash.dat paths; the fixtures are committed
so the suite never needs the recordings.

    .venv/bin/python scripts/extract_teeny_fixtures.py \
        2026-08-25=~/Downloads/teeny-holter-2026-08-26/flash.dat \
        2026-08-27=~/Downloads/flash2.dat

Beat times were read by eye from zoomed three-channel plots (see the
2026-08-26 detector spec) and snapped to the steepest sample of the
cleaned analysis lead within +/-80 ms. PVC times (the 2026-09-01 lead
agreement spec) are the detector's fiducial for the beat that is wider
and differently shaped on all three leads.
"""
import os
import sys
from datetime import datetime, timedelta
import numpy as np
from canine_holter.ingest.loader import load_recording

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")

# recording date: {name: (offset from recording start in s, duration s, beat times, pvc times)}
# offsets may be given as a wall-clock "HH:MM:SS" string, resolved against the header
WINDOWS = {
    "2026-08-25": {
        # 08:47:52.45, sinus tachycardia ~150 bpm; NeuroKit finds ~16 of these 23
        "tachy": (44560.45, 9.55, [0.222, 0.556, 0.900, 1.267, 1.633, 2.006, 2.394, 2.794, 3.194, 3.594, 3.994, 4.389, 4.800, 5.206, 5.639, 6.111, 6.617, 7.139, 7.589, 7.978, 8.344, 8.844, 9.361], []),
        # 15:22:53, lying down: a 0.3 mV QRS spike then a 0.7 mV T trough 0.2-0.35 s later
        "lying": (68257.0, 20.0, [1.006, 2.900, 4.167, 4.950, 7.078, 9.122, 10.406, 11.172, 13.028, 15.317, 16.578, 17.372, 19.606], []),
        # 15:12:00, lying down with a wandering pacemaker: the T waves NeuroKit
        # detects as beats, and a QRS that is a 0.03-0.3 mV notch on Ch 1 (3.5 mV
        # on Ch 3) so the detector lands on the tall P wave 130-170 ms earlier in
        # alternate beats. Beat times are the QRS, snapped on Ch 3.
        "lying_t": (67604.0, 18.3, [0.694, 2.128, 3.567, 5.267, 6.928, 8.522, 10.528, 11.989, 13.283, 14.978, 16.294, 17.378, 18.183], []),
        # 17:06:18, upright and still, with a real 4.67 s sinus pause
        "quiet": (74462.0, 16.0, [2.456, 4.483, 5.961, 7.367, 12.033, 13.839, 15.672], []),
        # 11:36:33-11:36:53: a PVC at 11:36:41.27, big and biphasic on all three
        # leads, after a run of small fast beats
        "pvc_run_end": ("11:36:33", 20.0, [], [8.267]),
    },
    "2026-08-27": {
        # 00:03:00 +40 s, asleep with sinus arrhythmia: channel 1 shows every
        # QRS as a fractured 90-130 ms wiggle, channels 0 and 2 as identical
        # narrow complexes. No PVCs.
        "midnight": ("00:03:00", 40.0, [], []),
        # 16:06:55-16:07:15: a PVC at 16:07:03.17, wide on all three leads
        "pvc": ("16:06:55", 20.0, [], [8.172]),
    },
}


def _offset(rec, when) -> float:
    if not isinstance(when, str):
        return float(when)
    clock = datetime.strptime(when, "%H:%M:%S").time()
    start = rec.start_time
    at = datetime.combine(start.date(), clock)
    if at < start:
        at += timedelta(days=1)
    return (at - start).total_seconds()


def main(args: list[str]) -> None:
    for arg in args:
        date, flash_path = arg.split("=", 1)
        rec = load_recording(os.path.expanduser(flash_path))
        out_dir = os.path.join(FIXTURES_DIR, f"teeny_{date}")
        os.makedirs(out_dir, exist_ok=True)
        for name, (when, duration, beat_times, pvc_times) in WINDOWS[date].items():
            offset = _offset(rec, when)
            i0, i1 = int(offset * rec.sample_rate), int((offset + duration) * rec.sample_rate)
            np.savez(
                os.path.join(out_dir, f"{name}.npz"),
                channels=rec.channels[:, i0:i1].astype(np.float32),
                sample_rate=rec.sample_rate,
                beat_times=np.array(beat_times),
                pvc_times=np.array(pvc_times),
                offset_sec=offset,
            )
            print(f"{date}/{name}: +{offset:.2f} s, {duration} s, {len(beat_times)} beats, {len(pvc_times)} PVCs")


if __name__ == "__main__":
    main(sys.argv[1:])
