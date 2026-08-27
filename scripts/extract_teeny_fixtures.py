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
    # 15:12:00, lying down with a wandering pacemaker: the T waves NeuroKit
    # detects as beats, and a QRS that is a 0.03-0.3 mV notch on Ch 1 (3.5 mV
    # on Ch 3) so the detector lands on the tall P wave 130-170 ms earlier in
    # alternate beats. Beat times are the QRS, snapped on Ch 3.
    "lying_t": (67604.0, 18.3, [0.694, 2.128, 3.567, 5.267, 6.928, 8.522, 10.528, 11.989, 13.283, 14.978, 16.294, 17.378, 18.183]),
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
