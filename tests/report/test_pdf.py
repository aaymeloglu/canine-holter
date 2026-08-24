import os
import re
import tempfile
from datetime import datetime
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.pdf import write_pdf


def _beat(time, rr, label):
    return Beat(time=time, rr_interval=rr, qrs_duration=0.08, label=label)


def _page_count(path):
    # matplotlib writes page objects uncompressed; /Pages is the tree root.
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", open(path, "rb").read()))


def _beats_with_couplets(n_runs):
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(200)]
    for r in range(n_runs):
        i = 20 + r * 30
        beats[i] = _beat(beats[i].time, 0.8, "V")
        beats[i + 1] = _beat(beats[i + 1].time, 0.8, "V")
    return beats


def _write(n_runs, samples=True):
    beats = _beats_with_couplets(n_runs)
    summary = summarize(beats)
    sig = np.sin(np.linspace(0, 2000, 160 * 100)) if samples else None
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "report.pdf")
        write_pdf(
            out,
            summary_lines=["- Total beats: 200"],
            beats=beats,
            summary=summary,
            start_time=datetime(2026, 8, 23, 15, 33, 8),
            samples=sig,
            sample_rate=100.0 if samples else None,
        )
        return _page_count(out), os.path.getsize(out)


def test_no_flagged_events_is_one_page():
    pages, size = _write(0)
    assert pages == 1
    assert size > 1000


def test_three_events_fit_on_one_strip_page():
    assert _write(3)[0] == 2


def test_four_events_spill_to_a_third_page():
    assert _write(4)[0] == 3


def test_no_samples_lists_events_without_strip_pages():
    assert _write(3, samples=False)[0] == 1
