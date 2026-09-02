import os
import re
import tempfile
from datetime import datetime
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.generate import build_content
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
        start = datetime(2026, 8, 23, 15, 33, 8)
        write_pdf(
            out,
            content=build_content(beats, summary, start),
            beats=beats,
            summary=summary,
            start_time=start,
            channels=sig[None, :] if samples else None,
            channel_names=("ECG",) if samples else (),
            sample_rate=100.0 if samples else None,
        )
        return _page_count(out), os.path.getsize(out)


# Every report with a waveform has a summary page, a timeline page, the
# strip primer, an extremes strip page (fastest/slowest heart rate), and the
# one-strip-per-hour page (these fixtures are under an hour: one strip);
# event sections add pages before the hourly one, two strips per page.
_BASE_PAGES = 5
_BASE_PAGES_TEXT = 4  # no waveform: no primer, and the extremes and hourly sections are one text page each


def test_no_flagged_events_is_summary_timeline_primer_and_extremes():
    pages, size = _write(0)
    assert pages == _BASE_PAGES
    assert size > 1000


def test_two_events_fit_on_one_strip_page():
    assert _write(2)[0] == _BASE_PAGES + 1


def test_three_events_spill_to_a_second_strip_page():
    assert _write(3)[0] == _BASE_PAGES + 2


def test_no_samples_lists_events_on_a_text_page_without_a_primer():
    assert _write(3, samples=False)[0] == _BASE_PAGES_TEXT + 1


def _beats_with_isolated_pvcs(n):
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(200)]
    for k in range(n):
        i = 10 + k * 5
        beats[i] = _beat(beats[i].time, 0.8, "V")
    return beats


def _write_isolated(n, samples=True):
    beats = _beats_with_isolated_pvcs(n)
    summary = summarize(beats)
    sig = np.sin(np.linspace(0, 2000, 160 * 100)) if samples else None
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "report.pdf")
        write_pdf(
            out,
            content=build_content(beats, summary, None),
            beats=beats,
            summary=summary,
            start_time=None,
            channels=sig[None, :] if samples else None,
            channel_names=("ECG",) if samples else (),
            sample_rate=100.0 if samples else None,
        )
        return _page_count(out)


def test_isolated_pvcs_get_their_own_strip_pages():
    assert _write_isolated(4) == _BASE_PAGES + 2  # two strip pages (2 + 2)


def test_isolated_pvcs_are_listed_on_a_text_page_without_samples():
    assert _write_isolated(4, samples=False) == _BASE_PAGES_TEXT + 1


def _write_hours(hours):
    """A recording of the given length at 60 bpm, no events; return its page count."""
    n = int(hours * 3600) + 1
    beats = [_beat(float(i), 1.0 if i else None, "N") for i in range(n)]
    summary = summarize(beats)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "report.pdf")
        write_pdf(
            out,
            content=build_content(beats, summary, None),
            beats=beats,
            summary=summary,
            start_time=None,
            channels=np.zeros((1, n * 10)),
            channel_names=("ECG",),
            sample_rate=10.0,
        )
        return _page_count(out)


# A day-long recording's hourly strips: 25 strips on 13 pages replace the one-strip page of the base count.
_HOURLY_STRIP_PAGES_24H = 13 - 1


def test_hourly_table_shares_the_timeline_page_for_a_day_long_recording():
    assert _write_hours(24.5) == _BASE_PAGES + _HOURLY_STRIP_PAGES_24H  # 25 rows fit under the timeline


def test_hourly_table_spills_to_its_own_page_past_the_first_page_rows():
    # 27 rows: 26 under the timeline, 1 on a new page; 27 strips on 14 pages
    assert _write_hours(26.5) == _BASE_PAGES + 1 + (14 - 1)


def test_summary_page_renders_groups_with_status_colors_and_footer():
    import matplotlib.pyplot as plt
    from canine_holter.report.pdf import STATUS_COLORS, _summary_page

    beats = _beats_with_couplets(1)
    content = build_content(beats, summarize(beats), None)
    fig = _summary_page(content)
    texts = {t.get_text(): t for t in fig.texts}
    assert "RECORDING" in texts and "VENTRICULAR ECTOPY" in texts and "Couplets" in texts
    assert texts["1"].get_color() == STATUS_COLORS["alert"]  # the couplet count
    assert any("not a diagnosis" in t for t in texts)
    plt.close(fig)


def test_summary_page_lays_the_six_panels_out_in_three_rows():
    import matplotlib.pyplot as plt
    from canine_holter.report.pdf import _PANEL_TOP, _summary_page

    assert len(_PANEL_TOP) == 3
    beats = _beats_with_couplets(1)
    fig = _summary_page(build_content(beats, summarize(beats), None))
    titles = [t.get_text() for t in fig.texts if t.get_text().isupper() and t.get_fontweight() == "bold"]
    assert titles == [
        "RECORDING", "HEART RATE", "VENTRICULAR ECTOPY", "SUPRAVENTRICULAR ECTOPY", "PAUSES", "RR VARIABILITY"
    ]
    # The footer sits below the third row of panels.
    footer_y = min(t.get_position()[1] for t in fig.texts if "not a diagnosis" in t.get_text())
    assert footer_y < _PANEL_TOP[2] - 0.1
    plt.close(fig)


def test_significance_line_wraps_instead_of_running_off_the_page():
    from canine_holter.report.generate import StripCaption
    from canine_holter.report.pdf import _CAPTION_WRAP, _significance_lines
    long = "4 PVCs in a row at 150 bpm: an accelerated idioventricular rhythm, generally less concerning than ventricular tachycardia."
    lines = _significance_lines(StripCaption("t", "w", long, "caution"))
    assert len(lines) == 2
    assert lines[0].startswith("\u25cf ")
    assert all(len(line) <= _CAPTION_WRAP for line in lines)
    assert " ".join(lines) == "\u25cf " + long
    assert _significance_lines(StripCaption("t", "w", "short", None)) == ["\u25cf short"]
