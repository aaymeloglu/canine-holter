from canine_holter.report.common import flagged_runs, isolated_pvcs, select_evenly, pvc_line
from canine_holter.types import Beat


def _beat(time, label):
    return Beat(time=time, rr_interval=0.8, qrs_duration=0.08, label=label)


def test_isolated_pvcs_are_single_v_runs_only():
    beats = [_beat(0, "N"), _beat(1, "V"), _beat(2, "N"), _beat(3, "V"), _beat(4, "V"), _beat(5, "N"), _beat(6, "V")]
    assert [[b.time for b in run] for run in isolated_pvcs(beats)] == [[1], [6]]
    assert [[b.time for b in run] for run in flagged_runs(beats)] == [[3, 4]]


def test_select_evenly_returns_everything_under_the_cap():
    assert select_evenly([1, 2, 3], 24) == [1, 2, 3]


def test_select_evenly_spreads_picks_across_the_list():
    items = list(range(30))
    picked = select_evenly(items, 4)
    assert picked == [0, 10, 19, 29]


def test_pvc_line_labels_an_isolated_pvc():
    from datetime import datetime
    start = datetime(2026, 8, 23, 15, 33, 8)
    assert pvc_line(2, [_beat(8232.8, "V")], start) == "PVC 3: isolated PVC at ~17:50:20 (t=8232.8s)"
