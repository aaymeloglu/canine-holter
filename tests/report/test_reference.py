"""Reference bands and the status each value gets. The bands come from the
ESVC Doberman DCM screening guidelines (Wess et al. 2017); the tests pin
the literal boundaries so a wording edit cannot silently move a band."""
from canine_holter.report.reference import (
    FOOTER_LINES,
    PAUSE_BAND,
    PVC_24H_BAND,
    RUN_RATE_BAND,
    analyzed_status,
    count_status,
    pause_status,
    pvc_24h_status,
    pvc_per_24h,
    run_rate_status,
)

H = 3600.0


def test_pvc_per_24h_not_computed_under_20_analyzed_hours():
    assert pvc_per_24h(65, 2.5 * H) is None
    assert pvc_per_24h(65, 19.9 * H) is None


def test_pvc_per_24h_equals_count_for_24_analyzed_hours():
    assert pvc_per_24h(120, 24 * H) == 120


def test_pvc_per_24h_scales_48_analyzed_hours_down():
    assert pvc_per_24h(120, 48 * H) == 60


def test_pvc_24h_status_bands():
    assert pvc_24h_status(49) == "ok"
    assert pvc_24h_status(50) == "caution"
    assert pvc_24h_status(300) == "caution"
    assert pvc_24h_status(301) == "alert"


def test_count_status_is_ok_only_at_zero():
    assert count_status(0) == "ok"
    assert count_status(1) == "alert"


def test_run_rate_status():
    assert run_rate_status(None) == "ok"
    assert run_rate_status(179.9) == "caution"
    assert run_rate_status(180.0) == "alert"


def test_pause_status_bands():
    assert pause_status(None) == "ok"
    assert pause_status(2.49) == "ok"
    assert pause_status(2.5) == "caution"
    assert pause_status(5.0) == "caution"
    assert pause_status(5.01) == "alert"


def test_analyzed_status_needs_20_hours():
    assert analyzed_status(20 * H) == "ok"
    assert analyzed_status(19.99 * H) == "caution"


def test_band_strings_and_footer_carry_the_guideline_values_and_source():
    assert PVC_24H_BAND == "<50 | 50-300 | >300"
    assert PAUSE_BAND == "<2.5 | 2.5-5 | >5 s"
    assert RUN_RATE_BAND == "<180 bpm"
    text = "\n".join(FOOTER_LINES)
    assert "Wess" in text and "2017" in text
    assert "not a diagnosis" in text
