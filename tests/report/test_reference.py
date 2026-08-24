"""Reference ranges printed beside the report numbers. The bands come from
the ESVC Doberman DCM screening guidelines (Wess et al. 2017); the tests
pin the literal values so a wording edit cannot silently move a band."""
from canine_holter.report.reference import pvc_per_24h, reference_lines, pvc_per_24h_line

H = 3600.0


def test_pvc_per_24h_not_computed_under_20_hours():
    assert pvc_per_24h(65, 2.5 * H) is None
    assert pvc_per_24h(65, 19.9 * H) is None


def test_pvc_per_24h_equals_count_for_a_24_hour_recording():
    assert pvc_per_24h(120, 24 * H) == 120


def test_pvc_per_24h_scales_a_48_hour_recording_down():
    assert pvc_per_24h(120, 48 * H) == 60


def test_pvc_per_24h_line_says_not_computed_for_short_recording():
    assert pvc_per_24h_line(65, 2 * H + 28 * 60) == (
        "- PVCs per 24 h: not computed (recording is 2h 28m; needs >= 20 h)"
    )


def test_pvc_per_24h_line_shows_scaled_count_for_long_recording():
    assert pvc_per_24h_line(120, 22 * H + 10 * 60) == "- PVCs per 24 h: 130 (scaled from 22h 10m)"


def test_reference_lines_carry_the_guideline_bands_and_source():
    text = "\n".join(reference_lines(24 * H))
    assert "under 50" in text
    assert "50-300" in text
    assert "over 300" in text
    assert "Couplets, triplets, or VT runs" in text
    assert "2.5 s" in text
    assert "Wess" in text and "2017" in text


def test_reference_lines_add_short_recording_note_only_under_20_hours():
    short = "\n".join(reference_lines(2 * H + 28 * 60))
    long = "\n".join(reference_lines(24 * H))
    assert "This recording is 2h 28m" in short
    assert "This recording is" not in long
