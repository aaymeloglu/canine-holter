import pytest
import canine_holter.report.generate as generate


@pytest.fixture
def report_text(monkeypatch):
    """Capture the text that reaches the PDF writer, as one string. The PDF
    itself is not greppable (matplotlib writes glyph codes), so end-to-end
    tests check the content handed to write_pdf and still let the real PDF
    be written."""
    captured = {}
    real = generate.write_pdf

    def spy(out_path, *, content, **kw):
        lines = []
        for group in content.summary_groups:
            lines.append(group.title)
            for row in group.rows:
                lines.append(f"{row.label}: {row.value}" + (f" ({row.reference})" if row.reference else ""))
        captured["text"] = "\n".join(
            lines
            + content.footer_lines
            + [line for s in content.sections for line in [s.heading, *s.labels]]
            + [line for s in content.sections for c in s.captions for line in (c.title, c.what, c.significance)]
            + [" | ".join(row) for row in [content.hourly_header, *content.hourly_rows]]
        )
        return real(out_path, content=content, **kw)

    monkeypatch.setattr(generate, "write_pdf", spy)
    return lambda: captured["text"]
