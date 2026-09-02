import pytest
import canine_holter.report.pdf as pdf


@pytest.fixture
def report_text(monkeypatch):
    """Capture the text that reaches the PDF writer, as one string. The PDF
    itself is not greppable (matplotlib writes glyph codes), so end-to-end
    tests check the content handed to write_pdf and still let the real PDF
    be written."""
    captured = {}
    real = pdf.write_pdf

    def spy(out_path, *, content, **kw):
        lines = []
        for group in content.summary_groups:
            lines.append(group.title)
            for row in group.rows:
                lines.append(f"{row.label}: {row.value}" + (f" ({row.reference})" if row.reference else ""))
        captured["text"] = "\n".join(
            lines
            + content.footer_lines
            + [s.heading for s in content.sections]
            + [
                line
                for s in content.sections
                for item in s.items
                for line in (item.caption.title, item.caption.what, item.caption.significance)
            ]
            + [" | ".join(row) for row in [content.hourly_header, *content.hourly_rows]]
        )
        return real(out_path, content=content, **kw)

    monkeypatch.setattr(pdf, "write_pdf", spy)
    return lambda: captured["text"]
