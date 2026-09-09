from app.services.pdf import extract_text_from_pdf


def test_extract_text_from_pdf():
    text = extract_text_from_pdf("data/documents/test.pdf")

    assert text
    assert "expected text" in text