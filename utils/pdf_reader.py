import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {pdf_path}: {e}")
    return text

def load_input_text(file_path: str) -> str:
    """Loads text from a .txt or .pdf file."""
    _, ext = os.path.splitext(file_path)

    if ext.lower() == ".pdf":
        print(f"[INFO] Extracting text from PDF: {file_path}")
        return extract_text_from_pdf(file_path)
    elif ext.lower() == ".txt":
        print(f"[INFO] Reading text from TXT: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format. Use .txt or .pdf")
