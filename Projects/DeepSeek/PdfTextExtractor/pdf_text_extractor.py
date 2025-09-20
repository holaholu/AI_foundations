import os
import shutil
import fitz  # PyMuPDF
import gradio as gr
import requests

from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# DeepSeek API URL
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def _find_poppler_path() -> str | None:
    """Try common Homebrew install paths for Poppler on macOS."""
    candidates = [
        "/opt/homebrew/bin",  # Apple Silicon default
        "/usr/local/bin",     # Intel macs
    ]
    # If pdfinfo is already on PATH, no need to return a path
    if shutil.which("pdfinfo"):
        return None
    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "pdfinfo")):
            return path
    return None


def _ensure_tesseract_path() -> str | None:
    """Ensure pytesseract knows where the tesseract binary is on macOS."""
    if shutil.which("tesseract"):
        return None
    candidates = [
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for exe in candidates:
        if os.path.isfile(exe):
            pytesseract.pytesseract.tesseract_cmd = exe
            return exe
    return None

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"

    return text if text.strip() else "No text found in the PDF."

def extract_text_with_ocr(pdf_file):
    """Extract text from a scanned PDF via OCR, handling Poppler/Tesseract availability."""
    poppler_path = _find_poppler_path()
    try:
        images = convert_from_path(pdf_file, poppler_path=poppler_path)
    except Exception as e:
        # Provide a clear message if Poppler is missing
        return (
            "OCR requires Poppler utilities. Install with: 'brew install poppler'. "
            f"Details: {e}"
        )

    tesseract_set = _ensure_tesseract_path()
    if not (tesseract_set or shutil.which("tesseract")):
        return (
            "Tesseract OCR not found. Install with: 'brew install tesseract'. "
            "Then re-run this tool."
        )

    extracted_text = "\n".join(pytesseract.image_to_string(img) for img in images)
    return extracted_text if extracted_text.strip() else "No text found via OCR."


def summarize_text(text):
    prompt = f"Summarize the following document text:\n\n{text}"
    
    payload = {
        "model": "deepseek-r1",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    return response.json().get("response", "No summary available.")

def extract_text_smart(pdf_file):
    """Try native text extraction first; fall back to OCR if needed."""
    try:
        text = extract_text_from_pdf(pdf_file)
    except Exception as e:
        text = ""
    if text and text.strip() and text.strip().lower() != "no text found in the pdf.":
        return text
    # Fall back to OCR
    return extract_text_with_ocr(pdf_file)

# Create Gradio interface
interface = gr.Interface(
    fn=extract_text_smart,
    inputs=gr.File(label="Upload PDF File"),
    outputs=gr.Textbox(label="Extracted Text", lines=15),
    title="AI-Powered PDF Text Extractor",
    description="Upload a PDF file, and AI will extract its text content."
)

# Launch the web app
if __name__ == "__main__":
    interface.launch()


# Test PDF Text Extraction
# if __name__ == "__main__":
#     pdf_path = "sample2.pdf"  # Provide a sample PDF file
#     print("### Summarized Extracted Text ###")
#     # print(extract_text_from_pdf(pdf_path))
#     print(summarize_text(extract_text_from_pdf(pdf_path)))






# Test PDF Image to Text Extraction
# if __name__ == "__main__":
#     pdf_path = "sample4ocr.pdf"  # Provide a sample PDF file
#     print("### Extracted Text ###")
#     print(extract_text_smart(pdf_path))
    










