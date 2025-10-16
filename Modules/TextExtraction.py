import os
import json
import uuid
import logging

import pytesseract
from pdf2image import convert_from_path

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

from config import DATA_DIR, BLOCKS_FILE, DOCS_FILE, ERROR_LOG, OCR_DPI

# =============== LOGGING SETUP =================
logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR)


def extract_text_digital(pdf_path: str, page_num: int) -> str:
    """
    Try extracting text using PyPDF2 for a specific page.
    """
    if PdfReader is None:
        logging.error("PyPDF2 not installed.")
        return ""

    try:
        reader = PdfReader(pdf_path)
        page = reader.pages[page_num]
        text = page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logging.error(f"Digital extraction failed: {pdf_path}, page {page_num} - {e}")
        return ""


def extract_text_ocr(pdf_path: str, page_num: int) -> str:
    """
    Fallback OCR text extraction using pytesseract + pdf2image.
    """
    try:
        images = convert_from_path(
            pdf_path, 
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=OCR_DPI
        )
        text = pytesseract.image_to_string(images[0])
        return text.strip()
    except Exception as e:
        logging.error(f"OCR failed: {pdf_path}, page {page_num} - {e}")
        return ""


def process_pdf(pdf_path: str, blocks_writer, docs_writer):
    """
    Process a single PDF file and extract text page-by-page.
    """
    filename = os.path.basename(pdf_path)
    doc_id = uuid.uuid4().hex

    if PdfReader is None:
        logging.error("PyPDF2 not installed, cannot process digital text.")
        return

    # Read number of pages
    try:
        reader = PdfReader(pdf_path)
        n_pages = len(reader.pages)
    except Exception as e:
        logging.error(f"Could not read PDF {pdf_path} - {e}")
        return

    aggregated_text = []
    total_words = 0
    pages_by_source = {"digital": 0, "ocr": 0, "error": 0}

    for i in range(n_pages):
        text = extract_text_digital(pdf_path, i)

        if not text or len(text.split()) < 3:  # Too short → fallback to OCR
            text = extract_text_ocr(pdf_path, i)
            source = "ocr" if text else "error"
        else:
            source = "digital"

        if not text:
            pages_by_source["error"] += 1
            logging.error(f"Page {i} in {filename} could not be extracted")
        else:
            word_count = len(text.split())
            total_words += word_count
            aggregated_text.append(text)
            pages_by_source[source] += 1

            blocks_writer.write(json.dumps({
                "doc_id": doc_id,
                "filename": filename,
                "page_index": i,
                "source": source,
                "text": text,
                "n_words": word_count
            }, ensure_ascii=False) + "\n")

    docs_writer.write(json.dumps({
        "doc_id": doc_id,
        "filename": filename,
        "n_pages": n_pages,
        "text_all": "\n".join(aggregated_text),
        "aggregates": {
            "total_words": total_words,
            "pages_by_source": pages_by_source
        }
    }, ensure_ascii=False) + "\n")


def run_extraction(input_dir: str = DATA_DIR):
    """
    Run extraction for all PDFs in the input folder.
    """
    with open(BLOCKS_FILE, "w", encoding="utf-8") as blocks_writer, \
         open(DOCS_FILE, "w", encoding="utf-8") as docs_writer:

        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, f)
                    print(f"Processing {pdf_path} ...")
                    process_pdf(pdf_path, blocks_writer, docs_writer)

    print("✅ Extraction complete!")
    print(f"Blocks → {BLOCKS_FILE}")
    print(f"Docs   → {DOCS_FILE}")
    print(f"Errors → {ERROR_LOG}")


if __name__ == "__main__":
    run_extraction()
