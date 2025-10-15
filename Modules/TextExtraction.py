import os
import json
import uuid
import logging
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# ================= CONFIG =================
DATA_DIR = r"Input Folder path"
OUT_DIR = "processed_jsonl"
LOG_DIR = "logs"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

BLOCKS_FILE = os.path.join(OUT_DIR, "text_blocks.jsonl")
DOCS_FILE = os.path.join(OUT_DIR, "text_docs.jsonl")
ERROR_LOG = os.path.join(LOG_DIR, "text_extraction_errors.log")

logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR)


def extract_text_digital(pdf_path, page_num):
    """Try extracting text using PyPDF2."""
    try:
        reader = PdfReader(pdf_path)
        page = reader.pages[page_num]
        text = page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logging.error(f"Digital extraction failed: {pdf_path}, page {page_num} - {e}")
        return ""


def extract_text_ocr(pdf_path, page_num):
    """Fallback: use OCR via pdf2image + pytesseract."""
    try:
        images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300)
        text = pytesseract.image_to_string(images[0])
        return text.strip()
    except Exception as e:
        logging.error(f"OCR failed: {pdf_path}, page {page_num} - {e}")
        return ""


def process_pdf(pdf_path, blocks_writer, docs_writer):
    filename = os.path.basename(pdf_path)
    doc_id = uuid.uuid4().hex

    if PdfReader is None:
        logging.error("PyPDF2 not installed, cannot process digital text.")
        return

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

        if not text or len(text.split()) < 3:  # too short → OCR
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

    # Save doc-level aggregate
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


def main():
    with open(BLOCKS_FILE, "w", encoding="utf-8") as blocks_writer, \
         open(DOCS_FILE, "w", encoding="utf-8") as docs_writer:

        for root, _, files in os.walk(DATA_DIR):
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
    main()
