import os
import json
import uuid
import logging
from collections import Counter
from unstructured.partition.pdf import partition_pdf
from ..config import (
    INPUT_DIR,
    OUTPUT_DIR,
    LAYOUT_OUTPUT_FILE,
    SUMMARY_OUTPUT_FILE,
    ERROR_LOG_FILE,
)


# ====================== ENVIRONMENT SETUP ======================

def setup_environment():
    """Ensure required directories and logging configuration are ready."""
    for directory in [OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Reset previous logging handlers (important if rerunning in notebooks)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(ERROR_LOG_FILE),
            logging.StreamHandler()
        ],
    )


# ====================== HELPER FUNCTIONS ======================

def map_element_type(element_category: str) -> str:
    """Map unstructured.io element categories to normalized block types."""
    type_map = {
        "Title": "title",
        "NarrativeText": "paragraph",
        "UncategorizedText": "paragraph",
        "ListItem": "paragraph",
        "Table": "table",
        "Image": "image",
        "FigureCaption": "image",
        "Footer": "footer",
        "Header": "footer",
    }
    return type_map.get(element_category, "unknown")


# ====================== CORE EXTRACTION ======================

def process_pdf(pdf_path: str):
    """
    Extracts layout blocks and summary stats from a single PDF.
    """
    filename = os.path.basename(pdf_path)
    logging.info(f"Processing: {filename}")

    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False,
        )

        doc_id = str(uuid.uuid4())
        blocks = []
        stats_counter = Counter()
        max_page_number = 0

        for i, el in enumerate(elements):
            block_type = map_element_type(el.category)
            stats_counter[block_type] += 1

            coords = el.metadata.coordinates.points if el.metadata.coordinates else None
            page_number = el.metadata.page_number or 0
            max_page_number = max(max_page_number, page_number)

            block_data = {
                "doc_id": doc_id,
                "filename": filename,
                "page_index": (page_number - 1) if page_number else 0,
                "block_index": i,
                "type": block_type,
                "bbox": [coords[0][0], coords[0][1], coords[2][0], coords[2][1]] if coords else None,
                "text": el.text,
            }
            blocks.append(block_data)

        summary = {
            "doc_id": doc_id,
            "filename": filename,
            "n_pages": max_page_number,
            "stats": dict(stats_counter),
        }

        logging.info(f"✅ Completed {filename}: {len(blocks)} blocks, {max_page_number} pages.")
        return blocks, summary

    except Exception as e:
        logging.error(f"❌ Failed: {filename} — {e}", exc_info=True)
        return None, None


# ====================== PIPELINE EXECUTION ======================

def extract_layouts():
    """Process all PDFs in input directory and save JSONL outputs."""
    setup_environment()

    # Clean up previous runs
    for file in [LAYOUT_OUTPUT_FILE, SUMMARY_OUTPUT_FILE]:
        if os.path.exists(file):
            os.remove(file)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logging.warning("No PDF files found in input directory.")
        return

    logging.info(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        blocks, summary = process_pdf(pdf_path)

        if blocks and summary:
            with open(LAYOUT_OUTPUT_FILE, "a", encoding="utf-8") as f_layout:
                for block in blocks:
                    f_layout.write(json.dumps(block, ensure_ascii=False) + "\n")

            with open(SUMMARY_OUTPUT_FILE, "a", encoding="utf-8") as f_summary:
                f_summary.write(json.dumps(summary, ensure_ascii=False) + "\n")

    logging.info(f"✅ Layout extraction complete!\n→ Layouts: {LAYOUT_OUTPUT_FILE}\n→ Summary: {SUMMARY_OUTPUT_FILE}")
