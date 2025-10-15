# # Install system dependency for PDF processing
# ! apt-get install poppler-utils

# # Install the machine learning model library with the correct syntax
# ! pip install "detectron2@git+[https://github.com/facebookresearch/detectron2.git@v0.6](https://github.com/facebookresearch/detect…

# # Install unstructured and its dependencies
# ! pip install "unstructured[local-inference]"

# # Pin the specific Pillow version required by this version of unstructured
# ! pip install Pillow==9.5.0

# ! apt-get update
# ! apt-get install -y poppler-utils

import os

os.makedirs("input_pdfs", exist_ok=True)
os.makedirs("processed_jsonl", exist_ok=True)
os.makedirs("logs", exist_ok=True)



import json
import logging
import uuid
from collections import Counter
from unstructured.partition.pdf import partition_pdf

# --- CONFIGURATION (for Google Colab) ---

# Define the paths for your project directories inside the Colab environment
INPUT_DIR = "input_pdfs"
OUTPUT_DIR = "processed_jsonl1"
LOG_DIR = "logs"

# Define the names for your output files
LAYOUT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "layout.jsonl")
SUMMARY_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "layout_summary.jsonl")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "layout_extraction_errors.log")

# --- SETUP ---

def setup_environment():
    """Create necessary directories and configure logging."""
    # Directories are created in a separate cell in Colab, but this ensures they exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Configure logging to write errors to a file
    # We need to clear previous handlers in Colab to avoid duplicate logs on re-runs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(ERROR_LOG_FILE),
            logging.StreamHandler() # Also print logs to the console
        ]
    )

# --- HELPER FUNCTIONS ---

def map_element_type(element_category):
    """Maps unstructured.io element types to the user-defined types."""
    type_map = {
        "Title": "title",
        "NarrativeText": "paragraph",
        "UncategorizedText": "paragraph",
        "ListItem": "paragraph",
        "Table": "table",
        "Image": "image",
        "FigureCaption": "image",
        "Footer": "footer",
        "Header": "footer"
    }
    return type_map.get(element_category, "unknown")

# --- CORE PROCESSING ---

def process_pdf(pdf_path):
    """
    Processes a single PDF file to extract layout elements.
    """
    filename = os.path.basename(pdf_path)
    logging.info(f"Starting processing for: {filename}")

    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )

        doc_id = str(uuid.uuid4())
        blocks = []
        stats_counter = Counter()
        max_page_number = 0

        for i, el in enumerate(elements):
            block_type = map_element_type(el.category)
            stats_counter[block_type] += 1

            coords = el.metadata.coordinates.points if el.metadata.coordinates else None
            page_number = el.metadata.page_number if el.metadata.page_number else 0

            if page_number > max_page_number:
                max_page_number = page_number

            block_data = {
                "doc_id": doc_id,
                "filename": filename,
                "page_index": page_number - 1 if page_number else 0,
                "block_index": i,
                "type": block_type,
                "bbox": [coords[0][0], coords[0][1], coords[2][0], coords[2][1]] if coords else None,
                "text": el.text
            }
            blocks.append(block_data)

        summary = {
            "doc_id": doc_id,
            "filename": filename,
            "n_pages": max_page_number,
            "stats": dict(stats_counter)
        }

        logging.info(f"Successfully processed {filename}. Found {len(blocks)} blocks across {max_page_number} pages.")
        return blocks, summary

    except Exception as e:
        logging.error(f"Failed to process {filename}. Error: {e}", exc_info=True)
        return None, None

# --- MAIN EXECUTION ---

def main():
    """
    Main function to find and process all PDFs in the input directory.
    """
    setup_environment()

    if os.path.exists(LAYOUT_OUTPUT_FILE):
        os.remove(LAYOUT_OUTPUT_FILE)
    if os.path.exists(SUMMARY_OUTPUT_FILE):
        os.remove(SUMMARY_OUTPUT_FILE)

    logging.info(f"Scanning for PDF files in '{INPUT_DIR}'...")
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]

    if not pdf_files:
        logging.warning("No PDF files found in the 'input_pdfs' folder. Please upload your files.")
        return

    logging.info(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        blocks, summary = process_pdf(pdf_path)

        if blocks and summary:
            with open(LAYOUT_OUTPUT_FILE, 'a', encoding=
                      'utf-8') as f_layout:
                for block in blocks:
                    f_layout.write(json.dumps(block) + '\n')

            with open(SUMMARY_OUTPUT_FILE, 'a', encoding='utf-8') as f_summary:
                f_summary.write(json.dumps(summary) + '\n')

    logging.info("All files processed. Check the 'processed_jsonl' and 'logs' directories for output.")

# --- RUN THE SCRIPT ---
# In Colab, we call the main function directly after defining it.
main()