import os
import json
import hashlib
import logging
from pathlib import Path
from PyPDF2 import PdfReader
from ..config import ROOT_DIR, OUTPUT_METADATA, ERROR_LOG


def compute_hash(file_path: Path) -> str:
    """Compute SHA1 hash of a file for uniqueness checking."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()


def process_pdf(file_path: Path):
    """Validate and extract metadata (page count) from a PDF."""
    try:
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        return num_pages
    except Exception as e:
        logging.error(f"Invalid PDF {file_path}: {e}")
        return None


def ingest_metadata():
    """Walk through input directory, extract metadata, and save JSONL output."""
    logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR, format="%(asctime)s - %(message)s")

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as meta_file:
        for root, _, files in os.walk(ROOT_DIR):
            for filename in files:
                if not filename.lower().endswith(".pdf"):
                    continue

                file_path = Path(root) / filename
                filesize = os.path.getsize(file_path)
                n_pages = process_pdf(file_path)

                if n_pages is not None:
                    metadata = {
                        "filename": str(Path(file_path).relative_to(ROOT_DIR)),
                        "n_pages": n_pages,
                        "filesize": filesize,
                        "hash": compute_hash(file_path)
                    }
                    meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                else:
                    logging.error(f"Failed to process: {file_path}")

    print(f"✅ Metadata ingestion complete!\n→ Output: {OUTPUT_METADATA}\n→ Errors: {ERROR_LOG}")
