import os
import hashlib
import json
from pathlib import Path
from PyPDF2 import PdfReader

# ========== CONFIG ==========
# ROOT_DIR = r"C:\Users\optimal\Documents\Classification\data\Test_input"  # Main folder containing PDFs inside subfolders
ROOT_DIR = r"Input_Folder_path"
OUTPUT_METADATA = r"processed_jsonl\metadata.jsonl"
ERROR_LOG = r"Tasks\logs\ingestion_errors.log"

# Ensure output folders exist
os.makedirs(os.path.dirname(OUTPUT_METADATA), exist_ok=True)
os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)



def compute_hash(file_path):
    """Compute SHA1 hash of file"""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()



def process_pdf(file_path):
    """Validate and extract metadata from PDF"""
    try:
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)  # If file invalid, this will raise error
        return num_pages
    except Exception as e:
        return f"INVALID: {str(e)}"


def main():
    with open(OUTPUT_METADATA, "w", encoding="utf-8") as meta_file, \
         open(ERROR_LOG, "w", encoding="utf-8") as error_file:

        for root, dirs, files in os.walk(ROOT_DIR):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(root, filename)
                    filesize = os.path.getsize(file_path)
                    result = process_pdf(file_path)

                    if isinstance(result, int):  # Valid PDF
                        metadata = {
                            "filename": os.path.relpath(file_path, ROOT_DIR),
                            "n_pages": result,
                            "filesize": filesize,
                            "hash": compute_hash(file_path)
                        }
                        meta_file.write(json.dumps(metadata) + "\n")
                    else:
                        error_file.write(f"{file_path} --> {result}\n")

if __name__ == "__main__":
    main()
