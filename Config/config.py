from pathlib import Path
import logging

# ===============================
# PROJECT STRUCTURE CONFIGURATION
# ===============================

# 🔹 Main Input Folder (update this path as needed)
DATA_DIR = Path(r"Input_Folder_Path")          # For text extraction / ingestion
INPUT_DIR = Path("input_pdfs")                 # For layout & table extraction

# 🔹 Output and Logs
OUT_DIR = Path("processed_jsonl")
VISUAL_DIR = Path("visualizations")
LOG_DIR = Path("logs")

# 🔹 Ensure directories exist
for d in [OUT_DIR, VISUAL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===============================
# MODULE-SPECIFIC FILE PATHS
# ===============================

# ---- TEXT EXTRACTION ----
TEXT_BLOCKS_FILE = OUT_DIR / "text_blocks.jsonl"
TEXT_DOCS_FILE = OUT_DIR / "text_docs.jsonl"
TEXT_EXTRACTION_LOG = LOG_DIR / "text_extraction_errors.log"
OCR_DPI = 300

# ---- TABLE EXTRACTION ----
TABLES_JSONL = OUT_DIR / "tables.jsonl"
TABLES_SUMMARY_JSONL = OUT_DIR / "tables_summary.jsonl"
TABLE_EXTRACTION_DPI = 200

# ---- METADATA INGESTION ----
OUTPUT_METADATA = OUT_DIR / "metadata.jsonl"
INGESTION_LOG = LOG_DIR / "ingestion_errors.log"

# ---- LAYOUT EXTRACTION ----
LAYOUT_OUTPUT_FILE = OUT_DIR / "layout.jsonl"
LAYOUT_SUMMARY_FILE = OUT_DIR / "layout_summary.jsonl"
LAYOUT_ERROR_LOG = LOG_DIR / "layout_extraction_errors.log"

# ===============================
# LOGGING CONFIGURATION
# ===============================

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# ===============================
# UTILITY SETTINGS
# ===============================

# Default DPI values for OCR and Table Extraction
DEFAULT_OCR_DPI = 300
DEFAULT_TABLE_DPI = 200

# Default Visualization Settings
SHOW_VISUALS = True

# ===============================
# SUMMARY PRINT
# ===============================
if __name__ == "__main__":
    print("✅ Configuration loaded successfully!")
    print(f"Input folder: {DATA_DIR}")
    print(f"Output folder: {OUT_DIR}")
    print(f"Log folder: {LOG_DIR}")
