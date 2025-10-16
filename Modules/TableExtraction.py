import os
import json
import uuid
import logging
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from unstructured.partition.pdf import partition_pdf

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    VISUAL_DIR,
    LOG_LEVEL,
    TABLES_JSONL,
    SUMMARY_JSONL,
    DPI,
)

# ==================== LOGGING ====================
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()),
                    format="%(asctime)s - %(levelname)s - %(message)s")


# ==================== PDF REPAIR ====================
def repair_pdf(input_path: str, output_path: str) -> str | None:
    """
    Attempt to repair a PDF using pikepdf.
    """
    import pikepdf
    try:
        with pikepdf.open(input_path) as pdf:
            pdf.save(output_path)
        return output_path
    except Exception as e:
        logging.error(f"Could not repair PDF {input_path}: {e}")
        return None


# ==================== LAYOUT EXTRACTION ====================
def extract_layout(pdf_path: str):
    """
    Extract layout elements from a PDF using the unstructured library.
    Falls back to a repaired version if parsing fails.
    """
    try:
        return partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )
    except Exception as e:
        logging.warning(f"Error in hi_res parsing for {pdf_path} ({e}), attempting repair...")
        fixed_pdf = os.path.splitext(pdf_path)[0] + "_fixed.pdf"
        if repaired := repair_pdf(pdf_path, fixed_pdf):
            try:
                return partition_pdf(
                    filename=repaired,
                    strategy="hi_res",
                    infer_table_structure=True,
                    extract_images_in_pdf=False
                )
            except Exception as e2:
                logging.error(f"Failed even after repair: {e2}")
        return []


# ==================== TABLE EXTRACTION ====================
def save_to_jsonl_with_strong_table_extraction(pdf_path: str, elements: list, doc_id: str):
    """
    Detect tables in PDF pages, extract cells via OCR, and save to JSONL.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_name = os.path.basename(pdf_path)
    pages_with_tables, table_records = [], []

    try:
        pages = convert_from_path(pdf_path, dpi=DPI)
    except Exception as e:
        logging.error(f"Could not read PDF pages for {pdf_name}: {e}")
        return

    for page_index, page_img in enumerate(pages):
        try:
            page_np = np.array(page_img)
            page_gray = cv2.cvtColor(page_np, cv2.COLOR_RGB2GRAY)

            # Find tables for this page
            page_tables = [
                el for el in elements
                if getattr(el, "category", "").lower() == "table"
                and getattr(el.metadata, "page_number", None) == page_index + 1
            ]

            for table_index, table in enumerate(page_tables):
                coords = getattr(table.metadata, "coordinates", None)
                if not coords or not coords.points:
                    continue

                x0, y0 = map(int, coords.points[0])
                x1, y1 = map(int, coords.points[2])
                table_crop = page_gray[y0:y1, x0:x1]

                # Detect grid lines using morphology
                thresh = cv2.adaptiveThreshold(
                    table_crop, 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, 15, 10
                )
                horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
                vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
                table_grid = cv2.addWeighted(horiz, 0.5, vert, 0.5, 0.0)

                contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cell_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 200]
                if not cell_boxes:
                    raise ValueError("No cells found in detected table")

                # Group cells into rows
                cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))
                rows, current_row = [], [cell_boxes[0]]
                for b in cell_boxes[1:]:
                    if abs(b[1] - current_row[-1][1]) < 20:
                        current_row.append(b)
                    else:
                        rows.append(sorted(current_row, key=lambda x: x[0]))
                        current_row = [b]
                rows.append(sorted(current_row, key=lambda x: x[0]))

                n_rows, n_cols = len(rows), max(len(r) for r in rows)
                cells_text = []
                for row in rows:
                    row_text = []
                    for (cx, cy, cw, ch) in row:
                        cell_crop = table_crop[cy:cy+ch, cx:cx+cw]
                        text = pytesseract.image_to_string(cell_crop, config="--psm 6").strip()
                        row_text.append(text)
                    cells_text.append(row_text)

                record = {
                    "doc_id": doc_id,
                    "filename": pdf_name,
                    "page_index": page_index,
                    "table_index": table_index,
                    "bbox": [x0, y0, x1, y1],
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "cells": cells_text
                }

                # Write individual table record
                with open(TABLES_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                table_records.append(record)
                pages_with_tables.append(page_index)

        except Exception as e:
            logging.error(f"Table extraction failed on page {page_index} of {pdf_name}: {e}")

    # Write summary record
    summary_record = {
        "doc_id": doc_id,
        "filename": pdf_name,
        "n_tables": len(table_records),
        "pages_with_tables": sorted(list(set(pages_with_tables)))
    }

    with open(SUMMARY_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

    logging.info(f"✅ Saved {len(table_records)} tables for {pdf_name}")


# ==================== MAIN PIPELINE ====================
def run_pipeline(input_dir: str = INPUT_DIR):
    """
    Run the full pipeline: extract layout, detect tables, and save JSONL outputs.
    """
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logging.warning(f"No PDF files found in {input_dir}")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        doc_id = str(uuid.uuid4())
        logging.info(f"Processing {pdf_file} ...")

        elements = extract_layout(pdf_path)
        save_to_jsonl_with_strong_table_extraction(pdf_path, elements, doc_id)

    logging.info("🎯 Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()















# import os
# import json
# import uuid
# import logging
# import cv2
# import pytesseract
# import numpy as np
# from unstructured.partition.pdf import partition_pdf
# from pdf2image import convert_from_path
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # ---------------- CONFIG ----------------
# INPUT_DIR = "input_pdfs"
# OUTPUT_DIR = "processed_jsonl"
# VISUAL_DIR = "visualizations"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(VISUAL_DIR, exist_ok=True)

# # ---------------- HELPER ----------------
# def repair_pdf(input_path, output_path):
#     import pikepdf
#     try:
#         with pikepdf.open(input_path) as pdf:
#             pdf.save(output_path)
#         return output_path
#     except Exception as e:
#         logging.error(f"Could not repair PDF {input_path}: {e}")
#         return None

# # ---------------- LAYOUT EXTRACTION ----------------
# def extract_layout(pdf_path):
#     try:
#         return partition_pdf(
#             filename=pdf_path,
#             strategy="hi_res",
#             infer_table_structure=True,
#             extract_images_in_pdf=False
#         )
#     except Exception as e:
#         logging.warning(f"Error in hi_res parsing ({e}), trying repair...")
#         fixed_pdf = os.path.splitext(pdf_path)[0] + "_fixed.pdf"
#         if repair_pdf(pdf_path, fixed_pdf):
#             try:
#                 return partition_pdf(
#                     filename=fixed_pdf,
#                     strategy="hi_res",
#                     infer_table_structure=True,
#                     extract_images_in_pdf=False
#                 )
#             except Exception as e2:
#                 logging.error(f"Failed even after repair: {e2}")
#         return []

# # # ---------------- VISUALIZATION ----------------
# # def visualize_layout(pdf_path, elements, output_dir=VISUAL_DIR, dpi=200):
# #     pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
# #     pdf_output_dir = os.path.join(output_dir, pdf_name)
# #     os.makedirs(pdf_output_dir, exist_ok=True)
# #     pages = convert_from_path(pdf_path, dpi=dpi)

# #     for page_num, page_img in enumerate(pages, start=1):
# #         fig, ax = plt.subplots(figsize=(12, 12))
# #         ax.imshow(page_img, cmap="gray")
# #         ax.axis("off")
# #         for el in elements:
# #             if el.metadata.page_number == page_num and el.metadata.coordinates:
# #                 coords = el.metadata.coordinates.points
# #                 if coords:
# #                     x0, y0 = coords[0]
# #                     x1, y1 = coords[2]
# #                     width, height = x1 - x0, y1 - y0
# #                     rect = patches.Rectangle(
# #                         (x0, y0), width, height,
# #                         linewidth=2,
# #                         edgecolor="red" if el.category == "Table" else "blue",
# #                         facecolor="none"
# #                     )
# #                     ax.add_patch(rect)
# #         out_path = os.path.join(pdf_output_dir, f"page_{page_num:03d}.png")
# #         plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
# #         plt.close(fig)
# #     logging.info(f"Saved visualization for {pdf_name}")

# # ---------------- TABLE EXTRACTION ----------------
# def save_to_jsonl_with_strong_table_extraction(pdf_path, elements, doc_id, output_dir=OUTPUT_DIR, dpi=200):
#     os.makedirs(output_dir, exist_ok=True)
#     table_jsonl_path = os.path.join(output_dir, "tables.jsonl")
#     summary_jsonl_path = os.path.join(output_dir, "tables_summary.jsonl")
#     pdf_name = os.path.basename(pdf_path)
#     pages_with_tables, table_records = [], []
#     pages = convert_from_path(pdf_path, dpi=dpi)

#     for page_index, page_img in enumerate(pages):
#         page_np = np.array(page_img)
#         page_gray = cv2.cvtColor(page_np, cv2.COLOR_RGB2GRAY)
#         page_tables = [
#             el for el in elements
#             if getattr(el, "category", "").lower() == "table"
#             and getattr(el.metadata, "page_number", None) == page_index + 1
#         ]

#         for table_index, table in enumerate(page_tables):
#             try:
#                 coords = getattr(table.metadata, "coordinates", None)
#                 if not coords or not coords.points:
#                     continue
#                 x0, y0 = map(int, coords.points[0])
#                 x1, y1 = map(int, coords.points[2])
#                 table_crop = page_gray[y0:y1, x0:x1]
#                 thresh = cv2.adaptiveThreshold(table_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
#                 horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
#                 vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
#                 horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
#                 vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
#                 table_grid = cv2.addWeighted(horiz, 0.5, vert, 0.5, 0.0)
#                 contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#                 cell_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 200]
#                 if not cell_boxes:
#                     raise ValueError("No cells found")

#                 cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))
#                 rows, current_row = [], [cell_boxes[0]]
#                 for b in cell_boxes[1:]:
#                     if abs(b[1] - current_row[-1][1]) < 20:
#                         current_row.append(b)
#                     else:
#                         rows.append(sorted(current_row, key=lambda x: x[0]))
#                         current_row = [b]
#                 rows.append(sorted(current_row, key=lambda x: x[0]))
#                 n_rows, n_cols = len(rows), max(len(r) for r in rows)
#                 cells_text = []
#                 for row in rows:
#                     row_text = []
#                     for (cx, cy, cw, ch) in row:
#                         cell_crop = table_crop[cy:cy+ch, cx:cx+cw]
#                         text = pytesseract.image_to_string(cell_crop, config="--psm 6").strip()
#                         row_text.append(text)
#                     cells_text.append(row_text)

#                 record = {
#                     "doc_id": doc_id,
#                     "filename": pdf_name,
#                     "page_index": page_index,
#                     "table_index": table_index,
#                     "bbox": [x0, y0, x1, y1],
#                     "n_rows": n_rows,
#                     "n_cols": n_cols,
#                     "cells": cells_text
#                 }
#                 with open(table_jsonl_path, "a", encoding="utf-8") as f:
#                     f.write(json.dumps(record, ensure_ascii=False) + "\n")
#                 table_records.append(record)
#                 pages_with_tables.append(page_index)
#             except Exception as e:
#                 logging.error(f"Table extraction failed on page {page_index}: {e}")
#                 continue

#     summary_record = {
#         "doc_id": doc_id,
#         "filename": pdf_name,
#         "n_tables": len(table_records),
#         "pages_with_tables": sorted(list(set(pages_with_tables)))
#     }
#     with open(summary_jsonl_path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
#     logging.info(f"✅ Saved {len(table_records)} tables for {pdf_name}")

# # ---------------- MAIN PIPELINE ----------------
# def run_pipeline():
#     pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(INPUT_DIR, pdf_file)
#         doc_id = str(uuid.uuid4())
#         logging.info(f"Processing {pdf_file} ...")
#         elements = extract_layout(pdf_path)
#         # visualize_layout(pdf_path, elements)
#         save_to_jsonl_with_strong_table_extraction(pdf_path, elements, doc_id)

#     logging.info("🎯 Pipeline completed!")

# if __name__ == "__main__":
#     run_pipeline()
