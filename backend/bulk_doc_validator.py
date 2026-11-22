"""
bulk_doc_validator.py
=====================
Bulk (Group) Document Validation — web-facing module.

This file contains the bulk-processing functions from the authoritative
uploaded Python source (Bulk Tamil Nadu DGE Marksheet Extraction + QR
Verification).  The core OCR, QR verification, comparison, and certificate-
detection logic lives in doc_validator.py and is imported here unchanged.

Allowed adaptations (web integration only):
  - Tkinter file-chooser calls replaced by the path passed from Flask.
  - Terminal input() / print() output is preserved for server-side audit logs
    but results are also captured and returned as structured dicts.
  - main() is not executed on import (guarded by __name__ == "__main__").
  - download_drive_pdf() uses a session-specific folder so concurrent web
    requests cannot overwrite each other's files.
  - save_bulk_report() appends a session_id suffix for the same reason.

No OCR algorithm, QR algorithm, verdict rule, threshold, regex, alias, or
comparison condition has been changed.
"""

import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Import all core functions from doc_validator (unchanged originals).
# ---------------------------------------------------------------------------
from doc_validator import (
    # OCR / extraction
    configure_tesseract,
    TESSERACT_PATH,
    detect_certificate_type,
    extract_marksheet_rows,
    display_extracted_rows,
    # QR + comparison
    scan_qr_and_download,
    compare_uploaded_with_entered,
    compare_uploaded_with_official,
    subject_key,
    unique_path,
    find_edge,
    # Folder constants
    BASE_DIR,
    UPLOAD_FOLDER,
    VERIFIED_FOLDER,
)

# ---------------------------------------------------------------------------
# Folder constants (bulk-specific)
# ---------------------------------------------------------------------------
REPORT_FOLDER        = BASE_DIR / "Bulk_Verification_Reports"
DRIVE_DOWNLOAD_FOLDER = BASE_DIR / "Drive_Marksheets"
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
DRIVE_DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPREADSHEET READING  (verbatim from uploaded source)
# ============================================================

def read_student_sheet(path):
    """Read CSV/Excel and normalize column names without changing cell data."""
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        # utf-8-sig handles CSV files saved by Excel with a BOM.
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="cp1252")
    elif suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path)
    else:
        raise ValueError("Please select a CSV, XLSX, or XLS file.")

    frame.columns = [str(col).strip() for col in frame.columns]

    # Find the Drive-link column even if the header has extra spaces.
    drive_column = next(
        (col for col in frame.columns
         if re.sub(r"[^A-Z]", "", col.upper()) in
         {"MARKSHEET", "MARKSHEETLINK", "DRIVELINK", "PDFLINK"}),
        None,
    )
    name_column = next(
        (col for col in frame.columns
         if re.sub(r"[^A-Z]", "", col.upper()) in
         {"NAME", "STUDENTNAME", "CANDIDATENAME"}),
        None,
    )

    if not name_column:
        raise ValueError("The spreadsheet must contain a Name column.")
    if not drive_column:
        raise ValueError("The spreadsheet must contain a MarkSheet / Drive link column.")

    # Pandas renames repeated headers: Mark, Mark.1, Mark.2, ...
    # Pair the Mark columns in their original left-to-right order with Subject 1,
    # Subject 2, etc. This supports the exact uploaded CSV layout.
    subject_headers = []
    for col in frame.columns:
        match = re.fullmatch(r"\s*Subject\s*(\d+)\s*", str(col), re.IGNORECASE)
        if match:
            subject_headers.append((int(match.group(1)), col))
    subject_headers.sort(key=lambda item: item[0])

    mark_headers = [
        col for col in frame.columns
        if re.fullmatch(r"\s*Mark(?:\.\d+)?\s*", str(col), re.IGNORECASE)
    ]

    if len(mark_headers) < len(subject_headers):
        raise ValueError(
            f"Found {len(subject_headers)} Subject columns but only "
            f"{len(mark_headers)} Mark columns. Each subject needs a mark column."
        )

    subject_columns = [
        (number, subject_col, mark_headers[position])
        for position, (number, subject_col) in enumerate(subject_headers)
    ]

    if not subject_columns:
        raise ValueError(
            "No Subject columns were found. Expected headers such as "
            "'Subject 1', 'Subject 2', etc."
        )

    return frame, name_column, drive_column, subject_columns


# ============================================================
# CELL HELPERS  (verbatim from uploaded source)
# ============================================================

def clean_cell(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_entered_rows(student_record, subject_columns):
    entered = []
    seen = set()

    for _, subject_col, mark_col in subject_columns:
        subject = clean_cell(student_record.get(subject_col, ""))
        raw_mark = clean_cell(student_record.get(mark_col, ""))

        if not subject or subject == "-":
            continue
        if not raw_mark or raw_mark == "-":
            continue

        # Accept spreadsheet numeric values such as 97 or 97.0.
        try:
            numeric = float(raw_mark)
            mark = int(numeric)
            if numeric != mark or not 0 <= mark <= 100:
                raise ValueError
        except (TypeError, ValueError):
            print(
                f"WARNING: Invalid entered mark for {subject}: {raw_mark}. "
                "This subject will be skipped."
            )
            continue

        key = subject_key(subject)
        if key in seen:
            print(f"WARNING: Duplicate subject '{subject}' in spreadsheet; skipped.")
            continue

        entered.append({"subject": subject, "mark": mark})
        seen.add(key)

    return entered


# ============================================================
# GOOGLE DRIVE DOWNLOAD  (verbatim + session-folder adaptation)
# ============================================================

def drive_file_id(url):
    """Extract a Google Drive file ID for caching and diagnostics."""
    if not url:
        return None
    patterns = (
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"/uc\?id=([A-Za-z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_drive_pdf(url, student_name, cache, folder=None):
    """
    Download a shared Google Drive PDF.

    folder : optional Path; when given, files are stored there instead of
             DRIVE_DOWNLOAD_FOLDER so concurrent requests stay isolated.
    """
    if not url:
        raise ValueError("Google Drive marksheet link is empty.")

    target_folder = folder if folder else DRIVE_DOWNLOAD_FOLDER
    file_id = drive_file_id(url)
    cache_key = (file_id or url) + str(target_folder)

    if cache_key in cache and Path(cache[cache_key]).is_file():
        return Path(cache[cache_key])

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Missing gdown. Install it using: pip install gdown"
        ) from exc

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", student_name).strip("_")
    destination = unique_path(
        target_folder,
        safe_name or "student_marksheet",
        ".pdf",
    )

    # Older gdown releases do not support the ``fuzzy`` keyword.
    try:
        downloaded = gdown.download(
            url=url,
            output=str(destination),
            quiet=True,
            fuzzy=True,
        )
    except TypeError as exc:
        if "fuzzy" not in str(exc):
            raise
        downloaded = gdown.download(
            url=url,
            output=str(destination),
            quiet=True,
        )

    if not downloaded or not destination.is_file():
        raise RuntimeError(
            "Google Drive download failed. Check the link and its sharing permission."
        )

    # Reject HTML confirmation/error pages saved with a .pdf suffix.
    with open(destination, "rb") as handle:
        signature = handle.read(5)
    if signature != b"%PDF-":
        try:
            destination.unlink()
        except OSError:
            pass
        raise RuntimeError(
            "The Drive link did not return a PDF. Check that it points directly "
            "to a PDF and that access is permitted."
        )

    cache[cache_key] = str(destination)
    return destination


# ============================================================
# COMPARISON ROW BUILDERS  (verbatim from uploaded source)
# ============================================================

def entered_comparison_rows(uploaded_rows, entered_rows):
    """Build report details using the same subject normalization/fuzzy threshold."""
    from difflib import SequenceMatcher

    uploaded_by_key = {subject_key(r["subject"]): r for r in uploaded_rows}
    entered_by_key  = {subject_key(r["subject"]): r for r in entered_rows}

    unmatched_uploaded = set(uploaded_by_key)
    unmatched_entered  = set(entered_by_key)
    matches = []

    for key in list(unmatched_entered):
        if key in unmatched_uploaded:
            matches.append((key, key))
            unmatched_uploaded.remove(key)
            unmatched_entered.remove(key)

    for entered_key in list(unmatched_entered):
        candidates = sorted(
            [
                (SequenceMatcher(None, entered_key, uploaded_key).ratio(), uploaded_key)
                for uploaded_key in unmatched_uploaded
            ],
            reverse=True,
        )
        if candidates and candidates[0][0] >= 0.88:
            best_ratio, uploaded_key = candidates[0]
            unique_best = (
                len(candidates) == 1
                or best_ratio - candidates[1][0] >= 0.06
            )
            if unique_best:
                matches.append((uploaded_key, entered_key))
                unmatched_uploaded.remove(uploaded_key)
                unmatched_entered.remove(entered_key)

    details = []
    mismatch_count = 0

    for uploaded_key, entered_key in matches:
        uploaded = uploaded_by_key[uploaded_key]
        entered  = entered_by_key[entered_key]
        same = int(uploaded["mark"]) == int(entered["mark"])
        if not same:
            mismatch_count += 1
        details.append({
            "Subject":            uploaded["subject"],
            "Uploaded Mark":      uploaded["mark"],
            "Entered Mark":       entered["mark"],
            "Entered Comparison": "SAME" if same else "MISMATCH",
        })

    for key in sorted(unmatched_uploaded):
        details.append({
            "Subject":            uploaded_by_key[key]["subject"],
            "Uploaded Mark":      uploaded_by_key[key]["mark"],
            "Entered Mark":       "",
            "Entered Comparison": "NOT ENTERED",
        })

    for key in sorted(unmatched_entered):
        details.append({
            "Subject":            entered_by_key[key]["subject"],
            "Uploaded Mark":      "",
            "Entered Mark":       entered_by_key[key]["mark"],
            "Entered Comparison": "NOT FOUND IN UPLOADED",
        })

    passed = (
        bool(matches)
        and mismatch_count == 0
        and not unmatched_uploaded
        and not unmatched_entered
    )
    return details, ("ALL SAME" if passed else "MISMATCH / INCOMPLETE")


def official_comparison_rows(uploaded_rows, official_rows):
    """Build the subject-level details for uploaded PDF vs official DGE PDF."""
    if not uploaded_rows or not official_rows:
        return []

    uploaded_by_key = {subject_key(r["subject"]): r for r in uploaded_rows}
    official_by_key = {subject_key(r["subject"]): r for r in official_rows}

    details = []
    for key in sorted(set(uploaded_by_key) | set(official_by_key)):
        uploaded = uploaded_by_key.get(key)
        official = official_by_key.get(key)

        if uploaded is None:
            details.append({
                "Subject":           official["subject"],
                "Uploaded Mark":     "NOT FOUND",
                "Official DGE Mark": official["mark"],
                "Official Comparison": "SUBJECT MISSING IN UPLOADED",
            })
        elif official is None:
            details.append({
                "Subject":           uploaded["subject"],
                "Uploaded Mark":     uploaded["mark"],
                "Official DGE Mark": "NOT FOUND",
                "Official Comparison": "SUBJECT MISSING IN OFFICIAL",
            })
        else:
            same = int(uploaded["mark"]) == int(official["mark"])
            details.append({
                "Subject":           uploaded["subject"],
                "Uploaded Mark":     uploaded["mark"],
                "Official DGE Mark": official["mark"],
                "Official Comparison": "SAME" if same else "MISMATCH",
            })

    return details


# ============================================================
# SINGLE-STUDENT PROCESSOR  (verbatim from uploaded source,
#   with session_folder parameter as minimum web adaptation)
# ============================================================

def process_one_student(student_record, row_number, name_column,
                        drive_column, subject_columns, drive_cache,
                        session_folder=None):
    """
    Process one student row from the spreadsheet.

    Returns (result_dict, detail_rows_list).
    The result_dict mirrors the original bulk workflow output exactly.
    session_folder is the only web-integration addition: when set, Drive
    downloads are stored there to prevent cross-request file collisions.
    """
    student_name = clean_cell(student_record.get(name_column, "")) or f"Row {row_number}"
    drive_url    = clean_cell(student_record.get(drive_column, ""))

    result = {
        "Student Name":        student_name,
        "Drive Link":          drive_url,
        "Certificate Type":    "",
        "Uploaded vs Entered": "NOT CHECKED",
        "Originality":         "UNVERIFIED",
        "Uploaded Total":      "",
        "Official Total":      "",
        "Status / Error":      "",
    }
    detail_rows    = []
    entered_details = []

    print("\n" + "#" * 88)
    print(f"STUDENT {row_number}: {student_name}")
    print("#" * 88)

    try:
        entered_rows = build_entered_rows(student_record, subject_columns)
        if not entered_rows:
            raise ValueError("No valid subject/mark entries found in this spreadsheet row.")

        print("\nSTEP 1 — Downloading marksheet from Google Drive")
        dl_folder    = session_folder if session_folder else DRIVE_DOWNLOAD_FOLDER
        uploaded_path = download_drive_pdf(drive_url, student_name, drive_cache, dl_folder)
        print("Downloaded PDF:", uploaded_path)

        print("\nSTEP 2 — Detecting certificate type")
        certificate_type = detect_certificate_type(str(uploaded_path))
        result["Certificate Type"] = certificate_type
        print("Certificate type:", certificate_type)

        if certificate_type == "UNKNOWN":
            raise ValueError("Could not reliably identify the certificate as 10th or 12th.")

        print("\nSTEP 3 — Extracting uploaded marksheet")
        uploaded_rows, uploaded_total = extract_marksheet_rows(str(uploaded_path))
        result["Uploaded Total"] = (
            uploaded_total if uploaded_total is not None else "Not identified"
        )
        display_extracted_rows(
            uploaded_rows,
            f"{student_name} — UPLOADED MARKSHEET EXTRACTION",
        )

        if not uploaded_rows:
            raise ValueError("No reliable subject marks were extracted from uploaded PDF.")

        print("\nSTEP 4 — Uploaded marksheet vs spreadsheet-entered marks")
        compare_uploaded_with_entered(uploaded_rows, entered_rows)
        entered_details, entered_status = entered_comparison_rows(
            uploaded_rows, entered_rows
        )
        result["Uploaded vs Entered"] = entered_status

        print("\nSTEP 5 — Scanning QR and retrieving official DGE verification")
        qr_url, official_pdf_path = scan_qr_and_download(str(uploaded_path))

        if not qr_url:
            result["Status / Error"] = "QR code not detected in uploaded marksheet."
            result["Originality"]    = "UNVERIFIED"
        elif not official_pdf_path:
            result["Status / Error"] = (
                "QR found, but official DGE verification PDF could not be downloaded."
            )
            result["Originality"] = "UNVERIFIED"
        else:
            print("\nSTEP 6 — Extracting official DGE marks")
            if certificate_type == "12TH":
                official_rows, official_total = extract_marksheet_rows(
                    str(official_pdf_path), second_year_only=True
                )
            else:
                official_rows, official_total = extract_marksheet_rows(
                    str(official_pdf_path), second_year_only=False
                )

            result["Official Total"] = (
                official_total if official_total is not None else "Not identified"
            )
            display_extracted_rows(
                official_rows,
                f"{student_name} — OFFICIAL DGE QR VERIFICATION",
            )

            print("\nSTEP 7 — Uploaded marksheet vs official DGE record")
            originality = compare_uploaded_with_official(uploaded_rows, official_rows)
            result["Originality"] = originality

            if not official_rows:
                result["Status / Error"] = "Official PDF had no reliable subject marks."
            elif originality == "ORIGINAL":
                result["Status / Error"] = "Uploaded marks match official DGE record."
            elif originality == "POSSIBLE_FORGERY":
                result["Status / Error"] = (
                    "Mark mismatch detected; manual authority verification required."
                )
            else:
                result["Status / Error"] = (
                    "Official comparison incomplete; manual review required."
                )

            official_dets = official_comparison_rows(uploaded_rows, official_rows)
            for item in official_dets:
                detail_rows.append({
                    "Student Name":    student_name,
                    "Comparison Type": "UPLOADED VS OFFICIAL DGE",
                    **item,
                })

        for item in entered_details:
            detail_rows.append({
                "Student Name":    student_name,
                "Comparison Type": "UPLOADED VS ENTERED",
                **item,
            })

    except Exception as exc:
        result["Originality"]    = "UNVERIFIED"
        result["Status / Error"] = str(exc)
        print(f"\nSTUDENT PROCESSING ERROR: {exc}")

    print("\nSTUDENT RESULT")
    print("Name:",                result["Student Name"])
    print("Uploaded vs Entered:", result["Uploaded vs Entered"])
    print("Originality:",         result["Originality"])
    print("Status:",              result["Status / Error"])

    return result, detail_rows


# ============================================================
# EXCEL REPORT  (verbatim from uploaded source, session_id suffix added)
# ============================================================

def save_bulk_report(summary_rows, detail_rows, session_id=None):
    """
    Generate the Excel bulk verification report.

    Sheet structure is identical to the original save_bulk_report():
      - Summary sheet
      - Subject Details sheet
    session_id is appended to the filename for uniqueness (web adaptation).
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix     = f"_{session_id}" if session_id else ""
    output_path = REPORT_FOLDER / f"Bulk_Marksheet_Verification_{timestamp}{suffix}.xlsx"

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(detail_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if details_df.empty:
            pd.DataFrame(
                columns=[
                    "Student Name", "Comparison Type", "Subject",
                    "Uploaded Mark", "Entered Mark", "Official DGE Mark",
                    "Entered Comparison", "Official Comparison",
                ]
            ).to_excel(writer, sheet_name="Subject Details", index=False)
        else:
            details_df.to_excel(writer, sheet_name="Subject Details", index=False)

        # Make the report easier to read in Excel.
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = max(
                    (len(str(cell.value)) if cell.value is not None else 0)
                    for cell in column_cells
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(max(max_length + 2, 12), 55)

    return output_path


# ============================================================
# WEB ENTRY POINT
# (minimum adaptation — calls original bulk functions above)
# ============================================================

def run_bulk_doc_verification(spreadsheet_path: str, session_id: str = None) -> dict:
    """
    Web-facing entry point for group (bulk) document verification.

    Parameters
    ----------
    spreadsheet_path : Absolute path to the saved CSV/XLSX/XLS file.
    session_id       : UUID string for this web request (isolates downloads/report).

    Returns
    -------
    dict with:
      summary_rows      : list of per-student result dicts
      detail_rows       : list of subject-level comparison dicts
      report_path       : str absolute path to the generated Excel file (or None)
      report_filename   : str basename of the report file (or None)
      total             : int
      original          : int
      possible_forgery  : int
      unverified        : int
      error             : str (only present on fatal setup errors)
    """
    if not TESSERACT_PATH:
        return {
            "error": (
                "Tesseract OCR not found. "
                r"Install it at C:\Program Files\Tesseract-OCR\ "
                "or set the TESSERACT_CMD environment variable."
            )
        }

    try:
        frame, name_column, drive_column, subject_columns = read_student_sheet(
            spreadsheet_path
        )
    except Exception as exc:
        return {"error": f"Could not read spreadsheet: {exc}"}

    if frame.empty:
        return {"error": "The spreadsheet contains no student rows."}

    # Isolated session folder for concurrent web requests.
    if session_id:
        session_folder = DRIVE_DOWNLOAD_FOLDER / f"session_{session_id}"
        session_folder.mkdir(parents=True, exist_ok=True)
    else:
        session_folder = DRIVE_DOWNLOAD_FOLDER

    summary_rows   = []
    all_detail_rows = []
    drive_cache    = {}

    for index, (_, student_record) in enumerate(frame.iterrows(), start=1):
        summary, details = process_one_student(
            student_record=student_record,
            row_number=index,
            name_column=name_column,
            drive_column=drive_column,
            subject_columns=subject_columns,
            drive_cache=drive_cache,
            session_folder=session_folder,
        )
        summary_rows.append(summary)
        all_detail_rows.extend(details)

    report_path = None
    try:
        report_path = save_bulk_report(summary_rows, all_detail_rows, session_id=session_id)
    except Exception as exc:
        print(f"Could not create Excel report: {exc}")

    original_count        = sum(1 for r in summary_rows if r["Originality"] == "ORIGINAL")
    possible_forgery_count = sum(1 for r in summary_rows if r["Originality"] == "POSSIBLE_FORGERY")
    unverified_count       = sum(1 for r in summary_rows if r["Originality"] == "UNVERIFIED")

    return {
        "summary_rows":    summary_rows,
        "detail_rows":     all_detail_rows,
        "report_path":     str(report_path) if report_path else None,
        "report_filename": report_path.name if report_path else None,
        "total":           len(summary_rows),
        "original":        original_count,
        "possible_forgery": possible_forgery_count,
        "unverified":      unverified_count,
    }


# ============================================================
# Desktop main() — NOT executed on web import
# ============================================================

def _desktop_main():
    """Original desktop bulk workflow (choose_spreadsheet → process → report)."""
    import tkinter as tk
    from tkinter import filedialog

    print("=" * 88)
    print("BULK TAMIL NADU DGE MARKSHEET EXTRACTION + QR VERIFICATION")
    print("=" * 88)

    if not TESSERACT_PATH:
        print("Tesseract OCR was not found.")
        print(r"Set TESSERACT_CMD or install it at C:\Program Files\Tesseract-OCR\tesseract.exe")
        return

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    spreadsheet_path = filedialog.askopenfilename(
        title="Select student marks CSV or Excel file",
        filetypes=[
            ("Spreadsheet files", "*.csv *.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
        ],
    )
    root.destroy()

    if not spreadsheet_path or not os.path.isfile(spreadsheet_path):
        print("No valid spreadsheet selected.")
        return

    result = run_bulk_doc_verification(spreadsheet_path)
    if "error" in result:
        print("ERROR:", result["error"])
        return

    print("\n" + "=" * 88)
    print("BULK VERIFICATION COMPLETED")
    print("=" * 88)
    print("Total students       :", result["total"])
    print("Original             :", result["original"])
    print("Possible forgery     :", result["possible_forgery"])
    print("Unverified           :", result["unverified"])
    if result["report_path"]:
        print("\nExcel report saved at:")
        print(result["report_path"])
    print("\nPROCESS COMPLETED")
    print("=" * 88)


if __name__ == "__main__":
    _desktop_main()
