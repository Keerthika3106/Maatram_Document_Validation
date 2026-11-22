"""
Maatram Foundation - Backend API
=================================
Endpoints:
  POST /api/photo/validate                   Individual photo validation
  POST /api/photo/group/validate             Group photo validation (spreadsheet)
  GET  /api/photo/group/results/<filename>   Download validated spreadsheet
  POST /api/document/individual/verify       Individual document (marksheet) verification
  POST /api/document/group/verify            Group document (bulk marksheet) verification
  GET  /api/document/group/report/<filename> Download bulk verification Excel report
  GET  /api/health                           Health check
"""

import os
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from photo_validator import validate_photo
from bulk_photo_validator import process_spreadsheet, MODEL_PATH, OUTPUT_FOLDER
from doc_validator import run_individual_doc_verification
from bulk_doc_validator import run_bulk_doc_verification, REPORT_FOLDER

app = Flask(__name__)

# Allow requests from the Vite dev server and common localhost origins
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://localhost:3000",
]}})

ALLOWED_IMAGE_EXTENSIONS      = {"jpg", "jpeg", "png"}
ALLOWED_SPREADSHEET_EXTENSIONS = {"csv", "xlsx", "xlsm"}


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# ---------------------------------------------------------------------------
# POST /api/photo/validate  (Individual)
# ---------------------------------------------------------------------------
@app.route("/api/photo/validate", methods=["POST"])
def photo_validate():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send the image as 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if _ext(file.filename) not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format '{file.filename}'. Only JPG, JPEG and PNG are accepted."
        }), 415

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        result = validate_photo(file_bytes, file.filename)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Validation error: {str(exc)}"}), 500

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# POST /api/photo/group/validate  (Group)
# ---------------------------------------------------------------------------
@app.route("/api/photo/group/validate", methods=["POST"])
def group_photo_validate():

    # 1. File presence
    if "file" not in request.files:
        return jsonify({"error": "No spreadsheet uploaded. Send the file as 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    # 2. Format check
    ext = _ext(file.filename)
    if ext not in ALLOWED_SPREADSHEET_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format '{file.filename}'. Only CSV, XLSX and XLSM are accepted."
        }), 415

    # 3. YuNet model check (fail fast before long processing)
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "YuNet model not found. Place face_detection_yunet_2023mar.onnx in the backend/ folder."
        }), 503

    # 4. Save spreadsheet to a named temp file (process_spreadsheet needs a real path)
    suffix = f".{ext}"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    try:
        file.save(tmp_path)
        tmp.close()

        # 5. Run the original process_spreadsheet() from bulk_photo_validator.py
        try:
            output_path, passed, failed = process_spreadsheet(tmp_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            return jsonify({"error": f"Processing error: {str(exc)}"}), 500

    finally:
        # Clean up input temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # 6. Build per-row results for the frontend table by reading the output file
    rows = _read_output_rows(output_path, ext)

    total  = passed + failed
    output_filename = Path(output_path).name

    # 7. Schedule cleanup of the output file after 10 minutes
    _schedule_cleanup(output_path, delay=600)

    return jsonify({
        "passed":          passed,
        "failed":          failed,
        "total":           total,
        "rows":            rows,
        "output_filename": output_filename,
    }), 200


def _read_output_rows(output_path: str, ext: str) -> list:
    """
    Read the validated output file produced by process_spreadsheet() and
    return a list of row dicts for the frontend table.
    Keys: id, name, photo_link, status, details
    """
    import csv as csv_mod
    from openpyxl import load_workbook as lw

    rows = []

    if ext == "csv":
        with open(output_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv_mod.DictReader(f)
            for i, row in enumerate(reader, start=1):
                rows.append(_make_table_row(i, row))

    else:  # xlsx / xlsm
        wb = lw(output_path, read_only=True)
        ws = wb.active
        headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        for i, excel_row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
            row_dict = {str(h): str(v) if v is not None else "" for h, v in zip(headers, excel_row)}
            rows.append(_make_table_row(i, row_dict))
        wb.close()

    return rows


def _make_table_row(index: int, row: dict) -> dict:
    """Normalize a spreadsheet row into the shape the frontend table expects."""
    import re as re_mod

    def _find(d: dict, keys: list) -> str:
        norm = {re_mod.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in d.items()}
        for k in keys:
            key = re_mod.sub(r"[^a-z0-9]", "", k.lower())
            if key in norm:
                return str(norm[key]) if norm[key] else ""
        return ""

    name        = _find(row, ["Name", "Full Name", "Candidate Name", "Student Name"])
    photo_link  = _find(row, ["Photo Link", "Photo URL", "Image Link", "Image URL", "Photo", "Link"])
    status_raw  = _find(row, ["Validation Status"])
    details     = _find(row, ["Failure Reason"])

    status = "pass" if status_raw.upper() == "PASS" else "fail"
    if status == "pass" and not details:
        details = "All validation checks passed."

    return {
        "id":         str(index).zfill(3),
        "name":       name or f"Row {index}",
        "photo_link": photo_link,
        "status":     status,
        "details":    details,
    }


def _schedule_cleanup(path: str, delay: int = 600):
    """Delete the output file after `delay` seconds in a background thread."""
    def _delete():
        time.sleep(delay)
        try:
            os.remove(path)
        except OSError:
            pass
    t = threading.Thread(target=_delete, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# GET /api/photo/group/results/<filename>  (Download validated spreadsheet)
# ---------------------------------------------------------------------------
@app.route("/api/photo/group/results/<filename>", methods=["GET"])
def download_group_results(filename: str):
    # Security: only serve files from OUTPUT_FOLDER, no path traversal
    safe_name = Path(filename).name
    file_path = os.path.join(OUTPUT_FOLDER, safe_name)

    if not os.path.isfile(file_path):
        return jsonify({"error": "Result file not found or already cleaned up."}), 404

    ext = _ext(safe_name)
    mime_map = {
        "csv":  "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsm": "application/vnd.ms-excel.sheet.macroenabled.12",
    }
    mime = mime_map.get(ext, "application/octet-stream")

    return send_file(
        file_path,
        mimetype=mime,
        as_attachment=True,
        download_name=safe_name,
    )


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    model_present = os.path.exists(MODEL_PATH)
    return jsonify({
        "status":      "ok",
        "yunet_model": "present" if model_present else "MISSING",
    })



# ---------------------------------------------------------------------------
# POST /api/document/individual/verify  (Individual Document / Marksheet)
# ---------------------------------------------------------------------------
ALLOWED_DOC_EXTENSIONS = {"pdf"}
MAX_DOC_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.route("/api/document/individual/verify", methods=["POST"])
def individual_doc_verify():
    # 1. File presence
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send the PDF as 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    # 2. Format check
    ext = _ext(file.filename)
    if ext not in ALLOWED_DOC_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format '{file.filename}'. Only PDF files are accepted."
        }), 415

    # 3. Read bytes and size check
    pdf_bytes = file.read()
    if not pdf_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400
    if len(pdf_bytes) > MAX_DOC_SIZE_BYTES:
        return jsonify({"error": "File is too large. Maximum allowed size is 10 MB."}), 413

    # 4. Parse manually entered subjects from form data
    #    Frontend sends: entered_subjects as a JSON string
    import json as _json
    entered_subjects = []
    raw_entered = request.form.get("entered_subjects", "")
    if raw_entered:
        try:
            parsed = _json.loads(raw_entered)
            if isinstance(parsed, list):
                for item in parsed:
                    subj = str(item.get("subject", "")).strip()
                    mark_raw = str(item.get("marks", item.get("mark", ""))).strip()
                    if subj and mark_raw.isdigit():
                        entered_subjects.append({
                            "subject": subj,
                            "mark": int(mark_raw),
                        })
        except (ValueError, TypeError):
            pass  # Ignore malformed JSON; entered comparison will be skipped

    # 5. Run the original verification engine
    try:
        result = run_individual_doc_verification(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            entered_subjects=entered_subjects,
        )
    except Exception as exc:
        return jsonify({"error": f"Verification engine error: {str(exc)}"}), 500

    # 6. Surface errors from the result dict (they always carry an "error" key)
    if "error" in result:
        step = result.get("step", "")
        # Map internal step names to appropriate HTTP codes
        if step in ("tesseract_check",):
            return jsonify(result), 503
        if step in ("uploaded_extraction",):
            return jsonify(result), 422
        return jsonify(result), 500

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# POST /api/document/group/verify  (Group / Bulk Document Verification)
# ---------------------------------------------------------------------------
ALLOWED_SHEET_EXTENSIONS = {"csv", "xlsx", "xls"}


@app.route("/api/document/group/verify", methods=["POST"])
def group_doc_verify():
    import uuid as _uuid

    # 1. File presence
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send the spreadsheet as 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    # 2. Format check
    ext = _ext(file.filename)
    if ext not in ALLOWED_SHEET_EXTENSIONS:
        return jsonify({
            "error": (
                f"Unsupported format '{file.filename}'. "
                "Only CSV, XLSX, and XLS files are accepted."
            )
        }), 415

    # 3. Save spreadsheet to a named temp file (run_bulk_doc_verification needs a real path)
    suffix = f".{ext}"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    session_id = _uuid.uuid4().hex  # Unique ID for this request

    try:
        file.save(tmp_path)
        tmp.close()

        # 4. Run the original bulk workflow (all core logic is unchanged)
        result = run_bulk_doc_verification(tmp_path, session_id=session_id)

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # 5. Surface fatal setup errors
    if "error" in result:
        return jsonify({"error": result["error"]}), 503

    # 6. Convert summary_rows into the shape the frontend table expects
    student_rows = []
    for i, row in enumerate(result.get("summary_rows", []), start=1):
        student_rows.append({
            "id":              str(i).zfill(3),
            "name":            row.get("Student Name", ""),
            "certificate_type": row.get("Certificate Type", ""),
            "uploaded_vs_entered": row.get("Uploaded vs Entered", "NOT CHECKED"),
            "originality":     row.get("Originality", "UNVERIFIED"),
            "uploaded_total":  str(row.get("Uploaded Total", "")),
            "official_total":  str(row.get("Official Total", "")),
            "status_error":    row.get("Status / Error", ""),
        })

    # 7. Build a subject-detail map keyed by student name for the frontend
    detail_map: dict = {}
    for item in result.get("detail_rows", []):
        name = item.get("Student Name", "")
        if name not in detail_map:
            detail_map[name] = []
        detail_map[name].append(item)

    # 8. Schedule cleanup of the Excel report after 30 minutes
    if result.get("report_path"):
        _schedule_cleanup(result["report_path"], delay=1800)

    return jsonify({
        "total":           result["total"],
        "original":        result["original"],
        "possible_forgery": result["possible_forgery"],
        "unverified":      result["unverified"],
        "rows":            student_rows,
        "detail_map":      detail_map,
        "report_filename": result.get("report_filename"),
    }), 200


# ---------------------------------------------------------------------------
# GET /api/document/group/report/<filename>  (Download bulk Excel report)
# ---------------------------------------------------------------------------
@app.route("/api/document/group/report/<filename>", methods=["GET"])
def download_group_doc_report(filename: str):
    # Security: only serve files from REPORT_FOLDER, no path traversal
    safe_name = Path(filename).name
    file_path = os.path.join(str(REPORT_FOLDER), safe_name)

    if not os.path.isfile(file_path):
        return jsonify({"error": "Report file not found or already cleaned up."}), 404

    return send_file(
        file_path,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=safe_name,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
