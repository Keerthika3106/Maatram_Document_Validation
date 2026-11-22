"""
doc_validator.py
================
Tamil Nadu DGE marksheet extraction and QR-based verification engine.

Source: Complete_Doc_Val.py (user-supplied authoritative implementation, v3).
Adaptation: tkinter / choose_pdf() / main() removed; web entry point
run_individual_doc_verification() added at the bottom. Every other function
and constant is an exact, verbatim copy of the uploaded source.
"""

import os
from pathlib import Path
import re
import shutil
from difflib import SequenceMatcher

import pymupdf
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from tabulate import tabulate

from datetime import datetime
from urllib.parse import urlparse
import subprocess
import time


# ============================================================
# CONSTANTS
# ============================================================

DPI = 250
PSM_MODES = (6, 4, 11, 12)
MIN_CONFIDENCE = 18

IGNORE_TOKENS = {
    "SUBJECT", "SUBJECTS", "THEORY", "PRACTICAL", "INTERNAL", "EXTERNAL",
    "MARK", "MARKS", "OBTAINED", "SCORE", "TOTAL", "OPTIONAL", "LANGUAGE",
    "ISSUED", "UNDER", "AUTHORITY", "GOVERNMENT", "CERTIFICATE",
    "CANDIDATE", "NAME", "DATE", "BIRTH", "ROLL", "NUMBER", "REGISTER",
    "SESSION", "YEAR", "SCHOOL", "COLLEGE", "BOARD", "EXAMINATION",
    "TAMILNADU", "DEPARTMENT", "PUBLIC", "SECONDARY", "HIGHER",
    "LEAVING", "APPEARED", "FOLLOWING", "FOR", "100", "OBTAINED",
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
    "EIGHT", "NINE", "PASS", "FAIL", "RESULT", "MEDIUM", "INSTRUCTION",
    "PERMANENT", "EMIS", "FATHER", "MOTHER", "GUARDIAN", "DOB",
    "PHOTOGRAPH", "SIGNATURE", "MEMBER", "SECRETARY", "CODE",
}
OCR_NOISE_TOKENS = {"KEY", "QWP DWE", "SIMLOWEV", "GUDUJE", "FE", "RS", "EE"}
# Used only to prefer a clearly recognized OCR reading over a corrupted reading
# for the same row; unknown subject names are still allowed.
KNOWN_SUBJECTS = {
    "TAMIL", "ENGLISH", "MATHEMATICS", "MATHS", "SCIENCE",
    "SOCIAL SCIENCE", "PHYSICS", "CHEMISTRY", "BIOLOGY",
    "COMPUTER SCIENCE", "COMPUTER APPLICATIONS", "COMMERCE",
    "ACCOUNTANCY", "ECONOMICS", "BUSINESS MATHEMATICS",
    "HISTORY", "GEOGRAPHY", "POLITICAL SCIENCE", "STATISTICS",
    "BOTANY", "ZOOLOGY", "ELECTRONICS", "INFORMATICS PRACTICES",
    "INFORMATION TECHNOLOGY", "ENGINEERING GRAPHICS",
}

# Many Indian marksheets print each score twice: as digits and as words
# (for example, 086 / ZERO EIGHT SIX). The word form is an independent
# reading channel and can recover a digit omitted or confused by OCR.
DIGIT_WORDS = {
    "ZERO": 0, "OH": 0, "O": 0,
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
    "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9,
}
TENS_WORDS = {
    "TEN": 10, "ELEVEN": 11, "TWELVE": 12, "THIRTEEN": 13,
    "FOURTEEN": 14, "FIFTEEN": 15, "SIXTEEN": 16,
    "SEVENTEEN": 17, "EIGHTEEN": 18, "NINETEEN": 19,
    "TWENTY": 20, "THIRTY": 30, "FORTY": 40, "FIFTY": 50,
    "SIXTY": 60, "SEVENTY": 70, "EIGHTY": 80, "NINETY": 90,
}

SUBJECT_ALIASES = {
    "MATHS": "MATHEMATICS",
    "MATH": "MATHEMATICS",
    "MATHEMATICS": "MATHEMATICS",
    "COMPUTERAPPLICATION": "COMPUTERAPPLICATIONS",
    "COMPAPP": "COMPUTERAPPLICATIONS",
    "SOCIALSTUDIES": "SOCIALSCIENCE",
}

SUBJECT_ALIASES_FOR_VERIFY = {
    "MATH": "MATHEMATICS",
    "MATHS": "MATHEMATICS",
    "MATHEMATICS": "MATHEMATICS",
    "COMPUTERAPPLICATION": "COMPUTERAPPLICATIONS",
    "COMPAPP": "COMPUTERAPPLICATIONS",
    "SOCIALSTUDIES": "SOCIALSCIENCE",
    "BUSINESSMATHS": "BUSINESSMATHEMATICS",
}

ALLOWED_DGE_HOST = "certverify.tndge.org"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "Uploaded_Marksheets"
VERIFIED_FOLDER = BASE_DIR / "verified_certificates"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
VERIFIED_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

def configure_tesseract():
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        r"D:\\OCR\\tesseract.exe",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


TESSERACT_PATH = configure_tesseract()


# ============================================================
# OCR AND IMAGE HELPERS
# ============================================================

def render_gray(page):
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def preprocess_variants(gray):
    """Generate complementary OCR images; no image files are written."""
    variants = [("gray", gray)]
    # Contrast enhancement helps faded photocopies.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("contrast", enhanced))
    # Adaptive threshold helps uneven backgrounds and security watermarks.
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )
    variants.append(("adaptive", adaptive))
    # Otsu threshold is useful for clean printed certificates.
    _, otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(("otsu", otsu))
    # Mild sharpening can recover thin printed characters.
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    variants.append(("sharp", sharp))
    return variants


def read_words(image, psm):
    data = pytesseract.image_to_data(
        image, output_type=Output.DICT,
        config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    )
    words = []
    for i, raw in enumerate(data["text"]):
        txt = raw.strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < MIN_CONFIDENCE:
            continue
        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        words.append({
            "text": txt, "x0": x, "x1": x + w, "xc": x + w / 2,
            "y0": y, "y1": y + h, "yc": y + h / 2, "conf": conf,
        })
    return words


def token(text):
    return re.sub(r"[^A-Z]", "", text.upper())


def parse_mark(text):
    s = text.strip().replace(",", "")
    # Marks in this table are generally 0-100; 097 is accepted as 97.
    if not re.fullmatch(r"\d{1,3}", s):
        return None
    value = int(s)
    return value if 0 <= value <= 100 else None


def _normalise_digit_word(raw):
    """Correct common OCR damage in printed number words conservatively."""
    t = token(raw)
    if t in DIGIT_WORDS or t in TENS_WORDS or t == "HUNDRED":
        return t
    # Single-character OCR fragments occur when a word is split by a seal.
    # Only accept a fuzzy match when it is unambiguous and sufficiently close.
    candidates = list(DIGIT_WORDS) + list(TENS_WORDS)
    ranked = sorted(
        ((SequenceMatcher(None, t, candidate).ratio(), candidate)
         for candidate in candidates), reverse=True
    )
    if ranked and ranked[0][0] >= 0.78:
        if len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.08:
            return ranked[0][1]
    return None


def parse_mark_words(words, mark_word, page_height):
    """Read the spelled-out score to the right of a numeric score.

    This is a secondary reading channel. It tolerates OCR splitting a word
    into fragments, but does not override a clearly read three-digit score.
    """
    row_tol = max(18, page_height * 0.009)
    nearby = [
        w for w in words
        if abs(w["yc"] - mark_word["yc"]) <= row_tol
        and w["x0"] >= mark_word["x1"] - 4
    ]
    nearby.sort(key=lambda w: w["x0"])

    tokens = []
    previous = None
    for w in nearby:
        t = _normalise_digit_word(w["text"])
        if t is None:
            # Ignore isolated OCR fragments, but stop at ordinary text after
            # a number-word sequence has started.
            if tokens:
                if len(token(w["text"])) <= 2:
                    continue
                break
            continue
        if previous is not None:
            gap = w["x0"] - previous["x1"]
            if gap > page_height * 0.055:
                break
        tokens.append(t)
        previous = w
        if len(tokens) >= 3:
            break

    if not tokens:
        return None
    if all(t in DIGIT_WORDS for t in tokens) and len(tokens) in (2, 3):
        value = int("".join(str(DIGIT_WORDS[t]) for t in tokens))
        return value if 0 <= value <= 100 else None
    if tokens[0] in TENS_WORDS:
        value = TENS_WORDS[tokens[0]]
        if len(tokens) > 1 and tokens[1] in DIGIT_WORDS:
            value += DIGIT_WORDS[tokens[1]]
        return value if 0 <= value <= 100 else None
    if len(tokens) == 1 and tokens[0] in TENS_WORDS:
        value = TENS_WORDS[tokens[0]]
        return value if 0 <= value <= 100 else None
    return None


def group_by_y(words, tolerance):
    lines = []
    for word in sorted(words, key=lambda w: (w["yc"], w["x0"])):
        line = next(
            (ln for ln in lines if abs(ln["yc"] - word["yc"]) <= tolerance),
            None
        )
        if line is None:
            lines.append({"yc": word["yc"], "words": [word]})
        else:
            line["words"].append(word)
            line["yc"] = sum(w["yc"] for w in line["words"]) / len(line["words"])
    for line in lines:
        line["words"].sort(key=lambda w: w["x0"])
    return lines


def find_total(words, height):
    candidates = [
        w for w in words
        if "TOTAL" in token(w["text"])
        and height * 0.38 < w["yc"] < height * 0.90
    ]
    if not candidates:
        return None, None
    # Prefer TOTAL MARKS when both words are visible.
    label = min(candidates, key=lambda w: w["yc"])
    nums = []
    for w in words:
        if abs(w["yc"] - label["yc"]) > height * 0.025:
            continue
        s = w["text"].replace(",", "")
        if re.fullmatch(r"\d{3,4}", s):
            v = int(s)
            if 100 <= v <= 1000:
                nums.append(w)
    right = [w for w in nums if w["xc"] > label["xc"]]
    if right:
        chosen = min(right, key=lambda w: w["xc"] - label["xc"])
        return int(chosen["text"].replace(",", "")), label["yc"]

    # If the printed total's numeric token is missed, try its number-in-words
    # representation on the same row (e.g. ZERO FIVE FIVE EIGHT = 0558).
    word_total = parse_mark_words(words, label, height)
    if word_total is not None:
        return word_total, label["yc"]
    return None, label["yc"]


def find_marks_header(words, height):
    """Locate the final-score header, avoiding the lower TOTAL MARKS row."""
    upper = [
        w for w in words
        if height * 0.25 < w["yc"] < height * 0.62
        and token(w["text"]) in {"MARK", "MARKS", "OBTAINED", "SCORE"}
    ]
    if not upper:
        return None
    # "OBTAINED" is often centered over the final marks column; otherwise MARKS.
    obtained = [w for w in upper if token(w["text"]) == "OBTAINED"]
    marks = [w for w in upper if token(w["text"]) in {"MARK", "MARKS", "SCORE"}]
    chosen = obtained if obtained else marks
    # Choose the rightmost relevant heading to avoid THEORY/PRACTICAL columns.
    return max(chosen, key=lambda w: w["xc"])


def find_numeric_clusters(words, width, top, bottom):
    pts = []
    for w in words:
        if not top <= w["yc"] <= bottom:
            continue
        val = parse_mark(w["text"])
        if val is not None:
            pts.append((w["xc"], val, w))
    if not pts:
        return []
    tolerance = max(18, width * 0.012)
    clusters = []
    for pt in sorted(pts, key=lambda p: p[0]):
        near = min(
            clusters,
            key=lambda c: abs(c["x"] - pt[0]),
            default=None
        )
        if near is None or abs(near["x"] - pt[0]) > tolerance:
            clusters.append({"x": pt[0], "items": [pt]})
        else:
            near["items"].append(pt)
            near["x"] = sum(p[0] for p in near["items"]) / len(near["items"])
    return clusters


def subject_text_near_mark(words, mark_word, mark_x, width, height, body_top, body_bottom):
    """Find the English subject label on the same printed row as a final mark."""
    row_band = max(18, height * 0.018)
    nearby = [
        w for w in words
        if body_top <= w["yc"] <= body_bottom
        and abs(w["yc"] - mark_word["yc"]) <= row_band
        and w["xc"] < mark_x - width * 0.12
        and re.fullmatch(r"[A-Za-z][A-Za-z&./'-]*", w["text"])
    ]
    if not nearby:
        return None

    lines = group_by_y(nearby, max(9, height * 0.008))
    candidates = []
    for line in lines:
        kept = []
        for w in line["words"]:
            t = token(w["text"])
            if len(t) < 3 or t in IGNORE_TOKENS or t in OCR_NOISE_TOKENS:
                continue
            # Reject OCR fragments and numeric/admin words.
            if not re.fullmatch(r"[A-Za-z][A-Za-z&./'-]*", w["text"]):
                continue
            kept.append(w)
        if not kept:
            continue
        # Subject labels are normally compact and left aligned.
        kept.sort(key=lambda w: w["x0"])
        selected = [kept[0]]
        for w in kept[1:]:
            if w["x0"] - selected[-1]["x1"] <= width * 0.045:
                selected.append(w)
        label = " ".join(w["text"] for w in selected)
        label = re.sub(r"[^A-Za-z&'./ -]", "", label).strip(" .-/")
        if not label or len(label) > 35:
            continue
        if len(label.split()) > 3:
            continue
        # Avoid lines that look like student names or administrative fields.
        if any(token(t) in IGNORE_TOKENS for t in label.split()):
            continue
        conf = sum(w["conf"] for w in selected) / len(selected)
        distance = abs(line["yc"] - mark_word["yc"])
        candidates.append((distance, -conf, label.upper()))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def extract_from_words(words, width, height):
    header = find_marks_header(words, height)
    if header is None:
        return None

    grand_total, total_y = find_total(words, height)
    # Include the first subject row, which can sit close to the marks header.
    body_top = header["yc"] + height * 0.006
    body_bottom = total_y - height * 0.012 if total_y else height * 0.82
    if body_bottom <= body_top:
        body_bottom = height * 0.82

    clusters = find_numeric_clusters(words, width, body_top, body_bottom)
    if not clusters:
        return None

    # Score each numeric column by proximity to the final-score header,
    # repetition across subject rows, and position to the right.
    def column_score(c):
        count = len(c["items"])
        distance = abs(c["x"] - header["xc"]) / max(width, 1)
        return count * 2.0 - distance * 8.0

    # Prefer repeated columns, but retain a one-row column as a fallback.
    clusters.sort(key=column_score, reverse=True)
    chosen = clusters[0]
    marks = sorted(chosen["items"], key=lambda p: p[2]["yc"])

    rows = []
    used_subjects = set()
    for _, value, mark_word in marks:
        # Prefer the independently printed word-form score when it is legible.
        # This fixes cases where OCR reads only one digit (e.g. 086 as 6/5).
        raw_mark = mark_word["text"].strip().replace(",", "")
        word_value = parse_mark_words(words, mark_word, height)
        # Trust a clearly recognised 2- or 3-digit numeric mark first. The
        # spelled-out form is used when OCR has dropped/garbled the number.
        if word_value is not None and not re.fullmatch(r"\d{2,3}", raw_mark):
            value = word_value
        subject = subject_text_near_mark(
            words, mark_word, chosen["x"], width, height, body_top, body_bottom
        )
        if not subject:
            continue
        key = re.sub(r"[^A-Z]", "", subject)
        if not key or key in used_subjects:
            continue
        used_subjects.add(key)
        rows.append({
            "subject": subject,
            "mark": value,
            "raw_mark": raw_mark,
            "word_mark": word_value,
            "confidence": mark_word["conf"],
            "mark_x": chosen["x"],
            "y": mark_word["yc"],
        })

    if not rows:
        return None

    # A pass that found more plausible subject rows is preferred.
    # Printed total is only a tie-breaker; no sum validation is performed.
    avg_conf = sum(r["confidence"] for r in rows) / len(rows)
    score = len(rows) * 1000 + avg_conf + (100 if grand_total is not None else 0)
    return {"rows": rows, "total": grand_total, "score": score}


def native_pdf_words(page):
    """Return positioned words from a PDF text layer, when one exists."""
    words = []
    try:
        for item in page.get_text("words"):
            x0, y0, x1, y1, text = item[:5]
            text = str(text).strip()
            if not text:
                continue
            words.append({
                "text": text, "x0": float(x0), "x1": float(x1),
                "xc": (float(x0) + float(x1)) / 2,
                "y0": float(y0), "y1": float(y1),
                "yc": (float(y0) + float(y1)) / 2,
                "conf": 99.0,
            })
    except Exception:
        return []
    return words


def recover_subject_from_row_crop(gray, row_y, mark_x):
    """
    Re-read only the subject-label cell for one detected marks row.

    This is independent of the full-page OCR layout analysis. It enlarges the
    left subject cell and tries complementary OCR preprocessing modes. Known
    subject names are accepted only when the crop OCR is a close match.
    """
    height, width = gray.shape[:2]
    if row_y is None:
        return None

    # The subject column is left of the final marks column. Keep a generous
    # margin but avoid capturing the numeric marks and roll-number columns.
    right = int(mark_x - width * 0.24) if mark_x else int(width * 0.40)
    right = max(int(width * 0.28), min(right, int(width * 0.58)))
    left = int(width * 0.055)
    top = max(0, int(row_y - height * 0.018))
    bottom = min(height, int(row_y + height * 0.018))

    crop = gray[top:bottom, left:right]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 20:
        return None

    # Upscaling is particularly important for low-resolution photocopies.
    crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9
    )
    _, otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    known = sorted(KNOWN_SUBJECTS, key=len, reverse=True)
    readings = []
    for image in (crop, enhanced, adaptive, otsu):
        for psm in (7, 6):
            try:
                raw = pytesseract.image_to_string(
                    image,
                    config=(
                        f"--oem 3 --psm {psm} "
                        "-c preserve_interword_spaces=1"
                    )
                )
            except Exception:
                continue

            # Strip borders/noise and retain plausible alphabetic words.
            cleaned = re.sub(r"[^A-Za-z ]", " ", raw).upper()
            words = [
                w for w in cleaned.split()
                if len(w) >= 2 and token(w) not in IGNORE_TOKENS
            ]
            if not words:
                continue

            # OCR may insert border/noise tokens before or after the true
            # label. Compare every contiguous 1-3 word span independently.
            spans = []
            for span_len in (1, 2, 3):
                for start in range(0, len(words) - span_len + 1):
                    spans.append(" ".join(words[start:start + span_len]))

            for label in spans:
                label_key = re.sub(r"[^A-Z]", "", label)
                if not label_key:
                    continue
                ranked = sorted(
                    [
                        (
                            SequenceMatcher(
                                None,
                                label_key,
                                re.sub(r"[^A-Z]", "", candidate)
                            ).ratio(),
                            candidate
                        )
                        for candidate in known
                    ],
                    reverse=True
                )
                if ranked and ranked[0][0] >= 0.72:
                    readings.append((ranked[0][0], ranked[0][1]))

    if not readings:
        return None

    # Require a meaningful lead over the runner-up to avoid forcing a wrong
    # standard subject name when the crop is genuinely unreadable.
    readings.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    best_score, best_name = readings[0]
    if best_score >= 0.90:
        return best_name

    second_score = next(
        (score for score, name in readings[1:] if name != best_name),
        0.0
    )
    if best_score >= 0.78 and best_score - second_score >= 0.06:
        return best_name
    return None


def process_page(page):
    """
    Extract using both native PDF text and OCR.
    OCR supplements native rows that the embedded text layer may omit.
    """
    native_result = None
    native_words = native_pdf_words(page)
    if native_words:
        native_result = extract_from_words(
            native_words, float(page.rect.width), float(page.rect.height)
        )

    gray = render_gray(page)

    # FAST PATH:
    # PSM 4 and PSM 11 usually recover the complete subject table from clean
    # DGE PDFs. Run these two on the original grayscale image first.
    candidates = []
    fast_passes = (4, 11)
    for psm in fast_passes:
        words = read_words(gray, psm)
        result = extract_from_words(words, gray.shape[1], gray.shape[0])
        if result:
            candidates.append(result)

    # ACCURACY FALLBACK:
    # Only generate enhanced images and run the remaining OCR modes when the
    # fast passes could not recover a sufficiently complete subject table.
    best_fast_count = max(
        (len(result["rows"]) for result in candidates),
        default=0
    )
    if best_fast_count < 5:
        variants = dict(preprocess_variants(gray))
        fallback_passes = [
            ("gray", 6), ("gray", 12),
            ("contrast", 4), ("contrast", 6),
            ("sharp", 4),
        ]
        for variant_name, psm in fallback_passes:
            words = read_words(variants[variant_name], psm)
            result = extract_from_words(words, gray.shape[1], gray.shape[0])
            if result:
                candidates.append(result)

    ocr_rows = []
    ocr_total = None

    if candidates:
        row_groups = []
        y_tolerance = max(18, gray.shape[0] * 0.012)

        for result in candidates:
            for row in result["rows"]:
                group = next(
                    (g for g in row_groups if abs(g["y"] - row["y"]) <= y_tolerance),
                    None
                )
                if group is None:
                    row_groups.append({"y": row["y"], "readings": [row]})
                else:
                    group["readings"].append(row)
                    group["y"] = sum(
                        r["y"] for r in group["readings"]
                    ) / len(group["readings"])

        merged_rows = []
        for group in sorted(row_groups, key=lambda g: g["y"]):
            readings = group["readings"]
            subject_counts = {}
            for row in readings:
                key = re.sub(r"\s+", " ", row["subject"].upper()).strip()
                subject_counts.setdefault(key, []).append(row)

            _, subject_readings = max(
                subject_counts.items(),
                key=lambda item: (
                    len(item[1]),
                    item[0] in KNOWN_SUBJECTS,
                    max(x["confidence"] for x in item[1]),
                ),
            )
            subject_choice = max(
                subject_readings, key=lambda row: row["confidence"]
            )

            mark_votes = {}
            for row in readings:
                value = row["mark"]
                raw = str(row.get("raw_mark", ""))
                digits = re.sub(r"\D", "", raw)
                score = max(1.0, row["confidence"])
                if len(digits) >= 2:
                    score += 30
                if len(digits) == 3:
                    score += 15
                if row.get("word_mark") is not None and row["word_mark"] == value:
                    score += 12
                mark_votes.setdefault(value, []).append(score)

            chosen_mark = max(
                mark_votes,
                key=lambda value: (
                    len(mark_votes[value]),
                    sum(mark_votes[value]),
                    value,
                ),
            )
            merged_rows.append({
                "subject": subject_choice["subject"],
                "mark": chosen_mark,
                "mark_x": subject_choice.get("mark_x"),
                "y": group["y"],
            })

        seen_ocr = set()
        for row in merged_rows:
            # Re-read the subject cell at high magnification. Do not stop at
            # the first full-page pass merely because six rows were detected.
            crop_subject = recover_subject_from_row_crop(
                gray, row.get("y"), row.get("mark_x")
            )
            if crop_subject:
                row["subject"] = crop_subject

            key = re.sub(r"[^A-Z]", "", row["subject"])
            if key and key not in seen_ocr:
                seen_ocr.add(key)
                ocr_rows.append({
                    "subject": row["subject"],
                    "mark": int(row["mark"]),
                    "y": row["y"],
                })

        totals = [
            result["total"] for result in candidates
            if result["total"] is not None
        ]
        if totals:
            ocr_total = max(
                set(totals),
                key=lambda value: (totals.count(value), value)
            )

    # Do not compare y-coordinates across native PDF points and OCR pixels.
    # Merge rows by normalized subject name instead.
    combined = []
    seen_subjects = set()

    if native_result:
        for row in native_result["rows"]:
            key = re.sub(r"[^A-Z]", "", row["subject"])
            if key and key not in seen_subjects:
                seen_subjects.add(key)
                combined.append({
                    "subject": row["subject"],
                    "mark": int(row["mark"]),
                    # Native PDF word coordinates are in PDF points.
                    "_y_norm": float(row.get("y", 0)) / max(float(page.rect.height), 1.0),
                })

    # Add any subject that OCR detected but the native PDF layer missed.
    for row in ocr_rows:
        key = re.sub(r"[^A-Z]", "", row["subject"])
        if key and key not in seen_subjects:
            seen_subjects.add(key)
            combined.append({
                "subject": row["subject"],
                "mark": int(row["mark"]),
                # OCR y coordinates are pixels; normalize by rendered height.
                "_y_norm": float(row.get("y", 0)) / max(float(gray.shape[0]), 1.0),
            })

    total = (
        native_result["total"]
        if native_result and native_result["total"] is not None
        else ocr_total
    )
    return combined, total


# ============================================================
# SUBJECT NORMALIZATION
# ============================================================

def normalize_subject_name(name):
    """Canonicalize harmless subject-name variations for comparison."""
    key = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    return SUBJECT_ALIASES.get(key, key)


def collect_user_marks():
    """Collect subject names and marks entered by the user (CLI only)."""
    print("\n" + "=" * 66)
    print("             ENTER YOUR SUBJECT MARKS")
    print("=" * 66)
    print("Enter one subject at a time.")
    print("Press ENTER on an empty subject name when you have finished.\n")

    entries = []
    seen = set()

    while True:
        subject = input("Subject name (ENTER to finish): ").strip()
        if not subject:
            break

        key = normalize_subject_name(subject)
        if not key:
            print("Please enter a valid subject name.")
            continue
        if key in seen:
            print("This subject has already been entered. Enter it only once.")
            continue

        while True:
            raw_mark = input(f"Your mark for {subject}: ").strip()
            if raw_mark.isdigit() and 0 <= int(raw_mark) <= 100:
                mark = int(raw_mark)
                break
            print("Enter a valid mark between 0 and 100.")

        entries.append({"subject": subject, "mark": mark})
        seen.add(key)
        print(f"Added: {subject} - {mark}\n")

    return entries


def compare_subject_marks(extracted_rows, user_rows):
    """Compare extracted and manually entered subjects and marks (CLI only)."""
    print("\n" + "=" * 78)
    print("                    SUBJECT MARK COMPARISON")
    print("=" * 78)

    if not user_rows:
        print("No manual subject marks were entered. Comparison skipped.")
        print("=" * 78)
        return

    extracted_by_key = {
        normalize_subject_name(row["subject"]): row
        for row in extracted_rows
    }
    user_by_key = {
        normalize_subject_name(row["subject"]): row
        for row in user_rows
    }

    # Match exact normalized names first. Then use conservative fuzzy matching
    # for minor OCR spelling differences, only when the best match is unique.
    unmatched_extracted = set(extracted_by_key)
    unmatched_user = set(user_by_key)
    matches = []

    for key in list(unmatched_user):
        if key in unmatched_extracted:
            matches.append((key, key, "exact"))
            unmatched_user.remove(key)
            unmatched_extracted.remove(key)

    for user_key in list(unmatched_user):
        candidates = []
        for extracted_key in unmatched_extracted:
            ratio = SequenceMatcher(None, user_key, extracted_key).ratio()
            if ratio >= 0.88:
                candidates.append((ratio, extracted_key))
        candidates.sort(reverse=True)

        if candidates:
            best_ratio, best_key = candidates[0]
            is_unique = len(candidates) == 1 or (
                best_ratio - candidates[1][0] >= 0.06
            )
            if is_unique:
                matches.append((best_key, user_key, "similar name"))
                unmatched_user.remove(user_key)
                unmatched_extracted.remove(best_key)

    comparison_table = []
    mismatch_count = 0

    for extracted_key, user_key, match_type in matches:
        extracted = extracted_by_key[extracted_key]
        entered = user_by_key[user_key]
        same = int(extracted["mark"]) == int(entered["mark"])
        status = "MATCH" if same else "MISMATCH"
        if not same:
            mismatch_count += 1
        comparison_table.append([
            extracted["subject"],
            extracted["mark"],
            entered["mark"],
            status
        ])

    if comparison_table:
        print(tabulate(
            comparison_table,
            headers=[
                "Subject",
                "PDF Extracted Mark",
                "Your Entered Mark",
                "Result"
            ],
            tablefmt="grid"
        ))
    else:
        print("No matching subject names were found between the PDF and your entries.")

    if unmatched_extracted:
        print("\nSubjects found in the PDF but not entered by you:")
        for key in sorted(unmatched_extracted):
            print(f"  - {extracted_by_key[key]['subject']}")

    if unmatched_user:
        print("\nSubjects entered by you but not found in the PDF:")
        for key in sorted(unmatched_user):
            print(f"  - {user_by_key[key]['subject']}")

    print("\n" + "-" * 78)
    print(f"Subjects compared : {len(matches)}")
    print(f"Matching marks    : {len(matches) - mismatch_count}")
    print(f"Mismatched marks  : {mismatch_count}")
    print(f"PDF-only subjects : {len(unmatched_extracted)}")
    print(f"User-only subjects: {len(unmatched_user)}")

    if mismatch_count == 0 and not unmatched_extracted and not unmatched_user:
        print("FINAL RESULT: All subject names and marks match.")
    else:
        print("FINAL RESULT: Differences found. Check the comparison table above.")
    print("=" * 78)


# ============================================================
# QR VERIFICATION
# ============================================================

def unique_path(folder, stem, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = folder / f"{stem}_{timestamp}{extension}"
    counter = 1
    while destination.exists():
        destination = folder / f"{stem}_{timestamp}_{counter}{extension}"
        counter += 1
    return destination


def find_edge():
    edge_path = shutil.which("msedge")
    if edge_path:
        return edge_path

    possible_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    for path in possible_paths:
        if os.path.isfile(path):
            return path
    return None


def save_website_as_pdf(url, marksheet_name):
    parsed = urlparse(url)

    # Security check: only allow the official Tamil Nadu DGE verification host.
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_DGE_HOST:
        print("QR URL is not from the recognized Tamil Nadu DGE verification host.")
        return None

    edge_path = find_edge()
    if not edge_path:
        print("Microsoft Edge was not found. Install Edge or add msedge to PATH.")
        return None

    pdf_path = unique_path(
        VERIFIED_FOLDER,
        marksheet_name + "_DGE_Verification",
        ".pdf"
    )

    # ----------------------------------------------------------------
    # EDGE HEADLESS — ROBUST INVOCATION
    #
    # Problem 1 — spaces in PDF path:
    #   Edge's --print-to-pdf rejects any path that contains a space
    #   (returns exit code 13).  The verified_certificates/ folder lives
    #   inside the project, which is under a user directory whose name
    #   contains spaces ("Keerthika  V").  Fix: write to a guaranteed
    #   space-free intermediate directory (C:\Temp or %SystemDrive%\Temp),
    #   then move the finished file to the real destination.
    #
    # Problem 2 — user data directory conflict:
    #   If Microsoft Edge is already open, --headless shares the same
    #   user profile and crashes (ACCESS VIOLATION / exit -1073741819).
    #   Fix: use a unique --user-data-dir per call.
    #
    # Problem 3 — --run-all-compositor-stages-before-draw +
    #             --virtual-time-budget on a live URL:
    #   These flags force an immediate paint before the network response
    #   arrives, causing a crash or an empty page on live URLs.
    #   Fix: remove both flags; use the default load-and-wait behaviour.
    #
    # Problem 4 — check=True raises on non-zero exit:
    #   Some Edge builds return non-zero even after writing the PDF.
    #   Fix: check=False; detect success by file presence and size.
    # ----------------------------------------------------------------
    import tempfile as _tempfile
    import uuid as _uuid

    # Space-free intermediate directory for the PDF.
    # Prefer C:\Temp (no spaces on any Windows system); fall back to
    # %SystemDrive%\Temp (e.g. D:\Temp), then %TEMP% as last resort.
    _sys_drive = os.environ.get("SystemDrive", "C:")
    _space_free_base = Path(f"{_sys_drive}\\Temp")
    try:
        _space_free_base.mkdir(parents=True, exist_ok=True)
    except OSError:
        _space_free_base = Path(_tempfile.gettempdir())

    _tmp_stem = f"dge_{_uuid.uuid4().hex}"
    tmp_pdf   = _space_free_base / f"{_tmp_stem}.pdf"

    # Unique Edge profile directory (also space-free).
    user_data_dir = _space_free_base / f"edge_profile_{_tmp_stem}"
    try:
        user_data_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        user_data_dir = Path(_tempfile.gettempdir()) / f"edge_profile_{_tmp_stem}"
        user_data_dir.mkdir(parents=True, exist_ok=True)

    command = [
        edge_path,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pdf-header-footer",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--window-size=1280,900",
        f"--user-data-dir={user_data_dir}",
        "--print-to-pdf=" + str(tmp_pdf),
        url,
    ]

    print("\nDownloading the official DGE verification page as PDF...")
    try:
        result = subprocess.run(
            command,
            check=False,          # detect success by file presence, not exit code
            timeout=120,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            stderr_text = (result.stderr or b"").decode(errors="replace").strip()
            print(f"Edge exited with code {result.returncode}. stderr: {stderr_text[:300]}")

        # Poll up to 30 seconds for the PDF to appear on disk.
        for _ in range(120):
            if tmp_pdf.is_file() and tmp_pdf.stat().st_size > 1000:
                # Move from space-free temp location to the real destination.
                try:
                    shutil.move(str(tmp_pdf), str(pdf_path))
                except Exception:
                    shutil.copy2(str(tmp_pdf), str(pdf_path))
                    tmp_pdf.unlink(missing_ok=True)
                print("Official DGE verification PDF saved:")
                print(pdf_path)
                return pdf_path
            time.sleep(0.25)

        print("The DGE page did not produce a valid PDF within the allowed time.")

    except Exception as exc:
        print("Could not download/save the DGE verification page:", exc)

    finally:
        # Clean up temporary files regardless of outcome.
        try:
            tmp_pdf.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            shutil.rmtree(user_data_dir, ignore_errors=True)
        except Exception:
            pass

    return None


def scan_qr_and_download(uploaded_pdf):
    """Scan the uploaded PDF's QR and return (QR URL, downloaded PDF path)."""
    detector = cv2.QRCodeDetector()

    try:
        with pymupdf.open(uploaded_pdf) as document:
            for page_number, page in enumerate(document, start=1):
                pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
                image = np.frombuffer(
                    pix.samples, dtype=np.uint8
                ).reshape(pix.height, pix.width, pix.n)

                if pix.n == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                qr_data = ""
                try:
                    success, decoded, points, _ = detector.detectAndDecodeMulti(image)
                    if success:
                        qr_data = next((value for value in decoded if value), "")
                except Exception:
                    pass

                if not qr_data:
                    qr_data, points, _ = detector.detectAndDecode(image)

                if qr_data:
                    print(f"\nQR code detected on page {page_number}.")
                    print("QR URL:", qr_data)

                    parsed = urlparse(qr_data)
                    if parsed.scheme != "https" or parsed.hostname != ALLOWED_DGE_HOST:
                        print("QR detected, but it is not a recognized official DGE URL.")
                        return qr_data, None

                    downloaded_pdf = save_website_as_pdf(
                        qr_data, Path(uploaded_pdf).stem
                    )
                    return qr_data, downloaded_pdf

    except Exception as exc:
        print("QR scanning failed:", exc)

    print("No QR code was detected in the uploaded marksheet.")
    return None, None


# ============================================================
# DISPLAY AND COMPARISON HELPERS
# ============================================================

def subject_key(name):
    key = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    return SUBJECT_ALIASES_FOR_VERIFY.get(key, key)


def display_extracted_rows(rows, heading):
    print("\n" + "=" * 72)
    print(heading)
    print("=" * 72)

    if not rows:
        print("No subject/mark rows were reliably extracted.")
        print("=" * 72)
        return

    table = [
        [index, row["subject"], row["mark"]]
        for index, row in enumerate(rows, 1)
    ]
    print(tabulate(
        table,
        headers=["S.No.", "Subject Name", "Extracted Mark"],
        tablefmt="grid"
    ))
    print("=" * 72)


def find_second_year_start_ratio(page):
    """
    Find the vertical position of the 'Second Year' heading.
    First use the PDF text layer; use OCR only if that heading is absent.
    Returns a normalized y position (0..1), or None if not found.
    """
    def locate_heading(words, page_height):
        if not words:
            return None

        # Convert word records to the common shape used by group_by_y().
        normalized_words = []
        for item in words:
            if isinstance(item, dict):
                txt = str(item.get("text", "")).strip()
                y0 = float(item.get("y0", item.get("yc", 0)))
                y1 = float(item.get("y1", item.get("yc", 0)))
                yc = float(item.get("yc", (y0 + y1) / 2))
            else:
                # PyMuPDF get_text("words") tuple.
                x0, y0, x1, y1, txt = item[:5]
                txt = str(txt).strip()
                yc = (float(y0) + float(y1)) / 2
            if txt:
                normalized_words.append({
                    "text": txt,
                    "x0": 0, "x1": 0, "x": 0,
                    "y0": y0, "y1": y1, "yc": yc,
                    "conf": 99.0
                })

        if not normalized_words:
            return None

        lines = group_by_y(
            normalized_words,
            max(3.0, page_height * 0.006)
        )
        for line in lines:
            line_tokens = [token(w["text"]) for w in line["words"]]
            # Match the standalone section heading only. The certificate title
            # also contains the words "SECOND YEAR" inside "SECONDARY SECOND YEAR".
            if line_tokens == ["SECOND", "YEAR"] or line_tokens == ["SECONDYEAR"]:
                return float(line["yc"]) / max(page_height, 1.0)
        return None

    native = page.get_text("words")
    found = locate_heading(native, float(page.rect.height))
    if found is not None:
        return found

    # OCR fallback for scanned/image-based official PDFs.
    gray = render_gray(page)
    ocr_words = read_words(gray, 6)
    found = locate_heading(ocr_words, float(gray.shape[0]))
    return found


def extract_dge_second_year_text_rows(page):
    """
    Read only the Second Year block from a Tamil Nadu DGE HSC certificate.
    The digital verification PDF prints one subject per text line and places
    the final mark immediately before the (roll/session/year) field.
    """
    text = page.get_text("text", sort=True)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]

    second_year_index = None
    for index, line in enumerate(lines):
        if re.fullmatch(r"SECOND\s+YEAR", line, flags=re.IGNORECASE):
            second_year_index = index
            break

    if second_year_index is None:
        return [], None, False

    subject_names = [
        "COMPUTER SCIENCE", "BUSINESS MATHEMATICS",
        "INFORMATICS PRACTICES", "COMPUTER APPLICATIONS",
        "MATHEMATICS", "ACCOUNTANCY", "ECONOMICS",
        "CHEMISTRY", "PHYSICS", "BIOLOGY", "ENGLISH",
        "TAMIL", "COMMERCE", "BOTANY", "ZOOLOGY",
        "HISTORY", "GEOGRAPHY", "STATISTICS",
    ]
    subject_pattern = "|".join(
        re.escape(name) for name in sorted(subject_names, key=len, reverse=True)
    )

    rows = []
    total = None
    for line in lines[second_year_index + 1:]:
        if re.search(r"\bTOTAL\s+MARKS\b", line, re.IGNORECASE):
            total_match = re.search(r"\b0?(\d{3})\b", line)
            if total_match:
                total = int(total_match.group(1))
            break

        subject_match = re.match(
            rf"^\s*({subject_pattern})\b",
            line,
            flags=re.IGNORECASE
        )
        if not subject_match:
            continue

        # DGE table row: ... final mark ( register / session / year )
        mark_match = re.search(
            r"\b(\d{2,3})\s*\(\s*\d{6,}\s*/",
            line
        )
        if not mark_match:
            continue

        mark = int(mark_match.group(1))
        if not 0 <= mark <= 100:
            continue

        subject = re.sub(r"\s+", " ", subject_match.group(1)).upper()
        rows.append({"subject": subject, "mark": mark})

    return rows, total, True


def extract_marksheet_rows(pdf_path, second_year_only=False):
    """
    Extract subject marks using the established process_page() algorithm.

    For a QR-downloaded Tamil Nadu HSC certificate, second_year_only=True
    reads only the block explicitly headed 'Second Year'. It never includes
    the First Year block in the final official comparison.
    """
    all_rows = []
    grand_total = None

    with pymupdf.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError("The PDF is password protected.")
        if len(document) == 0:
            raise ValueError("The PDF contains no pages.")

        for page_number, page in enumerate(document, 1):
            print(f"Reading page {page_number}/{len(document)}...")

            if second_year_only:
                second_rows, second_total, section_found = (
                    extract_dge_second_year_text_rows(page)
                )

                if section_found:
                    all_rows.extend(second_rows)
                    if grand_total is None and second_total is not None:
                        grand_total = second_total
                    if not second_rows:
                        print(
                            f"WARNING: Second Year heading found on page {page_number}, "
                            "but no subject marks could be parsed from its rows."
                        )
                    continue

                # Do not fall back to whole-page extraction: that could silently
                # mix First Year marks into a 12th-standard verification.
                print(
                    f"No standalone Second Year section found on page {page_number}; "
                    "page skipped."
                )
                continue

            page_rows, page_total = process_page(page)
            all_rows.extend(page_rows)
            if grand_total is None and page_total is not None:
                grand_total = page_total

    if second_year_only and not all_rows:
        raise ValueError(
            "Could not extract any subject marks from the official PDF's "
            "Second Year section. Verification stopped to avoid a false verdict."
        )

    # Deduplicate in reading order, preserving the first occurrence.
    final_rows = []
    seen = set()
    for row in all_rows:
        key = re.sub(r"[^A-Z]", "", row["subject"])
        if key and key not in seen:
            seen.add(key)
            final_rows.append({
                "subject": row["subject"],
                "mark": int(row["mark"])
            })

    return final_rows, grand_total


def compare_uploaded_with_entered(uploaded_rows, entered_rows):
    """Stage 1: compare the already-extracted uploaded PDF with user entries."""
    print("\n" + "=" * 78)
    print("UPLOADED MARKSHEET vs ENTERED MARKS")
    print("=" * 78)

    uploaded_by_key = {subject_key(r["subject"]): r for r in uploaded_rows}
    entered_by_key = {subject_key(r["subject"]): r for r in entered_rows}

    unmatched_uploaded = set(uploaded_by_key)
    unmatched_entered = set(entered_by_key)
    matches = []

    for key in list(unmatched_entered):
        if key in unmatched_uploaded:
            matches.append((key, key))
            unmatched_uploaded.remove(key)
            unmatched_entered.remove(key)

    # Conservative fuzzy matching for small subject-name OCR/typing differences.
    for entered_key in list(unmatched_entered):
        candidates = sorted(
            [
                (
                    SequenceMatcher(None, entered_key, uploaded_key).ratio(),
                    uploaded_key
                )
                for uploaded_key in unmatched_uploaded
            ],
            reverse=True
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

    table = []
    mismatch_count = 0
    for uploaded_key, entered_key in matches:
        uploaded = uploaded_by_key[uploaded_key]
        entered = entered_by_key[entered_key]
        same = int(uploaded["mark"]) == int(entered["mark"])
        if not same:
            mismatch_count += 1
        table.append([
            uploaded["subject"],
            uploaded["mark"],
            entered["mark"],
            "SAME" if same else "MISMATCH"
        ])

    if table:
        print(tabulate(
            table,
            headers=["Subject", "Uploaded PDF Mark", "Entered Mark", "Result"],
            tablefmt="grid"
        ))
    else:
        print("No subject names could be matched.")

    if unmatched_uploaded:
        print("\nSubjects in uploaded PDF but not entered:")
        for key in sorted(unmatched_uploaded):
            print("-", uploaded_by_key[key]["subject"])

    if unmatched_entered:
        print("\nSubjects entered but not found in uploaded PDF:")
        for key in sorted(unmatched_entered):
            print("-", entered_by_key[key]["subject"])

    passed = (
        bool(matches)
        and mismatch_count == 0
        and not unmatched_uploaded
        and not unmatched_entered
    )

    print("\nSTAGE 1 RESULT:", "ALL SAME" if passed else "MISMATCH / INCOMPLETE")
    print("=" * 78)


def compare_uploaded_with_official(uploaded_rows, official_rows):
    """Stage 2: compare uploaded marks only against the QR-downloaded DGE PDF."""
    print("\n" + "=" * 86)
    print("UPLOADED MARKSHEET vs QR-DOWNLOADED OFFICIAL DGE MARKSHEET")
    print("=" * 86)

    if not uploaded_rows or not official_rows:
        print("VERIFICATION INCONCLUSIVE: one or both PDFs have no reliable subject marks.")
        print("=" * 86)
        return "UNVERIFIED"

    uploaded_by_key = {subject_key(r["subject"]): r for r in uploaded_rows}
    official_by_key = {subject_key(r["subject"]): r for r in official_rows}

    uploaded_keys = set(uploaded_by_key)
    official_keys = set(official_by_key)

    table = []
    mismatch_count = 0
    for key in sorted(uploaded_keys | official_keys):
        uploaded = uploaded_by_key.get(key)
        official = official_by_key.get(key)

        if uploaded is None:
            table.append([
                official["subject"], "NOT FOUND", official["mark"],
                "SUBJECT MISSING IN UPLOADED"
            ])
            mismatch_count += 1
        elif official is None:
            table.append([
                uploaded["subject"], uploaded["mark"], "NOT FOUND",
                "SUBJECT MISSING IN OFFICIAL"
            ])
            mismatch_count += 1
        else:
            same = int(uploaded["mark"]) == int(official["mark"])
            table.append([
                uploaded["subject"],
                uploaded["mark"],
                official["mark"],
                "SAME" if same else "MISMATCH"
            ])
            if not same:
                mismatch_count += 1

    print(tabulate(
        table,
        headers=[
            "Subject",
            "Uploaded Mark",
            "Official QR Mark",
            "Comparison"
        ],
        tablefmt="grid"
    ))

    print("\n" + "-" * 86)

    # A missing subject is an extraction/coverage problem until independently
    # confirmed. It must NOT automatically produce a forgery verdict.
    if uploaded_keys != official_keys:
        print("FINAL RESULT: UNVERIFIED — SUBJECT LISTS ARE INCOMPLETE OR DIFFERENT")
        print("At least one subject is missing from one extraction. Check the official PDF/OCR before deciding.")
        result = "UNVERIFIED"
    elif mismatch_count == 0:
        print("FINAL RESULT: ORIGINAL / QR-VERIFIED MARKS MATCH")
        result = "ORIGINAL"
    else:
        print("FINAL RESULT: MARK MISMATCH DETECTED — POSSIBLE FORGERY")
        print("The subject lists match, but one or more extracted marks differ.")
        print("Manual verification against the official record is still required.")
        result = "POSSIBLE_FORGERY"

    print("-" * 86)
    return result


# ============================================================
# CERTIFICATE TYPE DETECTION
# ============================================================

def detect_certificate_type(pdf_path):
    """
    Identify 10th/12th standard from the uploaded certificate.
    Embedded PDF text is checked first; OCR is used only when text is absent
    or does not clearly identify the certificate type.
    """
    native_text_parts = []

    with pymupdf.open(pdf_path) as document:
        if document.is_encrypted or len(document) == 0:
            return "UNKNOWN"

        for page in list(document)[:2]:
            native_text = page.get_text("text", sort=True) or ""
            if native_text.strip():
                native_text_parts.append(native_text)

    def classify(text):
        normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
        is_12th = (
            "HIGHERSECONDARYCOURSEMARKCERTIFICATE" in normalized
            or "HIGHERSECONDARYSECON" in normalized
            or "HIGHERSECONDARY" in normalized
        )
        is_10th = (
            "SECONDARYSCHOOLLEAVINGCERTIFICATE" in normalized
            or "XSTANDARD" in normalized
            or "SSLC" in normalized
        )
        if is_12th:
            return "12TH"
        if is_10th:
            return "10TH"
        return "UNKNOWN"

    native_result = classify(" ".join(native_text_parts))
    if native_result != "UNKNOWN":
        return native_result

    # Scanned/image-only upload: OCR only until the certificate type is known.
    with pymupdf.open(pdf_path) as document:
        for page in list(document)[:2]:
            gray = render_gray(page)
            for psm in (6, 11):
                words = read_words(gray, psm)
                result = classify(" ".join(w["text"] for w in words))
                if result != "UNKNOWN":
                    return result

    return "UNKNOWN"


# ============================================================
# WEB API ENTRY POINT
# (web adaptation only — no original logic is modified)
# ============================================================

def run_individual_doc_verification(
    pdf_bytes: bytes,
    filename: str,
    entered_subjects: list,
) -> dict:
    """
    Web-facing entry point for individual document verification.

    Parameters
    ----------
    pdf_bytes        : Raw bytes of the uploaded PDF.
    filename         : Original filename (used for safe disk naming only).
    entered_subjects : List of {"subject": str, "mark": int} from the user form.
                       May be empty when the user skips manual entry.

    Returns
    -------
    A dict containing all result fields.
    On unrecoverable error an "error" key is present.
    Never raises; all exceptions are caught and returned as error dicts.
    """

    # 1. Tesseract availability check
    if not TESSERACT_PATH:
        return {
            "error": (
                "Tesseract OCR not found. "
                r"Install it at C:\Program Files\Tesseract-OCR\ "
                "or set the TESSERACT_CMD environment variable."
            ),
            "step": "tesseract_check",
        }

    # 2. Save PDF bytes to UPLOAD_FOLDER
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).stem)[:60] or "upload"
    saved_path = unique_path(UPLOAD_FOLDER, safe_stem + "_uploaded", ".pdf")
    try:
        saved_path.write_bytes(pdf_bytes)
    except OSError as exc:
        return {"error": f"Could not save uploaded PDF: {exc}", "step": "save"}

    # 3. Detect certificate type
    try:
        cert_type = detect_certificate_type(str(saved_path))
    except Exception as exc:
        return {"error": f"Certificate type detection failed: {exc}", "step": "cert_type"}

    if cert_type == "UNKNOWN":
        return {
            "error": (
                "Cannot determine whether this is a 10th or 12th marksheet. "
                "Verification stopped."
            ),
            "step": "cert_type",
        }

    # 4. Extract uploaded PDF
    try:
        uploaded_rows, uploaded_total = extract_marksheet_rows(str(saved_path))
    except Exception as exc:
        return {"error": f"Uploaded marksheet extraction failed: {exc}", "step": "uploaded_extraction"}

    if not uploaded_rows:
        return {
            "error": "No subject marks could be extracted from the uploaded PDF.",
            "step": "uploaded_extraction",
        }

    # 5. Entered marks comparison (inline — no console printing)
    entered_comparison: list = []
    if entered_subjects:
        up_by_key = {subject_key(r["subject"]): r for r in uploaded_rows}
        en_by_key = {subject_key(r["subject"]): r for r in entered_subjects}
        unmatched_up = set(up_by_key)
        unmatched_en = set(en_by_key)
        matches: list = []

        for key in list(unmatched_en):
            if key in unmatched_up:
                matches.append((key, key))
                unmatched_up.discard(key)
                unmatched_en.discard(key)

        for en_key in list(unmatched_en):
            cands = sorted(
                [(SequenceMatcher(None, en_key, uk).ratio(), uk) for uk in unmatched_up],
                reverse=True,
            )
            if cands and cands[0][0] >= 0.88:
                best_ratio, best_uk = cands[0]
                unique = len(cands) == 1 or best_ratio - cands[1][0] >= 0.06
                if unique:
                    matches.append((best_uk, en_key))
                    unmatched_up.discard(best_uk)
                    unmatched_en.discard(en_key)

        for up_key, en_key in matches:
            up_row = up_by_key[up_key]
            en_row = en_by_key[en_key]
            same = int(up_row["mark"]) == int(en_row["mark"])
            entered_comparison.append({
                "subject":       up_row["subject"],
                "uploaded_mark": int(up_row["mark"]),
                "entered_mark":  int(en_row["mark"]),
                "match":         same,
            })

    # 6. QR scan and official PDF download
    qr_url, official_pdf_path = scan_qr_and_download(str(saved_path))

    partial = {
        "certificate_type":   cert_type,
        "uploaded_filename":  filename,
        "uploaded_rows":      [{"subject": r["subject"], "mark": int(r["mark"])} for r in uploaded_rows],
        "uploaded_total":     int(uploaded_total) if uploaded_total is not None else None,
        "entered_comparison": entered_comparison,
        "qr_url":             qr_url,
    }

    if not qr_url:
        return {
            **partial,
            "qr_status":           "not_found",
            "official_rows":       [],
            "official_total":      None,
            "official_comparison": [],
            "final_verdict":       "UNVERIFIED",
            "verdict_explanation": "QR code was not found in the uploaded marksheet.",
        }

    if not official_pdf_path:
        edge_found = find_edge() is not None
        extra = " Microsoft Edge was not found on this server." if not edge_found else ""
        return {
            **partial,
            "qr_status":           "found",
            "official_rows":       [],
            "official_total":      None,
            "official_comparison": [],
            "final_verdict":       "UNVERIFIED",
            "verdict_explanation": "Official DGE verification page could not be downloaded." + extra,
        }

    # 7. Extract official PDF
    second_year_only = (cert_type == "12TH")
    try:
        official_rows, official_total = extract_marksheet_rows(
            str(official_pdf_path), second_year_only=second_year_only
        )
    except Exception as exc:
        return {
            **partial,
            "qr_status":           "found",
            "official_rows":       [],
            "official_total":      None,
            "official_comparison": [],
            "final_verdict":       "UNVERIFIED",
            "verdict_explanation": f"Official DGE PDF extraction failed: {exc}",
        }

    # 8. Compare using the original compare_uploaded_with_official() verbatim
    #    (also prints to stdout for server-side audit logging)
    verdict = compare_uploaded_with_official(uploaded_rows, official_rows)

    # Build the official comparison list for the frontend table
    up_by_key  = {subject_key(r["subject"]): r for r in uploaded_rows}
    off_by_key = {subject_key(r["subject"]): r for r in official_rows}
    all_keys   = sorted(set(up_by_key) | set(off_by_key))
    official_comparison: list = []
    for key in all_keys:
        up_r  = up_by_key.get(key)
        off_r = off_by_key.get(key)
        display  = (up_r or off_r)["subject"]
        up_mark  = int(up_r["mark"])  if up_r  else None
        off_mark = int(off_r["mark"]) if off_r else None
        official_comparison.append({
            "subject":       display,
            "uploaded_mark": up_mark,
            "official_mark": off_mark,
            "match": up_mark is not None and off_mark is not None and up_mark == off_mark,
        })

    # 9. Verdict explanation
    if verdict == "ORIGINAL":
        explanation = (
            "All extracted subject names and marks match the official "
            "QR-downloaded DGE record."
        )
    elif verdict == "POSSIBLE_FORGERY":
        explanation = (
            "Subject lists match, but one or more marks differ from the official "
            "DGE record. Manual authority verification is recommended."
        )
    else:
        explanation = (
            "Subject lists are incomplete or different between the uploaded and "
            "official PDF. Manual verification required."
        )

    return {
        **partial,
        "qr_status":           "found",
        "official_rows":       [{"subject": r["subject"], "mark": int(r["mark"])} for r in official_rows],
        "official_total":      int(official_total) if official_total is not None else None,
        "official_comparison": official_comparison,
        "final_verdict":       verdict,
        "verdict_explanation": explanation,
    }
