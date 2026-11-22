"""
Maatram Foundation - Group Photo Validation Logic
==================================================
Original Bulk_Photo_Validation.txt implementation preserved exactly.
Only changes for web integration:
  - Removed: tkinter, filedialog, root = tk.Tk(), root.withdraw(),
              root.attributes(), root.destroy()
  - Removed: main execution block (sections 11 and 12)
  - MODEL_PATH and OUTPUT_FOLDER resolve relative to this file (unchanged logic)
  - All validation functions, constants, thresholds and decision logic unchanged.
"""

import cv2
from PIL import Image, ImageOps
import numpy as np
import os
import csv
import re
import tempfile
import requests
import gdown
import time
import contextlib
import io

from pathlib import Path
from openpyxl import load_workbook


# ============================================================
# 1. PROFESSIONAL PHOTO STANDARDS
# ============================================================

TARGET_RATIO = 4 / 5

MIN_WIDTH = 400
MIN_HEIGHT = 500

MAX_WIDTH = 4000
MAX_HEIGHT = 5000


# Preferred image-quality thresholds

MIN_BRIGHTNESS = 70
MAX_BRIGHTNESS = 200

MIN_CONTRAST = 25
MIN_SHARPNESS = 80


# Tolerance limits

MAX_RATIO_DEVIATION = 0.10

MIN_ALLOWED_BRIGHTNESS = 50
MAX_ALLOWED_BRIGHTNESS = 220

MIN_ALLOWED_CONTRAST = 15
MIN_ALLOWED_SHARPNESS = 30


# Download settings

MAX_IMAGE_SIZE_MB = 20
REQUEST_TIMEOUT = 30


# YuNet model

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_detection_yunet_2023mar.onnx"
)


# Output folder

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Output"
)


# ============================================================
# 2. SPREADSHEET COLUMN CONFIGURATION
# ============================================================

NAME_COLUMNS = [
    "Name",
    "Full Name",
    "Candidate Name",
    "Student Name"
]

PHOTO_COLUMNS = [
    "Photo Link",
    "Photo URL",
    "Image Link",
    "Image URL",
    "Photo",
    "Link"
]


# Results added to the spreadsheet

OUTPUT_HEADERS = [
    "Validation Status",
    "Failure Reason"
]


# ============================================================
# 3. GENERAL HELPER FUNCTIONS
# ============================================================

def normalize_header(value):

    if value is None:
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).strip().lower()
    )


def find_column(headers, possible_names):

    normalized_headers = {
        normalize_header(value): index
        for index, value in enumerate(headers)
        if value is not None
    }

    for name in possible_names:

        key = normalize_header(name)

        if key in normalized_headers:

            return normalized_headers[key]

    return None


def get_timestamp():

    return time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def make_error_result(error):

    result = {
        header: ""
        for header in OUTPUT_HEADERS
    }

    result["Validation Status"] = "FAIL"

    result["Failure Reason"] = str(error)

    result["Quality Grade"] = "NOT VALIDATED"

    result["Validation Date"] = get_timestamp()

    return result


# ============================================================
# 4. GOOGLE DRIVE PHOTO DOWNLOADING
# ============================================================

def download_image(photo_link):

    if photo_link is None or not str(photo_link).strip():

        raise ValueError(
            "Photo link is empty."
        )

    photo_link = str(photo_link).strip()

    if not photo_link.startswith(
        ("https://", "http://")
    ):

        raise ValueError(
            "Invalid photo URL."
        )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".img",
        delete=False
    )

    image_path = temp_file.name

    temp_file.close()

    try:

        # Google Drive links

        if (
            "drive.google.com" in photo_link.lower()
            or "docs.google.com" in photo_link.lower()
        ):

            downloaded = gdown.download(
                url=photo_link,
                output=image_path,
                quiet=True
            )

            if not downloaded:

                raise ValueError(
                    "Google Drive download failed. "
                    "Check sharing permissions."
                )

        # Other direct image links

        else:

            total_size = 0

            with requests.get(
                photo_link,
                timeout=REQUEST_TIMEOUT,
                stream=True,
                allow_redirects=True
            ) as response:

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if (
                    content_type
                    and not content_type.startswith("image/")
                    and "octet-stream" not in content_type
                ):

                    raise ValueError(
                        "The link does not return an image."
                    )

                with open(image_path, "wb") as file:

                    for chunk in response.iter_content(
                        chunk_size=8192
                    ):

                        if not chunk:
                            continue

                        total_size += len(chunk)

                        if total_size > (
                            MAX_IMAGE_SIZE_MB * 1024 * 1024
                        ):

                            raise ValueError(
                                "Image exceeds 20 MB."
                            )

                        file.write(chunk)

        if not os.path.exists(image_path):

            raise ValueError(
                "Image download failed."
            )

        file_size = os.path.getsize(image_path)

        if file_size == 0:

            raise ValueError(
                "Downloaded image is empty."
            )

        if file_size > MAX_IMAGE_SIZE_MB * 1024 * 1024:

            raise ValueError(
                "Image exceeds maximum allowed size."
            )

        return image_path

    except Exception:

        if os.path.exists(image_path):

            os.remove(image_path)

        raise


# ============================================================
# 5. LOAD IMAGE AND CORRECT EXIF ORIENTATION
# ============================================================

def load_image(image_path):

    try:

        with Image.open(image_path) as pil_image:

            pil_image = ImageOps.exif_transpose(
                pil_image
            )

            width, height = pil_image.size

            if width * height > 40_000_000:

                raise ValueError(
                    "Image resolution is too large."
                )

            image = np.array(
                pil_image.convert("RGB")
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        return image

    except Exception as error:

        raise ValueError(
            f"Invalid image: {error}"
        )


# ============================================================
# 6. SINGLE PHOTO VALIDATION
#    ORIGINAL VALIDATION LOGIC RETAINED
# ============================================================

def validate_photo(image_path):

    # --------------------------------------------------------
    # STEP 1: LOAD IMAGE
    # --------------------------------------------------------

    image = load_image(image_path)

    height, width = image.shape[:2]

    file_size_kb = (
        os.path.getsize(image_path) / 1024
    )

    aspect_ratio = width / height

    print("\n--- PROFESSIONAL PHOTO ANALYSIS ---")

    print(
        "File:",
        os.path.basename(image_path)
    )

    print(
        f"Dimensions: {width} x {height} pixels"
    )

    print(
        f"File size: {file_size_kb:.2f} KB"
    )

    print(
        f"Aspect ratio: {aspect_ratio:.4f}"
    )

    # --------------------------------------------------------
    # STEP 2: ORIENTATION CHECK
    # --------------------------------------------------------

    is_portrait = height > width

    orientation = (
        "Portrait" if is_portrait
        else "Landscape" if width > height
        else "Square"
    )

    print("\n--- ORIENTATION CHECK ---")

    print("Orientation:", orientation)

    print(
        "Orientation check:",
        "PASS" if is_portrait else "FAIL"
    )

    # --------------------------------------------------------
    # STEP 3: DIMENSION CHECKS
    # --------------------------------------------------------

    min_dimensions_ok = (
        width >= MIN_WIDTH
        and height >= MIN_HEIGHT
    )

    max_dimensions_ok = (
        width <= MAX_WIDTH
        and height <= MAX_HEIGHT
    )

    print("\n--- DIMENSION CHECK ---")

    print(
        f"Minimum: {MIN_WIDTH} x {MIN_HEIGHT}"
    )

    print(
        f"Maximum: {MAX_WIDTH} x {MAX_HEIGHT}"
    )

    print(
        "Minimum dimensions:",
        "PASS" if min_dimensions_ok else "FAIL"
    )

    print(
        "Maximum dimensions:",
        "PASS" if max_dimensions_ok else "FAIL"
    )

    # --------------------------------------------------------
    # STEP 4: ASPECT RATIO CHECK
    # --------------------------------------------------------

    ratio_difference = (
        abs(aspect_ratio - TARGET_RATIO)
        / TARGET_RATIO
    )

    ratio_ok = ratio_difference <= 0.05

    ratio_acceptable = (
        ratio_difference <= MAX_RATIO_DEVIATION
    )

    if ratio_ok:

        ratio_status = "PASS"

    elif ratio_acceptable:

        ratio_status = "ACCEPTABLE"

    else:

        ratio_status = "FAIL"

    print("\n--- ASPECT RATIO CHECK ---")

    print("Target ratio: 4:5")

    print(
        f"Ratio difference: {ratio_difference * 100:.2f}%"
    )

    print(
        "Aspect ratio:",
        ratio_status
    )

    # --------------------------------------------------------
    # STEP 5: IMAGE QUALITY ANALYSIS
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Average brightness

    brightness = float(
        np.mean(gray)
    )

    # Contrast

    contrast = float(
        np.std(gray)
    )

    # Sharpness

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # Extreme pixels

    dark_pixel_ratio = float(
        np.mean(gray <= 1)
    )

    bright_pixel_ratio = float(
        np.mean(gray >= 254)
    )

    clipped_pixel_ratio = (
        dark_pixel_ratio + bright_pixel_ratio
    )

    # Percentiles

    p5, p50, p95 = np.percentile(
        gray,
        [5, 50, 95]
    )

    # Preferred quality checks

    brightness_ok = (
        MIN_BRIGHTNESS
        <= brightness
        <= MAX_BRIGHTNESS
    )

    contrast_ok = (
        contrast >= MIN_CONTRAST
    )

    sharpness_ok = (
        sharpness >= MIN_SHARPNESS
    )

    # Acceptable quality limits

    brightness_acceptable = (
        MIN_ALLOWED_BRIGHTNESS
        <= brightness
        <= MAX_ALLOWED_BRIGHTNESS
    )

    contrast_acceptable = (
        contrast >= MIN_ALLOWED_CONTRAST
    )

    sharpness_acceptable = (
        sharpness >= MIN_ALLOWED_SHARPNESS
    )

    print("\n--- BASIC IMAGE QUALITY ---")

    print(
        f"Average brightness: {brightness:.2f}/255"
    )

    print(
        f"Contrast: {contrast:.2f}"
    )

    print(
        f"Sharpness: {sharpness:.2f}"
    )

    print("\n--- PIXEL CHARACTERISTICS ---")

    print(
        f"5th percentile intensity: {p5:.1f}"
    )

    print(
        f"Median intensity: {p50:.1f}"
    )

    print(
        f"95th percentile intensity: {p95:.1f}"
    )

    print(
        f"Very dark pixels: {dark_pixel_ratio * 100:.2f}%"
    )

    print(
        f"Very bright pixels: {bright_pixel_ratio * 100:.2f}%"
    )

    print(
        f"Combined extreme pixels: "
        f"{clipped_pixel_ratio * 100:.2f}%"
    )

    # --------------------------------------------------------
    # QUALITY CHECK RESULTS
    # --------------------------------------------------------

    if brightness_ok:

        brightness_status = "PASS"

    elif brightness_acceptable:

        brightness_status = "ACCEPTABLE"

    else:

        brightness_status = "FAIL"

    if contrast_ok:

        contrast_status = "PASS"

    elif contrast_acceptable:

        contrast_status = "ACCEPTABLE"

    else:

        contrast_status = "FAIL"

    if sharpness_ok:

        sharpness_status = "PASS"

    elif sharpness_acceptable:

        sharpness_status = "ACCEPTABLE"

    else:

        sharpness_status = "FAIL"

    print("\n--- QUALITY CHECK RESULTS ---")

    print(
        "Brightness:",
        brightness_status
    )

    print(
        "Contrast:",
        contrast_status
    )

    print(
        "Sharpness:",
        sharpness_status
    )

    print(
        f"Pixel clipping: INFORMATIONAL "
        f"({clipped_pixel_ratio * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # STEP 6: LOAD YUNET FACE DETECTOR
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "YuNet model not found. Place "
            "face_detection_yunet_2023mar.onnx "
            "in the same folder as this Python file."
        )

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (width, height),
        0.80,
        0.30,
        5000
    )

    # --------------------------------------------------------
    # STEP 7: FACE DETECTION
    # --------------------------------------------------------

    _, faces = detector.detect(image)

    face_count = (
        0 if faces is None else len(faces)
    )

    face_ok = face_count == 1

    print("\n--- FACE DETECTION ---")

    print(
        "Face identified"
        if face_ok
        else "Face check: FAIL"
    )

    print(
        "Number of faces detected:",
        face_count
    )

    # --------------------------------------------------------
    # STEP 8: FINAL VALIDATION
    # --------------------------------------------------------

    print(
        "\n--- FINAL PROFESSIONAL PHOTO RESULT ---"
    )

    structural_ok = (
        is_portrait
        and min_dimensions_ok
        and max_dimensions_ok
        and face_ok
    )

    preferred_quality_ok = all([
        ratio_ok,
        brightness_ok,
        contrast_ok,
        sharpness_ok
    ])

    minor_deviation_ok = all([
        ratio_acceptable,
        brightness_acceptable,
        contrast_acceptable,
        sharpness_acceptable
    ])

    # Original final-check logic

    final_checks = {

        "Portrait orientation": (
            is_portrait,
            is_portrait
        ),

        "Minimum dimensions": (
            min_dimensions_ok,
            min_dimensions_ok
        ),

        "Maximum dimensions": (
            max_dimensions_ok,
            max_dimensions_ok
        ),

        "Aspect ratio": (
            ratio_ok,
            ratio_acceptable
        ),

        "Brightness": (
            brightness_ok,
            brightness_acceptable
        ),

        "Contrast": (
            contrast_ok,
            contrast_acceptable
        ),

        "Sharpness": (
            sharpness_ok,
            sharpness_acceptable
        ),

        "Exactly one face detected": (
            face_ok,
            face_ok
        )
    }

    for name, (preferred, acceptable) in final_checks.items():

        if preferred:

            status = "PASS"

        elif acceptable:

            status = (
                "ACCEPTABLE - MINOR DEVIATION"
            )

        else:

            status = "FAIL"

        print(
            f"{name}: {status}"
        )

    print(
        f"Pixel clipping: INFORMATIONAL "
        f"({clipped_pixel_ratio * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    reasons = []

    if structural_ok and preferred_quality_ok:

        print("\nFINAL DECISION: ACCEPT PHOTO")

        print(
            "The photo passes all preferred quality checks."
        )

        validation_status = "PASS"

        quality_grade = "PREFERRED"

    elif structural_ok and minor_deviation_ok:

        print("\nFINAL DECISION: ACCEPT PHOTO")

        print(
            "The photo is acceptable with minor deviations."
        )

        validation_status = "PASS"

        quality_grade = (
            "ACCEPTABLE - MINOR DEVIATION"
        )

    else:

        print("\nFINAL DECISION: REJECT PHOTO")

        print("Reason(s):")

        validation_status = "FAIL"

        quality_grade = "NOT ACCEPTABLE"

        if not is_portrait:

            reasons.append(
                "Photo must be in portrait orientation."
            )

        if not min_dimensions_ok:

            reasons.append(
                "Photo is below the minimum dimensions."
            )

        if not max_dimensions_ok:

            reasons.append(
                "Photo exceeds the maximum dimensions."
            )

        if not face_ok:

            reasons.append(
                "Exactly one face must be detected."
            )

        if not ratio_acceptable:

            reasons.append(
                "Aspect ratio deviation is too high."
            )

        if not brightness_acceptable:

            reasons.append(
                "Brightness is outside acceptable limits."
            )

        if not contrast_acceptable:

            reasons.append(
                "Contrast is too low."
            )

        if not sharpness_acceptable:

            reasons.append(
                "Sharpness is too low."
            )

        for reason in reasons:

            print("-", reason)

    print(
        "\nNote: These are initial photo-screening thresholds."
    )

    print(
        "Actual acceptance depends on the target "
        "organization's requirements."
    )

    # --------------------------------------------------------
    # RETURN VALIDATION RESULTS
    # --------------------------------------------------------

    result = {

        "Validation Status": validation_status,

        "Failure Reason": "; ".join(reasons) if reasons else "",

        "Orientation": orientation,

        "Orientation Check": (
            "PASS" if is_portrait else "FAIL"
        ),

        "Minimum Dimensions": (
            "PASS" if min_dimensions_ok else "FAIL"
        ),

        "Maximum Dimensions": (
            "PASS" if max_dimensions_ok else "FAIL"
        ),

        "Aspect Ratio": round(
            aspect_ratio, 4
        ),

        "Ratio Deviation (%)": round(
            ratio_difference * 100, 2
        ),

        "Aspect Ratio Check": ratio_status,

        "Brightness": round(
            brightness, 2
        ),

        "Brightness Check": brightness_status,

        "Contrast": round(
            contrast, 2
        ),

        "Contrast Check": contrast_status,

        "Sharpness": round(
            sharpness, 2
        ),

        "Sharpness Check": sharpness_status,

        "Dark Pixel (%)": round(
            dark_pixel_ratio * 100, 2
        ),

        "Bright Pixel (%)": round(
            bright_pixel_ratio * 100, 2
        ),

        "Clipped Pixel (%)": round(
            clipped_pixel_ratio * 100, 2
        ),

        "Face Count": face_count,

        "Face Check": (
            "PASS" if face_ok else "FAIL"
        ),

        "Image Width": width,

        "Image Height": height,

        "File Size (KB)": round(
            file_size_kb, 2
        ),

        "Quality Grade": quality_grade,

        "Validation Date": get_timestamp()
    }

    return result


# ============================================================
# 7. PROCESS ONE PERSON'S PHOTO
# ============================================================

def process_photo(photo_link):

    image_path = None

    try:
        image_path = download_image(photo_link)

        # Hide detailed per-photo diagnostic output.
        # Validation calculations and spreadsheet values remain unchanged.
        with contextlib.redirect_stdout(io.StringIO()):
            return validate_photo(image_path)

    except Exception as error:
        return make_error_result(error)

    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass


def get_available_output_path(output_path):
    """Return the requested path, or a timestamped alternative if it is locked."""
    output_path = Path(output_path)
    if not output_path.exists():
        return str(output_path)

    try:
        # Test whether Windows allows writing to the existing file.
        with open(output_path, "a", encoding="utf-8"):
            pass
        return str(output_path)
    except PermissionError:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return str(
            output_path.with_name(
                f"{output_path.stem}_{stamp}{output_path.suffix}"
            )
        )


# ============================================================
# 8. UPDATE CSV FILE
# ============================================================

def process_csv(file_path):

    with open(
        file_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        rows = list(csv.reader(file))

    if not rows:
        raise ValueError("The CSV file is empty.")

    headers = list(rows[0])

    max_columns = max(len(row) for row in rows)

    while len(headers) < max_columns:
        headers.append(f"Extra Column {len(headers) + 1}")

    name_col = find_column(headers, NAME_COLUMNS)
    photo_col = find_column(headers, PHOTO_COLUMNS)

    if name_col is None or photo_col is None:
        raise ValueError(
            "Required columns not found. "
            "Expected Name and Photo Link."
        )

    output_headers = list(headers)

    for header in OUTPUT_HEADERS:
        if normalize_header(header) not in {
            normalize_header(value)
            for value in output_headers
        }:
            output_headers.append(header)

    column_map = {
        normalize_header(header): index
        for index, header in enumerate(output_headers)
    }

    output_rows = [output_headers]

    total = len(rows) - 1

    passed = 0
    failed = 0

    print("\n========== PHOTO VALIDATION ==========\n")

    for index, original_row in enumerate(rows[1:], start=1):

        row = list(original_row)

        row += [""] * (len(output_headers) - len(row))

        name = row[name_col] if name_col < len(row) else ""
        photo_link = row[photo_col] if photo_col < len(row) else ""

        if not any(str(value).strip() for value in original_row):
            output_rows.append(row)
            continue

        result = process_photo(photo_link)

        for header in OUTPUT_HEADERS:
            col_index = column_map[normalize_header(header)]
            row[col_index] = result.get(header, "")

        if result["Validation Status"] == "PASS":

            passed += 1

            print(f"{index}. {name} - PASS")

        else:

            failed += 1

            print(f"{index}. {name} - FAIL")

            print(
                f"   Reason: {result['Failure Reason']}"
            )

        output_rows.append(row)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        Path(file_path).stem + "_validated.csv"
    )
    output_path = get_available_output_path(output_path)

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)
        writer.writerows(output_rows)

    return output_path, passed, failed


# ============================================================
# 9. UPDATE EXCEL FILE
# ============================================================

def process_excel(file_path):

    extension = Path(file_path).suffix.lower()

    workbook = load_workbook(
        file_path,
        keep_vba=(extension == ".xlsm")
    )

    sheet = workbook.active

    headers = [

        sheet.cell(1, col).value

        for col in range(
            1,
            sheet.max_column + 1
        )
    ]

    name_col = find_column(
        headers,
        NAME_COLUMNS
    )

    photo_col = find_column(
        headers,
        PHOTO_COLUMNS
    )

    if name_col is None or photo_col is None:

        raise ValueError(
            "Required columns not found.\n"
            "The spreadsheet must contain Name "
            "and Photo Link columns."
        )

    # Identify existing columns
    existing_columns = {}

    for col in range(
        1,
        sheet.max_column + 1
    ):

        header = sheet.cell(1, col).value

        if header is not None:

            existing_columns[
                normalize_header(header)
            ] = col

    # Create missing result columns
    output_columns = {}

    for header in OUTPUT_HEADERS:

        key = normalize_header(header)

        if key in existing_columns:

            col = existing_columns[key]

        else:

            col = sheet.max_column + 1

            sheet.cell(
                row=1,
                column=col,
                value=header
            )

            existing_columns[key] = col

        output_columns[header] = col

    total = sheet.max_row - 1

    passed = 0
    failed = 0
    processed = 0

    print("\n========== PHOTO VALIDATION ==========\n")

    for row_num in range(2, sheet.max_row + 1):

        name_cell = sheet.cell(row_num, name_col + 1)
        photo_cell = sheet.cell(row_num, photo_col + 1)

        name = name_cell.value
        photo_link = photo_cell.value

        # Use the hyperlink target if the cell contains a hyperlink.
        if photo_cell.hyperlink:
            photo_link = photo_cell.hyperlink.target or photo_link

        if not name and not photo_link:
            continue

        processed += 1
        result = process_photo(photo_link)

        # Save only status and failure reason in the spreadsheet.
        for header in OUTPUT_HEADERS:
            sheet.cell(
                row=row_num,
                column=output_columns[header],
                value=result.get(header, "")
            )

        if result["Validation Status"] == "PASS":
            passed += 1
            print(f"{processed}. {name or 'Unnamed'} - PASS")
        else:
            failed += 1
            print(f"{processed}. {name or 'Unnamed'} - FAIL")
            print(f"   Reason: {result['Failure Reason']}")

    # Save a separate output workbook
    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        Path(file_path).stem + "_validated" + extension
    )
    output_path = get_available_output_path(output_path)

    workbook.save(output_path)

    return output_path, passed, failed


# ============================================================
# 10. MAIN SPREADSHEET PROCESSING
# ============================================================

def process_spreadsheet(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".csv":

        return process_csv(file_path)

    elif extension in [".xlsx", ".xlsm"]:

        return process_excel(file_path)

    else:

        raise ValueError(
            "Unsupported file format.\n"
            "Select CSV, XLSX or XLSM."
        )
