"""
Maatram Foundation - Individual Photo Validation Logic
======================================================
Original validation algorithm preserved exactly.
Only change: file input comes from a bytes/numpy array
instead of Tkinter filedialog.
"""

import cv2
from PIL import Image, ImageOps
import numpy as np
import os

# ---------------------------------------------------------------------------
# Professional photo standards
# ---------------------------------------------------------------------------
TARGET_RATIO = 4 / 5
RATIO_TOLERANCE = 0.05

MIN_WIDTH = 400
MIN_HEIGHT = 500
MAX_WIDTH = 4000
MAX_HEIGHT = 5000

# Preferred image-quality thresholds
MIN_BRIGHTNESS = 70
MAX_BRIGHTNESS = 200
MIN_CONTRAST = 25
MIN_SHARPNESS = 80
MAX_CLIPPED_PIXEL_RATIO = 0.40  # Increased from 20% to 40%

# Tolerance limits for minor deviations
MAX_RATIO_DEVIATION = 0.10
MIN_ALLOWED_BRIGHTNESS = 50
MAX_ALLOWED_BRIGHTNESS = 220
MIN_ALLOWED_CONTRAST = 15
MIN_ALLOWED_SHARPNESS = 30
MAX_ALLOWED_CLIPPED_PIXEL_RATIO = 0.40

# ---------------------------------------------------------------------------
# YuNet model path – same directory as this file
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_detection_yunet_2023mar.onnx"
)


def validate_photo(file_bytes: bytes, filename: str) -> dict:
    """
    Run the original Maatram Foundation photo validation on *file_bytes*.

    Parameters
    ----------
    file_bytes : bytes   Raw image file bytes received from the HTTP upload.
    filename   : str     Original filename (used in the response only).

    Returns
    -------
    dict  Structured result matching the original script's output.
          Raises ValueError / FileNotFoundError for known error conditions.
    """

    # ------------------------------------------------------------------
    # Step 1: Load image and correct EXIF orientation
    # (Original: Image.open(file_path) → now: Image.open(BytesIO(bytes)))
    # ------------------------------------------------------------------
    import io
    with Image.open(io.BytesIO(file_bytes)) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image)
        image = np.array(pil_image.convert("RGB"))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width = image.shape[:2]
    file_size_kb = len(file_bytes) / 1024
    aspect_ratio = width / height

    # ------------------------------------------------------------------
    # Step 2: Orientation check
    # ------------------------------------------------------------------
    is_portrait = height > width

    # ------------------------------------------------------------------
    # Step 3: Dimension checks
    # ------------------------------------------------------------------
    min_dimensions_ok = width >= MIN_WIDTH and height >= MIN_HEIGHT
    max_dimensions_ok = width <= MAX_WIDTH and height <= MAX_HEIGHT

    # ------------------------------------------------------------------
    # Step 4: Aspect ratio check
    # ------------------------------------------------------------------
    ratio_difference = abs(aspect_ratio - TARGET_RATIO) / TARGET_RATIO
    ratio_ok = ratio_difference <= RATIO_TOLERANCE

    # ------------------------------------------------------------------
    # Step 5: Image-quality analysis
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Average brightness
    brightness = float(np.mean(gray))

    # Contrast
    contrast = float(np.std(gray))

    # Sharpness
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Near-black and near-white pixels
    dark_pixel_ratio  = float(np.mean(gray <= 1))
    bright_pixel_ratio = float(np.mean(gray >= 254))
    clipped_pixel_ratio = dark_pixel_ratio + bright_pixel_ratio

    # Pixel intensity percentiles
    p5, p50, p95 = np.percentile(gray, [5, 50, 95])

    # Preferred quality checks
    brightness_ok = MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
    contrast_ok   = contrast >= MIN_CONTRAST
    sharpness_ok  = sharpness >= MIN_SHARPNESS
    clipping_ok   = clipped_pixel_ratio <= MAX_CLIPPED_PIXEL_RATIO

    # ------------------------------------------------------------------
    # Step 6: Load YuNet face detector
    # ------------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "YuNet model not found. Place "
            "face_detection_yunet_2023mar.onnx "
            "in the backend/ folder."
        )

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (width, height),
        0.80,
        0.30,
        5000
    )

    # ------------------------------------------------------------------
    # Step 7: Detect faces
    # ------------------------------------------------------------------
    _, faces = detector.detect(image)

    face_count = 0 if faces is None else len(faces)
    face_ok = face_count == 1

    # ------------------------------------------------------------------
    # Step 8: Final decision
    # ------------------------------------------------------------------
    checks = {
        "Portrait orientation":    is_portrait,
        "Minimum dimensions":      min_dimensions_ok,
        "Maximum dimensions":      max_dimensions_ok,
        "Aspect ratio":            ratio_ok,
        "Brightness":              brightness_ok,
        "Contrast":                contrast_ok,
        "Sharpness":               sharpness_ok,
        "Pixel clipping":          clipping_ok,
        "Exactly one face detected": face_ok,
    }

    # Structural requirements remain mandatory
    structural_ok = (
        is_portrait
        and min_dimensions_ok
        and max_dimensions_ok
        and face_ok
    )

    # Accept minor deviations in quality measurements
    minor_deviation_ok = (
        ratio_difference <= MAX_RATIO_DEVIATION
        and MIN_ALLOWED_BRIGHTNESS <= brightness <= MAX_ALLOWED_BRIGHTNESS
        and contrast >= MIN_ALLOWED_CONTRAST
        and sharpness >= MIN_ALLOWED_SHARPNESS
        and clipped_pixel_ratio <= MAX_ALLOWED_CLIPPED_PIXEL_RATIO
    )

    preferred_quality_ok = all([
        ratio_ok,
        brightness_ok,
        contrast_ok,
        sharpness_ok,
        clipping_ok,
    ])

    if structural_ok and preferred_quality_ok:
        decision = "ACCEPT PHOTO"
        accepted_via = "preferred"
        rejection_reasons = []
    elif structural_ok and minor_deviation_ok:
        decision = "ACCEPT PHOTO"
        accepted_via = "minor_deviation"
        rejection_reasons = []
    else:
        decision = "REJECT PHOTO"
        accepted_via = None
        rejection_reasons = []

        if not is_portrait:
            rejection_reasons.append("Photo must be in portrait orientation.")
        if not min_dimensions_ok:
            rejection_reasons.append("Photo is below the minimum dimensions.")
        if not max_dimensions_ok:
            rejection_reasons.append("Photo exceeds the maximum dimensions.")
        if not face_ok:
            rejection_reasons.append("Exactly one face must be detected.")
        if ratio_difference > MAX_RATIO_DEVIATION:
            rejection_reasons.append(
                "Aspect ratio deviation exceeds the allowed tolerance."
            )
        if not (MIN_ALLOWED_BRIGHTNESS <= brightness <= MAX_ALLOWED_BRIGHTNESS):
            rejection_reasons.append("Brightness exceeds the allowed tolerance.")
        if contrast < MIN_ALLOWED_CONTRAST:
            rejection_reasons.append("Contrast is too low.")
        if sharpness < MIN_ALLOWED_SHARPNESS:
            rejection_reasons.append("Sharpness is too low.")
        if clipped_pixel_ratio > MAX_ALLOWED_CLIPPED_PIXEL_RATIO:
            rejection_reasons.append(
                "Combined extreme pixels exceed the allowed 20%."
            )

    # ------------------------------------------------------------------
    # Return structured result
    # ------------------------------------------------------------------
    return {
        "filename":               filename,
        "decision":               decision,          # "ACCEPT PHOTO" | "REJECT PHOTO"
        "accepted_via":           accepted_via,       # "preferred" | "minor_deviation" | None
        "width":                  width,
        "height":                 height,
        "file_size_kb":           round(file_size_kb, 2),
        "aspect_ratio":           round(aspect_ratio, 4),
        "orientation":            (
            "Portrait" if is_portrait
            else "Landscape" if width > height
            else "Square"
        ),
        "brightness":             round(brightness, 2),
        "contrast":               round(contrast, 2),
        "sharpness":              round(sharpness, 2),
        "dark_pixel_ratio":       round(dark_pixel_ratio * 100, 2),   # as %
        "bright_pixel_ratio":     round(bright_pixel_ratio * 100, 2), # as %
        "clipped_pixel_ratio":    round(clipped_pixel_ratio * 100, 2),# as %
        "p5":                     round(float(p5), 1),
        "p50":                    round(float(p50), 1),
        "p95":                    round(float(p95), 1),
        "face_count":             face_count,
        "checks":                 checks,            # dict name → bool (original preferred checks)
        "preferred_quality_ok":   preferred_quality_ok,
        "minor_deviation_ok":     minor_deviation_ok,
        "structural_ok":          structural_ok,
        "rejection_reasons":      rejection_reasons,
    }
