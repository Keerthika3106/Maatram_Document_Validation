# Maatram Foundation – Backend

## Setup

### 1. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Place the YuNet model

Copy `face_detection_yunet_2023mar.onnx` into the `backend/` folder:

```
maatram-foundation/
└── backend/
    ├── app.py
    ├── photo_validator.py
    ├── requirements.txt
    └── face_detection_yunet_2023mar.onnx   ← place here
```

Download from: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

### 3. Start the backend

```bash
cd backend
python app.py
```

Backend runs on **http://localhost:5000**

### 4. Start the frontend (separate terminal)

```bash
cd ..        # project root
npm run dev
```

Frontend runs on **http://localhost:5173** and proxies `/api/*` to `localhost:5000`.

---

## API

### POST /api/photo/validate

Upload a photo for validation.

**Request:** `multipart/form-data` with field `file` (JPG, JPEG or PNG).

**Response (200):**
```json
{
  "filename": "photo.jpg",
  "decision": "ACCEPT PHOTO",
  "accepted_via": "preferred",
  "width": 800,
  "height": 1000,
  "file_size_kb": 245.6,
  "aspect_ratio": 0.8,
  "orientation": "Portrait",
  "brightness": 142.5,
  "contrast": 48.3,
  "sharpness": 120.4,
  "dark_pixel_ratio": 0.1,
  "bright_pixel_ratio": 0.2,
  "clipped_pixel_ratio": 0.3,
  "p5": 60.0,
  "p50": 140.0,
  "p95": 220.0,
  "face_count": 1,
  "checks": {
    "Portrait orientation": true,
    "Minimum dimensions": true,
    "Maximum dimensions": true,
    "Aspect ratio": true,
    "Brightness": true,
    "Contrast": true,
    "Sharpness": true,
    "Pixel clipping": true,
    "Exactly one face detected": true
  },
  "preferred_quality_ok": true,
  "minor_deviation_ok": true,
  "structural_ok": true,
  "rejection_reasons": []
}
```

`accepted_via`:
- `"preferred"` – all preferred quality checks passed
- `"minor_deviation"` – some preferred checks failed but values are within tolerance
- `null` – photo rejected

### GET /api/health

Returns `{ "status": "ok", "yunet_model": "present" | "MISSING" }`.
