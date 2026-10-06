import os
import re
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
YOLOV7_DIR = ROOT / "yolov7"
if str(YOLOV7_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV7_DIR))

from yolov7.utils.datasets import letterbox

# PyTorch 2.x compatibility for legacy YOLOv7 weights.
_ORIGINAL_TORCH_LOAD = torch.load

def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _compat_torch_load

from models.experimental import attempt_load  # noqa: E402
from utils.general import non_max_suppression, scale_coords  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR = ROOT / "uploads"
RESULT_DIR = ROOT / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

allowed_extensions = {".jpg", ".jpeg", ".png"}


def resolve_model_path():
    model_path = ROOT / "model" / "YOLOv7.pt"
    if model_path.exists():
        return str(model_path)
    raise FileNotFoundError("YOLOv7 model not found in model/YOLOv7.pt")


def detect_device_label():
    return "NVIDIA GPU" if torch.cuda.is_available() else "CPU"


def device_for_model():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_class_name(name):
    value = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if "helmet" in value:
        return "helmet"
    if "reflective" in value and "jacket" in value:
        return "reflective_jacket"
    return value if value else "unknown"


MODEL = None


def load_model_once():
    global MODEL
    if MODEL is not None:
        return MODEL

    path = resolve_model_path()
    device = device_for_model()
    model = attempt_load(path, map_location=device)
    model.to(device)
    model.eval()
    MODEL = model
    return MODEL


load_model_once()


@app.route("/")
def index():
    return render_template("index.html", device_label=detect_device_label())


@app.route("/results/<path:filename>")
def result_image(filename):
    return send_from_directory(RESULT_DIR, filename)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Please upload an image."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Please upload an image."}), 400

        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        if ext not in allowed_extensions:
            return jsonify({"success": False, "error": "Unsupported image format. Please upload JPG or PNG."}), 400

        try:
            image = Image.open(file)
            image.verify()
            file.stream.seek(0)
        except Exception:
            return jsonify({"success": False, "error": "The uploaded file is not a valid image."}), 400

        unique_name = f"{uuid.uuid4().hex}{ext}"
        upload_path = UPLOAD_DIR / unique_name
        file.save(upload_path)

        img_bgr = cv2.imread(str(upload_path))
        if img_bgr is None:
            return jsonify({"success": False, "error": "Unable to read the uploaded image."}), 400

        conf_threshold = float(request.args.get("conf", 0.50))
        conf_threshold = max(0.0, min(conf_threshold, 1.0))

        detections, result_image_bgr = run_inference(img_bgr, conf_threshold)
        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = RESULT_DIR / result_filename
        success = cv2.imwrite(str(result_path), result_image_bgr)
        if not success:
            return jsonify({"success": False, "error": "Failed to save the detection image."}), 500

        stats = summarize_detections(detections)
        response = {
            "success": True,
            "image_url": f"/results/{result_filename}",
            "detections": detections,
            "statistics": stats,
        }
        return jsonify(response)
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    except Exception as exc:
        app.logger.exception("YOLOv7 detection failed")
        return jsonify({"success": False, "error": "Unable to process the image."}), 500


def run_inference(image_bgr, conf_threshold):
    model = load_model_once()
    device = device_for_model()
    names = model.names if hasattr(model, "names") else ["object"]

    img = image_bgr.copy()
    img_resized, _, _ = letterbox(img, new_shape=(640, 640), auto=False, stride=32)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]

    pred = non_max_suppression(pred, conf_thres=conf_threshold, iou_thres=0.45)
    pred = pred[0]

    detections = []
    output_image = image_bgr.copy()
    if pred is not None:
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], image_bgr.shape).round()
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            class_id = int(cls)
            class_name = names[class_id] if class_id < len(names) else "Unknown"
            confidence = float(conf)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (67, 216, 182), 2)
            label = f"{class_name} {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(output_image, (x1, max(0, y1 - label_h - 10)), (x1 + label_w + 10, y1), (67, 216, 182), -1)
            cv2.putText(output_image, label, (x1 + 5, max(10, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    return detections, output_image


def summarize_detections(detections):
    stats = {"total": len(detections), "helmet": 0, "reflective_jacket": 0}
    confidences = []

    for item in detections:
        confidence = float(item["confidence"])
        confidences.append(confidence)
        class_name = item["class_name"]
        key = normalize_class_name(class_name)
        if key == "helmet":
            stats["helmet"] += 1
        elif key == "reflective_jacket":
            stats["reflective_jacket"] += 1

    stats["average_confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    stats["highest_confidence"] = round(max(confidences), 4) if confidences else 0.0
    return stats


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
