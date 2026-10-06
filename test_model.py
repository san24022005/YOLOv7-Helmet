import os
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent
YOLOV7_DIR = ROOT / "yolov7"
if str(YOLOV7_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV7_DIR))

_ORIGINAL_TORCH_LOAD = torch.load


def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _compat_torch_load

from models.experimental import attempt_load  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402


def resolve_model_path():
    candidates = [ROOT / "model" / "YOLOv7.pt", ROOT / "model" / "model.pt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No YOLOv7.pt or model.pt file found in the model folder.")


def main():
    model_path = resolve_model_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Model loaded successfully")
    print("Model: YOLOv7")
    print(f"Device: {'NVIDIA GPU' if torch.cuda.is_available() else 'CPU'}")

    model = attempt_load(str(model_path), map_location=device)
    model.to(device)
    model.eval()

    names = model.names if hasattr(model, "names") else []
    print("Classes:")
    for index, name in enumerate(names):
        print(f"{index} - {name}")

    test_image = YOLOV7_DIR / "inference" / "images" / "zidane.jpg"
    if not test_image.exists():
        print("No test image available in yolov7/inference/images.")
        return

    image = cv2.imread(str(test_image))
    if image is None:
        raise RuntimeError(f"Could not read test image: {test_image}")

    image_resized, _, _ = __import__('yolov7.utils.datasets', fromlist=['letterbox']).letterbox(image, new_shape=(640, 640), auto=False, stride=32)
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor, augment=False)[0]

    predictions = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)[0]
    print("Test inference successful")
    print(f"Detections: {0 if predictions is None else len(predictions)}")

    if predictions is not None:
        for item in predictions:
            conf = float(item[4])
            cls_id = int(item[5])
            print(f"{names[cls_id]} {conf:.2f}")


if __name__ == "__main__":
    main()
