# Safety Equipment Detection with YOLOv7

This project runs a Flask web app that uses a pretrained YOLOv7 model to detect safety equipment such as helmets and reflective jackets from uploaded images.

## Project structure

```text
YOLOv7_Project/
├── model/
│   ├── YOLOv7.pt   # preferred file name if your weights are renamed
│   └── model.pt    # current file in this workspace
├── yolov7/
│   └── ...         # official YOLOv7 source code used for inference
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── uploads/
├── results/
├── app.py
├── test_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Model loading

The app checks for the model in this order:

1. `model/YOLOv7.pt`
2. `model/model.pt`

This keeps compatibility with the current workspace while respecting the requirement that the original model file not be overwritten.

## Run locally in VS Code

### 1) Open the project in VS Code

### 2) Create a virtual environment

```powershell
python -m venv .venv
```

### 3) Activate the environment

```powershell
.venv\Scripts\activate
```

### 4) Install dependencies

```powershell
pip install -r requirements.txt
```

### 5) Run the app

```powershell
python app.py
```

### 6) Open the browser

```text
http://127.0.0.1:5000
```

## Test the model directly

```powershell
python test_model.py
```

This script loads the YOLOv7 checkpoint, prints the device, class names, and runs a quick inference test using the sample images bundled with the YOLOv7 repository.

## Notes

- The model is loaded once at application startup for faster inference.
- CUDA is used automatically when available; otherwise CPU is used.
- The app does not retrain or replace the original model.
