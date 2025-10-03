Weapon detection ML helper scripts
=================================

This folder contains small, mature scripts to run detection and training using Ultralytics YOLOv8.

Quick start
-----------

- Create and activate your venv (already done in the workspace):

  PowerShell:

  ```powershell
  C:\luck\.venv_ml\Scripts\Activate.ps1
  ```

- Install dependencies (if not already installed):

  ```powershell
  pip install -r ml/requirements.txt
  ```

- Run detection on images:

  ```powershell
  python -m ml.detect --model yolov8n.pt --images samples/gun1.jpg samples/gun2.jpg
  ```

- Train on a YOLO-format dataset by providing a `data.yaml` that points to your train/val folders:

  ```powershell
  python -m ml.train --model yolov8n.pt --data data.yaml --epochs 50
  ```

Notes
-----
- These scripts use the `ultralytics` package which bundles YOLOv8.
- For custom weapon datasets, prepare labels in YOLO format (one .txt per image with class and bbox in normalized coords).
