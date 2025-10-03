"""Training scaffold for weapon detection using Ultralytics YOLOv8.

This script expects a dataset in the YOLO format with a dataset .yaml file
that points to train/val image directories and class names.

It's intentionally small: it calls the ultralytics `model.train()` API with
reasonable defaults while exposing common arguments.
"""
from pathlib import Path
from typing import Optional


def train(model: str = "yolov8n.pt", data: str = "data.yaml", epochs: int = 50, imgsz: int = 640, batch: int = 16):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics package is required. Install with 'pip install ultralytics'") from e

    model = YOLO(model)
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train YOLOv8 on a YOLO-format dataset")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)

    args = p.parse_args()
    train(args.model, args.data, args.epochs, args.imgsz, args.batch)
