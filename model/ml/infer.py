"""Run inference programmatically and report structured results.

This script loads a YOLO model via ultralytics and prints detections
in a concise table: image, class_name, confidence, bbox.
"""
from pathlib import Path
from typing import List, Dict, Any


def infer(model_spec: str, images: List[Path], conf_thres: float = 0.25) -> Dict[Path, List[Dict[str, Any]]]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required. Install via 'pip install ultralytics'") from e

    model = YOLO(model_spec)
    results = {}
    for img in images:
        res_list = []
        out = model.predict(source=str(img), conf=conf_thres)
        for r in out:
            # r.boxes is a Boxes object; convert to simple dicts
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for b in boxes:
                cls = int(b.cls.cpu().numpy()) if hasattr(b, 'cls') else None
                conf = float(b.conf.cpu().numpy()) if hasattr(b, 'conf') else None
                xyxy = b.xyxy.cpu().numpy().tolist() if hasattr(b, 'xyxy') else None
                res_list.append({
                    'class': cls,
                    'confidence': conf,
                    'bbox_xyxy': xyxy,
                })
        results[img] = res_list
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--model', default='yolov8n.pt')
    p.add_argument('images', nargs='+')
    p.add_argument('--conf', type=float, default=0.25)
    args = p.parse_args()

    images = [Path(p) for p in args.images]
    res = infer(args.model, images, args.conf)
    for img, detections in res.items():
        print(f"Image: {img}")
        if not detections:
            print("  No detections")
            continue
        for d in detections:
            print(f"  class={d['class']} conf={d['confidence']:.3f} bbox={d['bbox_xyxy']}")
