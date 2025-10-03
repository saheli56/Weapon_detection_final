"""Simple detection runner using Ultralytics YOLO models.

This script provides a small wrapper to run inference on images and
save visualized results. Designed to be simple, testable, and
production-adaptable.
"""
from pathlib import Path
from typing import Iterable, Optional

def run_detection(model_path: str, inputs: Iterable[Path], output_dir: Optional[Path] = None, conf: float = 0.25):
    """Run detection with a YOLO model saved at `model_path`.

    Args:
        model_path: path-like spec that ultralytics will accept (e.g., 'yolov8n.pt' or local .pt/.yaml)
        inputs: iterable of Path objects pointing to images.
        output_dir: directory to write result images. If None, uses './runs/detect'.
        conf: confidence threshold.
    Returns:
        list of output file paths written.
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics package is required. Install with 'pip install ultralytics'") from e

    output_dir = Path(output_dir or Path.cwd() / "runs" / "detect").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    written = []
    for img in inputs:
        res = model.predict(source=str(img), conf=conf, save=True, save_txt=False)
        # ultralytics saves outputs under runs/detect automatically; move the latest file to our output_dir
        # res is a list of Results; pick the first result's path if available
        for r in res:
            # r.orig_img contains numpy array; r.path is attr in newer ultralytics
            out_path = getattr(r, 'path', None)
            if out_path is None:
                # fallback: ultralytics writes to default path pattern. We'll search for a new file in runs/detect
                # to keep this code robust, just append a note
                continue
            out = Path(out_path)
            target = output_dir / out.name
            if not target.exists():
                out.replace(target)
            written.append(target)

    return written

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run YOLO detection on sample images.")
    p.add_argument("--model", default="yolov8n.pt", help="Model spec for ultralytics YOLO (default: yolov8n.pt)")
    p.add_argument("--images", nargs='+', required=True, help="Image paths to run inference on")
    p.add_argument("--out", default=None, help="Output directory")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = p.parse_args()
    inputs = [Path(x) for x in args.images]
    outs = run_detection(args.model, inputs, Path(args.out) if args.out else None, args.conf)
    print("Wrote:")
    for o in outs:
        print(" -", o)
