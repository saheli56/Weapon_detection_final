"""Evaluate fine-tuned YOLO model on the test split and produce a JSON report.

Saves annotated images to ml/runs/real_test_detect and writes ml/test_results.json
with structured detections per image.
"""
from pathlib import Path
import json
from typing import List, Dict, Any


def run_eval(model_path: str, test_images_dir: Path, out_dir: Path, conf: float = 0.25) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required. Install via 'pip install ultralytics'") from e

    model = YOLO(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    imgs = sorted(test_images_dir.glob('*.jpg'))
    for img in imgs:
        res_list = []
        outs = model.predict(source=str(img), conf=conf, save=True)
        # Move saved image(s) into our out_dir and collect detections
        for r in outs:
            saved_path = getattr(r, 'path', None)
            if saved_path:
                saved = Path(saved_path)
                target = out_dir / saved.name
                if not target.exists():
                    try:
                        saved.replace(target)
                    except Exception:
                        # fallback: copy
                        import shutil
                        shutil.copy(saved, target)

            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for b in boxes:
                # b.cls, b.conf, b.xyxy
                cls = int(b.cls.cpu().numpy().item()) if hasattr(b, 'cls') else None
                confv = float(b.conf.cpu().numpy().item()) if hasattr(b, 'conf') else None
                xyxy = b.xyxy.cpu().numpy().tolist() if hasattr(b, 'xyxy') else None
                res_list.append({'class': cls, 'confidence': confv, 'bbox_xyxy': xyxy})

        results[str(img)] = res_list

    return results


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--testdir', default='ml/processed_dataset/test/images')
    p.add_argument('--out', default='ml/runs/real_test_detect')
    p.add_argument('--conf', type=float, default=0.25)
    args = p.parse_args()

    model = args.model
    testdir = Path(args.testdir)
    outdir = Path(args.out)

    report = run_eval(model, testdir, outdir, conf=args.conf)
    report_path = Path('ml') / 'test_results.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'Report written to {report_path}. Annotated images in {outdir}')
