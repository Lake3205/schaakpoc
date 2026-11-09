"""Evaluate the trained chess model on the labeled test set (datasets/chess/test).

Default mode prints test metrics. Use --show to visually check random test images:
model predictions (left) next to the ground-truth labels (right).
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEST_DIR = Path("datasets/chess/test")


def run_metrics(model, args, device):
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=device,
        workers=args.workers,
        project=str(Path("runs").resolve()),
        name="chess_test",
        exist_ok=True,
        plots=True,
    )

    box = metrics.box
    print("\n=== Test results ===")
    print(f"Precision : {box.mp:.4f}")
    print(f"Recall    : {box.mr:.4f}")
    print(f"mAP50     : {box.map50:.4f}")
    print(f"mAP50-95  : {box.map:.4f}")

    print(f"\n{'Class':<15}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        print(f"{metrics.names[int(cls)]:<15}{p:>8.3f}{r:>8.3f}{ap50:>8.3f}{ap:>10.3f}")

    print(f"\nPlots and confusion matrix saved to: {metrics.save_dir}")
    return metrics


def draw_ground_truth(image, label_path, names):
    """Draw YOLO-format ground-truth boxes (class cx cy w h, normalized) on a copy of the image."""
    out = image.copy()
    h, w = out.shape[:2]
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, names.get(cls, str(cls)), (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
    return out


def add_title(image, title):
    bar = np.zeros((30, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def run_show(model, args, device):
    images = sorted(p for p in (TEST_DIR / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {TEST_DIR / 'images'}")

    rng = random.Random(args.seed)
    picks = rng.sample(images, min(args.num, len(images)))
    print(f"Showing {len(picks)} random test images. Any key = next, q = quit.")

    for i, path in enumerate(picks, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"Could not read image: {path}")
            continue

        result = model(image, conf=args.show_conf, device=device, verbose=False)[0]
        pred = add_title(result.plot(), f"Prediction ({len(result.boxes)})")

        label_path = TEST_DIR / "labels" / f"{path.stem}.txt"
        n_gt = len(label_path.read_text().splitlines()) if label_path.exists() else 0
        truth = add_title(draw_ground_truth(image, label_path, result.names), f"Ground truth ({n_gt})")

        print(f"[{i}/{len(picks)}] {path.name}: predicted {len(result.boxes)}, labeled {n_gt}")
        cv2.imshow("Prediction vs ground truth (any key = next, q = quit)", np.hstack([pred, truth]))
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test the chess model on the test set")
    parser.add_argument("--model", default="runs/chess/weights/best.pt", help="trained model")
    parser.add_argument("--data", default="chess.yaml", help="dataset config with a 'test' entry")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="confidence threshold for metrics")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--show", action="store_true", help="show random test images with boxes instead of metrics")
    parser.add_argument("--num", type=int, default=10, help="number of images to show with --show")
    parser.add_argument("--show-conf", type=float, default=0.5, help="confidence threshold for --show")
    parser.add_argument("--seed", type=int, default=None, help="random seed for --show (repeatable picks)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model} (run train.py first)")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    model = YOLO(args.model)

    if args.show:
        run_show(model, args, device)
    else:
        run_metrics(model, args, device)


if __name__ == "__main__":
    main()
