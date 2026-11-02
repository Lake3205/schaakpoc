"""Detect chess pieces with OpenCV + YOLO (runs on the GPU when available)."""
import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Detect chess pieces")
    parser.add_argument("--model", default="runs/chess/weights/best.pt", help="trained model")
    parser.add_argument("--source", default="0", help="webcam index, image or video path")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model} (run train.py first)")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    model = YOLO(args.model)

    def detect(frame):
        result = model(frame, conf=args.conf, device=device, verbose=False)[0]
        for box in result.boxes:
            name = result.names[int(box.cls)]
            print(f"{name}: {float(box.conf):.2f}")
        return result.plot()

    # Single image
    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.source}")
        cv2.imshow("Chess pieces", detect(frame))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Webcam or video
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Chess pieces (press q to quit)", detect(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
