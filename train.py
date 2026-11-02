"""Train a YOLO chess-piece detector on the GPU."""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train a chess piece detector")
    parser.add_argument("--data", default="chess.yaml", help="dataset yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="base model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(Path("runs").resolve()),
        name="chess",
        exist_ok=True,
    )
    print(f"Done. Best model: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
