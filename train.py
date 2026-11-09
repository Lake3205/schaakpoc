"""Train a YOLO chess-piece detector on the GPU."""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def print_metrics(trainer):
    m = trainer.metrics
    print(
        f"Epoch {trainer.epoch + 1}/{trainer.epochs}: "
        f"P={m.get('metrics/precision(B)', 0):.3f} "
        f"R={m.get('metrics/recall(B)', 0):.3f} "
        f"mAP50={m.get('metrics/mAP50(B)', 0):.3f} "
        f"mAP50-95={m.get('metrics/mAP50-95(B)', 0):.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train a chess piece detector")
    parser.add_argument("--data", default="chess.yaml", help="dataset yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="base model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="dataloader processes (0 avoids Windows multiprocessing memory errors)",
    )
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", print_metrics)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(Path("runs").resolve()),
        name="chess",
        exist_ok=True,
    )
    print(f"Done. Best model: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
