"""Train a YOLO chess-piece detector on the GPU."""
import argparse
from pathlib import Path

# Available YOLO detection model families (YOLO26 needs ultralytics>=8.4.0) and sizes.
YOLO_VERSIONS = ("yolov8", "yolo11", "yolo26")
MODEL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_YOLO = "yolo11"

# Augmentations that generalize to steep, oblique and rotated camera angles. They are purely
# geometric, so class ids (piece type and color) are preserved; vertical flips stay disabled.
AUGMENTATION = {"perspective": 0.0005, "degrees": 180.0, "scale": 0.5}


def base_model(yolo: str = DEFAULT_YOLO, size: str = "n", model: str | None = None) -> str:
    """Weights to start training from: ``model`` if given, else e.g. ``yolo26n.pt``."""
    if model:
        return model
    if yolo not in YOLO_VERSIONS:
        raise ValueError(f"Unknown YOLO version: {yolo} (choose from {', '.join(YOLO_VERSIONS)})")
    if size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size: {size} (choose from {', '.join(MODEL_SIZES)})")
    return f"{yolo}{size}.pt"


def train_kwargs(args: argparse.Namespace, device) -> dict:
    """Keyword arguments for ``YOLO.train``; each run is saved under ``<project>/<name>``
    (``weights/best.pt``, ``results.csv`` metrics and the ``args.yaml`` run config)."""
    return {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": device,
        "perspective": args.perspective,
        "degrees": args.degrees,
        "scale": args.scale,
        "flipud": 0.0,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "exist_ok": True,
    }


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
    parser.add_argument("--data", default="chess.yaml", type=Path,
                        help="dataset yaml (choose any split / merged dataset)")
    parser.add_argument("--yolo", default=DEFAULT_YOLO, choices=YOLO_VERSIONS,
                        help="YOLO version of the base model")
    parser.add_argument("--size", default="n", choices=MODEL_SIZES,
                        help="model size: n(ano), s(mall), m(edium), l(arge), x(-large)")
    parser.add_argument("--model", default=None,
                        help="custom base model (.pt/.yaml); overrides --yolo and --size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="dataloader processes (0 avoids Windows multiprocessing memory errors)",
    )
    parser.add_argument("--perspective", type=float, default=AUGMENTATION["perspective"],
                        help="perspective augmentation (oblique camera angles)")
    parser.add_argument("--degrees", type=float, default=AUGMENTATION["degrees"],
                        help="rotation augmentation in degrees (any board rotation)")
    parser.add_argument("--scale", type=float, default=AUGMENTATION["scale"],
                        help="scale augmentation gain")
    parser.add_argument("--project", default="runs", type=Path, help="directory for training runs")
    parser.add_argument("--name", default="chess", help="run name (saved to <project>/<name>)")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    model_name = base_model(args.yolo, args.size, args.model)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    print(f"Base model: {model_name}")
    model = YOLO(model_name)
    model.add_callback("on_fit_epoch_end", print_metrics)
    model.train(**train_kwargs(args, device))
    print(f"Done. Best model: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
