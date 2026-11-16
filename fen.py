"""Track a chess game from one image per move and export annotated images + FEN strings.

The first frame must show the starting position; it calibrates the (stationary) camera.
"""
import argparse
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path) -> list:
    """Sort ``move_2.jpg`` before ``move_10.jpg``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_frames(sources: list[str]) -> list[Path]:
    """Image files from the given files/folders, folders sorted in natural order."""
    frames = []
    for source in map(Path, sources):
        if source.is_dir():
            frames += sorted((p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                             key=natural_key)
        else:
            frames.append(source)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Chess pieces -> board state -> FEN per move")
    parser.add_argument("--model", default="runs/chess/weights/best.pt", type=Path,
                        help="trained detector")
    parser.add_argument("--source", nargs="+", default=["0"],
                        help="webcam index (space = capture after a move, q = quit), "
                             "or image files / folders with one image per move")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--out", default="runs/fen/game", type=Path,
                        help="output folder for move_<num>.jpg and moves.fen")
    parser.add_argument("--ext", default=".jpg", choices=(".jpg", ".png"), help="image format")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model} (run train.py first)")

    import cv2
    import torch
    from ultralytics import YOLO

    from schaakfen import CalibrationError, FenSession, detections_from_result

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))
    session = FenSession(args.out, image_ext=args.ext)

    def process(frame) -> bool:
        result = model(frame, conf=args.conf, device=device, verbose=False)[0]
        try:
            out = session.process(frame, detections_from_result(result))
        except CalibrationError as e:
            print(f"Calibration failed: {e}")
            return False
        move = f" {out.result.san}" if out.result.san else ""
        print(f"{out.image_path}{move}: {out.fen}")
        return True

    webcam = len(args.source) == 1 and args.source[0].isdigit()
    if not webcam:
        for path in list_frames(args.source):
            frame = cv2.imread(str(path))
            if frame is None:
                raise SystemExit(f"Could not read image: {path}")
            if not process(frame):
                raise SystemExit("The first frame must show the full starting position")
        print(f"FEN log: {session.fen_log}")
        return

    cap = cv2.VideoCapture(int(args.source[0]))
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam: {args.source[0]}")
    print("Press space after every move to capture the board, q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Chess game (space = capture, q = quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            process(frame)
    cap.release()
    cv2.destroyAllWindows()
    print(f"FEN log: {session.fen_log}")


if __name__ == "__main__":
    main()
