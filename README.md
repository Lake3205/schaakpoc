# schaakpoc

Simple chess piece detector in Python using **OpenCV** (camera / images / drawing)
and **YOLO** (Ultralytics) for detection, running on an NVIDIA RTX 5070 GPU.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Train a chess-piece detector on your GPU |
| `detect.py` | Detect pieces from a webcam, image or video |
| `chess.yaml` | Dataset config (12 classes: white/black king, queen, rook, bishop, knight, pawn) |

## 1. Setup (Windows or Linux, Python 3.10+)

The RTX 5070 (Blackwell) needs PyTorch built with **CUDA 12.8 or newer**.
Make sure you have a recent NVIDIA driver installed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check that the GPU is found:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Get a dataset

Download a chess pieces dataset in **YOLO format**, e.g. one of the free
chess pieces datasets on [Roboflow Universe](https://universe.roboflow.com/search?q=chess+pieces)
(export as "YOLOv8"), and unzip it so you get:

```
datasets/chess/train/images   datasets/chess/train/labels
datasets/chess/valid/images   datasets/chess/valid/labels
```

If your dataset uses different class names/order, update `names` in `chess.yaml`
(or just point `--data` at the `data.yaml` that comes with the dataset).

## 3. Train

```bash
python train.py              # defaults: 50 epochs, 640px, batch 16
python train.py --epochs 100 --batch 32
```

The trained model is saved to `runs/chess/weights/best.pt`.

## 4. Detect

```bash
python detect.py                          # webcam 0, press q to quit
python detect.py --source board.jpg       # single image, press any key to close
python detect.py --source game.mp4        # video file
python detect.py --conf 0.3               # lower confidence threshold
```

Detected pieces are drawn on the image and printed in the terminal.
