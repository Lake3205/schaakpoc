# schaakpoc

Simple chess piece detector in Python using **OpenCV** (camera / images / drawing)
and **YOLO** (Ultralytics) for detection, running on an NVIDIA RTX 5070 GPU.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Train a chess-piece detector on your GPU |
| `detect.py` | Detect pieces from a webcam, image or video |
| `detect_test.py` | Evaluate the trained model on the labeled test set |
| `chess.yaml` | Dataset config (12 classes: white/black king, queen, rook, bishop, knight, pawn) |
| `app.py` | **v2** desktop GUI (PyQt6) to train chess-piece *image classifiers* (see section 6) |
| `schaakgui/` | Code of the v2 GUI: projects, datasets, augmentations, quotas, checkpoints, training thread |

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
python train.py              # defaults: 50 epochs, 640px, batch 16, workers 0
python train.py --epochs 100 --batch 32
```

`workers=0` avoids Windows multiprocessing memory errors. You can increase it
with `--workers 2` if your system has enough memory.

After every epoch a summary line is printed with precision (P), recall (R),
mAP50 and mAP50-95 on the validation set. All per-epoch metrics are also saved
to `runs/chess/results.csv`.

The trained model is saved to `runs/chess/weights/best.pt`.

## 4. Detect

```bash
python detect.py                          # webcam 0, press q to quit
python detect.py --source board.jpg       # single image, press any key to close
python detect.py --source game.mp4        # video file
python detect.py --conf 0.3               # lower confidence threshold
```

Detected pieces are drawn on the image and printed in the terminal.

## 5. Test

Put a labeled test set in `datasets/chess/test/images` and `datasets/chess/test/labels`, then:

```bash
python detect_test.py
```

This prints precision, recall, mAP50 and mAP50-95 (overall and per class) and
saves plots and the confusion matrix to `runs/chess_test`.

To check the model yourself, open 10 random test images with boxes drawn
(prediction on the left, ground-truth labels on the right; any key = next, q = quit):

```bash
python detect_test.py --show
python detect_test.py --show --num 20 --show-conf 0.3 --seed 42
```

## 6. schaakpoc v2: training GUI for piece classifiers

`app.py` opens a desktop window for training image classifiers (one cropped piece per image)
with PyTorch/torchvision:

```bash
python app.py                      # workspace projects are stored in ./.projects
python app.py --projects D:/work   # use another workspace directory
```

Dataset folders use the `ImageFolder` layout, one sub-folder per class:

```
my_pieces/white-king/001.jpg
my_pieces/black-pawn/xyz.png
```

Features:

* **Projects** – create, switch, save and delete projects (top bar). Each project is a JSON file in
  `.projects/` and keeps its datasets, augmentations, quotas, checkpoints (`.projects/<name>/checkpoints/run_*`)
  and evaluation records.
* **Datasets** – add multiple training pools and test/validation folders and check the ones to use.
  Shuffle can be toggled and a fixed seed set. A bar chart shows the class balance of the checked pools.
  Without test folders, 20% of the training pool is held out for validation.
* **Augmentation** – checkboxes for horizontal/vertical flip, rotation, color jitter, affine/shear and
  random resized crop. They are composed into a `transforms.Compose` used only for the training split.
* **Training & Quotas** – epochs, batch size, learning rate, image size, architecture; accuracy floor
  (per epoch) and moving-average target, confidence floor (mean top-1 confidence per epoch) and moving-average
  target. Training stops (optional) once all targets are met. Full weights are exported every *N* epochs
  (`snapshots/epoch_XXXX.pt`) and early stopping halts training when validation loss/accuracy does not improve
  for *patience* epochs.
* **Checkpoints & Provenance** – resume training from any `.pt`/`.pth` checkpoint. Every checkpoint stores
  the cumulative epochs, quota milestones (e.g. "Achieved 82.0% Val Acc Floor (target 80%) at Epoch 15"),
  the folders used in each session, snapshots and the best weights. Selecting a model shows this in the
  *Model Provenance* card with achieved/pending milestone cards.
* **Live training** – training runs in a `QThread`; batch and run progress bars, a live log and
  Pause/Resume/Stop buttons.
* **Manual Review** – while paused or after training, shows random validation images with ground truth vs.
  top-3 predictions (green border = correct, red = incorrect) with Previous/Next buttons.
* **Analytics** – confusion matrix, normalized confusion matrix, instance count, precision per epoch and
  precision/recall vs. confidence threshold, updated after every validation step.

Run the tests with:

```bash
python -m pytest tests
```
