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
| `app.py` | **v3** desktop GUI (PyQt6) to train *object detectors* (default) or *image classifiers* (see section 6) |
| `schaakgui/` | Code of the GUI: projects, datasets, augmentations, detector/classifier training, quotas, checkpoints |
| `scan.py` | Garden chess scanner on the command line: photo → FEN, annotated photo, board diagram (see section 7) |

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

Switch between the available YOLO models with `--yolo` (`yolov8`, `yolo11` (default) or `yolo26`)
and `--size` (`n`, `s`, `m`, `l`, `x`). The pretrained weights are downloaded automatically:

```bash
python train.py --yolo yolo26              # YOLO26 nano (yolo26n.pt)
python train.py --yolo yolo26 --size s     # YOLO26 small (yolo26s.pt)
python train.py --model my_weights.pt      # any other base model (overrides --yolo/--size)
```

YOLO26 needs `ultralytics>=8.4.0` (`pip install -U ultralytics`).

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

`app.py` opens a desktop window for training models on your own datasets (not only chess):

```bash
python app.py                      # workspace projects are stored in ./.projects
python app.py --projects D:/work   # use another workspace directory
```

The **Task** selector in the top bar chooses what a project trains (stored per project):

* **Detection (default)** – Ultralytics YOLO object detection: many objects per image, each with its own
  box and class. Trains on YOLO folders (`images/` + `labels/`, class names from `data.yaml`).
* **Classification** – one label per image (or per cropped YOLO box) with PyTorch/torchvision.

### Detection task (default)

* **Datasets** – check YOLO training, validation and test folders (add each split, e.g.
  `datasets/chess/train`, separately; folders without `labels/` are ignored). All folders must use the
  same class names: run **🔄 Synchronize YAML class names…** first if they differ. Without validation
  folders the training folders are used for validation.
* **Training & Quotas → Detection training** – base model (default `yolo26m.pt`, downloaded
  automatically), epochs, image size, batch size, early stopping *patience* (stops when mAP has not
  improved for that many epochs; 0 = off), save weights every *N* epochs, data loader workers and mosaic.
  There are no quotas for detection.
* **Augmentation** – the same checkboxes, applied box-aware by Ultralytics: flips → `fliplr`/`flipud`,
  rotation/affine degrees → `degrees`, color jitter → `hsv_h/s/v` (brightness, saturation; contrast is not
  available), affine → `translate`/`shear`, resized crop → `scale = 1 − minimum scale`. The preview line
  shows the resulting settings.
* **▶ Start / ⏸ Pause / ⏹ Stop** – training runs in a background thread. Pause stops after the current
  batch and saves the current weights (`weights/paused.pt`) so Manual Review and Test own picture can
  inspect them; Stop finishes after the current batch and keeps the best weights.
* **Results** – every run is saved in `.projects/<name>/checkpoints/detectors/detector_<stamp>/`
  (`weights/best.pt`, `last.pt`, Ultralytics plots, `results.csv` and `schaakgui_run.json` with the
  provenance: data folders, base model, augmentation, metrics). The *Checkpoints & Provenance* tab lists
  the project's detectors; **Continue from** fine-tunes a new run from one of them, and **Import
  detector…** copies existing YOLO detector weights (e.g. `runs/chess/weights/best.pt` from `train.py`)
  into the project (classifier checkpoints are rejected).
* **Live analytics** – precision, recall, mAP50 and mAP50-95 per epoch, the confusion matrix (with a
  *background* class for missed and extra boxes), the normalized matrix with Δ against a baseline, and
  precision/recall vs. confidence.
* **🎓 Final exam** – evaluates the selected detector on the checked test folders (precision, recall, mAP,
  per-class results, confusion matrix; recorded in the project).
* **🔍 Manual Review** – random labeled images with the prediction (left) next to the ground truth
  (right): green = correct box, orange = wrong class / extra box / missed object, plus the counts.
* **🖼 Test own picture** – pick a **Detector** (the project's detectors, newest or selected first, plus
  *➕ Add detector file…*); it finds and names every piece by itself, no classifier needed. Click or
  drop a picture: every object gets a
  numbered box, listed on the right. The *Min. confidence* slider trades missed objects against false
  boxes. Mapping boxes to board squares (8×8 grid) is a possible later, optional feature.

### Classification task

Dataset folders use the `ImageFolder` layout, one sub-folder per class:

```
my_pieces/white-king/001.jpg
my_pieces/black-pawn/xyz.png
```

YOLO splits (the detector dataset from section 2) can be added directly, e.g. `datasets/chess/train`
(or its `images` folder). Every box in `labels/*.txt` is cropped from its image and becomes one sample
labeled with its class name from the `names` of `data.yaml` (looked up in the split folder and its two
parent folders; unknown ids become `class_<id>`). Add each split separately, not the dataset root.

Features:

* **Projects** – create, switch, save and delete projects (top bar). Each project is a JSON file in
  `.projects/` and keeps its datasets, augmentations, quotas, checkpoints (`.projects/<name>/checkpoints/run_*`)
  and evaluation records.
* **Datasets** – add multiple training pools, validation sets and test sets and check the ones to use.
  * *Validation sets* are evaluated after every epoch (metrics, quotas, best model, early stopping).
    Without validation folders, 20% of the training pool is held out for validation.
  * *Test sets* are never used during training. They are the final exam after training: the
    **🎓 Final exam (test sets)** button evaluates the selected model on them (accuracy, loss, confidence,
    precision/recall, confusion matrix; recorded in the project's evaluations), and Manual Review shows
    test images after training.

  Shuffle can be toggled and a fixed seed set. A bar chart shows the class balance of the checked pools.
  A green ✓ badge next to a folder shows that a `.yaml` configuration (with class `names`) was detected.
  **🔄 Synchronize YAML class names…** lists every tag of the detected `.yaml` files with the files it
  comes from; pick or type one unified name per tag (e.g. map `W-R` and `White-Rook` to `white-rook`).
  *Apply* writes the same unified `names` list to every `.yaml` and re-indexes the class ids of all YOLO
  label files belonging to them (files on disk are modified).
* **Augmentation** – checkboxes for horizontal/vertical flip, rotation, color jitter, affine/shear and
  random resized crop. They are composed into a `transforms.Compose` used only for the training split.
* **Training & Quotas** – epochs, batch size, learning rate, image size, architecture (torchvision
  ResNet/MobileNet/EfficientNet or Ultralytics YOLO26 classifiers `yolo26{n,s,m,l,x}-cls`); accuracy floor
  (per epoch) and moving-average target, confidence floor (mean top-1 confidence per epoch) and moving-average
  target. Training stops (optional) once all targets are met. Full weights are exported every *N* epochs
  (`snapshots/epoch_XXXX.pt`) and early stopping halts training when the monitored metric does not improve
  for *patience* epochs. The monitor can be validation loss, validation accuracy, or *both* (stops once
  neither loss nor accuracy improved for *patience* epochs).
* **Checkpoints & Provenance** – resume training from any `.pt`/`.pth` checkpoint. Every checkpoint stores
  the cumulative epochs, quota milestones (e.g. "Achieved 82.0% Val Acc Floor (target 80%) at Epoch 15"),
  the folders used in each session, snapshots and the best weights. Selecting a model shows this in the
  *Model Provenance* card with achieved/pending milestone cards.
* **Live training** – training runs in a `QThread`; batch and run progress bars, a live log and
  Pause/Resume/Stop buttons. The *Live quota check* shows every quota objective as `current / target`
  with the share of the target reached (e.g. `82.0% / 90.0% (91%)`); the *Class Quotas* tab shows per class
  `correct / needed` validation samples for the accuracy floor (e.g. `9 / 8 (112%)`).
* **Manual Review** – while paused (validation sets) or after training (test sets), shows random images with ground truth vs.
  top-3 predictions (green border = correct, red = incorrect) with Previous/Next buttons.
* **Test own picture** – also available here; it uses a project detector only (boxes and names).
* **Analytics** – confusion matrix, normalized confusion matrix, instance count, precision per epoch and
  precision/recall vs. confidence threshold, updated after every validation step. The normalized confusion
  matrix shows the change against the previous run (or a baseline chosen in *Normalized matrix Δ baseline*)
  in every cell, e.g. `0.85 (+3.2%)`: green = improvement (diagonal gain / off-diagonal drop), red =
  regression. The precision plot is zoomed to `0.75–1.01` and only extends lower when a value falls below.

Run the tests with:

```bash
python -m pytest tests
```

## 7. Garden chess scanner (photo → FEN)

Five explainable steps, each saved as an image with `--debug` / **Export debug images**:

```text
photo ─► 1 board corners ─► 2 straighten (homography) ─► 3 detect pieces ─► 4 pieces → squares ─► 5 FEN + exports
         (YOLO-pose model,   (8×8 grid, 800 px)           (YOLO detector)     (foot point of each   (python-chess,
          OpenCV fallback,                                                     box, 1 per square)    PNG/TXT)
          or drag by hand)
```

Only two networks are used: the 12-class piece **detector** (task *Detection*) and a 4-keypoint
**board corner** model (task *Board corners*, YOLO-pose, base model `yolo26s-pose.pt`).

**GUI:** **♟ Scan board** opens the scan screen: open a photo, check/drag the 4 corners, detect the
pieces, confirm where a1 is (⟳ Rotate a1), click diagram squares to correct them, then export the FEN,
annotated photo, diagram, debug images or **Save as Training Sample** (photo + YOLO boxes + corner
label, per split). **🧰 Data tools…** generates synthetic board-corner datasets and imports ChessReD2K;
the created folders are added to the dataset lists.

**CLI:**

```bash
python scan.py photo.jpg --pieces pieces.pt --board board.pt --out out/ --debug
python scan.py photo.jpg --pieces pieces.pt --corners 260,150,640,150,820,620,80,620 --a1 3 --out out/
python scan.py --evaluate chessred/annotations.json --images chessred --pieces pieces.pt --board board.pt --out eval/
```

Outputs: `fen.txt`, `annotated.png`, `diagram.png`, `scan.json` (+ `1_corners.png` … `5_diagram.png`
with `--debug`; `evaluation.json` with `--evaluate`). See `SCANNER_IMPLEMENTATION.md` for what is
implemented and which data you still have to collect yourself.
