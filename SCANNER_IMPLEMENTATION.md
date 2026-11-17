# Garden chess scanner (V3.1.1): what is implemented and what you still have to do

Scanner pipeline: photo → board corners → straightened board → pieces → squares → FEN, annotated
photo, board diagram and debug images. The design is in `CHESSBOARD_SCAN_PLAN.md`; user documentation
is in `README.md` section 7.

## ✅ Implemented in this branch

### Core logic (plain Python, no PyQt6)

| Module | Contents |
| --- | --- |
| `schaakgui/board.py` | `Point`/`Corners`/`Cell` types, `BOARD_PX = 800`, `order_corners`, `board_homography`, `to_board`/`to_image`, `warp_board`, `cell_at`, `edge_distance`, `grid_lines`, OpenCV fallback `find_corners_opencv` + `inner_to_outer`, `CornerProposal`, `load_board_model` (checks pose + 4 keypoints), `BoardFinder.propose` (board model → OpenCV → `None`) |
| `schaakgui/position.py` | `PIECE_SYMBOLS` / `piece_symbol`, `foot_point`, `square_name` / `cell_of` for all four a1 corners, `place_pieces`, `corner_square_is_dark`, `suggest_a1_corner`, `BoardState` (conflicts, uncertain, off-board, user edits) |
| `schaakgui/fen.py` | `GameInfo`, `placement_fen`, `full_fen`, `suggest_castling`, `check_position` (python-chess warnings), `squares_from_fen` |
| `schaakgui/render.py` | `draw_corners` (step 1), `draw_feet` (step 4), `draw_scan` (annotated photo; green/orange/red), `draw_diagram` (Unicode glyphs with a letter fallback), `diagram_square` |
| `schaakgui/scan_session.py` | `ScanSession`, the controller shared by the GUI and the CLI. It holds all actions (corners, detection, re-placement without detecting again, orientation, edits, boxes), all exports, and `save_training_sample` |
| `schaakgui/keypoints.py` | `BOARD_KPT_SHAPE`, `BOARD_FLIP_IDX`, and functions to read and write YOLO-pose labels and `data.yaml` |
| `schaakgui/synthetic.py` | `generate_board_dataset`: synthetic board images with perspective, backgrounds (your own folder or procedural noise), occluders, shadows, lighting and noise; writes YOLO-pose `train/` + `valid/` |
| `schaakgui/importers.py` | `import_chessred2k` → `pieces/{train,valid,test}` + `board/{train,valid,test}`; `chessred_test_cases` |
| `schaakgui/scan_eval.py` | `corner_error`, `evaluate_board_model`, `square_accuracy`, `evaluate_scans` (square accuracy, exact boards, ≤ 1 error, failures per step) |

### CLI

- `scan.py` scans one photo, or evaluates a ChessReD set with `--evaluate`.
- See README section 7 for examples.

### GUI

- **`schaakgui/scan_dialog.py`:** the `ScanDialog` scan screen.
  - Workflow: 5 steps, draggable corners, drawing missed boxes, and selecting/relabelling/removing boxes.
  - Detection: runs in a background thread, with live confidence and foot-lift sliders.
  - Review: rotate or suggest a1, click the diagram to correct a square, game info box (side to move, castling, en passant, clocks), problem list.
  - Output: exports and **Save as Training Sample**.
- **`schaakgui/data_tools_dialog.py`:** `DataToolsDialog` with two tabs, **Synthetic Boards** and **Import ChessReD2K**, plus a progress bar and a cancel button.
- **`schaakgui/main_window.py`:**
  - New **♟ Scan board** and **🧰 Data tools…** buttons. Folders created by Data tools are added to the training, validation and test lists.
  - New **Board corners** task (YOLO-pose):
    - its own base model (`yolo26s-pose.pt`) and its own model to continue from;
    - a **Freeze first N layers** setting for fine-tuning;
    - models are saved in `checkpoints/boards/`;
    - final exam with corner error.
- **`schaakgui/review_dialog.py`:** Manual Review draws predicted (and true) corners for board models.

### Changes to existing modules

| Module | Change |
| --- | --- |
| `projects.py` | `TASKS["board"]`, `DetectorConfig.freeze` / `board_base_model` / `board_resume_from`, `default_base_model`, `ScanSettings` |
| `data.py` | `yolo_kpt_shape`, pose-aware `read_yolo_labels(..., kpt_shape)` |
| `detector.py` | Pose models, `checkpoints/boards/`, `kpt_shape`/`flip_idx` in `data.yaml`, `freeze`, pose metrics + corner error |
| `worker.py` | `CallWorker` / `start_call` |
| `requirements.txt` | `chess` (python-chess) |

### Tests (`QT_QPA_PLATFORM=offscreen python -m pytest tests -q`)

- **Core:** `test_board.py`, `test_position.py`, `test_fen.py`, `test_keypoints_and_data.py`, `test_synthetic.py`, `test_importers.py`.
- **Scanner and CLI:** `test_scan_session.py`, which covers the full pipeline with a mock detector, plus the CLI.
- **GUI:** `test_gui.py`, which covers the scan dialog, the data tools dialog, the board task and the main window buttons.

## 🧑 Still to do yourself (data and training)

The code cannot download gated datasets or take photos. Do these steps on your own machine:

1. **Install the new dependency:** `pip install -r requirements.txt` (adds `chess`).
2. **Clean the existing piece dataset (`datasets/chess`):**
   - Run **Synchronize YAML class names** in the GUI to remove the empty class 12 (`Black`).
   - 12 classes must remain: `white-king` … `black-pawn`.
3. **Download ChessReD2K:**
   - Get the images and `annotations.json` from [4TU.ResearchData](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f).
   - Import them with **🧰 Data tools… → Import ChessReD2K**. This gives `pieces/…` for the detector and `board/…` for the corner model.
4. **Background / negative photos:**
   - Take about 30–50 photos of grass, paving and garden with no board and no pieces.
   - Use them as the background folder in **Synthetic Boards**.
   - Also add them as empty-label (negative) images to the piece detector's training set.
5. **Garden chess photo sessions (phase C):**
   - Take 100–200 piece photos and 50–100 corner photos over **at least 5 sessions** with different light and angles.
   - Camera at least 2 m high, tilted 30°–70°, with all 4 corners visible.
   - Save them with **Save as Training Sample**, split by session: train = sessions 1–3, valid = session 4, test = session 5.
6. **Train the models** in the GUI:
   - **Board corner model:** task *Board corners*, base `yolo26s-pose.pt`. First train on the synthetic + ChessReD `board/` folders, then fine-tune on your garden corner samples (continue from that model, freeze ≈ 10).
   - **Piece detector:** task *Detection*. Train on `datasets/chess` + ChessReD `pieces/`, then fine-tune on the garden piece samples with freeze ≈ 10.
7. **Evaluate:**
   - Use **🎓 Final exam** on the test folders. Board models also report the corner error.
   - Run `python scan.py --evaluate annotations.json --images … --pieces … --board …` to get square accuracy and failures per step.

No trained weights or datasets are included in the repository.
