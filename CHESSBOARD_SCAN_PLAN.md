# Garden chess scanner: photo → FEN, annotated photo and board diagram

## Goal

Take a photo of a **garden chess set** (large pieces on a black-and-white
checkered board on the ground), review the recognised position, and export:

- FEN notation.
- The original photo with the board outlined and every piece boxed.
- A clean, reconstructed board diagram.

Build on the existing **Test own picture** screen
([schaakgui/review_dialog.py](schaakgui/review_dialog.py)) and the YOLO trainer
([schaakgui/detector.py](schaakgui/detector.py)).

## Situation and constraints

| Fact | Consequence for the plan |
| --- | --- |
| No photos of the garden set yet (no piece data, no full-board photos). | **Phase A: pre-train** on public and synthetic data. **Phase C: fine-tune** later on own photos. |
| Board is plain black/white squares. | Board data can be **generated** (synthetic), and OpenCV's checkerboard finder works on an empty board. |
| Pieces are tall (roughly 30–100 cm) and stand on the ground. | They hide squares and lean in the photo. Use the **foot point** of a box, not its centre, and take photos from a **high viewpoint**. |
| Outdoor: sun, shadows, grass, paving. | Strong colour/brightness augmentation, shadow simulation, and background-only images. |
| School project: must be easy to explain. | Small, separate steps. Each step has one module, one job, and its own **debug image**. Only two neural networks; everything else is plain geometry. |

## The pipeline in five steps (the explanation for the report)

```
 photo ──► 1. find board corners ──► 2. straighten board ──► 3. detect pieces ──► 4. pieces → squares ──► 5. FEN + exports
           (pose model / OpenCV /      (homography,            (YOLO detector,      (foot point of each        (python-chess,
            or user clicks)             8×8 grid)               existing)            box, 1 piece per square)    PNG exports)
```

| Step | What it does | How | Module | Debug image |
| --- | --- | --- | --- | --- |
| 1 | Find the 4 outer corners of the playing area | YOLO-pose model → OpenCV checkerboard finder → user drags corners | `board.py` | `1_corners.png` |
| 2 | Compute the 8×8 grid | Perspective transform (homography) from 4 points | `board.py` | `2_warped.png` |
| 3 | Box and name every piece | Existing YOLO detector | `detector.py` | `3_pieces.png` |
| 4 | Put each piece on a square | Foot point → transform → `col = x // square` | `position.py` | `4_feet.png` |
| 5 | Make the result | Board → FEN, validate, draw, save | `fen.py`, `render.py` | `5_diagram.png` |

Explainability rules for the implementation:

- **Qt-free logic, thin GUI.** All logic lives in plain Python modules
  (testable, usable from the command line). The dialogs only display and forward
  clicks. This matches the existing pattern of `ImageTester` + `ImageTestDialog`.
- **One controller per screen.** `ScanSession` holds the state of one scan and
  offers one method per user action.
- **Two models only.** A piece detector (boxes) and a board corner model (4
  points). There is no end-to-end "photo → FEN" network, because every
  intermediate result must be visible and checkable.
- **Every step can be saved as an image** (`ScanSession.debug_images()`, CLI
  `--debug`) for the report and for finding errors.

## Phase A: pre-training without own data

### A1. Piece detector (pre-train)

Data:

1. **Existing** `datasets/chess` (Roboflow *rookeye* v29, ~18.7k images, 12
   piece classes). Class 12 `Black` has zero labels: remove it with
   **Synchronize YAML class names**.
2. **ChessReD2K** piece boxes ([4TU.ResearchData](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f)):
   2,078 real smartphone photos from many angles. Importer: `importers.py`.
3. **Background-only images** (empty label files): photos of grass, paving and
   the garden *without* pieces. Ultralytics uses them to learn "nothing here"
   and to reduce false boxes outdoors. This is cheap and needs no labelling.
4. Optional: search Roboflow Universe for *giant chess* / *garden chess* /
   *outdoor chess*. Check the licence and class names before use.

All sets are unified to the same 12 names (`white-king` … `black-pawn`) with
the existing YAML sync.

Training settings (Detection task, existing trainer):

- Base `yolo26s.pt` (fast enough to explain and run; `m` if accuracy is low).
- Augmentation: flips on, colour jitter **strong** (outdoor light), scale/resized
  crop on (pieces appear large in garden photos), small rotation, mosaic on.
- Expectation for the report: good on public test sets, weaker on the garden set.
  That gap is the motivation for Phase C.

### A2. Board corner model (pre-train)

The model is YOLO-pose with 1 class `board` and **4 keypoints in image order:
top-left, top-right, bottom-right, bottom-left** (sorted clockwise from the corner
nearest the image's top-left). The model does **not** learn which corner is a1.
Orientation is a separate, simple step (section B3). Reasons:

- Synthetic boards and an empty black/white board look the same from all four
  sides, so a1 cannot be learnt from them anyway.
- Left-right flip augmentation stays valid with `flip_idx: [1, 0, 3, 2]`.
- It is easier to explain: "the model finds 4 points; the rules decide the names".

Data, in order of importance:

1. **Synthetic boards** (`synthetic.py`). This is the main source, made for a
   black/white board:
   - draw a perfect 8×8 black/white board (random border, slightly varied
     white/grey and black/dark-grey, optional tile seams);
   - warp it with a random, camera-like perspective (tilt 20–70°, rotation
     ≤ ±30°, random scale/position, all 4 corners inside the image) onto a
     random background photo;
   - add piece-like occluders (tall dark/light blobs standing on random squares)
     and random shadows, blur, noise and brightness;
   - the corner labels are exactly known, so no labelling is needed.
   Backgrounds: a folder of any outdoor photos. Best: ~50 photos of the actual
   garden/school yard *without* the board. Fallback: generated noise textures.
2. **ChessReD2K corners** (real photos, regular boards). ChessReD's
   White-relative corners are converted to image order. They add real camera
   effects and real pieces covering corners.
3. **Classical fallback without a model:** `cv2.findChessboardCornersSB` finds
   the 7×7 inner corners of an (almost) empty board. The outer corners are one
   square further out. This is useful for the first own photos and as a
   cross-check.

Training: new task **Board corners** in the GUI (same trainer, pose model).
Base `yolo26s-pose.pt`, `kpt_shape: [4, 3]`, `flip_idx: [1, 0, 3, 2]`, rotation
augmentation ≤ 10° (bigger rotations change which corner is "top-left").

## Phase B: the scan workflow (GUI + CLI)

### B1. Load photo and find corners

- Open or drop a photo (CLI: `python scan.py photo.jpg`).
- **Auto corners:** `BoardFinder.propose()` tries the pose model, then the
  OpenCV checkerboard finder. It shows which method succeeded.
- The user can always **drag** the 4 corner handles. The grid overlay updates
  live. **Manual corners always work**, so the first milestone needs no board
  model.

### B2. Detect pieces and place them on squares

- Run the selected piece detector (in a background thread).
- **Foot point** = bottom centre of the box, lifted by `lift × box height`
  (default 0.08). For a tall piece the box centre lies over another square; the
  foot does not.
- Transform the foot point with the homography → `(col, row)` in an image-ordered
  grid. Outside the board = **off-board** list (shown, not used in the FEN).
- One piece per square: the highest confidence wins, others are listed as
  **conflicts**. Low confidence (< threshold) or foot near a square edge
  (< 15 % of a square) is marked **uncertain** (orange).

### B3. Orientation (which corner is a1)

- Internally squares are stored as image-grid cells `(col, row)`. Chess names
  are computed only at the end, so rotating never re-runs detection.
- **Suggestion rule** (easy to explain):
  1. a1 is a **dark** square. Of the 4 corner squares, keep the 2 dark ones
     (a1 and h8), measured in the straightened image.
  2. a1 is on **White's side**: choose the dark corner nearest the average
     position of the white pieces.
- The user confirms or presses **⟳ Rotate** (cycles a1 through the 4 corners).

### B4. Review and correct

| Photo view | Board diagram |
| --- | --- |
| Board outline, optional 8×8 grid, 4 draggable corners | Clean top-down diagram (White at the bottom by default, flip button) |
| Box per piece with `white-knight / f3 / 94%` | Click a square → menu: empty or one of the 12 pieces |
| Orange = uncertain, red = conflict/off-board | Orange/red marks on the same squares |
| Mode **Draw box** (for training samples): drag a box, choose a class | |

### B5. FEN

- Placement from the reviewed board.
- Controls: side to move, castling rights (suggested from king/rook home
  squares), en-passant square, halfmove clock, fullmove number. These are
  **user-supplied**: one photo cannot show them.
- `python-chess` builds the FEN and checks it. Warnings (e.g. "no white king",
  "pawn on rank 8", "side not to move is in check") are shown and **never**
  silently fixed.

### B6. Export

- **Copy FEN**, **Save FEN** (`.txt`).
- **Save annotated photo** (PNG, original resolution: outline, grid, boxes,
  labels).
- **Save board diagram** (PNG).
- **Save debug steps** (the 5 step images, for the report).
- **Save as training sample** (Phase C): photo + corrected piece boxes + corners.

Exports always reflect the current reviewed state, including corrections.

## Phase C: own photos and fine-tuning

1. **Capture protocol** (also part of the report):
   - camera high (≥ 2 m: standing on a bench, phone on a pole, or from a
     window), all 4 board corners visible, 30–70° above the horizon;
   - different sessions: sunny/cloudy, morning/afternoon, different sides of the
     board;
   - positions: start position, random middle-game positions, few pieces;
   - target: **100–200 photos** for pieces (×10–32 pieces each = thousands of
     boxes), **50–100** for corners.
2. **Fast labelling with the pre-trained models:** open each photo in the scan
   screen. The pre-trained models already propose corners and boxes. Correct
   them (drag corners, relabel, draw missing boxes) and press **Save as
   training sample**. This writes YOLO labels to
   `<sample root>/pieces/<split>/` and `<sample root>/board/<split>/`.
3. **Split by session, not by photo:** photos from one session are very similar.
   Use e.g. sessions 1–3 for train, 4 for valid and 5 for test, otherwise the
   test score is too optimistic.
4. **Fine-tune** with the existing **Continue from** option, starting from the
   pre-trained weights, with fewer epochs (30–50) and **`freeze` = 10** (the first
   layers keep their general features; only the later layers adapt). Keep a
   small part of the public data in the training set so the model does not
   forget.
5. **Compare before/after** on the own test session: piece mAP50, corner error,
   square accuracy, exact-board accuracy. This table is the core result of the
   report.

## Architecture: new and changed controllers and functions

### Overview

| Module | Type | Responsibility |
| --- | --- | --- |
| `schaakgui/board.py` | new, Qt-free | Corners, homography, grid, corner finding (step 1–2) |
| `schaakgui/position.py` | new, Qt-free | Foot points, squares, orientation, board state (step 4) |
| `schaakgui/fen.py` | new, Qt-free | FEN build/parse, game info, validation (step 5) |
| `schaakgui/render.py` | new, Qt-free | Annotated photo, diagram, debug images |
| `schaakgui/scan_session.py` | new, Qt-free | **Controller** of one scan (one method per user action) |
| `schaakgui/keypoints.py` | new, Qt-free | Read/write YOLO-pose board labels and `data.yaml` |
| `schaakgui/synthetic.py` | new, Qt-free | Synthetic board dataset generator |
| `schaakgui/importers.py` | new, Qt-free | ChessReD2K → YOLO (pieces + board) |
| `schaakgui/scan_eval.py` | new, Qt-free | Corner error, square/exact-board accuracy |
| `schaakgui/scan_dialog.py` | new, Qt | Scan screen (view) + its widgets |
| `schaakgui/data_tools_dialog.py` | new, Qt | Generate synthetic boards, import ChessReD2K |
| `scan.py` | new, CLI | Same pipeline from the command line (report demos) |
| `projects.py`, `data.py`, `detector.py`, `review_dialog.py`, `worker.py`, `main_window.py` | changed | Board task, pose labels, fine-tune freeze, new buttons |

Shared types (in `board.py`):

```python
Point = tuple[float, float]                    # pixel (x, y) in the original photo
Corners = tuple[Point, Point, Point, Point]    # image order: TL, TR, BR, BL
Cell = tuple[int, int]                         # (col, row) in the straightened grid, 0..7, row 0 = top
BOARD_PX = 800                                 # side of the straightened board image
```

### `schaakgui/board.py`: board geometry (steps 1–2)

| Function / class | Purpose |
| --- | --- |
| `order_corners(points: Sequence[Point]) -> Corners` | Sort 4 points clockwise starting top-left (used by every corner source). |
| `board_homography(corners: Corners, size: int = BOARD_PX) -> np.ndarray` | 3×3 matrix photo → straight board (`cv2.getPerspectiveTransform`). |
| `to_board(h: np.ndarray, point: Point) -> Point` | Map one photo point onto the straight board. |
| `to_image(h: np.ndarray, point: Point) -> Point` | Inverse mapping (for drawing grid and squares on the photo). |
| `warp_board(image: Image, corners: Corners, size: int = BOARD_PX) -> Image` | Straightened top-down board image (debug step 2, orientation rule). |
| `cell_at(board_point: Point, size: int = BOARD_PX) -> Optional[Cell]` | `(col, row)` of a straight-board point, `None` outside. |
| `edge_distance(board_point: Point, size: int = BOARD_PX) -> float` | Distance to the nearest square edge as a fraction of a square (uncertainty). |
| `grid_lines(corners: Corners) -> list[tuple[Point, Point]]` | 9 + 9 grid lines in photo coordinates for the overlay. |
| `inner_to_outer(inner: np.ndarray) -> Corners` | 7×7 inner checkerboard corners → 4 outer corners (one square further). |
| `find_corners_opencv(image: Image) -> Optional[CornerProposal]` | Classical: `cv2.findChessboardCornersSB(gray, (7, 7))` + `inner_to_outer`. |
| `@dataclass CornerProposal(corners: Corners, method: str, score: float)` | `method` is `"model"`, `"opencv"` or `"manual"`. |
| `load_board_model(path) -> YOLO` | Load and validate: task `pose`, 4 keypoints, else a clear error. |
| `class BoardFinder` | Tries the corner sources in order. |
| `BoardFinder.__init__(model_path: Optional[str] = None)` | |
| `BoardFinder.load_model(path)` | Select another board model. |
| `BoardFinder.propose(image: Image, conf: float = 0.25) -> Optional[CornerProposal]` | Model → OpenCV → `None` (the user then clicks). |
| `BoardFinder._from_model(image, conf) -> Optional[CornerProposal]` | Best `board` detection's 4 keypoints, ordered. |

Stretch only (not needed for the school version): Hough-line lattice fitting
for boards full of pieces without a model.

### `schaakgui/position.py`: pieces → squares (step 4)

| Function / class | Purpose |
| --- | --- |
| `PIECE_SYMBOLS: dict[str, str]` | `"white-king" → "K"` … `"black-pawn" → "p"`. |
| `piece_symbol(label: str) -> Optional[str]` | Tolerant mapping (`White-Knight`, `white_knight` → `N`), reuses `same_class` normalisation. |
| `foot_point(box, lift: float = 0.08) -> Point` | Bottom centre lifted by `lift × height`. |
| `square_name(cell: Cell, a1_corner: int) -> str` | Chess name of a grid cell for a chosen a1 corner (0 = TL … 3 = BL). |
| `cell_of(square: str, a1_corner: int) -> Cell` | Inverse of `square_name`. |
| `@dataclass Placement(index, label, symbol, conf, box, foot, cell, edge)` | One detection with its foot point, cell (`None` = off-board) and edge distance. |
| `place_pieces(detections: list[dict], corners: Corners, lift: float = 0.08) -> list[Placement]` | Steps 2 + 4 for all detections. |
| `corner_square_is_dark(warped: Image, corner: int) -> bool` | Mean brightness of one corner square (orientation rule 1). |
| `suggest_a1_corner(warped: Image, placements: list[Placement]) -> int` | Dark corner nearest the white pieces (rules 1 + 2). Falls back to the dark corner at the bottom. |
| `@dataclass BoardState` | `cells: dict[Cell, str]`, `conflicts: dict[Cell, list[Placement]]`, `uncertain: set[Cell]`, `off_board: list[Placement]`, `edited: set[Cell]`. |
| `BoardState.from_placements(placements, min_conf: float = 0.5, min_edge: float = 0.15) -> BoardState` | One piece per cell, conflicts, uncertainty. |
| `BoardState.set_cell(cell: Cell, symbol: Optional[str])` | User correction (`None` = empty). Marks the cell as edited. |
| `BoardState.squares(a1_corner: int) -> dict[str, str]` | `{"e1": "K", …}` for FEN and diagram. |

### `schaakgui/fen.py`: FEN (step 5)

| Function / class | Purpose |
| --- | --- |
| `@dataclass GameInfo(side="w", castling="-", en_passant="-", halfmove=0, fullmove=1)` | User-supplied state. |
| `placement_fen(squares: dict[str, str]) -> str` | Piece placement field only. |
| `full_fen(squares: dict[str, str], info: GameInfo) -> str` | All six fields. |
| `suggest_castling(squares: dict[str, str]) -> str` | `KQkq` subset from king/rook home squares. |
| `check_position(squares: dict[str, str], info: GameInfo) -> list[str]` | Readable warnings from `chess.Board.status()`. Never modifies anything. |
| `squares_from_fen(fen: str) -> dict[str, str]` | Parse (tests, known positions, evaluation). |

### `schaakgui/render.py`: images

| Function | Purpose |
| --- | --- |
| `draw_scan(image, corners, placements, state, a1_corner, grid: bool = True) -> Image` | Annotated photo at original resolution (outline, grid, boxes, `label / square / %`, colour = status). |
| `draw_corners(image, corners, color=(0, 200, 0), labels: Sequence[str] = ("TL", "TR", "BR", "BL")) -> Image` | Corner overlay (step 1, review of the board model). |
| `draw_feet(warped, placements) -> Image` | Straight board with foot points and grid (step 4). |
| `draw_diagram(squares: dict[str, str], size: int = 640, white_bottom: bool = True, marks: dict[str, str] = {}) -> Image` | Clean diagram with PIL (piece glyph if the font has it, else letter), orange/red marks. |

### `schaakgui/scan_session.py`: the controller

`ScanSession` holds the state of one scan: photo, corners and method, a1 corner,
detections, placements, `BoardState`, `GameInfo`, confidence, lift. It is used by
both the GUI and the CLI.

| Method | User action / purpose |
| --- | --- |
| `__init__(tester: ImageTester, finder: BoardFinder)` | Reuses the existing `ImageTester` for pieces. |
| `load_photo(path) -> None` | Open/drop a photo. Resets the state. |
| `auto_corners() -> Optional[CornerProposal]` | **Auto corners** button. |
| `set_corners(corners: Sequence[Point], method: str = "manual") -> None` | Corners from any source. Re-places pieces **without** re-detecting. |
| `move_corner(index: int, point: Point) -> None` | Drag one handle. |
| `detect_pieces(conf: float) -> None` | Slow: run piece detector (called in a worker thread). |
| `replace() -> None` | Rebuild placements + state after corners/lift/conf change (keeps user edits). |
| `suggest_orientation() -> int` / `set_a1_corner(i)` / `rotate()` | Orientation step. |
| `set_square(square: str, symbol: Optional[str]) -> None` | Diagram click correction. |
| `add_box(box, label) -> None` / `remove_detection(index)` / `relabel(index, label)` | Box edits (training samples). |
| `squares() -> dict[str, str]` / `fen() -> str` / `warnings() -> list[str]` | Results. |
| `annotated_image()`, `diagram_image()`, `debug_images() -> dict[str, Image]` | Images (steps 1–5). |
| `export_fen(path)`, `export_annotated(path)`, `export_diagram(path)`, `export_debug(folder)` | Exports. |
| `save_training_sample(root, split) -> dict[str, str]` | Writes the photo + `pieces/<split>` box labels + `board/<split>` pose label (+ `data.yaml` if missing). |
| `to_dict() -> dict` | JSON summary (photo, corners, method, FEN, warnings), saved next to exports. |

### `schaakgui/keypoints.py`: YOLO-pose board labels

| Function | Purpose |
| --- | --- |
| `BOARD_KPT_SHAPE = [4, 3]`, `BOARD_FLIP_IDX = [1, 0, 3, 2]` | Constants written to every board `data.yaml`. |
| `board_label_line(corners, width, height) -> str` | `0 xc yc w h x1 y1 2 … x4 y4 2` (normalised). |
| `write_board_label(path, corners, width, height)` / `read_board_label(path, width, height) -> Optional[Corners]` | |
| `write_board_yaml(dataset_root, splits=("train", "valid", "test"))` | `names: {0: board}`, `kpt_shape`, `flip_idx`. |

### `schaakgui/synthetic.py`: synthetic board generator

| Function / class | Purpose |
| --- | --- |
| `@dataclass SyntheticConfig(count=3000, image_size=640, val_fraction=0.15, backgrounds_dir=None, max_rotation=30, tilt=(20, 70), occluders=True, shadows=True, seed=None)` | |
| `checkerboard(square_px: int, border_px: int, light: RGB, dark: RGB, seams: bool) -> Image` | Flat 8×8 board, a1 dark. |
| `random_view(rng, image_size) -> Corners` | Camera-like corner positions (tilt, rotation, scale), all inside the image. |
| `paste_board(background, board, corners) -> Image` | Perspective warp + paste (`cv2.warpPerspective` + mask). |
| `add_occluders(image, corners, rng) -> Image` | Piece-like blobs standing on random squares (cover lines and corners). |
| `add_lighting(image, rng) -> Image` | Shadows, brightness, blur, noise, JPEG artefacts. |
| `load_backgrounds(folder) -> list[Path]` / `noise_background(rng, size) -> Image` | Background sources. |
| `make_sample(rng, cfg, backgrounds) -> tuple[Image, Corners]` | One image + exact corners. |
| `generate_board_dataset(out_dir, cfg, progress: Callable[[int, int], None] = None, stop: Callable[[], bool] = None) -> dict` | Writes `train/` + `valid/` (images, labels) and `data.yaml`. Returns counts and paths. |

### `schaakgui/importers.py`: public data

| Function | Purpose |
| --- | --- |
| `CHESSRED_NAMES: dict[int, str]` | ChessReD category id → unified name (`white-pawn`, …). |
| `load_chessred(annotation_json) -> dict` | Read the annotation file once. |
| `chessred_corners(annotation: dict) -> Corners` | White-relative corners → image order (`order_corners`). |
| `import_chessred2k(annotation_json, images_root, out_dir, progress=None, stop=None) -> dict` | Only images with boxes + corners. Writes `out_dir/pieces/{train,valid,test}` (detect, 12 classes) and `out_dir/board/{train,valid,test}` (pose), keeping ChessReD's split. |
| `chessred_test_cases(annotation_json, images_root) -> list[dict]` | `(image path, FEN placement)` for the end-to-end benchmark. |

### `schaakgui/scan_eval.py`: evaluation for the report

| Function | Purpose |
| --- | --- |
| `corner_error(pred: Corners, truth: Corners) -> float` | Mean corner distance as % of the board diagonal. |
| `evaluate_board_model(weights, folders) -> dict` | Pose mAP + mean/median corner error + % within 2 %. |
| `square_accuracy(pred: dict[str, str], truth: dict[str, str]) -> float` | Correct squares / 64. |
| `evaluate_scans(cases, piece_weights, board_weights, conf=0.25) -> dict` | Full pipeline on known positions: square accuracy, exact-board %, boards with ≤ 1 error, per-step failure counts (no corners / wrong orientation / wrong piece). |

### `schaakgui/scan_dialog.py`: scan screen (view)

| Class / method | Purpose |
| --- | --- |
| `class PhotoView(QWidget)` | Scaled photo with overlays. |
| ` signals cornerMoved(int, float, float)`, `boxDrawn(float, float, float, float)`, `detectionClicked(int)` | Photo coordinates, not widget coordinates. |
| ` set_photo(image)`, `set_overlay(corners, placements, state, a1_corner, grid)`, `set_mode("corners" \| "boxes")` | |
| ` _to_photo(pos) -> Point` / `_to_widget(point) -> QPointF` | One place for coordinate conversion. |
| ` mousePressEvent / mouseMoveEvent / mouseReleaseEvent / paintEvent` | Drag the nearest corner (within 20 px) or draw a box. |
| `class BoardDiagram(QWidget)` | Clickable 8×8 diagram. |
| ` squareClicked = pyqtSignal(str)`, `set_position(squares, marks, white_bottom)`, `paintEvent`, `mousePressEvent` | |
| `class PieceMenu(QMenu)` / `PieceMenu.choose(parent, global_pos) -> Optional[str] \| Ellipsis` | Empty + 12 pieces. `Ellipsis` = cancelled. |
| `class GameInfoBox(QGroupBox)` / `changed`, `info() -> GameInfo`, `set_info(info)` | Side to move, castling, en passant, clocks. |
| `class ScanDialog(QDialog)` | Step bar: **1 Photo → 2 Corners → 3 Pieces → 4 Orientation → 5 Review & Export**. |
| ` __init__(session, detectors, board_models, sample_root, import_model=None, parent=None)` | Combos for piece detector and board model (+ *Add model file…*, like the current dialog). |
| ` on_open_photo`, `on_auto_corners`, `on_corner_moved`, `on_mode_changed` | Steps 1–2. |
| ` on_detect` → `start_call(session.detect_pieces, conf)` → `on_detect_done` / `on_failed` | Step 3, off the GUI thread. |
| ` on_conf_changed`, `on_lift_changed` | Re-place without re-detecting (conf filters existing detections). |
| ` on_rotate`, `on_flip_diagram` | Step 4. |
| ` on_square_clicked`, `on_box_drawn`, `on_detection_clicked`, `on_info_changed` | Corrections. |
| ` refresh()` | Redraw both views, problem list, FEN line and warnings from the session. |
| ` copy_fen`, `save_fen`, `save_annotated`, `save_diagram`, `save_debug`, `save_sample` | Export buttons. `save_sample` asks the split (train/valid/test). |

### `schaakgui/data_tools_dialog.py`: dataset tools

| Class / method | Purpose |
| --- | --- |
| `class DataToolsDialog(QDialog)` | Two tabs; long jobs in a worker with progress and stop. |
| ` _build_synthetic_tab()` / `on_generate()` | Count, image size, backgrounds folder, seed, output folder → `generate_board_dataset`. |
| ` _build_chessred_tab()` / `on_import()` | Annotation JSON, images folder, output folder → `import_chessred2k`. |
| ` on_progress(i, total)`, `on_done(result)`, `on_failed(error)` | |
| ` created_folders() -> dict[str, list[str]]` | `{"train": […], "valid": […], "test": […]}` for MainWindow to add to the project. |

### `scan.py`: command line

```
python scan.py photo.jpg --pieces P.pt [--board B.pt] [--corners x1,y1,x2,y2,x3,y3,x4,y4]
               [--a1 0..3] [--conf 0.25] [--lift 0.08] [--side w] --out out/ [--debug]
```

The CLI writes `fen.txt`, `annotated.png`, `diagram.png`, `scan.json`, and with
`--debug` the 5 step images. It only calls `ScanSession` methods.
`python scan.py --evaluate chessred.json --images … --pieces … --board …` runs
`evaluate_scans`.

### Changes to existing modules

| Module | Change |
| --- | --- |
| `projects.py` | `TASKS["board"] = "Board corners (4 keypoints per board)"`. `DetectorConfig.freeze: int = 0` (fine-tune). `default_base_model(task) -> str` (`yolo26m.pt` / `yolo26s-pose.pt`, applied when switching task and the base doesn't fit). `@dataclass ScanSettings(piece_model, board_model, conf=0.25, lift=0.08, sample_root)` + `Project.scan`. `list_checkpoints` also excludes `checkpoints/boards/`. |
| `data.py` | `yolo_kpt_shape(split_dir) -> Optional[list[int]]`. `read_yolo_labels(label_file, kpt_shape=None)`: with `kpt_shape`, the first 4 values are the box (currently a pose line would be read as a polygon and give a wrong box). `scan_yolo_folder` passes the shape. |
| `detector.py` | `load_yolo_model(path, task="detect")` (accepts `detect` or `pose`; `load_yolo_detector` stays as a wrapper). `models_dir(project_dir, task)`, `list_models(project_dir, task)` (`detectors/` vs `boards/`), `import_model(project_dir, source, task)`. `build_data_yaml(..., task="detect")` copies and checks `kpt_shape`/`flip_idx`. `DetectorJob.task`. Run name `board_<stamp>` for pose. `train(..., freeze=cfg.freeze or None)`. `pose_metrics(raw)` (mAP50(P), mAP50-95(P)). `evaluate_detector` adds `corner_error` for pose. `DetectionReviewSession.predict` returns predicted/true corners for pose. |
| `review_dialog.py` | `DetectionReviewDialog._show` draws corners via `render.draw_corners` for board models. `ImageTester` unchanged (used by `ScanSession`). |
| `worker.py` | Generic `CallWorker(QObject)` (`finished(object)`, `progress(int, int)`, `failed(str)`) + `start_call(func, *args, **kwargs) -> tuple[QThread, CallWorker]` for detection, generation and import. |
| `main_window.py` | `apply_task_ui` handles 3 tasks (`board` uses the detection UI, hides class balance and confusion). New buttons **♟ Scan board** → `open_scan()`, **🧰 Data tools…** → `open_data_tools()`. `project_board_models()`, `import_project_model(source, task)`, `add_dataset_folders(created: dict)`. |
| `requirements.txt` | Add `chess` (python-chess). |
| `README.md` | New section: garden chess workflow, capture tips, exports, data tools, fine-tuning. |

### Data flow of one scan (sequence)

```
ScanDialog.on_open_photo ─► session.load_photo ─► session.auto_corners ─► BoardFinder.propose
ScanDialog.on_detect ─► start_call(session.detect_pieces) ─► ImageTester.detect
                     ─► session.replace ─► place_pieces ─► BoardState.from_placements
                     ─► session.suggest_orientation ─► suggest_a1_corner
ScanDialog.refresh ◄─ session.squares / fen / warnings / annotated_image / diagram_image
user drags corner ─► session.move_corner ─► session.replace ─► refresh   (no new detection)
user clicks square ─► PieceMenu.choose ─► session.set_square ─► refresh
```

## Project layout for the experiments

| Project | Task | Train data |
| --- | --- | --- |
| `garden-pieces` | Detection | rookeye + ChessReD2K pieces + background images → later + own `pieces/` |
| `garden-board` | Board corners | synthetic boards + ChessReD2K board → later + own `board/` |

Models live in `.projects/<name>/checkpoints/detectors/` and
`.projects/<name>/checkpoints/boards/`. The scan screen can pick models from
any file via *Add model file…*.

## Implementation order (milestones)

1. **Core geometry + FEN** (no GUI): `board.py` (homography part),
   `position.py`, `fen.py`, `render.py`, with tests on synthetic fixtures.
2. **Controller + CLI with manual corners**: `ScanSession`, `scan.py --corners`.
   Works with the existing `runs/chess/weights/best.pt` detector.
3. **Scan screen**: `ScanDialog` with corner dragging, orientation, corrections,
   exports, debug steps.
4. **Pre-training data**: `keypoints.py`, `synthetic.py`, `importers.py`,
   `DataToolsDialog`, pose-aware `data.py`.
5. **Board task**: pose training in `detector.py`/`main_window.py`, then
   `BoardFinder` (model → OpenCV) as **Auto corners**.
6. **Pre-train both models** (Phase A) and record the baseline metrics.
7. **Own data**: box drawing + **Save as training sample**, `freeze`
   fine-tuning, before/after evaluation (Phase C).
8. Optional: webcam snapshot, Hough-lattice fallback.

Milestones 1–3 already give a usable tool (manual corners). Each later
milestone improves it without changing the earlier code.

## Verification

Unit tests (one test file per new module, Qt-free, fast):

- `test_board.py`: `order_corners` for shuffled points; homography maps the 4
  corners to the square corners; `cell_at`/`edge_distance` on known points;
  `find_corners_opencv` on a rendered empty board.
- `test_position.py`: foot point vs. centre on a tall box; `square_name` for
  all four `a1_corner` values; conflicts, off-board and uncertain cells;
  `suggest_a1_corner` on a rendered start position in all 4 rotations.
- `test_fen.py`: start position and known FENs round-trip; castling
  suggestion; warnings for missing king / pawn on rank 8.
- `test_keypoints.py` + `data.py`: pose labels round-trip;
  `read_yolo_labels` with `kpt_shape` returns the correct box.
- `test_synthetic.py`: generated labels match the drawn corners (pixel check
  on the warped board); `data.yaml` has `kpt_shape`/`flip_idx`.
- `test_importers.py`: a 2-image ChessReD-style JSON fixture → correct YOLO
  files and corner order.
- `test_scan_session.py`: with a fake detector, perspective photo → expected
  FEN; corner move and rotate don't re-detect; corrections survive rotation;
  exports exist at original resolution; training sample labels are correct.
- `test_gui.py`: scan dialog opens, a square correction changes the FEN,
  export buttons write files (offscreen Qt).

Experiments (for the report):

- Piece detector: mAP50 on the public test sets and on the own test session,
  **before and after** fine-tuning.
- Board model: corner error on synthetic valid, ChessReD2K test and own test.
- Full pipeline: square accuracy and exact-board accuracy on the ChessReD test
  split and on own photos with manually recorded positions, plus a breakdown of
  the failure step.

## Limitations (state them in the report)

- One photo cannot show pieces that are fully hidden. A high viewpoint is
  required, and garden pieces make this more important than on a table.
- The homography assumes a flat board. A soft mat on uneven grass bends the
  grid.
- Castling rights, en passant, side to move and clocks are user input.
- Synthetic and ChessReD data look different from the garden. Fine-tuning on own
  photos (Phase C) is expected to give the largest improvement.
