"""Garden chess scanner on the command line: photo → FEN, annotated photo, diagram (+ step images).

    python scan.py photo.jpg --pieces P.pt [--board B.pt] [--corners x1,y1,x2,y2,x3,y3,x4,y4]
                   [--a1 0..3] [--conf 0.25] [--lift 0.08] [--side w] --out out/ [--debug]
    python scan.py --evaluate annotations.json --images chessred/images --pieces P.pt --board B.pt

Writes ``fen.txt``, ``annotated.png``, ``diagram.png`` and ``scan.json`` (with ``--debug`` also the
images of the 5 steps). It only calls :class:`schaakgui.scan_session.ScanSession` methods.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_corners(text: str) -> list[tuple[float, float]]:
    values = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if len(values) != 8:
        raise argparse.ArgumentTypeError("--corners needs 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4")
    return [(values[i], values[i + 1]) for i in range(0, 8, 2)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("photo", nargs="?", help="photo of the board")
    parser.add_argument("--pieces", help="YOLO piece detector weights (.pt)")
    parser.add_argument("--board", help="YOLO-pose board corner model (.pt); else OpenCV fallback")
    parser.add_argument("--corners", type=parse_corners,
                        help="manual corners in pixels: x1,y1,x2,y2,x3,y3,x4,y4 (any order)")
    parser.add_argument("--a1", type=int, choices=range(4),
                        help="corner of a1: 0=top-left 1=top-right 2=bottom-right 3=bottom-left "
                             "(default: suggested)")
    parser.add_argument("--conf", type=float, default=0.25, help="piece confidence threshold")
    parser.add_argument("--lift", type=float, default=0.08, help="foot point lift (x box height)")
    parser.add_argument("--side", choices=("w", "b"), default="w", help="side to move")
    parser.add_argument("--castling", default=None, help="castling rights (default: suggested)")
    parser.add_argument("--out", default="scan_out", help="output folder")
    parser.add_argument("--debug", action="store_true", help="also save the 5 step images")
    parser.add_argument("--evaluate", metavar="ANNOTATIONS_JSON",
                        help="ChessReD annotations.json: evaluate the full pipeline on its test split")
    parser.add_argument("--images", help="ChessReD images root (with --evaluate)")
    parser.add_argument("--split", default="test", help="ChessReD split to evaluate (default test)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate at most N photos (0 = all)")
    return parser


def run_scan(args) -> dict:
    from schaakgui.board import BoardFinder
    from schaakgui.detector import ImageTester
    from schaakgui.scan_session import ScanSession

    session = ScanSession(ImageTester(args.pieces), BoardFinder(args.board),
                          conf=args.conf, lift=args.lift)
    session.load_photo(args.photo)
    if args.corners:
        session.set_corners(args.corners, "manual")
    elif session.auto_corners() is None:
        raise SystemExit("No board corners found: pass --corners x1,y1,...,x4,y4 or a --board model.")
    print(f"Corners ({session.corner_method}): "
          + "  ".join(f"({x:.0f}, {y:.0f})" for x, y in session.corners))
    session.detect_pieces(args.conf)
    if args.a1 is None:
        session.suggest_orientation()
    else:
        session.set_a1_corner(args.a1)
    session.info.side = args.side
    session.info.castling = args.castling if args.castling else session.suggest_castling()
    written = session.export_all(args.out, debug=args.debug)
    print(f"FEN: {session.fen()}")
    for problem in session.problems():
        print(f"  review: {problem}")
    for warning in session.warnings():
        print(f"  warning: {warning}")
    print(f"Saved to {Path(args.out).resolve()}")
    return written


def run_evaluate(args) -> dict:
    from schaakgui.importers import chessred_test_cases
    from schaakgui.scan_eval import evaluate_scans

    if not args.images or not args.pieces:
        raise SystemExit("--evaluate needs --images and --pieces (and preferably --board)")
    cases = chessred_test_cases(args.evaluate, args.images, args.split)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("No test photos found (check --images and --split).")
    report = evaluate_scans(cases, args.pieces, args.board, args.conf,
                            progress=lambda i, n: print(f"\r{i}/{n}", end="", flush=True))
    print()
    print(f"Boards: {report['boards']}  square accuracy {report['square_accuracy']:.1f}%  "
          f"exact {report['exact_boards']:.1f}%  ≤1 error {report['within_one_error']:.1f}%")
    print("Failures: " + ", ".join(f"{k} {v}" for k, v in report["failures"].items()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.evaluate:
        run_evaluate(args)
        return 0
    if not args.photo:
        build_parser().error("give a photo (or --evaluate)")
    if not args.pieces:
        build_parser().error("--pieces (YOLO piece detector weights) is required")
    run_scan(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
