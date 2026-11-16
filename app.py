"""schaakpoc v3 desktop app: train object detectors (YOLO boxes, many objects per image; default) or
image classifiers with a PyQt6 GUI. Switch with the *Task* selector in the project bar.

Usage:
    python app.py                 # projects are stored in ./.projects
    python app.py --projects DIR  # use another workspace directory
"""
import argparse
import sys

from PyQt6.QtWidgets import QApplication

from schaakgui.main_window import MainWindow
from schaakgui.projects import PROJECTS_DIR, ProjectManager


def main():
    parser = argparse.ArgumentParser(description="schaakpoc v3 training GUI (detection / classification)")
    parser.add_argument("--projects", default=str(PROJECTS_DIR), help="workspace directory")
    args, qt_args = parser.parse_known_args()

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("schaakpoc v3")
    window = MainWindow(ProjectManager(args.projects))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
