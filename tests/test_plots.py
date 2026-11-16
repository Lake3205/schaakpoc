import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from PyQt6.QtWidgets import QApplication  # noqa: E402

from schaakgui.plots import PlotCanvas, delta_color, precision_ylim  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_precision_ylim_zoomed_and_never_clipping():
    assert precision_ylim([]) == (0.75, 1.01)
    assert precision_ylim([0.8, 0.83, 1.0]) == (0.75, 1.01)
    assert precision_ylim([0.8, 0.75]) == (0.75, 1.01)
    low, high = precision_ylim([0.9, 0.62, float("nan")])
    assert low == pytest.approx(0.6) and high == 1.01
    assert precision_ylim([0.0])[0] == 0.0


def test_delta_colors():
    assert delta_color(3.2, diagonal=True) == delta_color(-1.5, diagonal=False)  # improvements
    assert delta_color(-3.2, diagonal=True) == delta_color(1.5, diagonal=False)  # regressions
    assert delta_color(3.2, diagonal=True) != delta_color(-3.2, diagonal=True)
    assert delta_color(0.01, diagonal=True) not in (delta_color(1, True), delta_color(-1, True))


def test_plots_render_zoom_and_deltas(app):
    canvas = PlotCanvas()
    history = [{"epoch": 1, "precision": 0.80, "recall": 0.78, "val_acc": 0.79},
               {"epoch": 2, "precision": 0.83, "recall": 0.81, "val_acc": 0.82}]
    canvas.plot_precision(history)
    assert canvas.ax.get_ylim() == pytest.approx((0.75, 1.01))
    canvas.plot_precision(history + [{"epoch": 3, "precision": 0.5, "recall": 0.4, "val_acc": 0.45}])
    assert canvas.ax.get_ylim()[0] == pytest.approx(0.35)

    baseline = {"confusion": [[8, 2], [1, 9]], "classes": ["a", "b"], "label": "previous run"}
    canvas.plot_confusion([[9, 1], [2, 8]], ["a", "b"], normalized=True, baseline=baseline)
    texts = [t.get_text() for t in canvas.ax.texts]
    assert "0.90" in texts and "(+10.0%)" in texts and "(-10.0%)" in texts
    assert "previous run" in canvas.ax.get_title()
    colors = {t.get_position(): t.get_color() for t in canvas.ax.texts}
    assert colors[(0, 0.22)] == delta_color(10, True, True)  # diagonal gain: green
    assert colors[(0, 1.22)] == delta_color(10, False, False)  # more confusion b -> a: red
    canvas.plot_confusion([[9, 1], [2, 8]], ["a", "b"], normalized=True)
    assert "90%" in [t.get_text() for t in canvas.ax.texts]
