"""Regenerate the SafeTriageNet cover image at higher resolution.

Produces a 1680 x 840 PNG (same 2:1 aspect ratio the Kaggle cover field expects)
with a crisp vector-style composition: dark navy backdrop, red ECG trace, a
green medical shield, a lightweight stacking / ensemble motif, and the
SafeTriageNet title block.

Run from the repository root:
    python build/build_cover_image.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "assets" / "cover_image.png"

# Target raster size. Kaggle competition spec: 560 x 280 px, 2:1 aspect ratio.
W_PX, H_PX = 560, 280
DPI = 140

BG = "#0a1929"        # dark navy
ECG = "#ef4444"       # clean red
SHIELD = "#22c55e"    # vivid green
NET = "#22c55e"
GRID = "#132238"


def ecg_waveform(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays for a stylised multi-beat ECG trace."""
    x = np.linspace(0, 1, n)
    y = np.zeros_like(x)

    # Baseline with mild noise
    rng = np.random.default_rng(7)
    y += rng.normal(0.0, 0.008, n)

    def beat(center: float, amp: float = 1.0) -> np.ndarray:
        """PQRST-ish shape centered at `center` in [0, 1] units."""
        t = (x - center) * 60  # stretch the beat
        shape = np.zeros_like(t)
        # P wave (small bump)
        shape += 0.12 * amp * np.exp(-((t + 1.6) ** 2) / 0.45)
        # Q dip
        shape += -0.08 * amp * np.exp(-((t + 0.35) ** 2) / 0.05)
        # R spike
        shape += 1.0 * amp * np.exp(-((t - 0.0) ** 2) / 0.012)
        # S dip
        shape += -0.22 * amp * np.exp(-((t - 0.35) ** 2) / 0.05)
        # T wave
        shape += 0.22 * amp * np.exp(-((t - 1.6) ** 2) / 0.6)
        return shape

    # Place several beats across the width
    for c in (0.08, 0.22, 0.36, 0.70, 0.84):
        y += beat(c)

    return x, y


def draw_shield(ax, cx: float, cy: float, size: float) -> None:
    """Draw a medical shield with a white cross at (cx, cy)."""
    # Shield outline via a Bezier path
    w = size
    h = size * 1.15
    verts = [
        (cx - w / 2, cy + h * 0.40),       # top-left
        (cx + w / 2, cy + h * 0.40),       # top-right
        (cx + w / 2, cy - h * 0.05),       # right-mid
        (cx, cy - h * 0.55),               # bottom tip
        (cx - w / 2, cy - h * 0.05),       # left-mid
        (cx - w / 2, cy + h * 0.40),       # close
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 4 + [MplPath.CLOSEPOLY]
    path = MplPath(verts, codes)
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=SHIELD,
            linewidth=4.5,
            joinstyle="round",
        )
    )

    # Medical cross inside the shield (white)
    cross_arm = size * 0.22
    cross_thick = size * 0.08
    ax.add_patch(
        FancyBboxPatch(
            (cx - cross_arm, cy - cross_thick / 2),
            cross_arm * 2,
            cross_thick,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=SHIELD,
            edgecolor="none",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (cx - cross_thick / 2, cy - cross_arm),
            cross_thick,
            cross_arm * 2,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=SHIELD,
            edgecolor="none",
        )
    )


def draw_stacking_motif(ax, left: float, right: float, top: float, bottom: float) -> None:
    """Draw a small 3-layer feed-forward graph to suggest the stacking ensemble."""
    layers = [5, 5, 3]  # input / hidden / output columns of nodes
    n_layers = len(layers)
    xs = np.linspace(left, right, n_layers)
    positions: list[list[tuple[float, float]]] = []
    for x, n in zip(xs, layers):
        ys = np.linspace(bottom, top, n)
        positions.append([(x, y) for y in ys])

    # Edges first (so nodes draw on top)
    for li in range(n_layers - 1):
        for src in positions[li]:
            for dst in positions[li + 1]:
                ax.plot(
                    [src[0], dst[0]],
                    [src[1], dst[1]],
                    color=NET,
                    alpha=0.35,
                    linewidth=1.3,
                )

    # Nodes
    r = (right - left) * 0.022
    for layer in positions:
        for (x, y) in layer:
            ax.add_patch(
                Circle(
                    (x, y),
                    r,
                    facecolor=BG,
                    edgecolor=NET,
                    linewidth=2.2,
                )
            )


def main() -> None:
    fig = plt.figure(
        figsize=(W_PX / DPI, H_PX / DPI),
        dpi=DPI,
        facecolor=BG,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)   # 2:1 aspect
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Faint horizontal rule along the ECG baseline
    ax.axhline(0.215, color=GRID, linewidth=1.0, zorder=1)

    # ECG trace
    x, y = ecg_waveform()
    y_scaled = 0.215 + y * 0.085  # amplitude tuned for the 0..0.5 vertical range
    ax.plot(x, y_scaled, color=ECG, linewidth=3.2, zorder=3)

    # Centered shield (above the ECG baseline)
    draw_shield(ax, cx=0.50, cy=0.22, size=0.085)

    # Stacking motif on the right
    draw_stacking_motif(ax, left=0.70, right=0.965, top=0.33, bottom=0.11)

    # Title block at the top-center. Font sizes scale with the figure width
    # so the same composition works whether we render 560x280 or larger.
    title_size = 44 * (W_PX / 1680)
    subtitle_size = 18 * (W_PX / 1680)
    ax.text(
        0.5, 0.435,
        "SafeTriageNet",
        color="white",
        fontsize=title_size,
        fontweight="bold",
        ha="center",
        va="center",
        family="DejaVu Sans",
    )
    ax.text(
        0.5, 0.38,
        "Safety-Aware Multimodal Triage",
        color="#cbd5e1",
        fontsize=subtitle_size,
        ha="center",
        va="center",
        family="DejaVu Sans",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT,
        dpi=DPI,
        facecolor=BG,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)

    # Verify size
    from PIL import Image
    im = Image.open(OUT)
    print(f"Wrote {OUT} at {im.size[0]}x{im.size[1]} px")


if __name__ == "__main__":
    main()
