# %% CELL EXT.FIG.0 — Figure infrastructure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
FIGDIR = EXT_OUT / "figures"
STATUS_COLORS = {"pass":"#2e7d32","warning":"#f9a825","fatal":"#c62828",
                 "blocker":"#7b1fa2","scope_excluded":"#9e9e9e","na":"#e0e0e0"}
def save(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"[EXT.FIG] saved {name}.pdf/.png")
