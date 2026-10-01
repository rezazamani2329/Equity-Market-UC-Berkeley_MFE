"""Redraw Part 3 report figures from the saved results CSVs (no data pull needed).

    python scripts/make_p3_figures.py

Reads results/p3_horizon_ic.csv and writes figures/p3_ic_by_horizon.png.
Rerun after scripts/run_part3.py whenever the Part 3 results change.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Short legend labels keyed by the signal names run_part3.py writes.
LABELS = {
    "conditional_signal (common - shock)": "Composite (common − shock)",
    "common_lag (continuation leg)": "Common leg",
    "shock_lag (reversion leg, raw sign)": "Shock leg (raw sign)",
    "leader_ret_lag (part 1 baseline, unconditional)": "Part 1 baseline (leader return)",
    "reversal_signal (naive, -own_lag)": "Naive own-return reversal",
}
STYLE = {
    "Composite (common − shock)": dict(color="#c0392b", lw=2.6, marker="o"),
    "Common leg": dict(color="#1f6fb2", lw=2.0, marker="s"),
    "Shock leg (raw sign)": dict(color="#7f8c8d", lw=1.6, marker="^", ls="--"),
    "Part 1 baseline (leader return)": dict(color="#2c3e50", lw=1.6, marker="D", ls=":"),
    "Naive own-return reversal": dict(color="#e67e22", lw=2.0, marker="v"),
}


def ic_by_horizon() -> Path:
    ic = pd.read_csv(RESULTS / "p3_horizon_ic.csv")
    ic["label"] = ic["signal"].map(LABELS).fillna(ic["signal"])
    horizons = sorted(ic["horizon"].unique())
    xpos = {h: i for i, h in enumerate(horizons)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    panels = [
        (axes[0], [l for l in STYLE if l != "Naive own-return reversal"],
         "Leader-based signals"),
        (axes[1], list(STYLE), "All signals, incl. naive reversal"),
    ]
    for ax, labels, title in panels:
        for label in labels:
            d = ic[ic["label"] == label].sort_values("horizon")
            if d.empty:
                continue
            ax.plot(d["horizon"].map(xpos), d["mean_ic"], label=label, **STYLE[label])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"{h}d" for h in horizons])
        ax.set_xlabel("Forward horizon (trading days after signal date)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean daily rank IC")
    axes[1].legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.suptitle("Part 3: mean rank IC by horizon (corrected one-day alignment)",
                 fontsize=12.5, y=1.0)
    fig.text(0.01, -0.02, "Source: results/p3_horizon_ic.csv. Horizons beyond 1d use "
             "overlapping forward returns.", fontsize=8, color="#555")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "p3_ic_by_horizon.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", ic_by_horizon())
