#!/usr/bin/env python3
"""Render the five paper-facing plots from a reproduced summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#17669b"
ORANGE = "#b85b25"
GRAY = "#515963"


def save(fig, target):
    fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white", metadata={"Software": None})
    plt.close(fig)


def get_result(rows, model=None, contrast=None):
    return next(row for row in rows if (model is None or row.get("model") == model) and row["contrast"] == contrast)


def make_concepts(out):
    fig, ax = plt.subplots(figsize=(8.4, 2.4))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [
        (0.03, "Mean stability", "Do aggregate outcomes agree?"),
        (0.365, "Behavioral stability", "Do the same tasks succeed?"),
        (0.70, "Estimand stability", "Do treatment gains agree?"),
    ]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.28), 0.27, 0.52, boxstyle="round,pad=0.018", facecolor="#f7f9fb", edgecolor=BLUE, linewidth=1.2))
        ax.text(x + 0.135, 0.62, title, ha="center", va="center", fontweight="bold", color=BLUE)
        ax.text(x + 0.135, 0.43, body, ha="center", va="center", fontsize=9)
    for x in (0.305, 0.64):
        ax.annotate("", (x + 0.05, 0.54), (x, 0.54), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.4})
    ax.text(0.5, 0.08, "These properties are distinct.", ha="center", fontsize=9, color=GRAY)
    save(fig, out / "figure1_stability_concepts.png")


def make_k(summary, out):
    row = summary["k_transition"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].barh(["Losses", "Gains"], [-row["losses"], row["gains"]], color=[ORANGE, BLUE], height=0.55)
    axes[0].axvline(0, color=GRAY, linewidth=0.8)
    axes[0].set(xlim=(-500, 500), xlabel="Paired outcomes", title="Corrective transition")
    axes[0].text(-row["losses"] - 12, 0, str(row["losses"]), ha="right", va="center", fontweight="bold")
    axes[0].text(row["gains"] + 12, 1, str(row["gains"]), ha="left", va="center", fontweight="bold")
    axes[1].barh(["K4 to K4R"], [100 * row["turnover"]], color=BLUE, height=0.45)
    axes[1].set(xlim=(0, 35), xlabel="Paired outcome switches (%)", title=f"{row['switches']:,} of {row['pairs']:,} pairs")
    axes[1].text(100 * row["turnover"] + 1, 0, f"{100 * row['turnover']:.2f}%", va="center")
    fig.tight_layout()
    save(fig, out / "figure2_k_switching.png")


def make_r2(summary, out):
    rows = [row for row in summary["r2"]["strict_results"] if row["row_type"] == "tau"]
    refs = ["BASE", "MASK_SENTINEL", "METADATA_TOKEN_MATCHED_TO_MASK"]
    labels = ["BASE", "MASK", "Metadata"]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), sharex=True)
    for ax, model in zip(axes, ["gemma", "llama", "deepseek"]):
        selected = [get_result(rows, model, ref) for ref in refs]
        x = [100 * row["estimate"] for row in selected]
        lo = [100 * (row["estimate"] - row["simultaneous_ci_low"]) for row in selected]
        hi = [100 * (row["simultaneous_ci_high"] - row["estimate"]) for row in selected]
        ax.errorbar(x, [2, 1, 0], xerr=[lo, hi], fmt="o", color=BLUE, capsize=3)
        ax.set(yticks=[2, 1, 0], yticklabels=labels, title=model.title(), xlabel="Strict gain (pp)", xlim=(-7, 42), ylim=(-0.6, 2.6))
        ax.axvline(0, color=GRAY, linewidth=0.8)
        ax.grid(axis="x", alpha=0.16)
    fig.tight_layout()
    save(fig, out / "figure3_r2_effects.png")


def make_interaction(summary, out):
    discoveries = summary["r2"]["strict_results"]
    confirmations = summary["gemma_confirmation"]["strict_results"]
    r2 = get_result(discoveries, "gemma", "BASE_minus_MASK_SENTINEL")
    confirm = get_result(confirmations, contrast="Delta_BASE_MASK")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for y, row, color, marker, label in ((1, r2, ORANGE, "s", "R2 discovery"), (0, confirm, BLUE, "o", "Prospective confirmation")):
        x = 100 * row["estimate"]
        lo = 100 * (row["estimate"] - row["simultaneous_ci_low"])
        hi = 100 * (row["simultaneous_ci_high"] - row["estimate"])
        ax.errorbar([x], [y], xerr=[[lo], [hi]], fmt=marker, color=color, capsize=4, label=label)
        ax.text(x, y + 0.16, f"{x:.2f} pp", ha="center", color=color, fontsize=9)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.axvline(5, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set(yticks=[1, 0], yticklabels=["R2 discovery", "Prospective confirmation"], xlabel="BASE minus MASK interaction (pp)", xlim=(-5, 32), ylim=(-0.5, 1.5))
    ax.grid(axis="x", alpha=0.16)
    fig.tight_layout()
    save(fig, out / "figure4_discovery_confirmation.png")


def make_evidence_flow(summary, out):
    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, title, body, color=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018", facecolor="#f7f9fb", edgecolor=color, linewidth=1.1))
        ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center", fontweight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=8.5, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.2})

    k = summary["k_transition"]
    confirm = summary["gemma_confirmation"]
    s1b = summary["s1b"]
    box(0.03, 0.56, 0.25, 0.34, "K-series", f"{k['switches']:,} paired switches\n+0.052 pp aggregate drift", GRAY)
    box(0.375, 0.56, 0.25, 0.34, "R2 discovery", "Three models\nHeterogeneous evidence")
    box(0.72, 0.56, 0.25, 0.34, "Gemma confirmation", f"{confirm['unique_tasks']} untouched tasks\nMagnitude confirmed")
    arrow(0.29, 0.73, 0.36, 0.73)
    arrow(0.64, 0.73, 0.705, 0.73)
    box(0.10, 0.06, 0.34, 0.31, "Portability boundary", "Stage B qualification failed\n8 of 27 exceeded tolerance", GRAY)
    box(0.56, 0.06, 0.34, 0.31, "Endpoint boundary", f"S1 BASE–MASK: {100 * next(r['estimate'] for r in summary['s1']['strict_results'] if r['contrast']=='Delta_BASE_MASK'):.2f} pp\nS1B status: {s1b['status']}", GRAY)
    arrow(0.49, 0.53, 0.31, 0.39)
    arrow(0.82, 0.53, 0.74, 0.39)
    fig.tight_layout()
    save(fig, out / "figure5_evidence_flow.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/reproduced_summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    make_concepts(args.out)
    make_k(summary, args.out)
    make_r2(summary, args.out)
    make_interaction(summary, args.out)
    make_evidence_flow(summary, args.out)
    print(f"Rendered five figures in {args.out}")


if __name__ == "__main__":
    main()
