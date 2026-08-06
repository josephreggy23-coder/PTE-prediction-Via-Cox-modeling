"""Draw the model framework figure used in the README.

This is documentation, not analysis. It renders the causal and analytic
chain the study tests: mechanical injury -> cortical network state ->
inhibitory reserve -> latency to PTZ-evoked stage III seizure, and the
validation scaffold wrapped around it.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "docs/model_framework.png"

INK = "#14181d"
SUB = "#5b6670"
BRAIN = "#7b2d26"          # injury / biology
SIGNAL = "#2d5f7c"         # electrophysiology
LATENT = "#4c6478"         # the unobserved state
OUTCOME = "#41613a"        # seizure outcome
FRAME = "#8a7a52"          # analysis scaffold
PAPER = "#ffffff"
FAINT = "#eef1f4"

TITLE_DY = 0.042           # title baseline below box top
BODY_DY = 0.086            # first body line below box top
LINE_DY = 0.0305           # line spacing inside a box


def box(ax, x, y, w, h, title, lines, edge, fill,
        title_size=10.6, body_size=8.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - TITLE_DY, title, ha="center", va="center",
            fontsize=title_size, color=edge, weight="bold", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - BODY_DY - i * LINE_DY, ln, ha="center",
                va="center", fontsize=body_size,
                color=SUB if ln.startswith("(") else INK, zorder=3,
                style="italic" if ln.startswith("(") else "normal")


def arrow(ax, p0, p1, color=INK, label=None, lw=1.7):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=color, zorder=4, shrinkA=1, shrinkB=1))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.016, label,
                ha="center", va="bottom", fontsize=7.9, color=SUB,
                style="italic", zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(15.0, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    ax.text(0.010, 0.982,
            "Latent-period model: from mechanical injury to seizure "
            "susceptibility",
            fontsize=16, color=INK, weight="bold", va="top")
    ax.text(0.010, 0.947,
            "Larval zebrafish, 6 dpf, n = 100.  Outcome: latency to "
            "PTZ-evoked stage III seizure, right-censored at 1800 s.",
            fontsize=10.0, color=SUB, va="top")

    # ================= biological chain ==============================
    ax.text(0.010, 0.908, "THE BIOLOGICAL CHAIN THE MODEL ENCODES",
            fontsize=9.6, color=INK, weight="bold", va="top")

    Y, H = 0.606, 0.255
    box(ax, 0.010, Y, 0.213, H, "Mechanical injury",
        ["weight-drop, 100 g",
         "0 / 27 / 54 / 108 cm",
         "peak pressure 0-290 kPa",
         "(the dose of trauma)"], BRAIN, "#fbf4f3")

    box(ax, 0.258, Y, 0.246, H, "Cortical network state",
        ["aperiodic exponent = E:I balance",
         "flattening = toward excitation",
         "variance + lag-1 autocorrelation",
         "= critical slowing",
         "interictal-like discharge rate",
         "(recorded 2-4 h post-injury)"], SIGNAL, "#f2f7fa")

    box(ax, 0.539, Y, 0.196, H, "Inhibitory reserve",
        ["the latent state itself,",
         "not directly observable",
         "",
         "how much GABAergic restraint",
         "the network still has left"], LATENT, FAINT)

    box(ax, 0.770, Y, 0.220, H, "Seizure latency",
        ["PTZ 15 mM is a GABA-A",
         "antagonist: it titrates",
         "that reserve directly",
         "",
         "short latency = little reserve"], OUTCOME, "#f4f7f2")

    mid = Y + H / 2
    arrow(ax, (0.223, mid), (0.258, mid), BRAIN, "perturbs")
    arrow(ax, (0.504, mid), (0.539, mid), SIGNAL, "indexes")
    arrow(ax, (0.735, mid), (0.770, mid), LATENT, "unmasked by PTZ")

    # the comparison the study actually runs
    ax.add_patch(FancyArrowPatch(
        (0.116, Y - 0.006), (0.880, Y - 0.006), arrowstyle="-|>",
        mutation_scale=14, linewidth=1.6, color=BRAIN,
        linestyle=(0, (5, 3)), connectionstyle="arc3,rad=0.05", zorder=4))
    ax.text(0.498, 0.528,
            "M0  =  impact pressure alone.        M1  =  impact pressure + "
            "the LFP block.",
            ha="center", fontsize=10.2, color=BRAIN, weight="bold")
    ax.text(0.498, 0.499,
            "Does the brain's own signal know anything the impact gauge "
            "does not?",
            ha="center", fontsize=9.6, color=BRAIN, style="italic")

    # ================= analysis scaffold =============================
    ax.text(0.010, 0.474, "THE ANALYSIS BUILT AROUND IT",
            fontsize=9.6, color=INK, weight="bold", va="top")

    Y2, H2 = 0.198, 0.238
    box(ax, 0.010, Y2, 0.229, H2, "Survival model",
        ["elastic-net Cox PH",
         "censored larvae stay in the",
         "risk set, never recoded",
         "as non-seizing",
         "clutch = stratified baseline"], FRAME, "#fbf9f4")

    box(ax, 0.259, Y2, 0.229, H2, "Every choice in-fold",
        ["leave-one-fish-out, 100 folds",
         "scaling, tuning, selection",
         "all fitted inside the fold",
         "",
         "a canary test fails if any peeks"], FRAME, "#fbf9f4")

    box(ax, 0.508, Y2, 0.229, H2, "Null and diagnostics",
        ["1000 permutations, the whole",
         "pipeline refitted each time",
         "Schoenfeld residuals check PH",
         "(Weibull AFT if violated)",
         "events per predictor reported"], FRAME, "#fbf9f4")

    box(ax, 0.757, Y2, 0.233, H2, "Biology-first readout",
        ["every hazard ratio carries its",
         "expected direction, and an",
         "inversion is flagged, not hidden",
         "",
         "discrimination + backwards biology",
         "=  fitted noise"], OUTCOME, "#f4f7f2")

    mid2 = Y2 + H2 / 2
    for x in (0.239, 0.488, 0.737):
        arrow(ax, (x, mid2), (x + 0.020, mid2), FRAME, lw=1.4)

    # ================= deliberate exclusions =========================
    ax.add_patch(FancyBboxPatch(
        (0.010, 0.022), 0.980, 0.148,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.2, edgecolor=SUB, facecolor=FAINT, zorder=2))
    ax.text(0.026, 0.150, "HELD OUT OF THE MODEL ON PURPOSE",
            fontsize=9.0, color=INK, weight="bold", va="center")
    ax.text(
        0.026, 0.118,
        "condition  —  a label for the drop height, not a variable; using it "
        "would let the design itself leak in as a predictor",
        fontsize=8.8, color=INK, va="center")
    ax.text(
        0.026, 0.087,
        "lfp_clean_fraction, dlc_tracking_error_px  —  recording and tracking "
        "quality; they describe the measurement, not the animal",
        fontsize=8.8, color=INK, va="center")
    ax.text(
        0.026, 0.056,
        "beh_baseline_*  —  tested separately against impact pressure "
        "(question 6), because behaviour that merely restates the dose "
        "carries no independent information",
        fontsize=8.8, color=INK, va="center")

    fig.savefig(OUT, dpi=200, facecolor=PAPER, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
