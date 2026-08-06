"""The only two analysis figures: permutation null and calibration curve."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402

INK = "#1b1b1b"
ACCENT = "#b3341f"
MUTED = "#9aa0a6"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK, labelsize=9, length=4, width=0.9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def permutation_null(null_results: list, path):
    """One panel per model: null distribution with the observed value."""
    n = len(null_results)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.1), squeeze=False)

    for ax, res in zip(axes[0], null_results):
        null = np.asarray(res["null_distribution"])
        obs = res["observed_c_index"]

        ax.hist(null, bins=34, color=MUTED, alpha=0.75,
                edgecolor="white", linewidth=0.5)
        ax.axvline(np.percentile(null, 95), color=INK, ls=":", lw=1.2,
                   label=f"null 95th pct = {np.percentile(null, 95):.3f}")
        ax.axvline(obs, color=ACCENT, lw=2.2,
                   label=f"observed c = {obs:.3f}")
        ax.axvline(0.5, color=INK, ls="--", lw=0.8, alpha=0.5,
                   label="chance = 0.500")

        ax.set_xlabel("cross-validated concordance index")
        ax.set_ylabel("permutations")
        ax.set_title(
            f"{res['label']}: outcome-shuffled null "
            f"({res['n_permutations']} refits)\n"
            f"p = {res['p_value']:.4f},  z = {res['z_vs_null']:+.2f}",
            fontsize=10.5, loc="left",
        )
        # The observed value sits in the right tail by construction, so
        # the legend goes left or it lands on top of the finding.
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        _style(ax)

    fig.suptitle(
        "Permutation null: the whole pipeline refitted end to end on shuffled "
        "outcomes",
        fontsize=11.5, color=INK, x=0.01, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def calibration(cal: dict, cv_result, path):
    """Left: grouped predicted vs observed. Right: per-larva scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.0))

    # ---- grouped calibration -----------------------------------------
    pts = [c for c in cal["curve"]
           if c["predicted_median_s"] and c["observed_km_median_s"]]
    if pts:
        px = [c["predicted_median_s"] for c in pts]
        py = [c["observed_km_median_s"] for c in pts]
        lim = [0, max(max(px), max(py)) * 1.15]
        ax1.plot(lim, lim, ls="--", lw=1.0, color=INK, alpha=0.6,
                 label="perfect calibration")
        ax1.plot(px, py, "o-", color=ACCENT, ms=8, lw=1.4,
                 label="risk quartiles")
        for c in pts:
            ax1.annotate(f"Q{c['risk_group']}",
                         (c["predicted_median_s"], c["observed_km_median_s"]),
                         textcoords="offset points", xytext=(7, -11),
                         fontsize=8.5, color=INK)
        ax1.set_xlim(lim)
        ax1.set_ylim(lim)
    ax1.set_xlabel("predicted median latency (s)")
    ax1.set_ylabel("observed Kaplan-Meier median latency (s)")
    ax1.set_title(
        f"Grouped calibration\nslope (risk domain) = "
        f"{cal['calibration_slope_risk_domain']:.3f} "
        f"[{cal['calibration_slope_ci'][0]:.3f}, "
        f"{cal['calibration_slope_ci'][1]:.3f}], target 1.000",
        fontsize=10.5, loc="left",
    )
    ax1.legend(frameon=False, fontsize=8.5)
    _style(ax1)

    # ---- per-larva ----------------------------------------------------
    t, e = cv_result.time, cv_result.event
    pred = cv_result.pred_median_s
    horizon = cal["observation_window_s"]
    finite = np.isfinite(pred)

    m_ev = (e == 1) & finite
    m_cn = (e == 0) & finite
    ax2.scatter(pred[m_ev], t[m_ev], s=26, color=ACCENT, alpha=0.75,
                edgecolor="white", linewidth=0.5,
                label=f"reached stage III (n={m_ev.sum()})")
    ax2.scatter(pred[m_cn], t[m_cn], s=42, facecolor="none", color=INK,
                marker="^", linewidth=1.1,
                label=f"censored at {int(horizon)} s (n={m_cn.sum()})")

    hi = max(np.nanmax(pred[finite]) if finite.any() else horizon, horizon)
    ax2.plot([0, hi], [0, hi], ls="--", lw=1.0, color=INK, alpha=0.6)
    ax2.axhline(horizon, color=MUTED, lw=0.9, ls=":")
    ax2.set_xlabel("predicted median latency (s)")
    ax2.set_ylabel("observed latency (s)")
    ax2.set_title(
        f"Per larva (held out)\nmedian absolute error = "
        f"{cal['median_absolute_error_s']:.0f} s,  bias = "
        f"{cal['mean_signed_bias_s']:+.0f} s",
        fontsize=10.5, loc="left",
    )
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax2)

    fig.suptitle(
        "Calibration: are the predicted seizure latencies right in seconds, "
        "not just in order?",
        fontsize=11.5, color=INK, x=0.01, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path
