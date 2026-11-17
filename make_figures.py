"""Figures 1-4 (+ comparison figure) from the logs in results/logs/. Every number plotted is read
from a log written during training/evaluation; nothing is hard-coded."""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "results", "logs")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# validated categorical slots 1-3 (one colour per entity, fixed everywhere)
C = {"sft": "#2a78d6", "ppo": "#eb6834", "dpo": "#1baf7a", "ref": "#898781"}
INK, INK2, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "font.size": 9.5, "lines.linewidth": 2.0, "legend.frameon": False,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
})


def _read(name):
    with open(os.path.join(LOG, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _smooth(y, k=25):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig1_sft():
    rows = _read("sft_log.jsonl")
    steps = [r for r in rows if "step_loss" in r]
    ev = [r for r in rows if r.get("val_loss") is not None]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot([r["step"] for r in steps], [r["step_loss"] for r in steps], color=C["sft"], alpha=0.18, lw=1)
    ax.plot([r["step"] for r in steps], _smooth([r["step_loss"] for r in steps]), color=C["sft"],
            label="train loss (25-step moving avg.)")
    ax.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], color=INK, marker="o", ms=5,
            label="validation loss (500 pairs)")
    ax.set(xlabel="optimizer step (batch 16)", ylabel="NLL per summary token",
           title="Figure 1 - SFT training and validation loss")
    ax.legend()
    return _save(fig, "fig1_sft_loss.png")


def fig2_reward():
    rows = _read("reward_log.jsonl")
    steps = [r for r in rows if "step_loss" in r]
    ev = [r for r in rows if r.get("val_acc") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot([r["step"] for r in steps], [r["step_loss"] for r in steps], color=C["sft"], alpha=0.15, lw=1)
    ax.plot([r["step"] for r in steps], _smooth([r["step_loss"] for r in steps], 50), color=C["sft"],
            label="train loss (50-step avg.)")
    ax.axhline(np.log(2), color=AXIS, lw=1, ls="--")
    ax.text(steps[-1]["step"], np.log(2), " log 2", va="bottom", ha="right", color=INK2)
    ax.set(xlabel="optimizer step (8 pairs)", ylabel="Bradley-Terry loss", title="Reward model - training loss")
    ax.legend()
    ax = axes[1]
    ax.plot([r["step"] for r in ev], [100 * r["val_acc"] for r in ev], color=INK, marker="o", ms=5)
    ax.axhline(50, color=AXIS, lw=1, ls="--")
    ax.text(0, 50.4, "chance (50%)", color=INK2)
    ax.annotate(f"{100 * ev[-1]['val_acc']:.1f}%", (ev[-1]["step"], 100 * ev[-1]["val_acc"]),
                textcoords="offset points", xytext=(-10, -4), ha="right", va="top", color=INK)
    ax.set(xlabel="optimizer step", ylabel="pairwise accuracy (%)",
           title="Reward model - validation accuracy (1,000 pairs)")
    ax.set_ylim(48, max(100 * r["val_acc"] for r in ev) + 3)
    fig.suptitle("Figure 2 - Reward model", fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "fig2_reward_model.png")


def fig3_ppo():
    rows = _read("ppo_log.jsonl")
    it = [r["iter"] for r in rows]
    panels = [("score", "mean $r_\\phi$ score", "Mean reward-model score"),
              ("kl", "nats (sum over summary)", "Mean KL to reference"),
              ("length", "tokens", "Mean summary length"),
              ("clipfrac", "share of tokens", "Clip fraction (mean of K=4 epochs)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (key, ylab, title) in zip(axes.flat, panels):
        y = [r[key] for r in rows]
        ax.plot(it, y, color=C["ppo"], alpha=0.25, lw=1)
        ax.plot(it, _smooth(y, 10), color=C["ppo"], label="10-iteration moving avg.")
        ax.set(xlabel="PPO iteration", ylabel=ylab, title=title)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(f"Figure 3 - PPO training ({len(rows)} iterations x 16 prompts)", fontweight="bold", color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    p = _save(fig, "fig3_ppo_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (key, title) in zip(axes, [("pg_loss", "Policy loss  $-L^{CLIP}$"),
                                       ("vf_loss", "Value loss  $E[(V-R)^2]$"),
                                       ("entropy", "Entropy  $E[H]$")]):
        y = [r[key] for r in rows]
        ax.plot(it, y, color=C["ppo"], alpha=0.25, lw=1)
        ax.plot(it, _smooth(y, 10), color=C["ppo"])
        ax.set(xlabel="PPO iteration", title=title)
    fig.suptitle("Figure 3b - PPO loss terms", fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "fig3b_ppo_losses.png")
    return p


def fig4_dpo():
    rows = _read("dpo_log.jsonl")
    steps = [r for r in rows if "step_loss" in r]
    ev = [r for r in rows if r.get("val_acc") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot([r["step"] for r in steps], [r["step_loss"] for r in steps], color=C["dpo"], alpha=0.15, lw=1)
    ax.plot([r["step"] for r in steps], _smooth([r["step_loss"] for r in steps], 50), color=C["dpo"],
            label="train loss (50-step avg.)")
    ax.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], color=INK, marker="o", ms=5,
            label="validation loss")
    ax.axhline(np.log(2), color=AXIS, lw=1, ls="--")
    ax.set(xlabel="optimizer step (8 pairs)", ylabel="DPO loss", title="DPO - loss")
    ax.legend()
    ax = axes[1]
    ax.plot([r["step"] for r in ev], [100 * r["val_acc"] for r in ev], color=INK, marker="o", ms=5)
    ax.axhline(50, color=AXIS, lw=1, ls="--")
    ax.annotate(f"{100 * ev[-1]['val_acc']:.1f}%", (ev[-1]["step"], 100 * ev[-1]["val_acc"]),
                textcoords="offset points", xytext=(-10, -4), ha="right", va="top", color=INK)
    ax.set(xlabel="optimizer step", ylabel="implicit reward accuracy (%)",
           title="DPO - validation implicit reward accuracy")
    ax.set_ylim(-3, max(100 * r["val_acc"] for r in ev) + 8)
    ax.annotate("step 0: every margin is exactly 0\n(policy = reference), so no pair counts as > 0",
                (ev[0]["step"], 100 * ev[0]["val_acc"]), xytext=(ev[-1]["step"] * 0.3, 12),
                color=INK2, fontsize=8, va="bottom", arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    fig.suptitle("Figure 4 - DPO", fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "fig4_dpo.png")


def fig_comparison():
    """Small multiples (one axis each) of the Table 3 metrics."""
    with open(os.path.join(ROOT, "results", "table3_comparison.json"), encoding="utf-8") as f:
        t = json.load(f)
    rows = t["rows"]
    names = [r["policy"] for r in rows]
    cols = [C[n.lower()] for n in names]
    metrics = [("mean_score", "Mean $r_\\phi$"), ("kl_to_sft", "KL to SFT (nats)"),
               ("length", "Length (tokens)"), ("rougeL", "ROUGE-L F1")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, (k, title) in zip(axes, metrics):
        vals = [r[k] for r in rows]
        bars = ax.bar(names, vals, color=cols, width=0.6, edgecolor=SURF, linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -4), ha="center", va="bottom" if v >= 0 else "top",
                        color=INK)
        lo, hi = min(vals + [0]), max(vals + [0])
        ax.set_ylim(lo - 0.18 * (hi - lo) if lo < 0 else 0, hi + 0.12 * (hi - lo))
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    if t.get("reference_mean_score") is not None:
        axes[0].axhline(t["reference_mean_score"], color=C["ref"], ls="--", lw=1)
        axes[0].text(2.45, t["reference_mean_score"], "human ref.", color=INK2, ha="right", va="bottom")
    fig.suptitle("Figure 6 - SFT vs PPO vs DPO on 300 held-out posts", fontweight="bold", color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "fig6_comparison.png")


def fig_length_hist(prompt_lens, summary_lens):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, data, title, lim in [(axes[0], prompt_lens, "Prompt length (SFT train)", 448),
                                 (axes[1], summary_lens, "Summary length incl. EOS (SFT train)", 48)]:
        ax.hist(data, bins=40, color=C["sft"], edgecolor=SURF, linewidth=1)
        ax.axvline(np.mean(data), color=INK, lw=1.5)
        ax.text(np.mean(data), ax.get_ylim()[1] * 0.92, f" mean {np.mean(data):.1f}", color=INK)
        ax.axvline(lim, color=AXIS, ls="--", lw=1)
        ax.set(xlabel="tokens", ylabel="posts", title=title)
    fig.suptitle("Figure 0 - Token lengths of the 20,000 SFT training pairs", fontweight="bold", color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "fig0_token_lengths.png")


if __name__ == "__main__":
    for f in (fig1_sft, fig2_reward, fig3_ppo, fig4_dpo, fig_comparison):
        try:
            print(f())
        except FileNotFoundError as e:
            print("skip", f.__name__, "-", e)
