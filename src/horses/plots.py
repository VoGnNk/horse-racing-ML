"""Графіки порівняння моделей (кожна функція зберігає рисунок у файл і показує його)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import ci, metrics_row, paired


def dot_ci(ax, names, boots, scs, metric, refs=(), title="", ylabel=""):
    ys = [metrics_row(k, scs[k])[metric] for k in names]
    lo = [ci(boots, k, metric)[0] for k in names]
    hi = [ci(boots, k, metric)[1] for k in names]
    x = np.arange(len(names))
    ax.errorbar(x, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt="o", capsize=5, color="C0")
    for j, ref in enumerate(refs):
        ax.axhline(metrics_row(ref, scs[ref])[metric], color=f"C{j + 1}", ls="--", label=ref)
    if refs:
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (A)", "").replace(" (B)", "") for n in names], rotation=15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def plot_model_comparison(scs, boots, a_names, b_names, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    dot_ci(axes[0, 0], a_names, boots, scs, "top1", ["random 1/N", "market"], "A (без ринку): top-1", "частка вгаданих переможців")
    dot_ci(axes[0, 1], b_names, boots, scs, "top1", ["market"], "B (з ринком): top-1")
    dot_ci(axes[1, 0], a_names, boots, scs, "log_loss", ["random 1/N", "market"], "A (без ринку): log-loss (менше = краще)", "log-loss")
    dot_ci(axes[1, 1], b_names, boots, scs, "log_loss", ["market", "market + logistic"], "B (з ринком): log-loss (менше = краще)")
    fig.suptitle("Порівняння моделей на validation (2017–2018), 95% ДІ по забігах")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def plot_hit_by_rank(scs, a_names, b_names, path, max_k=6):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ks = np.arange(1, max_k + 1)
    for ax, names, ref_name, title in ((axes[0], a_names, "random 1/N", "A: переможець серед k найкращих оцінок"),
                                       (axes[1], b_names, "market", "B: переможець серед k найкращих оцінок")):
        for k in names + [ref_name]:
            wr = scs[k]["rt"]["win_rank"]
            ax.plot(ks, [(wr <= j).mean() for j in ks], marker="o", ls="--" if k == ref_name else "-", label=k)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("частка забігів")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def reliability(sc, bins=20):
    d = pd.DataFrame({"p": sc["p"], "y": sc["y"]})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"))
    return g["p"].to_numpy(), g["y"].to_numpy()


def plot_calibration(scs, names, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for k in names:
        px, py = reliability(scs[k])
        ax.plot(px, py, marker="o", ms=4, label=k)
    lim = [0.005, 0.6]
    ax.plot(lim, lim, color="gray", ls="--", label="ідеальне калібрування")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("передбачена ймовірність (нормована по забігу)")
    ax.set_ylabel("фактична частота перемог")
    ax.set_title("Калібрування на validation (квантильні біни)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def forest(ax, models, boots, metric, scale, refs=("market", "market + logistic")):
    seen = set()
    for i, k in enumerate(models):
        used = [r for r in refs if r != k]
        for j, ref in enumerate(used):
            est, lo, hi = paired(boots, k, ref, metric)
            ax.errorbar(est * scale, i + (j - (len(used) - 1) / 2) * 0.25, fmt="o", capsize=4, color=f"C{refs.index(ref)}",
                        xerr=[[(est - lo) * scale], [(hi - est) * scale]],
                        label=None if ref in seen else f"відносно {ref}")
            seen.add(ref)
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)


def plot_paired_differences(boots, models, suptitle, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, metric, scale, label in ((axes[0], "top1", 100, "Δ top-1, в.п. (більше = краще)"),
                                     (axes[1], "log_loss", 1000, "Δ log-loss ×1000 (менше = краще)")):
        forest(ax, models, boots, metric, scale)
        ax.set_xlabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
