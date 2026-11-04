"""Аналіз помилок і важливість ознак."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as cfg


def segment_frame(val, scs, best_a, best_b):
    seg = (val.groupby("rid").agg(n=("rid", "size"), race_type=("race_type", "first"), fav_p=("p_mkt", "max"))
              .sort_index())
    seg["random 1/N"] = 1.0 / seg["n"]
    seg["market"] = (scs["market"]["rt"]["win_rank"].to_numpy() == 1)
    seg[f"{best_a} (A)"] = (scs[f"{best_a} (A)"]["rt"]["win_rank"].to_numpy() == 1)
    seg[f"{best_b} (B)"] = (scs[f"{best_b} (B)"]["rt"]["win_rank"].to_numpy() == 1)
    seg["Розмір поля"] = pd.cut(seg["n"], [0, 7, 10, 14, 100], labels=["≤7", "8–10", "11–14", "15+"])
    seg["Упевненість ринку (макс. p)"] = pd.cut(seg["fav_p"], [0, 0.2, 0.3, 0.4, 1.0],
                                                labels=["<0.2", "0.2–0.3", "0.3–0.4", ">0.4"])
    seg["Тип забігу"] = seg["race_type"]
    methods = ["random 1/N", f"{best_a} (A)", "market", f"{best_b} (B)"]
    return seg, methods


SEGMENT_COLUMNS = ["Розмір поля", "Упевненість ринку (макс. p)", "Тип забігу"]


def segment_tables(seg, methods, columns=SEGMENT_COLUMNS):
    out = []
    for col in columns:
        tab = seg.groupby(col, observed=True)[methods].mean()
        size = seg.groupby(col, observed=True).size().rename("забігів")
        out.append((col, pd.concat([tab.round(3), size], axis=1)))
    return out


def plot_segments(seg, methods, path, columns=SEGMENT_COLUMNS):
    fig, axes = plt.subplots(1, len(columns), figsize=(15, 4), sharey=True)
    for ax, col in zip(axes, columns):
        seg.groupby(col, observed=True)[methods].mean().plot(kind="bar", ax=ax, rot=0, legend=(col == columns[0]))
        ax.set_title(f"Top-1 за сегментами: {col}")
        ax.set_xlabel("")
    axes[0].set_ylabel("частка вгаданих переможців")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def top_misses(val, sc, n=10):
    dd = val[["rid", "horseName", "jockeyName", "course", "date", "p_mkt", cfg.TARGET]].copy()
    dd["p_model"] = sc["p"]
    dd = dd.sample(frac=1, random_state=cfg.SEED)
    dd["rank"] = dd.groupby("rid")["p_model"].rank(ascending=False, method="first")
    pick = dd[dd["rank"] == 1].set_index("rid")
    winner = dd[dd[cfg.TARGET] == 1].set_index("rid")
    miss = pick[pick[cfg.TARGET] == 0].sort_values("p_model", ascending=False).head(n)
    return pd.DataFrame({
        "дата": miss["date"], "траса": miss["course"], "обраний кінь": miss["horseName"],
        "p моделі": miss["p_model"].round(3), "p ринку": miss["p_mkt"].round(3),
        "переможець": winner.loc[miss.index, "horseName"],
        "p моделі для переможця": winner.loc[miss.index, "p_model"].round(3)})


def permutation_importances(pipe, cols, val, n_rows=100_000, n_repeats=3):
    smp = val.sample(min(n_rows, len(val)), random_state=cfg.SEED)
    pi = permutation_importance(pipe, smp[cols], smp[cfg.TARGET], scoring="neg_log_loss",
                                n_repeats=n_repeats, random_state=cfg.SEED, n_jobs=1)
    return pd.Series(pi.importances_mean, index=cols).sort_values()


def plot_importances(imp, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("зростання log-loss після перемішування ознаки")
    ax.set_title("Важливість ознак, LightGBM (A, без ринку), validation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
