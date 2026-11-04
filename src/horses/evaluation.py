"""Метрики рівня забігу, bootstrap по забігах і парні порівняння моделей.

Принципи:
- ймовірності нормуються в межах забігу (сума = 1), бо переможець рівно один;
- нічиї в оцінках розв'язуються ВИПАДКОВО (рядки перемішуються), інакше порядок рядків дав би хибну перевагу;
- довірчі інтервали рахуються ресемплінгом ЗАБІГІВ (коні одного забігу залежні), усі моделі на тих самих ресемплах.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as cfg


def normalise_in_race(part, p):
    s = pd.Series(np.asarray(p, dtype="float64"), index=part.index)
    return (s / s.groupby(part["rid"]).transform("sum")).to_numpy()


def race_table(part, p):
    d = pd.DataFrame({"rid": part["rid"].to_numpy(), "y": part[cfg.TARGET].to_numpy(),
                      "p": p, "price": part["decimalPrice"].to_numpy()})
    d = d.sample(frac=1, random_state=cfg.SEED)                    
    d["rank"] = d.groupby("rid")["p"].rank(ascending=False, method="first")
    pc = d["p"].clip(cfg.EPS, 1 - cfg.EPS)
    d["ll"] = -(d["y"] * np.log(pc) + (1 - d["y"]) * np.log(1 - pc))
    g = d.groupby("rid")
    top = d[d["rank"] == 1].set_index("rid")
    return pd.DataFrame({
        "n": g.size(),
        "ll_sum": g["ll"].sum(),
        "win_rank": d[d["y"] == 1].groupby("rid")["rank"].min(),
        "roi": top["y"] / top["price"] - 1,
    }).sort_index()


def score(part, p_raw, normalise=True):
    p = normalise_in_race(part, p_raw) if normalise else np.asarray(p_raw, dtype="float64")
    return {"p": p, "y": part[cfg.TARGET].to_numpy(), "rt": race_table(part, p)}


def metrics_row(name, sc):
    rt = sc["rt"]
    return {"model": name,
            "top1": (rt["win_rank"] == 1).mean(),
            "top3": (rt["win_rank"] <= 3).mean(),
            "log_loss": rt["ll_sum"].sum() / rt["n"].sum(),
            "pr_auc": average_precision_score(sc["y"], sc["p"]),
            "roc_auc": roc_auc_score(sc["y"], sc["p"]),
            "roi_top1": rt["roi"].mean()}


def bootstrap(scs, n_boot=cfg.N_BOOT, seed=cfg.SEED):
    names = list(scs)
    R = len(scs[names[0]]["rt"])
    arr = {}
    for k in names:
        rt = scs[k]["rt"]
        arr[k] = {"top1": (rt["win_rank"].to_numpy() == 1).astype(float),
                  "top3": (rt["win_rank"].to_numpy() <= 3).astype(float),
                  "ll": rt["ll_sum"].to_numpy(), "n": rt["n"].to_numpy().astype(float),
                  "roi": rt["roi"].to_numpy()}
    rng = np.random.default_rng(seed)
    res = {k: {m: np.empty(n_boot) for m in ("top1", "top3", "log_loss", "roi")} for k in names}
    p = np.full(R, 1.0 / R)
    for b in range(n_boot):
        c = rng.multinomial(R, p)
        for k in names:
            a = arr[k]
            res[k]["top1"][b] = (c * a["top1"]).sum() / R
            res[k]["top3"][b] = (c * a["top3"]).sum() / R
            res[k]["log_loss"][b] = (c * a["ll"]).sum() / (c * a["n"]).sum()
            res[k]["roi"][b] = (c * a["roi"]).sum() / R
    return res


def ci(boots, k, m):
    lo, hi = np.percentile(boots[k][m], [2.5, 97.5])
    return lo, hi


def paired(boots, a, b, m):
    d = boots[a][m] - boots[b][m]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi


def summary_table(scs, boots):
    rows = []
    for k, sc in scs.items():
        r = metrics_row(k, sc)
        for m in ("top1", "log_loss", "roi"):
            lo, hi = ci(boots, k, m)
            r[m + "_95ci"] = f"[{lo:.4f}; {hi:.4f}]"
        rows.append(r)
    cols = ["model", "top1", "top1_95ci", "top3", "log_loss", "log_loss_95ci",
            "pr_auc", "roc_auc", "roi_top1", "roi_95ci"]
    return pd.DataFrame(rows)[cols].round(4)
