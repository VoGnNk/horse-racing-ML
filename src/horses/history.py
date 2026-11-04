"""Історичні ознаки коня, жокея й тренера БЕЗ витоку.

Статистика сутності на день d рахується лише за результатами СТРОГО раніших днів:
результати того ж дня (зокрема того самого забігу, де в тренера може бути кілька коней) не враховуються.
Історія береться з повних даних 1990..last_year, щоб навчальні роки не мали штучного «холодного старту».
"""
import numpy as np
import pandas as pd

from . import config as cfg

RAW_COLS = ["rid", "year", "date", "horseName", "jockeyName", "trainerName", cfg.TARGET]


def load_history_source(last_year=cfg.LAST_YEAR):
    raw = pd.read_parquet(cfg.MERGED_PATH, columns=RAW_COLS, filters=[("year", "<=", last_year)])
    raw["date_dt"] = pd.to_datetime(raw["date"], format="%y/%m/%d")
    return raw


def entity_history(raw, key, prefix):
    day = (raw.groupby([key, "date_dt"])[cfg.TARGET].agg(day_wins="sum", day_runs="size")
              .reset_index().sort_values([key, "date_dt"]))
    g = day.groupby(key)
    wins_before = g["day_wins"].cumsum() - day["day_wins"]
    runs_before = g["day_runs"].cumsum() - day["day_runs"]
    day[f"{prefix}_races"] = runs_before
    day[f"{prefix}_win_rate"] = (wins_before + cfg.K * cfg.PRIOR) / (runs_before + cfg.K)
    return day[[key, "date_dt", f"{prefix}_races", f"{prefix}_win_rate"]]


def merge_history(frame, raw):
    out = frame.copy()
    for key, prefix in cfg.HIST_SPECS:
        out = out.merge(entity_history(raw, key, prefix), on=[key, "date_dt"], how="left")
    return out


def add_history_features(model_df, raw):
    table = model_df.copy()
    table["date_dt"] = pd.to_datetime(table["date"], format="%y/%m/%d")
    return merge_history(table, raw).drop(columns=["date_dt"])


def leak_test(raw, cutoff="2015-01-01", seed=cfg.SEED):
    sub = raw[raw["year"].between(2012, 2016)].reset_index(drop=True)
    keys = sub[["rid", "horseName", "jockeyName", "trainerName", "date_dt"]]
    base = merge_history(keys, sub)
    pert = sub.copy()
    fut = pert["date_dt"] >= pd.Timestamp(cutoff)
    pert.loc[fut, cfg.TARGET] = pert.loc[fut, cfg.TARGET].sample(frac=1, random_state=seed).to_numpy()
    new = merge_history(keys, pert)
    before = (sub["date_dt"] <= pd.Timestamp(cutoff)).to_numpy()
    a, b = base[cfg.HIST_FEATURES].to_numpy(float), new[cfg.HIST_FEATURES].to_numpy(float)
    return (np.allclose(a[before], b[before], equal_nan=True),
            not np.allclose(a[~before], b[~before], equal_nan=True))
