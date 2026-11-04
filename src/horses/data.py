"""Відбір рядків, чищення, побудова модельної таблиці й розбиття за часом.

Тут лише безстанові перетворення: кожен рядок або забіг обробляється окремо,
без обчислення статистик по всьому датасету, тож витоку між train і test бути не може.
Усе, що навчається (медіана, масштаб, кодування), живе в Pipeline (див. models.py).
"""
import numpy as np
import pandas as pd

from . import config as cfg


def select_modelling_rows(df, first_year=cfg.FIRST_YEAR, last_year=cfg.LAST_YEAR):
    n_wins = df.groupby("rid")[cfg.TARGET].transform("sum")
    in_years = df["year"].between(first_year, last_year)
    mask = in_years & (n_wins == 1)

    print(f"Рядків усього: {len(df):,}")
    print(f"У роках {first_year}-{last_year}: {in_years.sum():,}")
    print(f"Після видалення забігів без єдиного переможця: {mask.sum():,}")
    work = df.loc[mask].copy()
    print("Забігів:", work["rid"].nunique())
    return work


def add_clean_features(work):
    rep = {}

    zero_w = work["weightSt"] == 0
    work["weight_lb"] = (work["weightSt"] * 14 + work["weightLb"]).where(~zero_w)
    rep["weightSt == 0 -> NaN"] = int(zero_w.sum())

    bad_age = ~work["age"].between(2, 20) & work["age"].notna()
    work["age_clean"] = work["age"].where(work["age"].between(2, 20))
    rep["age поза [2, 20] -> NaN"] = int(bad_age.sum())

    zero_s = work["saddle"] == 0
    work["saddle_clean"] = work["saddle"].where(work["saddle"] > 0)
    rep["saddle == 0 -> NaN"] = int(zero_s.sum())

    work["dist_m"] = work["metric"].where(work["metric"] > 0).astype("float32")
    rep["metric відсутня або <= 0"] = int(work["dist_m"].isna().sum())

    work["rclass_num"] = pd.to_numeric(work["rclass"].str.extract(r"(\d+)")[0])

    h = work["hurdles"].str.lower()
    work["race_type"] = np.select(
        [h.str.contains("fence", na=False), h.str.contains("hurdle", na=False)],
        ["chase", "hurdle"], default="none")

    work["has_headgear"] = work["headGear"].notna().astype("int8")
    work["OR_missing"] = work["OR"].isna().astype("int8")

    work["p_mkt"] = work["decimalPrice"] / work.groupby("rid")["decimalPrice"].transform("sum")
    work["logit_p_mkt"] = np.log(work["p_mkt"] / (1 - work["p_mkt"]))
    return work, rep


def make_model_table(work):
    keep = (cfg.ID_COLS + [cfg.TARGET] + cfg.NUM_FEATURES + cfg.CAT_FEATURES
            + cfg.FLAG_FEATURES + cfg.MARKET)
    return work[keep].sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)


def split_by_year(model_df):
    return (model_df[model_df["year"].isin(cfg.TRAIN_YEARS)],
            model_df[model_df["year"].isin(cfg.VAL_YEARS)],
            model_df[model_df["year"].isin(cfg.TEST_YEARS)])
