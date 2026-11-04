"""Моделі й пайплайни. Усе, що навчається (медіана, масштаб, кодування), живе всередині Pipeline
і підганяється лише на навчальній частині.

- LogReg: медіана + індикатор пропуску + StandardScaler (чутлива до масштабу й пропусків);
- дерева (RandomForest, LightGBM): NaN нативно, масштабування не потрібне.
"""
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg

MODEL_NAMES = ["LogReg", "RandomForest", "LightGBM"]


def make_estimator(name, **override):
    if name == "LogReg":
        return LogisticRegression(max_iter=500, C=1.0)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=100,
                                      n_jobs=-1, random_state=cfg.SEED)
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=cfg.SEED, n_jobs=-1, verbose=-1)
    params.update(override)
    return LGBMClassifier(**params)


def make_pipeline(name, exp, **override):
    linear = name == "LogReg"
    num = cfg.NUM_FEATURES + cfg.HIST_FEATURES
    if exp == "B":
        num = num + (["logit_p_mkt"] if linear else ["p_mkt"])
    if linear:
        num_tr = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scale", StandardScaler())])
        cat_tr = OneHotEncoder(handle_unknown="ignore")
    else:
        num_tr = "passthrough"                      # дерева працюють із NaN нативно, масштаб не потрібен
        cat_tr = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer([("num", num_tr, num), ("cat", cat_tr, cfg.CAT_FEATURES),
                             ("flag", "passthrough", cfg.FLAG_FEATURES)])
    cols = num + cfg.CAT_FEATURES + cfg.FLAG_FEATURES
    return Pipeline([("prep", pre), ("model", make_estimator(name, **override))]), cols
