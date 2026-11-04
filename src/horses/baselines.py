"""Baseline-и: constant, випадковий вибір 1/N, ринок, ринок з перекалібруванням."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config as cfg
from .evaluation import score


def fit_market_recalibrator(train):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train[["logit_p_mkt"]], train[cfg.TARGET])
    return lr


def baseline_scores(part, train_rate, recalibrator):
    return {
        "constant (не нормована)": score(part, np.full(len(part), train_rate), normalise=False),
        "random 1/N": score(part, 1.0 / part["runners"]),
        "market": score(part, part["p_mkt"]),
        "market + logistic": score(part, recalibrator.predict_proba(part[["logit_p_mkt"]])[:, 1]),
    }
