"""Допоміжні функції для текстових висновків, що рахуються з результатів."""
from .evaluation import paired


def verdict(lo, hi, higher_is_better):
    good = (lo > 0) if higher_is_better else (hi < 0)
    bad = (hi < 0) if higher_is_better else (lo > 0)
    return "**краще за ринок**" if good else ("**гірше за ринок**" if bad else "статистично **не відрізняється від ринку**")


def line(boots, a, b, label):
    parts = []
    for m, hib, scale, unit in (("log_loss", False, 1000, "×10⁻³"), ("top1", True, 100, "в.п.")):
        est, lo, hi = paired(boots, a, b, m)
        parts.append(f"{'log-loss' if m == 'log_loss' else 'top-1'}: Δ = {est * scale:+.2f} {unit} "
                     f"[{lo * scale:+.2f}; {hi * scale:+.2f}] — {verdict(lo, hi, hib)}")
    return f"- {label}: " + "; ".join(parts)
