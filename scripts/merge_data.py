"""Об'єднує horses_YYYY.csv і races_YYYY.csv (Kaggle hwaitt/horse-racing) в один файл.

Запуск (з кореня репозиторію):
    python scripts/merge_data.py

Вхід:  data/raw/horses_*.csv, data/raw/races_*.csv  (forward.csv не використовується)
Вихід: data/merged.parquet  (рядок = кінь у забігу; ключ з'єднання rid)
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "merged.parquet"
KEY_CANDIDATES = ["rid", "race_id", "raceId"]


def load_kind(kind):
    files = sorted(RAW_DIR.glob(f"{kind}_*.csv"))
    if not files:
        found = [p.name for p in RAW_DIR.glob("*")][:10] if RAW_DIR.exists() else []
        raise SystemExit(f"Не знайшов {kind}_*.csv у {RAW_DIR}. Є там: {found}")
    parts = []
    for f in files:
        year = int(re.search(r"_(\d{4})\.csv$", f.name).group(1))
        df = pd.read_csv(f, low_memory=False)
        df["year"] = year
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    print(f"{kind:7s}: {len(parts)} файлів, {out.shape[0]:>9,} рядків, {out.shape[1]} колонок")
    return out


def main():
    races = load_kind("races")
    horses = load_kind("horses")

    common = [c for c in races.columns if c in horses.columns and c != "year"]
    key = next((k for k in KEY_CANDIDATES if k in common), None)
    if key is None:
        raise SystemExit(f"Не знайшов ключ з'єднання серед спільних колонок: {common}")
    print("Ключ з'єднання:", key)

    if races[key].duplicated().any():
        raise SystemExit("Ключ у races не унікальний: з'єднання дало б дублікати рядків.")
    print("Рядків horses без забігу в races:", int((~horses[key].isin(races[key])).sum()))
    print("Забігів без жодного коня в horses:", int((~races[key].isin(horses[key])).sum()))

    n_before = len(horses)
    df = horses.merge(races.drop(columns=["year"]), on=key, how="inner",
                      validate="m:1", suffixes=("", "_race"))
    print(f"horses до з'єднання: {n_before:,} | після: {len(df):,} | втрачено: {n_before - len(df):,}")

    for c in df.columns:
        if df[c].dtype == object:
            types = df[c].dropna().map(type).value_counts()
            if len(types) > 1:
                print(f"Змішані типи в {c}: {dict((t.__name__, int(n)) for t, n in types.items())}")
            df[c] = df[c].map(lambda v: v if pd.isna(v) else str(v)).astype("string")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(OUT_PATH, index=False)
    except ImportError as e:
        raise SystemExit(f"Для запису parquet потрібен pyarrow: pip install pyarrow ({e})")
    print(f"Збережено: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.0f} МБ), розмір таблиці: {df.shape}")


if __name__ == "__main__":
    main()
