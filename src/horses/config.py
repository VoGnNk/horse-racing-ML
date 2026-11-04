"""Константи проєкту: шляхи, seed, періоди розбиття, списки ознак."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MERGED_PATH = DATA_DIR / "merged.parquet"
MODEL_TABLE_PATH = DATA_DIR / "model_table.parquet"
FIG_DIR = ROOT / "figures"

SEED = 42

TARGET = "res_win"
FIRST_YEAR, LAST_YEAR = 2005, 2019          
TRAIN_YEARS = range(2005, 2017)             
VAL_YEARS = range(2017, 2019)               
TEST_YEARS = [2019]                         

ID_COLS = ["rid", "year", "date", "horseName", "jockeyName", "trainerName", "course"]
NUM_FEATURES = ["age_clean", "weight_lb", "saddle_clean", "OR", "runners", "dist_m", "rclass_num"]
CAT_FEATURES = ["race_type", "condition"]
FLAG_FEATURES = ["has_headgear", "OR_missing"]
MARKET = ["decimalPrice", "p_mkt", "logit_p_mkt"]


HIST_SPECS = [("horseName", "horse"), ("jockeyName", "jockey"), ("trainerName", "trainer")]
HIST_FEATURES = [f"{p}_{s}" for _, p in HIST_SPECS for s in ("races", "win_rate")]
PRIOR, K = 0.10, 10                         


EPS = 1e-6                                  
N_BOOT = 1000                               
