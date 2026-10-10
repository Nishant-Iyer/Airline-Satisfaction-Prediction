"""
Predicting Airline Satisfaction (Kaggle Playground S6E10) - v2 pipeline.

Route profile on Flight Distance + fold-safe target encoding + original-data teacher,
then LightGBM / XGBoost / CatBoost on a shared 5-fold CV and an OOF-weighted rank blend.

Usage:
    python src/download_data.py      # puts competition + original data under ./data
    python src/train.py              # writes submission.csv, OOF/test preds and experiments.csv

Same code as notebooks/02_route_profile_te_teacher.ipynb (cells in order).
"""

# ---------------- Settings ----------------
QUICK_MODE   = False   # True = subsample train + 3 folds for fast experiments
QUICK_ROWS   = 100_000
N_FOLDS      = 5
SEED         = 42
USE_MODELS   = ["lgb", "xgb", "cat"]

# Feature blocks (switch off one at a time to measure what each adds)
RATING_AGGS  = False   # row-wise rating aggregates (v1 had these; the forum found them neutral)
USE_ROUTE    = True    # route profile keyed on Flight Distance
USE_TE       = True    # fold-safe target encoding (FD, Age, FD x Class, FD x Type of Travel)
USE_TEACHER  = True    # model trained only on the original data -> feature; plus original per-route rates
EXPERIMENT_NOTE = "v2: route profile + fold-safe TE + original teacher"

import os, glob, re, shutil, time, itertools, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 60)
HAS_GPU = shutil.which("nvidia-smi") is not None
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def is_cat(s):
    # True for text/category columns on any pandas version
    return not pd.api.types.is_numeric_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)

print("GPU available:", HAS_GPU, "| output dir:", OUT_DIR)

# ======================================================================
# 1. Load data
# ======================================================================

COMP_TAGS = ("s6e10", "playground")

def all_csvs():
    return glob.glob("/kaggle/input/**/*.csv", recursive=True) + glob.glob("./**/*.csv", recursive=True)

def find_file(name):
    hits = [p for p in all_csvs() if os.path.basename(p) == name]
    comp = [p for p in hits if any(t in p for t in COMP_TAGS)]
    assert comp or hits, f"Could not find {name}. Add the competition data via 'Add Input'."
    return (comp or hits)[0]

train = pd.read_csv(find_file("train.csv"))
test  = pd.read_csv(find_file("test.csv"))
sub   = pd.read_csv(find_file("sample_submission.csv"))
TARGET = "satisfaction"
ID = "id"
print("train", train.shape, "| test", test.shape)

# Original dataset (used as a teacher / lookup, NOT appended as rows)
orig_paths = [p for p in all_csvs() if "airline-passenger-satisfaction" in p and not any(t in p for t in COMP_TAGS)]
if orig_paths:
    orig = pd.concat([pd.read_csv(p) for p in orig_paths], ignore_index=True)
    orig = orig.drop(columns=[c for c in orig.columns if c.lower().startswith("unnamed") or c == ID])
    print("original rows:", len(orig), "from", [os.path.basename(p) for p in orig_paths])
else:
    orig = None
    print("Original dataset not attached -> teacher features will be skipped.")
print(train.head())

# ======================================================================
# 2. Quick EDA
# ======================================================================

print("Target distribution:")
print(train[TARGET].value_counts(normalize=True).round(4))
print("\nMissing values (train):")
miss = train.isna().sum(); print(miss[miss > 0] if miss.sum() else "none")
print("\nColumn types:")
summary = pd.DataFrame({"dtype": train.dtypes.astype(str), "nunique": train.nunique(), "example": train.iloc[0]})
print(summary)

# Target encoding to 0/1. Positive class = "satisfied" (the class the submission asks a probability for).
y_raw = train[TARGET]
if is_cat(y_raw):
    classes = sorted(y_raw.astype(str).unique())
    pos = [c for c in classes if "satisf" in c.lower() and "dis" not in c.lower()]
    pos = pos[0] if pos else classes[-1]
    y = (y_raw.astype(str) == pos).astype(int).values
    print(f"String target -> positive class = '{pos}'")
else:
    y = y_raw.astype(int).values
print("positive rate:", y.mean().round(4))

# Which raw features separate the classes best? (single-feature AUC, quick signal check)
signal = {}
for c in train.columns.drop([TARGET, ID], errors="ignore"):
    s = train[c]
    if is_cat(s):
        s = s.astype(str).map(pd.Series(y, index=train.index).groupby(s.astype(str)).mean())
    s = s.fillna(s.median())
    if s.nunique() > 1:
        a = roc_auc_score(y, s); signal[c] = max(a, 1 - a)
print(pd.Series(signal).sort_values(ascending=False).round(4).to_frame("single_feature_auc"))

# ======================================================================
# 3. Feature engineering
# ======================================================================

def clean_name(c):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", c).strip("_")

def to01(s):
    if is_cat(s):
        return s.astype(str).str.lower().eq("satisfied").astype(int).values
    return s.astype(int).values

full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
full.columns = [clean_name(c) for c in full.columns]
ID_C = clean_name(ID)
n_train = len(train)

cat_cols = [c for c in full.columns if is_cat(full[c])]
num_cols = [c for c in full.columns if c not in cat_cols + [ID_C]]
rating_cols = [c for c in num_cols if full[c].dropna().between(0, 5).all() and full[c].nunique() <= 6]
delay_cols = [c for c in num_cols if "delay" in c.lower()]
FD  = next(c for c in num_cols if "distance" in c.lower())
AGE = next((c for c in num_cols if c.lower() == "age"), None)
RAW = [c for c in full.columns if c != ID_C]
print("categorical:", cat_cols)
print("ratings:", len(rating_cols), "| delays:", delay_cols, "| route key:", FD)

# --- Delays (cheap, kept from v1)
if len(delay_cols) == 2:
    arr, dep = sorted(delay_cols, key=lambda c: "arriv" not in c.lower())
    full[arr] = full[arr].fillna(full[dep])
    full["delay_diff"] = full[arr] - full[dep]
    full["delay_max"]  = full[delay_cols].max(axis=1)

# --- Optional row-wise rating aggregates
if RATING_AGGS and rating_cols:
    R = full[rating_cols]
    full["rating_mean"] = R.replace(0, np.nan).mean(axis=1)
    full["rating_std"]  = R.std(axis=1)
    full["n_zero"]      = (R == 0).sum(axis=1)

# --- Route profile: Flight Distance acts like a route ID (computed on train+test, no target used)
if USE_ROUTE:
    g = full.groupby(FD)
    full["fd_count"] = g[FD].transform("size")
    prof_cols = rating_cols + ([AGE] if AGE else []) + delay_cols
    for c in prof_cols:
        m = g[c].transform("mean")
        full[f"fd_mean_{c}"] = m
        full[f"fd_resid_{c}"] = full[c] - m
    for c in cat_cols:
        for lvl in sorted(full[c].dropna().astype(str).unique())[1:]:      # drop one level per column
            full[f"fd_share_{c}_{clean_name(lvl)}"] = (full[c].astype(str) == lvl).groupby(full[FD]).transform("mean")
    print("route profile features added")

# --- Original-data teacher + per-route lookup (uses only the original labels -> no leakage)
if USE_TEACHER and orig is not None:
    import lightgbm as lgb
    o = orig.copy(); o.columns = [clean_name(c) for c in o.columns]
    y_o = to01(o[TARGET])
    shared = [c for c in RAW if c in o.columns]
    Xo, Xf = o[shared].copy(), full[shared].copy()
    for c in shared:
        if c in cat_cols:
            cats = sorted(set(Xf[c].dropna().astype(str)) | set(Xo[c].dropna().astype(str)))
            Xo[c] = pd.Categorical(Xo[c].astype(str), categories=cats)
            Xf[c] = pd.Categorical(Xf[c].astype(str), categories=cats)
    teacher = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 random_state=SEED, verbose=-1)
    teacher.fit(Xo, y_o)
    full["teacher_p"] = teacher.predict_proba(Xf)[:, 1]
    # original satisfaction rate per route (smoothed toward the original mean)
    prior_o, m_o = y_o.mean(), 10
    st = pd.DataFrame({"k": o[FD].values, "y": y_o}).groupby("k")["y"].agg(["sum", "count"])
    full["orig_fd_rate"]  = full[FD].map((st["sum"] + prior_o * m_o) / (st["count"] + m_o))
    full["orig_fd_count"] = full[FD].map(st["count"]).fillna(0)
    print(f"teacher trained on {len(o)} original rows using {len(shared)} shared columns")

for c in cat_cols:
    full[c] = full[c].fillna("NA").astype(str).astype("category")

# --- Keys for fold-safe target encoding (applied inside each CV fold, see run_cv)
TE_KEYS = {}
if USE_TE:
    TE_KEYS["te_fd"] = full[FD].astype(str)
    if AGE: TE_KEYS["te_age"] = full[AGE].astype(str)
    for c in cat_cols:
        if c.lower() in ("class", "type_of_travel"):
            TE_KEYS[f"te_fd_x_{c}"] = full[FD].astype(str) + "|" + full[c].astype(str)
    TE_KEYS = {k: pd.factorize(v)[0] for k, v in TE_KEYS.items()}
K      = pd.DataFrame(TE_KEYS).iloc[:n_train].reset_index(drop=True)
K_test = pd.DataFrame(TE_KEYS).iloc[n_train:].reset_index(drop=True)

FEATURES = [c for c in full.columns if c != ID_C]
X      = full.iloc[:n_train][FEATURES].reset_index(drop=True)
X_test = full.iloc[n_train:][FEATURES].reset_index(drop=True)
print("static features:", len(FEATURES), "| TE features per fold:", len(TE_KEYS))

if QUICK_MODE:
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), size=min(QUICK_ROWS, len(X)), replace=False)
    X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    K = K.iloc[idx].reset_index(drop=True)
    N_FOLDS = 3
    print("QUICK_MODE: using", len(X), "rows,", N_FOLDS, "folds")

is_orig = np.zeros(len(X), bool)   # kept for compatibility; original rows are not appended in v2
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))

# ======================================================================
# 4. Models (same folds for every model)
# ======================================================================

def fit_lgb(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=5000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m

def fit_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=5000, learning_rate=0.03, max_depth=7, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        tree_method="hist", device="cuda" if HAS_GPU else "cpu",
        enable_categorical=True, max_cat_to_onehot=8,
        eval_metric="auc", early_stopping_rounds=200, random_state=SEED)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m

def fit_cat(Xtr, ytr, Xva, yva):
    from catboost import CatBoostClassifier
    cats = [c for c in Xtr.columns if isinstance(Xtr[c].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(Xtr[c])]
    m = CatBoostClassifier(
        iterations=5000, learning_rate=0.06, depth=7, l2_leaf_reg=3,
        eval_metric="Logloss" if HAS_GPU else "AUC",
        task_type="GPU" if HAS_GPU else "CPU",
        od_type="Iter", od_wait=200, random_seed=SEED, verbose=0)
    m.fit(Xtr, ytr, cat_features=cats, eval_set=(Xva, yva), use_best_model=True)
    return m

FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}

def prep(df, model):
    # CatBoost wants plain strings for categoricals
    if model == "cat":
        df = df.copy()
        for c in df.columns:
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].astype(str).astype(object)
    return df

def te_map(keys_fit, y_fit, keys_apply, m=20):
    prior = y_fit.mean()
    st = pd.DataFrame({"k": keys_fit, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
    enc = (st["sum"] + prior * m) / (st["count"] + m)
    return pd.Series(keys_apply).map(enc).fillna(prior).values

def add_te(tr_idx, others):
    # Train rows: inner 5-fold OOF encoding. Validation/test rows: encoding fit on the whole training part.
    out_tr = pd.DataFrame(index=range(len(tr_idx)))
    out_o = [pd.DataFrame(index=range(len(KO))) for KO in others]
    ytr = y[tr_idx]
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
    for col in K.columns:
        ktr = K[col].values[tr_idx]
        enc = np.zeros(len(tr_idx))
        for a, b in inner.split(ktr, ytr):
            enc[b] = te_map(ktr[a], ytr[a], ktr[b])
        out_tr[col] = enc
        for o, KO in zip(out_o, others):
            o[col] = te_map(ktr, ytr, KO[col].values)
    return out_tr, out_o

def with_te(Xpart, te):
    if te is None or te.shape[1] == 0: return Xpart
    return pd.concat([Xpart.reset_index(drop=True), te.reset_index(drop=True)], axis=1)

def run_cv(model):
    oof = np.zeros(len(X)); pred = np.zeros(len(X_test)); t0 = time.time()
    Xm, Xtm = prep(X, model), prep(X_test, model)
    for k, (tr, va) in enumerate(folds):
        if K.shape[1]:
            te_tr, (te_va, te_te) = add_te(tr, [K.iloc[va], K_test])
        else:
            te_tr = te_va = te_te = None
        Xtr, Xva, Xte = with_te(Xm.iloc[tr], te_tr), with_te(Xm.iloc[va], te_va), with_te(Xtm, te_te)
        m = FITTERS[model](Xtr, y[tr], Xva, y[va])
        oof[va] = m.predict_proba(Xva)[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / len(folds)
        print(f"  {model} fold {k}: AUC {roc_auc_score(y[va], oof[va]):.5f}")
    mask = ~is_orig
    auc = roc_auc_score(y[mask], oof[mask])
    print(f"{model} CV AUC: {auc:.5f}  ({time.time()-t0:.0f}s)")
    return oof, pred, auc

results = {}
for name in USE_MODELS:
    results[name] = run_cv(name)

# ======================================================================
# 5. Blend (weights chosen on OOF, rank-averaged)
# ======================================================================

mask = ~is_orig
names = list(results)
oof_r  = {n: rankdata(results[n][0][mask]) / mask.sum() for n in names}
test_r = {n: rankdata(results[n][1]) / len(X_test) for n in names}

best = (-1, None)
grid = np.round(np.arange(0, 1.0001, 0.05), 2)
for w in itertools.product(grid, repeat=len(names)):
    if abs(sum(w) - 1) > 1e-9: continue
    auc = roc_auc_score(y[mask], sum(wi * oof_r[n] for wi, n in zip(w, names)))
    if auc > best[0]: best = (auc, w)

blend_auc, weights = best
print("Single models:", {n: round(float(results[n][2]), 5) for n in names})
print("Blend weights:", {n: float(w) for n, w in zip(names, weights)}, "| blend CV AUC:", round(blend_auc, 5))
blend_test = sum(wi * test_r[n] for wi, n in zip(weights, names))

# ======================================================================
# 6. Save submission, predictions and experiment log
# ======================================================================

sub[TARGET] = blend_test
sub.to_csv(f"{OUT_DIR}/submission.csv", index=False)

# Keep OOF/test predictions so future experiments can be blended with this one
for n in names:
    pd.DataFrame({"oof": results[n][0]}).to_csv(f"{OUT_DIR}/oof_{n}.csv", index=False)
    pd.DataFrame({ID: sub[ID], "pred": results[n][1]}).to_csv(f"{OUT_DIR}/test_{n}.csv", index=False)

log_path = f"{OUT_DIR}/experiments.csv"
row = {"time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "note": EXPERIMENT_NOTE,
       "quick_mode": QUICK_MODE, "n_features": len(FEATURES) + K.shape[1],
       **{f"cv_{n}": round(float(results[n][2]), 5) for n in names}, "cv_blend": round(float(blend_auc), 5),
       "public_lb": ""}
log = pd.concat([pd.read_csv(log_path), pd.DataFrame([row])]) if os.path.exists(log_path) else pd.DataFrame([row])
log.to_csv(log_path, index=False)
print(sub.head()); print(); print(log.tail())
