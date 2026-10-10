# Predicting Airline Satisfaction (Kaggle Playground S6E10)

A gradient-boosting pipeline for the **[Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10)** competition on Kaggle. The task is to predict the probability that an airline passenger is satisfied, from demographics, trip details, 13 service ratings and flight delays. Submissions are scored by ROC AUC.

**[Case study](https://www.konnectingnots.com/projects/airline-satisfaction.html)**

---

## 📊 Results

| Version | What changed | Blend CV AUC | Public LB | Rank |
|---|---|---|---|---|
| v1 | Raw features + row-wise rating/delay aggregates | 0.95885 | 0.95834 | #925 |
| v2 | + route profile, fold-safe target encoding, original-data teacher | **0.96093** | **0.96042** | **#577** |

CV and public leaderboard moved by the same +0.0021, so the 5-fold CV is a trustworthy guide for this data. Full numbers per model are in [`results/experiments.csv`](results/experiments.csv).

| Model (v2) | CV AUC |
|---|---|
| LightGBM | 0.96078 |
| XGBoost | 0.96076 |
| CatBoost | 0.96069 |
| Rank blend (0.35 / 0.30 / 0.35) | **0.96093** |

---

## 💡 Modeling Strategy

The training data (700k rows) is synthetic, generated from the public [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) dataset. Three ideas did most of the work.

### 1. Flight Distance is a route ID
`Flight Distance` has about 3,500 distinct values that repeat thousands of times, so it behaves more like an identifier for a route than a continuous number. For every distance value (computed over train + test, no target involved) the pipeline adds:
- how many passengers share that route,
- the route's average for each service rating, age and delay,
- each passenger's difference from their route's average,
- the route's mix of class, travel type, customer type and gender.

### 2. Fold-safe target encoding
Satisfaction rates by `Flight Distance`, `Age`, `Flight Distance × Class` and `Flight Distance × Type of Travel`, smoothed toward the global mean. The encoding is computed **inside** each CV fold: training rows get an inner 5-fold out-of-fold encoding, and validation and test rows get the encoding fitted on the training part only, so the CV score never sees its own labels.

### 3. Original data as a teacher, not as rows
Competitors on the forum measured that appending the original 130k rows to the training set makes things worse. Instead, a LightGBM trained only on the original data predicts every competition row, and that prediction becomes a feature. The original dataset's satisfaction rate per route is added the same way. Neither uses competition labels, so there is no leakage. This added about +0.0005 AUC on top of the route features.

### Models and blend
LightGBM, XGBoost and CatBoost (GPU) are trained on the same stratified 5 folds with early stopping. Out-of-fold predictions are rank-normalised and blended with weights chosen by a grid search on OOF AUC.

---

## 📁 Project Structure

```
├── notebooks/
│   ├── 01_baseline_gbdt_blend.ipynb       # v1: raw features + aggregates (Kaggle notebook)
│   └── 02_route_profile_te_teacher.ipynb  # v2: route profile + TE + teacher (Kaggle notebook)
├── src/
│   ├── download_data.py   # Downloads competition + original data via the Kaggle API
│   └── train.py           # v2 pipeline as a script (same code as notebook 02)
├── results/
│   └── experiments.csv    # CV and leaderboard history
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Execution

### On Kaggle (easiest)
Import a notebook from `notebooks/` into the competition, add the dataset `teejmahal20/airline-passenger-satisfaction` as an input, turn on a GPU accelerator and run all. A full run takes about 13 minutes on a T4.

### Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/download_data.py   # needs a Kaggle API token and the competition rules accepted
python src/train.py           # writes outputs/submission.csv, OOF/test predictions and experiments.csv
```
Set `QUICK_MODE = True` at the top of `src/train.py` for a fast 100k-row, 3-fold run while experimenting. Each feature block (`USE_ROUTE`, `USE_TE`, `USE_TEACHER`) can be switched off to measure its effect.

---

## 🔭 Next steps

- **Model diversity.** The three boosted models now score within 0.0001 of each other, so blending adds little. A neural net (e.g. RealMLP) or a tabular foundation model such as TabPFN should make different errors.
- **Ablation** of each feature block on the full data.
- **Tuning** LightGBM and CatBoost once the feature set is settled.

## 🙏 Credits

The route-profile, original-data teacher and feature-ladder ideas came from the competition's discussion forum, in particular posts by Dmitriy Kolmikov (Flight Distance as a route ID) and Koumei Maki (feature ladder, crediting @goodpjw2008, @evgendvorkin and @busyaprime). The implementation here is my own.
