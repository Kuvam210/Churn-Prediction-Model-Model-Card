# Part 3 — Churn Prediction Model & Model Card

> **Capstone project · D2C Customer Retention · Snapshot date: 2025-09-30**

Binary classifier that scores each customer on their probability of churning in the 60 days following the snapshot date. Final model: Logistic Regression trained on 25 pre-engineered RFM + behavioural features — selected over Gradient Boosting on both AUC and interpretability grounds.

---

## Repository structure

```
repo_part3_churn_model/
├── churn_model.ipynb       # end-to-end training + evaluation notebook
├── model.pkl               # serialised model bundle (model + encoders + metadata)
├── metrics.json            # all evaluation metrics
├── error_analysis.md       # 10 named customer FP/FN case studies
├── model_card.md           # intended use, limitations, ethics
├── figures/
│   ├── 01_churn_distribution.png
│   ├── 02_roc_pr_curves.png
│   ├── 03_confusion_matrices.png
│   ├── 04_threshold_sweep.png
│   ├── 05_feature_importance.png
│   ├── 06_score_distribution.png
│   └── 07_error_distribution.png
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Place CSVs in a data/ subfolder next to the notebook:
# repo_part3_churn_model/data/rfm_modeling_snapshot.csv
jupyter notebook churn_model.ipynb
# Run All — model.pkl, metrics.json, and figures/ are auto-generated
```

The notebook auto-detects its own directory. No path editing needed on Windows, Mac, or Linux.

---

## Problem framing

| Item | Detail |
|---|---|
| Target | `churn_next_60d` — 1 if no purchase in 60 days after 2025-09-30 |
| Input | `rfm_modeling_snapshot.csv` — 2,400 customers, pre-split |
| Splits | Train 1,728 · Val 336 · Test 336 (pre-assigned, stratified) |
| Class balance | ~47% retained / 53% churned — no oversampling needed |
| Decision threshold | **0.40** (recall-biased — see §Threshold) |

---

## Model comparison

| Model | Val AUC | Val F1 | Test AUC | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.8795 | 0.7871 | **0.8932** | **0.8315** | 0.7872 | **0.8810** |
| Gradient Boosting | 0.8770 | 0.7672 | 0.8646 | 0.7932 | 0.7564 | 0.8333 |

Gradient Boosting assigns 57% of its feature importance to `recency_days` alone — a near-linear relationship that Logistic Regression handles equally well. LR wins on every test metric, trains in milliseconds, and produces directly interpretable coefficients. Selected as the final model.

---

## ROC & Precision-Recall curves

![ROC and PR curves](figures/02_roc_pr_curves.png)

Both models clear 0.87 AUC on validation. LR extends its lead on the held-out test set (0.8932 vs 0.8646) and maintains a stronger PR curve across the high-recall operating range — exactly where the retention team works.

---

## Confusion matrices — test set (threshold = 0.40)

![Confusion matrices](figures/03_confusion_matrices.png)

At threshold 0.40, Logistic Regression catches **148 of 168 true churners** (recall 0.881) while producing 40 false positives. Gradient Boosting misses 8 more churners and generates 5 more unnecessary campaigns — strictly worse on both error types.

---

## Threshold selection

![Threshold sweep](figures/04_threshold_sweep.png)

The threshold was deliberately set to **0.40** rather than the standard 0.50:

- A **missed churner (FN)** costs Rs. 500–2,000 in lost lifetime value
- An **unnecessary campaign (FP)** costs Rs. 20–40

The 25–100× cost asymmetry justifies a recall-biased operating point. At 0.40, recall = 0.881 and precision stays above 0.78. Dropping below 0.35 floods the retention team with low-risk customers and degrades campaign ROI without meaningful recall gain.

---

## Feature importance

![Feature importance](figures/05_feature_importance.png)

The two models surface different signals:

**Gradient Boosting** — `recency_days` dominates at 0.57 importance. The tree is essentially a recency-threshold rule with small corrections from `monetary_180d`.

**Logistic Regression** — `return_rate_180d` and `negative_ticket_rate_90d` carry the largest coefficients. The model picks up active dissatisfaction signals the tree misses. Web engagement (`campaign_clicks_30d`, `category_diversity`) also contributes — digital disengagement precedes transactional churn.

The LR coefficients are directly actionable: prioritise customers with high return rates and unresolved support issues before they go silent.

---

## model.pkl — bundle contents

```python
import pickle

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

# bundle["model"]      → fitted LogisticRegression
# bundle["encoders"]   → dict of LabelEncoder per categorical column
# bundle["features"]   → ordered list of 25 feature names
# bundle["threshold"]  → 0.40
# bundle["cat_cols"]   → categorical column names
# bundle["num_cols"]   → numeric column names

prob = bundle["model"].predict_proba(X_encoded)[:, 1]
pred = (prob >= bundle["threshold"]).astype(int)
```

---

## Error analysis summary

10 specific customers (5 FP, 5 FN) are examined in detail in `error_analysis.md`.

**False Positives** (flagged as risk, stayed) are mostly cycle buyers with ~75-day purchase intervals, or browsing-heavy customers who haven't converted yet. Low-cost campaigns, low regret even if wrong.

**False Negatives** (missed churners) are the expensive failures. Two patterns dominate:

- **Sudden silent churners** — recent purchase + high sessions, but something broke post-snapshot. CUST00727 bought 8 days before snapshot at Rs. 2,696 and churned anyway. Structurally uncatchable from pre-snapshot features alone.
- **Loyalty fatigue** — Gold/Platinum tier customers churning despite high historical spend. CUST00188 is the clearest case: the loyalty programme isn't delivering perceived value at the top tier.

**Recommended rule-based supplement:** regardless of model score, flag any customer with `monetary_180d > median` AND `ticket_count_90d >= 1` as high-priority. This catches the sudden-silent-churner pattern the model cannot.

---

## Known limitations

- `recency_days` dominance means cycle buyers (~75-90 day intervals) are systematically over-flagged near the end of their gap
- Post-snapshot events (bad delivery, unresolved complaint raised after cutoff) cannot be captured
- Static single-snapshot model — seasonal shifts and campaign changes erode performance over time
- Retrain trigger: test AUC < 0.82 or PSI > 0.20 on `recency_days`. Mandatory quarterly retraining regardless

---

## Requirements

```
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
```
