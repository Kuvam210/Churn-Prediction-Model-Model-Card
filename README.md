# Part 3 — Churn Prediction Model & Model Card

> **Capstone project · D2C Customer Retention · Snapshot date: 2025-09-30**

Binary classifier that scores each customer on their probability of churning in the 60 days following the snapshot date. The final model is a Logistic Regression trained on 25 pre-engineered RFM + behavioural features, selected over Gradient Boosting on both AUC and interpretability grounds.

---

## Repository structure

```
Churn Prediction Model & Model Card/
├── churn_model.ipynb       # end-to-end training + evaluation notebook
├── model.pkl               # serialised model bundle (see below)
├── metrics.json            # all evaluation metrics in one place
├── error_analysis.md       # 10 named customer FP/FN case studies
├── model_card.md           # model card (intended use, limits, ethics)
├── figures/                # all chart exports (PNG, 150 dpi)
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
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place all CSVs inside a data/ subfolder next to the notebook
#    repo_part3_churn_model/
#    └── data/
#        ├── rfm_modeling_snapshot.csv   ← required
#        └── ... (other capstone CSVs)

# 3. Open and run all cells
jupyter notebook churn_model.ipynb
# or in VS Code: open the file and click "Run All"
```

The notebook auto-detects its own location — no path editing required on Windows, Mac, or Linux.

---

## Problem framing

| Item | Detail |
|---|---|
| Target | `churn_next_60d` — 1 if no purchase in the 60 days after 2025-09-30 |
| Input | `rfm_modeling_snapshot.csv` — 2,400 customers, pre-split |
| Splits | Train 1,728 · Val 336 · Test 336 (pre-assigned, stratified) |
| Class balance | ~47% retained / 53% churned — no oversampling needed |
| Threshold | 0.40 (recall-biased; see §Threshold below) |

---

## Model comparison

Two models were trained and evaluated head-to-head. Logistic Regression won on every test metric.

| Model | Val AUC | Val F1 | **Test AUC** | **Test F1** | Test Precision | Test Recall |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.8795 | 0.7871 | **0.8932** | **0.8315** | 0.7872 | 0.8810 |
| Gradient Boosting | 0.8770 | 0.7672 | 0.8646 | 0.7932 | 0.7564 | 0.8333 |

Gradient Boosting assigns 57% of its feature importance to `recency_days` alone — a near-linear relationship that Logistic Regression handles just as well. The complexity cost of GBM buys nothing here, so LR was selected as the final model.

---

## ROC & Precision-Recall curves

![ROC and PR curves](figures/02_roc_pr_curves.png)

Both models clear 0.87 AUC on validation. LR edges ahead on the test set (0.8932 vs 0.8646) and maintains a stronger PR curve throughout the high-recall operating range, which is where the retention team actually operates.

---

## Confusion matrices — test set

![Confusion matrices](figures/03_confusion_matrices.png)

At threshold 0.40, Logistic Regression catches **148 of 168 true churners** (recall 0.881) while generating 40 false positives. Gradient Boosting misses 8 more churners (FN=28) and generates 5 more unnecessary campaigns (FP=45) — strictly worse on both error types.

---

## Threshold selection

![Threshold sweep](figures/04_threshold_sweep.png)

The threshold was set at **0.40** rather than the standard 0.50 for a deliberate business reason:

- A **missed churner (FN)** costs Rs. 500–2,000 in lost lifetime value.
- An **unnecessary campaign (FP)** costs Rs. 20–40.

The asymmetry justifies a recall-biased operating point. Lowering below 0.35 floods the retention team with low-risk customers and degrades campaign ROI. At 0.40, recall is 0.881 and precision stays above 0.78 — an acceptable trade-off.

---

## Feature importance

![Feature importance](figures/05_feature_importance.png)

The two models tell slightly different stories about what drives churn:

**Gradient Boosting** — `recency_days` dominates (0.57 importance), followed by `monetary_180d`. The model is essentially a recency-threshold rule with corrections.

**Logistic Regression** — `return_rate_180d` and `negative_ticket_rate_90d` carry the largest coefficients, meaning the model captures dissatisfaction signals that the tree misses. Web engagement features (`campaign_clicks_30d`, `category_diversity`) also contribute, reinforcing that digital disengagement precedes transactional churn.

The LR feature weights are directly actionable: target customers with high return rates and negative support experiences first, before they go silent.

---

## model.pkl — bundle contents

```python
import pickle

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

# Keys:
# bundle["model"]      → fitted LogisticRegression
# bundle["encoders"]   → dict of LabelEncoder per categorical column
# bundle["features"]   → ordered list of 25 feature names
# bundle["threshold"]  → 0.40
# bundle["cat_cols"]   → categorical column names
# bundle["num_cols"]   → numeric column names
```

Usage:

```python
prob = bundle["model"].predict_proba(X_encoded)[:, 1]
pred = (prob >= bundle["threshold"]).astype(int)
```

---

## Error analysis summary

10 specific customers (5 FP, 5 FN) were examined in `error_analysis.md`. Key patterns:

**False Positives** (flagged as churn risk, stayed) tend to be cycle buyers with purchase intervals around 75–90 days, or browsing-heavy customers who haven't converted yet. These are low-cost interventions even if wrong.

**False Negatives** (missed churners) are the harder and more expensive failure mode. Two patterns dominate:
- **Sudden silent churners** — recent purchase + high sessions, but something went wrong after the snapshot (unresolved ticket, bad product experience). CUST00727 bought 8 days before snapshot with Rs. 2,696 in revenue and churned anyway.
- **Loyalty programme fatigue** — Gold or Platinum tier customers who churned despite high historical value. CUST00188 is the clearest example: the programme isn't delivering enough perceived value to retain high-spend customers.

**Recommended rule-based supplement:** flag any customer with `monetary_180d > median` AND `ticket_count_90d >= 1` as high-priority regardless of model score. This catches the sudden-silent-churner pattern the model structurally cannot.

---

## Known limitations

- `recency_days` dominates — cycle buyers (purchase every 60–90 days) will be over-flagged near the end of their natural gap.
- Sudden post-snapshot events (bad delivery, unresolved complaint) are not capturable from pre-snapshot features.
- Static model on a single snapshot — churn drivers shift with seasons and campaigns.
- Retrain if test AUC drops below 0.82 or PSI > 0.20 on `recency_days`. Mandatory quarterly retraining regardless.

---

## Requirements

```
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

Full pinned versions in `requirements.txt`.
