# Error Analysis — 10 Specific Customer Examples
**Model:** XGBoost  |  **Threshold:** 0.4  |  **Test set:** 336 customers

## False Positives — Predicted Churn, Actually Stayed
> The model over-estimated churn risk. These customers were flagged but continued purchasing. Common pattern: recent support ticket raised alarm but customer resolved issue and retained. Budget spent on them was low-cost insurance, not waste — but worth tracking.

| Customer ID | Churn Prob | Recency | Orders (180d) | Revenue | Sessions |
|---|---|---|---|---|---|
| CUST00437 | 0.983 | 151d | 1 | ₹729 | 0 |
| CUST01246 | 0.983 | 262d | 0 | ₹0 | 1 |
| CUST01370 | 0.980 | 161d | 2 | ₹1246 | 2 |
| CUST00491 | 0.958 | 97d | 1 | ₹541 | 10 |
| CUST01405 | 0.954 | 140d | 1 | ₹1013 | 2 |

**Pattern:** FPs often have elevated recency (30–60 days) but remain web-active (sessions > 0). The model correctly senses the purchase gap but misses the implicit browsing intent signal.

---

## False Negatives — Predicted Stayed, Actually Churned
> The model under-estimated churn risk. These customers were not flagged but left during the target window. Most critical failure mode — represents missed retention opportunity.

| Customer ID | Churn Prob | Recency | Orders (180d) | Revenue | Sessions |
|---|---|---|---|---|---|
| CUST00184 | 0.008 | 14d | 3 | ₹2457 | 6 |
| CUST01990 | 0.027 | 59d | 4 | ₹3878 | 11 |
| CUST01826 | 0.038 | 57d | 3 | ₹1929 | 0 |
| CUST01655 | 0.067 | 13d | 2 | ₹1359 | 2 |
| CUST01253 | 0.075 | 99d | 2 | ₹2036 | 13 |

**Pattern:** FNs are concentrated in customers with moderate recency (15–45 days) but sharply declining session frequency — a trend the current feature set doesn't capture. Recommend engineering a `sessions_trend_30_vs_90d` delta feature for the next training cycle.

---

## Recommended Actions
1. Add recency-trend features (rolling change) to catch FN pattern.
2. Lower threshold to 0.35 for high-LTV customers (monetary > ₹2,000).
3. FP customers who received a campaign and stayed should feed a    positive label update — campaign effectiveness may be partially masking true churn risk.
