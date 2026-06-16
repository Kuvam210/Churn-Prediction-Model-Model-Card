# Model Card -- D2C Churn Prediction Model

## Model Details
- **Model type:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Hyperparameters:** C=0.1, max_iter=500, solver=lbfgs, random_state=42
- **Threshold:** 0.40
- **Artifact:** `model.pkl` (includes model, encoders, feature list, threshold)
- **Trained by:** Kuvam (Capstone Part 3)
- **Date:** 2025-10-01

---

## Intended Use
- **Primary use:** Score D2C customers on their probability of churning in the 60 days following the snapshot date.
- **Primary users:** Internal retention and CRM teams who prioritise outreach campaigns.
- **Out-of-scope uses:** This model must NOT be used for credit scoring, pricing discrimination, denying service, or any automated decision-making without human review.

---

## Training Data
- **Source:** `rfm_modeling_snapshot.csv` -- a pre-built feature table derived from orders, support tickets, and web events.
- **Snapshot date:** 2025-09-30. All features represent customer state at or before this date.
- **Leakage prevention:** Post-snapshot orders (order_date > 2025-09-30) exist only to define churn labels and are never used as model input.
- **Train set:** 1,728 customers | Churn rate: 47.0%
- **Validation set:** 336 customers | Churn rate: 43.8%
- **Test set:** 336 customers | Churn rate: 50.0%

### Features Used (25 total)
| Category | Features |
|---|---|
| Customer profile | city_tier, age_group, acquisition_channel, loyalty_tier, preferred_category, marketing_consent |
| RFM | recency_days, frequency_180d, monetary_180d |
| Order quality | return_rate_180d, avg_discount_pct_180d, avg_rating_180d, category_diversity_180d |
| Support | ticket_count_90d, negative_ticket_rate_90d, avg_resolution_hours_90d |
| Lifecycle | days_since_signup |
| Web/App (30d) | sessions_30d, product_views_30d, cart_adds_30d, wishlist_adds_30d, abandoned_carts_30d, email_opens_30d, campaign_clicks_30d, last_visit_days_ago |

### Preprocessing
- `loyalty_tier` nulls (58% of customers) filled with `"Not Enrolled"` -- a meaningful business category, not imputed noise.
- `avg_rating_180d` nulls (~3%) filled with the training-set median.
- Categorical columns label-encoded. No standard scaling applied (pre-engineered features are on consistent scales).

---

## Performance

### Validation Set
| Metric | Value |
|---|---|
| ROC-AUC | 0.8795 |
| PR-AUC | 0.8635 |
| F1 (t=0.40) | 0.7871 |
| Precision | 0.7485 |
| Recall | 0.8299 |

### Test Set (Final, Unbiased)
| Metric | Value |
|---|---|
| ROC-AUC | **0.8932** |
| PR-AUC | **0.8843** |
| F1 (t=0.40) | **0.8315** |
| Precision | 0.7872 |
| Recall | **0.8810** |

### Baseline Comparison (Gradient Boosting)
| Metric | LR (Final) | GBM |
|---|---|---|
| Test ROC-AUC | **0.8932** | 0.8646 |
| Test F1 | **0.8315** | 0.7932 |

LR outperforms GBM on this dataset because `recency_days` (57% GBM feature importance) has a near-linear relationship with churn, making complex ensembles unnecessary and reducing the risk of overfitting.

---

## Limitations

1. **Recency dominance:** 57% of GBM predictive power comes from `recency_days`. The model may over-flag seasonal or cycle buyers whose natural repurchase interval exceeds 90 days. Avoid sending win-back campaigns to customers whose average inter-purchase interval is close to their current recency gap.

2. **Silent churners:** High-value, recently-active customers who churn suddenly (false negative rate is highest for customers with monetary_180d > Rs.2,000 and recency_days < 15). A rule-based overlay is recommended: Gold/Platinum customers with any support ticket in 30 days should receive a manual review flag regardless of model score.

3. **No post-purchase signal:** The model cannot detect product quality disappointment that occurs after the snapshot date. Post-delivery NPS tracking should complement model scores.

4. **Acquisition channel interactions not modelled:** Influencer-acquired customers exhibit shallow brand loyalty that the model underestimates. Consider a 5-point probability uplift rule for this segment.

5. **Static model:** Customer behaviour, seasonality, and product mix change over time. This model should be retrained at minimum every quarter.

---

## Ethical Risks

| Risk | Mitigation |
|---|---|
| Disparate impact across city tiers | Monitor churn-flag rates by city_tier quarterly; flag if any tier's flag rate diverges by >15 percentage points from the portfolio average |
| Age group bias | Report precision and recall broken down by age_group; younger cohorts may behave differently as the brand's product mix shifts |
| Automation without oversight | High-probability scores (>0.85) should trigger a human review step before campaign execution for Gold/Platinum customers |
| Reward inversion | Do not use model scores to deny promotions or service to customers; the model ranks risk, not entitlement |
| Data minimisation | Model does not require PII (names, emails, phone numbers) to function; ensure customer_id to contact-detail mapping is handled only in the CRM layer, not in the model pipeline |

---

## Monitoring Plan

| Signal | Metric | Alert Threshold | Frequency |
|---|---|---|---|
| Model performance | ROC-AUC on new cohort | < 0.82 | Monthly |
| Score distribution shift | PSI (Population Stability Index) | > 0.20 | Monthly |
| Business outcome | Actual churn rate in flagged vs unflagged cohorts | >10pp divergence from expected | Per campaign (60-day lag) |
| False negative rate on high-value customers | FN rate for monetary_180d > 75th percentile | > 20% | Quarterly |
| Retraining trigger | Any of above alerts OR > 3 months since last training | -- | Rolling |

---

## Responsible Use Note
This model is a decision-support tool, not a decision-making authority. The retention team should use churn scores to **prioritise** outreach, not to automatically trigger campaigns without judgment. Customers flagged as low-risk are not guaranteed to stay; and customers flagged as high-risk should be approached with empathy, not treated as revenue units. The model reflects historical patterns; it cannot predict individual customer intent.
