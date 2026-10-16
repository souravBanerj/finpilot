# FinPilot self-evaluation

Synthetic data: **20 seeded personas × 12 months** (12766 transactions) with ground-truth labels (`eval/generate_data.py`). Regenerate with `python -m eval.run_eval`.

## 1. Categorization (offline: memory-free rules + recurrence, no LLM)

- **Accuracy: 91.7%**   **Macro-F1: 0.969**

| Layer | Share of txns | Accuracy |
|---|---|---|
| fallback | 8.3% | 0.0% |
| recurrence | 3.8% | 100.0% |
| rule | 88.0% | 100.0% |

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Income | 1.00 | 1.00 | 1.00 | 291 |
| Rent | 1.00 | 1.00 | 1.00 | 240 |
| Groceries | 1.00 | 0.93 | 0.96 | 1464 |
| Food Delivery | 1.00 | 0.88 | 0.94 | 2073 |
| Dining | 1.00 | 0.88 | 0.93 | 975 |
| Transport | 1.00 | 0.87 | 0.93 | 3420 |
| Subscriptions | 1.00 | 1.00 | 1.00 | 1920 |
| Utilities | 1.00 | 1.00 | 1.00 | 720 |
| Shopping | 1.00 | 0.89 | 0.94 | 808 |
| Entertainment | 1.00 | 0.86 | 0.93 | 441 |
| Health | 1.00 | 0.92 | 0.96 | 129 |
| Travel | 1.00 | 1.00 | 1.00 | 45 |
| Savings | 1.00 | 1.00 | 1.00 | 240 |

Top confusions (honest failure modes): `Transport -> Other` ×429; `Food Delivery -> Other` ×247; `Dining -> Other` ×121; `Groceries -> Other` ×103; `Shopping -> Other` ×87; `Entertainment -> Other` ×60; `Health -> Other` ×10

_LLM fallback not evaluated in this run (no MESH_API_KEY). Unknown merchants land in `Other` here; set the key and re-run to measure the LLM layer._

## 2. Anomaly / pattern detection (injected ground truth)

| Detector | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| category_spike | 16 | 12 | 4 | 57% | 80% |
| duplicate_charge | 40 | 0 | 0 | 100% | 100% |
| hidden_subscription_caught | 40 | 0 | 0 | 100% | 100% |
| subscription_overlap | 40 | 0 | 0 | 100% | 100% |
| subscription_price_hike | 20 | 0 | 0 | 100% | 100% |
| txn_outlier | 40 | 5 | 0 | 89% | 100% |

**Spike-detector sensitivity** (threshold = std-devs above the category's own history; production default = 1.5). Monthly order counts are noisy (~35%), so this is a genuine precision/recall trade-off:

| σ threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.0 | 48% | 100% | 0.65 |
| 1.0 | 58% | 95% | 0.72 |
| 1.5 | 57% | 80% | 0.67 |
| 2.0 | 61% | 70% | 0.65 |
| 2.5 | 67% | 50% | 0.57 |

## 3. Forecast backtest (rolling origin, train on months 1..t-1, predict month t)

| Method (total monthly expenses) | MAPE |
|---|---|
| **FinPilot model** | **14.89%** |
| Naive (last month) | 18.47% |
| Mean of last 3 months | 16.06% |

- Income MAPE: 2.82%
- 80% prediction-band coverage: **87.5%** (target 80%)

| Category | Model MAPE | Naive MAPE |
|---|---|---|
| Dining | 130.8% | 154.8% |
| Entertainment | 59.6% | 87.4% |
| Food Delivery | 52.2% | 60.9% |
| Groceries | 56.4% | 68.3% |
| Health | 70.0% | 97.9% |
| Other | 161.0% | 168.8% |
| Rent | 0.0% | 0.0% |
| Shopping | 138.2% | 166.7% |
| Subscriptions | 0.5% | 0.5% |
| Transport | 30.1% | 37.3% |
| Travel | 82.4% | 98.0% |
| Utilities | 9.4% | 6.7% |
