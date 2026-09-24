# Data validation and 70/15/15 split report

## Indian FMCG retail sales 2024

Split method: **chronological (day-aligned)**

| set | rows | share | at-risk rate | range |
|---|---|---|---|---|
| train | 70,116 | 70.1% | 1.76% | 2024-01-01 .. 2024-09-12 |
| val | 14,932 | 14.9% | 1.69% | 2024-09-13 .. 2024-11-05 |
| test | 14,952 | 14.9% | 1.62% | 2024-11-06 .. 2024-12-30 |

| check | status | detail |
|---|---|---|
| row count | PASS | 100,000 rows, 28 columns |
| dates parse | PASS | range 2024-01-01 00:00:00 .. 2024-12-30 23:57:00 |
| no future-dated sales (EC-05) | PASS | ok |
| units > 0 (EC-03) | PASS | min=1 |
| prices > 0 | PASS | ok |
| selling >= cost | PASS | no negative margin |
| lead time > 0 (FR-010 / EC-16) | PASS | 3..14 days |
| stock on hand >= 0 (EC-11) | PASS | ok |
| revenue = units x price | PASS | 0 mismatches |
| no fully duplicated lines (EC-02) | PASS | 0 duplicates |
| Invoice_ID unique | WARN | 62 repeated ids with different content -> id collisions, not replays; composite external_txn_id generated |
| external_txn_id unique | PASS | composite idempotency key |
| missing Customer_Age | WARN | 40.1% null (not used by forecasting) |
| missing Customer_Gender | WARN | 5.0% null (not used by forecasting) |
| intermittent demand (EC-52) | PASS | 1.3% of SKU-days have zero sales |

Drift (PSI vs train; <0.1 stable): Units val=0.0002 test=0.0001, Selling_Price val=0.0005 test=0.0013, Lead_Time_Days val=0.001 test=0.001, Stock_On_Hand val=0.0012 test=0.0009

SMOTE experiment (train-only oversampling; threshold tuned on validation):

| model | SMOTE | fit rows | pos. rate | val F1 | test precision | test recall | test F1 | test PR-AUC |
|---|---|---|---|---|---|---|---|---|
| logreg | False | 70,116 | 0.02 | 0.034 | 0.016 | 0.946 | 0.032 | 0.016 |
| logreg | True | 137,762 | 0.50 | 0.034 | 0.016 | 0.909 | 0.032 | 0.016 |
| rf | False | 70,116 | 0.02 | 0.034 | 0.014 | 0.331 | 0.028 | 0.016 |
| rf | True | 137,762 | 0.50 | 0.034 | 0.016 | 0.777 | 0.032 | 0.017 |

## E-Grocery inventory snapshot

Split method: **stratified by ['Category', 'stock_risk'] (seed=42)**

| set | rows | share | at-risk rate | range |
|---|---|---|---|---|
| train | 704 | 70.4% | 24.57% | - .. - |
| val | 148 | 14.8% | 23.65% | - .. - |
| test | 148 | 14.8% | 23.65% | - .. - |

| check | status | detail |
|---|---|---|
| row count | PASS | 1,000 rows, 39 columns |
| SKU_ID unique (unique constraint, TRD 11.2) | PASS | 1 row per SKU |
| numeric fields parse (EU format) | PASS | unparseable: none |
| dates parse | PASS | ok |
| expiry after receipt | PASS | ok |
| lead time > 0 (EC-16) | PASS | 1..14 days |
| safety stock >= 0 (EC-59) | PASS | ok |
| on-hand >= 0 (EC-11) | PASS | ok |
| reserved <= on-hand (FR-005) | WARN | 28 SKUs reserve more than on-hand -> available clamped to 0 |
| supplier id -> name is 1:1 | PASS | ok |
| cost > 0 | PASS | ok (needed for order-value limit, EC-64) |
| class balance (target=stock_risk) | PASS | at-risk 24.3%; raw status counts {'In Stock': 428, 'Expiring Soon': 329, 'Low Stock': 241, 'Out of Stock': 2} |
| no sales history in file | WARN | single snapshot per SKU -> cannot train ARIMA/Prophet on it; used for master data, opening balances and the stock-risk classifier |

Drift (PSI vs train; <0.1 stable): Avg_Daily_Sales val=0.0809 test=0.1134, Lead_Time_Days val=0.0339 test=0.0218, Unit_Cost_USD val=0.0954 test=0.0433, Safety_Stock val=0.0388 test=0.0936

SMOTE experiment (train-only oversampling; threshold tuned on validation):

| model | SMOTE | fit rows | pos. rate | val F1 | test precision | test recall | test F1 | test PR-AUC |
|---|---|---|---|---|---|---|---|---|
| logreg | False | 704 | 0.25 | 0.757 | 0.885 | 0.657 | 0.754 | 0.859 |
| logreg | True | 1,062 | 0.50 | 0.763 | 0.815 | 0.629 | 0.710 | 0.860 |
| rf | False | 704 | 0.25 | 0.917 | 1.000 | 0.829 | 0.906 | 0.958 |
| rf | True | 1,062 | 0.50 | 0.919 | 0.969 | 0.886 | 0.925 | 0.976 |

## Seasonality signal in FMCG demand

{"weekday_units_cv": 0.0088, "month_units_cv": 0.026, "note": "coefficient of variation of total units by weekday / month (0 = no seasonality)"}
