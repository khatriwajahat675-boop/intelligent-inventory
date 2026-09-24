# Data augmentation report (retrieval-grounded synthesis)

Embedding backend used: **tfidf** (see reports/augmentation/embedding_model_comparison.json for how it was selected)

Requested synthetic volume: **2,000,000** rows, split proportionally to original dataset size: FMCG 1,980,198, grocery 19,802.

Only the TRAIN split was augmented; validation and test sets are 100% real (see docs/AUGMENTATION.md).

## fmcg

- real train rows: 70,116
- synthetic rows generated: 1,980,198
- combined (real + synthetic) train rows: 2,050,314
- generation time: 14.68s (134,898.5/sec)
- output file: `data/processed/fmcg/train_augmented.csv.gz` (220.1 MB gzip)
- real positive rate (stock_risk): 1.76%  |  synthetic positive rate: 0.40%
- validation checks failed: 0

| check | status | detail |
|---|---|---|
| row count | PASS | 2,050,314 rows, 32 columns |
| dates parse | PASS | range 2024-01-01 00:00:00 .. 2024-09-12 23:54:00 |
| no future-dated sales (EC-05) | PASS | ok |
| units > 0 (EC-03) | PASS | min=1 |
| prices > 0 | PASS | ok |
| selling >= cost | PASS | no negative margin |
| lead time > 0 (FR-010 / EC-16) | PASS | 3..14 days |
| stock on hand >= 0 (EC-11) | PASS | ok |
| revenue = units x price | PASS | 0 mismatches |
| no fully duplicated lines (EC-02) | PASS | 0 duplicates |
| Invoice_ID unique | WARN | 32 repeated ids with different content -> id collisions, not replays; composite external_txn_id generated |
| external_txn_id unique | PASS | composite idempotency key |
| missing Customer_Age | WARN | 65.6% null (not used by forecasting) |
| missing Customer_Gender | WARN | 4.9% null (not used by forecasting) |
| intermittent demand (EC-52) | PASS | 1.3% of SKU-days have zero sales |

## grocery

- real train rows: 704
- synthetic rows generated: 19,802
- combined (real + synthetic) train rows: 20,506
- generation time: 0.24s (83,795.8/sec)
- output file: `data/processed/grocery/train_augmented.csv.gz` (2.8 MB gzip)
- real positive rate (stock_risk): 24.57%  |  synthetic positive rate: 24.31%
- validation checks failed: 0

| check | status | detail |
|---|---|---|
| row count | PASS | 20,506 rows, 43 columns |
| SKU_ID unique (unique constraint, TRD 11.2) | PASS | 1 row per SKU |
| numeric fields parse (EU format) | PASS | unparseable: none |
| dates parse | PASS | ok |
| expiry after receipt | PASS | ok |
| lead time > 0 (EC-16) | PASS | 1..14 days |
| safety stock >= 0 (EC-59) | PASS | ok |
| on-hand >= 0 (EC-11) | PASS | ok |
| reserved <= on-hand (FR-005) | WARN | 18 SKUs reserve more than on-hand -> available clamped to 0 |
| supplier id -> name is 1:1 | PASS | ok |
| cost > 0 | PASS | ok (needed for order-value limit, EC-64) |
| class balance (target=stock_risk) | PASS | at-risk 24.3%; raw status counts {'In Stock': 15289, 'Low Stock': 4962, 'Expiring Soon': 231, 'Out of Stock': 24} |
| no sales history in file | WARN | single snapshot per SKU -> cannot train ARIMA/Prophet on it; used for master data, opening balances and the stock-risk classifier |
