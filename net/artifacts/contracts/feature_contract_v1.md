# Feature Contract v1

- Contract: `fraud_model_feature_contract`
- Dataset stage: `tabular_model_input`
- Feature count: `14`
- Raw features: `2`
- Derived features: `12`

## Conventions
- Transaction ID: `transaction_id` as `string`
- Label: `is_fraud` as `boolean`

## Preprocessing Expectations
- Numeric coercion: `pd.to_numeric(errors='coerce')`
- Null fill before scaling: `0.0`
- Scaling: `StandardScaler` fit on `train_only`
- Feature selection: `VarianceThreshold+SelectKBest(f_classif)`

## Feature Order
- `00` `amount` [source, float64]
- `01` `is_flagged_fraud` [source, float64]
- `02` `previous_transaction_amount` [derived, float64]
- `03` `prior_5_transaction_count` [derived, float64]
- `04` `prior_10_transaction_count` [derived, float64]
- `05` `prior_5_amount_sum` [derived, float64]
- `06` `prior_5_amount_mean` [derived, float64]
- `07` `prior_5_amount_std` [derived, float64]
- `08` `prior_10_amount_sum` [derived, float64]
- `09` `prior_10_amount_mean` [derived, float64]
- `10` `prior_10_amount_std` [derived, float64]
- `11` `seconds_since_previous_transaction` [derived, float64]
- `12` `prior_5_unique_name_dest_count` [derived, float64]
- `13` `prior_10_unique_name_dest_count` [derived, float64]
