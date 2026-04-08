CREATE TABLE IF NOT EXISTS transactions (
  transaction_id        VARCHAR(64) PRIMARY KEY,
  user_id               VARCHAR(64),
  person_id             VARCHAR(64),
  account_id            VARCHAR(64),
  amount                DECIMAL(18,2),
  currency              VARCHAR(10),
  timestamp             TIMESTAMP,
  channel               VARCHAR(32),
  destination_account   VARCHAR(64),
  country               VARCHAR(8),

  score                 INTEGER,
  decision              VARCHAR(16),

  flag_amount           BOOLEAN,
  flag_country          BOOLEAN,
  flag_velocity         BOOLEAN,
  flag_destination      BOOLEAN,

  profile_avg_amount    DECIMAL(18,2),
  profile_std_dev       DECIMAL(18,2),
  profile_typical_countries VARCHAR(256),
  profile_tx_last_10min INTEGER,

  model_version         VARCHAR(32),
  calibrated_score      DECIMAL(10,6),
  processor_version     VARCHAR(16),
  processed_at          TIMESTAMP
);
