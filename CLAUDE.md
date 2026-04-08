# Fraud Detection Processor — Context for Claude Code

## What this is
A fraud detection processor that receives payment transactions, scores them in real time,
and decides whether to allow, block, or challenge each one.

This is an MVP to demo to VISA on Wednesday. Keep it simple and working over perfect.

---

## Architecture

```
TransactionEvent (protobuf via gRPC)
        ↓
   Load Balancer
        ↓
   [PROCESSOR]  ←→  DynamoDB (user profile read/write)
        ↓
   RabbitMQ (publishes enriched event)
        ↓
   Batch Consumer
        ↓
   Redshift (analytics + dashboard)
        ↓
   API REST → Dashboard (VISA)

S3 ← backup (Dynamo + Redshift automated)
Neptune ← post-MVP (money laundering / graph fraud)
```

---

## Transaction Event (proto)

```proto
message TransactionEvent {
  string transaction_id = 1;
  string user_id        = 2;
  string person_id      = 3;
  string account_id     = 4;
  double amount         = 5;
  string currency       = 6;
  string timestamp      = 7;
  string channel        = 8;
  string destination_account = 9;
  string country        = 10;
}
```

---

## Processor flow (in order)

1. Receive TransactionEvent via gRPC
2. Generate correlation_id for tracing
3. GET user profile from DynamoDB (single item, low latency)
4. Compute fraud features and score
5. Decide: allow / block / challenge
6. Publish enriched event to RabbitMQ
7. UPDATE user profile in DynamoDB (incremental, no full history)

---

## DynamoDB — User Profile

Table: `user_profiles`
Primary key: `user_id` (string)

Each item represents the current statistical profile of a user.
Never store full transaction history here — only aggregated state.

```json
{
  "user_id": "u_123",
  "avg_amount": 4500.00,
  "std_dev_amount": 1200.00,
  "tx_count": 142,
  "tx_last_hour": 2,
  "tx_last_10min": 0,
  "typical_countries": ["AR", "UY"],
  "typical_channels": ["web", "mobile"],
  "known_destinations": ["acc_456", "acc_789"],
  "last_country": "AR",
  "last_timestamp": "2026-04-03T10:22:00Z"
}
```

Profile is updated AFTER the decision is made using incremental statistics
(Welford's online algorithm for mean/stddev — no need to store all transactions).

---

## Fraud rules (MVP — no graph DB needed)

All rules produce a boolean flag. Final score is a weighted sum of flags.

| Flag | Rule | Field from event | Field from profile |
|------|------|------------------|--------------------|
| flag_amount | amount > avg + 2 * std_dev | amount | avg_amount, std_dev_amount |
| flag_country | country not in typical_countries | country | typical_countries |
| flag_velocity | tx_last_10min >= 3 | timestamp | tx_last_10min |
| flag_destination | destination_account not in known_destinations | destination_account | known_destinations |

Score = weighted sum. Example weights (tunable):
- flag_amount: 30
- flag_country: 25
- flag_velocity: 35
- flag_destination: 10

Decision thresholds:
- score < 40  → allow
- score 40–69 → challenge
- score >= 70  → block

---

## Enriched event (published to RabbitMQ → consumed by Batch Consumer → Redshift)

```json
{
  "transaction_id": "tx_001",
  "user_id": "u_123",
  "person_id": "p_456",
  "account_id": "acc_123",
  "amount": 15000.00,
  "currency": "ARS",
  "timestamp": "2026-04-03T10:22:00Z",
  "channel": "web",
  "destination_account": "acc_999",
  "country": "BR",

  "score": 85,
  "decision": "blocked",

  "flag_amount": true,
  "flag_country": true,
  "flag_velocity": false,
  "flag_destination": true,

  "profile_avg_amount": 4500.00,
  "profile_std_dev": 1200.00,
  "profile_typical_countries": ["AR", "UY"],
  "profile_tx_last_10min": 1,

  "processor_version": "1.0.0",
  "processed_at": "2026-04-03T10:22:00.123Z"
}
```

---

## Redshift — Transactions table

```sql
CREATE TABLE transactions (
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

  processor_version     VARCHAR(16),
  processed_at          TIMESTAMP
);
```

---

## RabbitMQ

Queue: `enriched_transactions`
Exchange: direct
Message format: JSON (the enriched event above)

The Batch Consumer reads from this queue, batches messages, and inserts into Redshift.

---

## Tech stack

- Language: Java (Spring Boot) or Go — team decides
- gRPC: receive TransactionEvent
- AWS SDK v2: DynamoDB client
- RabbitMQ client: Spring AMQP (Java) or amqp091-go (Go)
- Redshift: JDBC with PostgreSQL driver (Redshift is Postgres-compatible)

---

## Infrastructure

- EC2 on AWS Learner Lab
- Docker + docker-compose for all services
- GitHub repo, deploy via git pull + docker compose up --build

```
/
├── processor/
│   ├── Dockerfile
│   └── src/
├── batch-consumer/
│   ├── Dockerfile
│   └── src/
├── dashboard/
│   ├── Dockerfile
│   └── src/
└── docker-compose.yml
```

---

## What's out of scope for Wednesday

- Neptune (graph DB for money laundering) — mention in demo as roadmap
- Kinesis / Flink (real-time streaming) — RabbitMQ covers MVP needs
- Device fingerprinting — no device field in proto yet
- VPN detection — use country field from event as-is

---

## Key decisions and why

| Decision | Reason |
|----------|--------|
| DynamoDB for profiles | Single-key lookup, microsecond latency, no joins needed |
| RabbitMQ as buffer | Decouples processor from Redshift, messages persist if consumer fails |
| Redshift for analytics | Columnar, optimized for aggregation queries the dashboard needs |
| INSERT per transaction to Redshift | MVP volume is low, simpler than S3+COPY for now |
| S3 for backups | Automated, cheap, future source for ML retraining |
| Neptune post-MVP | Money laundering detection requires graph traversal, not needed Wednesday |

---

## How to use this file with Claude Code

Drop this file at the root of your repo as `CLAUDE.md`.
Claude Code reads it automatically at the start of every session.

Then you can say things like:
- "Implement the processor flow described in CLAUDE.md"
- "Create the DynamoDB client with the user profile schema"
- "Implement the fraud scoring rules"
- "Set up the RabbitMQ publisher for the enriched event"
- "Create the Redshift schema and JDBC insert"
- "Generate the proto file and gRPC server skeleton"
- "Write the docker-compose.yml for all services"
