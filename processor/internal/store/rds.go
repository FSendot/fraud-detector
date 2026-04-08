package store

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

type TransactionRecord struct {
	TransactionID      string
	UserID             string
	PersonID           string
	AccountID          string
	Amount             float64
	Currency           string
	Timestamp          string
	Channel            string
	DestinationAccount string
	Country            string

	Score    int32
	Decision string

	FlagAmount      bool
	FlagCountry     bool
	FlagVelocity    bool
	FlagDestination bool

	ProfileAvgAmount         float64
	ProfileStdDev            float64
	ProfileTypicalCountries  []string
	ProfileTxLast10Min       int

	ModelVersion     string
	CalibratedScore  float64
	ProcessorVersion string
	ProcessedAt      time.Time
}

type RDSClient struct {
	db *sql.DB
}

func NewRDSClient() (*RDSClient, error) {
	host := os.Getenv("RDS_HOST")
	port := os.Getenv("RDS_PORT")
	user := os.Getenv("RDS_USER")
	password := os.Getenv("RDS_PASSWORD")
	dbname := os.Getenv("RDS_DBNAME")

	if port == "" {
		port = "5432"
	}
	if dbname == "" {
		dbname = "fraud"
	}

	dsn := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=require",
		host, port, user, password, dbname)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("rds open: %w", err)
	}

	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("rds ping: %w", err)
	}

	return &RDSClient{db: db}, nil
}

func (c *RDSClient) InsertTransaction(ctx context.Context, r *TransactionRecord) error {
	_, err := c.db.ExecContext(ctx, `
		INSERT INTO transactions (
			transaction_id, user_id, person_id, account_id,
			amount, currency, timestamp, channel, destination_account, country,
			score, decision,
			flag_amount, flag_country, flag_velocity, flag_destination,
			profile_avg_amount, profile_std_dev, profile_typical_countries, profile_tx_last_10min,
			model_version, calibrated_score, processor_version, processed_at
		) VALUES (
			$1, $2, $3, $4,
			$5, $6, $7, $8, $9, $10,
			$11, $12,
			$13, $14, $15, $16,
			$17, $18, $19, $20,
			$21, $22, $23, $24
		) ON CONFLICT (transaction_id) DO NOTHING`,
		r.TransactionID, r.UserID, r.PersonID, r.AccountID,
		r.Amount, r.Currency, r.Timestamp, r.Channel, r.DestinationAccount, r.Country,
		r.Score, r.Decision,
		r.FlagAmount, r.FlagCountry, r.FlagVelocity, r.FlagDestination,
		r.ProfileAvgAmount, r.ProfileStdDev, strings.Join(r.ProfileTypicalCountries, ","), r.ProfileTxLast10Min,
		r.ModelVersion, r.CalibratedScore, r.ProcessorVersion, r.ProcessedAt,
	)
	if err != nil {
		return fmt.Errorf("rds insert transaction: %w", err)
	}
	return nil
}

func (c *RDSClient) Close() {
	_ = c.db.Close()
}
