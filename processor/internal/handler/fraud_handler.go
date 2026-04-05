package handler

import (
	"context"
	"log"
	"time"

	"github.com/google/uuid"

	"github.com/fraud-detector/processor/internal/dynamo"
	"github.com/fraud-detector/processor/internal/scoring"
	pb "github.com/fraud-detector/processor/proto"
)

const processorVersion = "1.0.0"

type FraudHandler struct {
	pb.UnimplementedFraudProcessorServer
	dynamo *dynamo.Client
}

func NewFraudHandler(dynamoClient *dynamo.Client) *FraudHandler {
	return &FraudHandler{dynamo: dynamoClient}
}

func (h *FraudHandler) ProcessTransaction(ctx context.Context, req *pb.TransactionEvent) (*pb.TransactionResponse, error) {
	correlationID := uuid.New().String()

	log.Printf("[%s] processing tx=%s user=%s amount=%.2f country=%s",
		correlationID, req.TransactionId, req.UserId, req.Amount, req.Country)

	// 1. GET user profile from DynamoDB
	profile, err := h.dynamo.GetProfile(ctx, req.UserId)
	if err != nil {
		log.Printf("[%s] error getting profile: %v", correlationID, err)
		return nil, err
	}

	// 2-3. Compute fraud score and decide
	result := scoring.Evaluate(req, profile)

	log.Printf("[%s] score=%d decision=%s flags=[amount=%v country=%v velocity=%v dest=%v]",
		correlationID, result.Score, result.Decision,
		result.FlagAmount, result.FlagCountry, result.FlagVelocity, result.FlagDestination)

	// 4. TODO: Publish enriched event to RabbitMQ
	_ = buildEnrichedEvent(req, profile, result, correlationID)

	// 5. UPDATE user profile in DynamoDB
	if err := h.dynamo.UpdateProfile(ctx, profile, req.Amount, req.Country, req.Channel, req.DestinationAccount, req.Timestamp); err != nil {
		log.Printf("[%s] error updating profile: %v", correlationID, err)
		// Non-fatal — we still return the decision
	}

	return &pb.TransactionResponse{
		TransactionId: req.TransactionId,
		Decision:      result.Decision,
		Score:         int32(result.Score),
		CorrelationId: correlationID,
	}, nil
}

// EnrichedEvent is the full event published to RabbitMQ for downstream consumption.
type EnrichedEvent struct {
	TransactionID      string   `json:"transaction_id"`
	UserID             string   `json:"user_id"`
	PersonID           string   `json:"person_id"`
	AccountID          string   `json:"account_id"`
	Amount             float64  `json:"amount"`
	Currency           string   `json:"currency"`
	Timestamp          string   `json:"timestamp"`
	Channel            string   `json:"channel"`
	DestinationAccount string   `json:"destination_account"`
	Country            string   `json:"country"`

	Score    int    `json:"score"`
	Decision string `json:"decision"`

	FlagAmount      bool `json:"flag_amount"`
	FlagCountry     bool `json:"flag_country"`
	FlagVelocity    bool `json:"flag_velocity"`
	FlagDestination bool `json:"flag_destination"`

	ProfileAvgAmount         float64  `json:"profile_avg_amount"`
	ProfileStdDev            float64  `json:"profile_std_dev"`
	ProfileTypicalCountries  []string `json:"profile_typical_countries"`
	ProfileTxLast10Min       int      `json:"profile_tx_last_10min"`

	ProcessorVersion string `json:"processor_version"`
	ProcessedAt      string `json:"processed_at"`
}

func buildEnrichedEvent(tx *pb.TransactionEvent, profile *dynamo.UserProfile, result scoring.Result, correlationID string) *EnrichedEvent {
	return &EnrichedEvent{
		TransactionID:      tx.TransactionId,
		UserID:             tx.UserId,
		PersonID:           tx.PersonId,
		AccountID:          tx.AccountId,
		Amount:             tx.Amount,
		Currency:           tx.Currency,
		Timestamp:          tx.Timestamp,
		Channel:            tx.Channel,
		DestinationAccount: tx.DestinationAccount,
		Country:            tx.Country,

		Score:    result.Score,
		Decision: result.Decision,

		FlagAmount:      result.FlagAmount,
		FlagCountry:     result.FlagCountry,
		FlagVelocity:    result.FlagVelocity,
		FlagDestination: result.FlagDestination,

		ProfileAvgAmount:        profile.AvgAmount,
		ProfileStdDev:           profile.StdDevAmount,
		ProfileTypicalCountries: profile.TypicalCountries,
		ProfileTxLast10Min:      profile.TxLast10Min,

		ProcessorVersion: processorVersion,
		ProcessedAt:      time.Now().UTC().Format(time.RFC3339Nano),
	}
}
