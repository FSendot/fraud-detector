package handler

import (
	"context"
	"log"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/fraud-detector/processor/internal/dynamo"
	"github.com/fraud-detector/processor/internal/scoring"
	pb "github.com/fraud-detector/processor/proto"
)

const processorVersion = "1.0.0"

type FraudHandler struct {
	pb.UnimplementedFraudProcessorServer
	dynamo        *dynamo.Client
	scoringEngine *scoring.Engine
}

func NewFraudHandler(dynamoClient *dynamo.Client, scoringEngine *scoring.Engine) *FraudHandler {
	return &FraudHandler{
		dynamo:        dynamoClient,
		scoringEngine: scoringEngine,
	}
}

func (h *FraudHandler) ProcessTransaction(ctx context.Context, req *pb.ProcessTransactionRequest) (*pb.TransactionResponse, error) {
	correlationID := uuid.New().String()
	if err := validateRequest(req); err != nil {
		return nil, err
	}
	transaction := req.GetTransaction()
	trace := req.GetTrace()

	log.Printf("[%s] processing transaction=%s request_id=%s source_system=%s user=%s feature_count=%d",
		correlationID,
		transaction.GetTransactionId(),
		trace.GetRequestId(),
		trace.GetSourceSystem(),
		transaction.GetUserId(),
		len(req.GetFeatures()),
	)

	profile, err := h.dynamo.GetProfile(ctx, transaction.GetUserId())
	if err != nil {
		log.Printf("[%s] error getting profile: %v", correlationID, err)
		return nil, err
	}

	result, err := h.scoringEngine.ScoreTransaction(req, correlationID)
	if err != nil {
		return nil, err
	}

	log.Printf("[%s] scored transaction %s with model=%s calibrated_score=%.4f action=%s decision=%s",
		correlationID, transaction.GetTransactionId(), result.ModelVersion, result.Calibrated, result.RecommendedAct, result.Decision)

	// 4. TODO: Publish enriched event to RabbitMQ
	amount := req.GetFeatures()["amount"]
	_ = buildEnrichedEvent(req, profile, result, correlationID, amount)

	// 5. UPDATE user profile in DynamoDB
	if err := h.dynamo.UpdateProfile(
		ctx,
		profile,
		amount,
		transaction.GetCountry(),
		transaction.GetChannel(),
		transaction.GetDestinationAccount(),
		transaction.GetEventTimestamp(),
	); err != nil {
		log.Printf("[%s] error updating profile: %v", correlationID, err)
	}

	return &pb.TransactionResponse{
		TransactionId:   transaction.GetTransactionId(),
		Decision:        result.Decision,
		Score:           result.Score,
		CorrelationId:   correlationID,
		RequestId:       trace.GetRequestId(),
		SourceSystem:    trace.GetSourceSystem(),
		ModelVersion:    result.ModelVersion,
		CalibratedScore: result.Calibrated,
	}, nil
}

type EnrichedEvent struct {
	TransactionID      string            `json:"transaction_id"`
	RequestID          string            `json:"request_id"`
	CorrelationID      string            `json:"correlation_id"`
	SourceSystem       string            `json:"source_system"`
	SourceComponent    string            `json:"source_component"`
	SourceRegion       string            `json:"source_region"`
	UserID             string            `json:"user_id"`
	PersonID           string            `json:"person_id"`
	AccountID          string            `json:"account_id"`
	Amount             float64           `json:"amount"`
	Currency           string            `json:"currency"`
	Timestamp          string            `json:"timestamp"`
	Channel            string            `json:"channel"`
	DestinationAccount string            `json:"destination_account"`
	Country            string            `json:"country"`
	Decision           string            `json:"decision"`
	Score              int32             `json:"score"`
	CalibratedScore    float64           `json:"calibrated_score"`
	ModelVersion       string            `json:"model_version"`
	ProfileAvgAmount   float64           `json:"profile_avg_amount"`
	ProfileStdDev      float64           `json:"profile_std_dev"`
	ProfileTxLast10Min int               `json:"profile_tx_last_10min"`
	MetadataLabels     map[string]string `json:"metadata_labels"`
	ProcessorVersion   string            `json:"processor_version"`
	ProcessedAt        string            `json:"processed_at"`
}

func buildEnrichedEvent(req *pb.ProcessTransactionRequest, profile *dynamo.UserProfile, result scoring.Result, correlationID string, amount float64) *EnrichedEvent {
	transaction := req.GetTransaction()
	trace := req.GetTrace()
	return &EnrichedEvent{
		TransactionID:      transaction.GetTransactionId(),
		RequestID:          trace.GetRequestId(),
		CorrelationID:      correlationID,
		SourceSystem:       trace.GetSourceSystem(),
		SourceComponent:    trace.GetSourceComponent(),
		SourceRegion:       trace.GetSourceRegion(),
		UserID:             transaction.GetUserId(),
		PersonID:           transaction.GetPersonId(),
		AccountID:          transaction.GetAccountId(),
		Amount:             amount,
		Currency:           transaction.GetCurrency(),
		Timestamp:          transaction.GetEventTimestamp(),
		Channel:            transaction.GetChannel(),
		DestinationAccount: transaction.GetDestinationAccount(),
		Country:            transaction.GetCountry(),
		Decision:           result.Decision,
		Score:              result.Score,
		CalibratedScore:    result.Calibrated,
		ModelVersion:       result.ModelVersion,
		ProfileAvgAmount:   profile.AvgAmount,
		ProfileStdDev:      profile.StdDevAmount,
		ProfileTxLast10Min: profile.TxLast10Min,
		MetadataLabels:     req.GetMetadataLabels(),
		ProcessorVersion:   processorVersion,
		ProcessedAt:        time.Now().UTC().Format(time.RFC3339Nano),
	}
}

func validateRequest(req *pb.ProcessTransactionRequest) error {
	if req == nil {
		return status.Error(codes.InvalidArgument, "request is required")
	}
	if req.GetTrace() == nil {
		return status.Error(codes.InvalidArgument, "trace is required")
	}
	if req.GetTrace().GetRequestId() == "" {
		return status.Error(codes.InvalidArgument, "trace.request_id is required")
	}
	if req.GetTrace().GetSourceSystem() == "" {
		return status.Error(codes.InvalidArgument, "trace.source_system is required")
	}
	if req.GetTransaction() == nil {
		return status.Error(codes.InvalidArgument, "transaction is required")
	}
	if req.GetTransaction().GetTransactionId() == "" {
		return status.Error(codes.InvalidArgument, "transaction.transaction_id is required")
	}
	if len(req.GetFeatures()) == 0 {
		return status.Error(codes.InvalidArgument, "features must include the packaged scoring inputs")
	}
	return nil
}
