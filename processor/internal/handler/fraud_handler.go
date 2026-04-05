package handler

import (
	"context"
	"log"

	"github.com/google/uuid"

	pb "github.com/fraud-detector/processor/proto"
)

type FraudHandler struct {
	pb.UnimplementedFraudProcessorServer
}

func NewFraudHandler() *FraudHandler {
	return &FraudHandler{}
}

func (h *FraudHandler) ProcessTransaction(ctx context.Context, req *pb.TransactionEvent) (*pb.TransactionResponse, error) {
	correlationID := uuid.New().String()

	log.Printf("[%s] processing transaction %s for user %s — amount=%.2f country=%s",
		correlationID, req.TransactionId, req.UserId, req.Amount, req.Country)

	// TODO: 1. GET user profile from DynamoDB
	// TODO: 2. Compute fraud features and score
	// TODO: 3. Decide: allow / block / challenge
	// TODO: 4. Publish enriched event to RabbitMQ
	// TODO: 5. UPDATE user profile in DynamoDB

	// Placeholder — allow everything until scoring is wired up
	return &pb.TransactionResponse{
		TransactionId: req.TransactionId,
		Decision:      "allowed",
		Score:         0,
		CorrelationId: correlationID,
	}, nil
}
