package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/fraud-detector/processor/proto"
)

func main() {
	conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewFraudProcessorClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.ProcessTransaction(ctx, &pb.TransactionEvent{
		TransactionId:      "tx_001",
		UserId:             "u_123",
		PersonId:           "p_456",
		AccountId:          "acc_123",
		Amount:             15000.00,
		Currency:           "ARS",
		Timestamp:          "2026-04-03T10:22:00Z",
		Channel:            "web",
		DestinationAccount: "acc_999",
		Country:            "BR",
	})
	if err != nil {
		log.Fatalf("error: %v", err)
	}

	fmt.Printf("Response:\n")
	fmt.Printf("  Transaction ID:  %s\n", resp.TransactionId)
	fmt.Printf("  Decision:        %s\n", resp.Decision)
	fmt.Printf("  Score:           %d\n", resp.Score)
	fmt.Printf("  Correlation ID:  %s\n", resp.CorrelationId)
}
