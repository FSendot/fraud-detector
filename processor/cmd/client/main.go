package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/fraud-detector/processor/proto"
)

func main() {
	addr := flag.String("addr", "localhost:50051", "server address")
	txID := flag.String("tx-id", "", "transaction ID")
	userID := flag.String("user-id", "", "user ID")
	personID := flag.String("person-id", "", "person ID")
	accountID := flag.String("account-id", "", "account ID")
	amount := flag.Float64("amount", 0, "transaction amount")
	currency := flag.String("currency", "ARS", "currency code")
	ts := flag.String("timestamp", time.Now().UTC().Format(time.RFC3339), "transaction timestamp")
	channel := flag.String("channel", "web", "channel (web, mobile, atm)")
	dest := flag.String("dest", "", "destination account")
	country := flag.String("country", "", "country code")
	requestID := flag.String("request-id", "", "traceable upstream request ID")
	sourceSystem := flag.String("source-system", "processor-client", "originating system name")
	sourceComponent := flag.String("source-component", "cli", "originating component")
	sourceRegion := flag.String("source-region", "local", "originating region")
	flag.Parse()

	if *txID == "" || *userID == "" || *amount == 0 {
		log.Fatal("required flags: -tx-id, -user-id, -amount")
	}
	if *requestID == "" {
		*requestID = fmt.Sprintf("req-%s", *txID)
	}

	conn, err := grpc.Dial(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewFraudProcessorClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.ProcessTransaction(ctx, &pb.ProcessTransactionRequest{
		Trace: &pb.RequestTrace{
			RequestId:       *requestID,
			SourceSystem:    *sourceSystem,
			SourceComponent: *sourceComponent,
			SourceRegion:    *sourceRegion,
		},
		Transaction: &pb.TransactionContext{
			TransactionId:      *txID,
			UserId:             *userID,
			PersonId:           *personID,
			AccountId:          *accountID,
			DestinationAccount: *dest,
			Currency:           *currency,
			Country:            *country,
			Channel:            *channel,
			EventTimestamp:     *ts,
		},
		Features: map[string]float64{
			"amount": *amount,
		},
		MetadataLabels: map[string]string{
			"channel":     *channel,
			"environment": "local-dev",
		},
	})
	if err != nil {
		log.Fatalf("error: %v", err)
	}

	fmt.Printf("Response:\n")
	fmt.Printf("  Transaction ID:  %s\n", resp.TransactionId)
	fmt.Printf("  Request ID:      %s\n", resp.RequestId)
	fmt.Printf("  Source System:   %s\n", resp.SourceSystem)
	fmt.Printf("  Decision:        %s\n", resp.Decision)
	fmt.Printf("  Score:           %d\n", resp.Score)
	fmt.Printf("  Model Version:   %s\n", resp.ModelVersion)
	fmt.Printf("  Cal. Score:      %.4f\n", resp.CalibratedScore)
	fmt.Printf("  Correlation ID:  %s\n", resp.CorrelationId)
}
