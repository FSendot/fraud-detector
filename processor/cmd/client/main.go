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
	flag.Parse()

	if *txID == "" || *userID == "" || *amount == 0 {
		log.Fatal("required flags: -tx-id, -user-id, -amount")
	}

	conn, err := grpc.Dial(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewFraudProcessorClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.ProcessTransaction(ctx, &pb.TransactionEvent{
		TransactionId:      *txID,
		UserId:             *userID,
		PersonId:           *personID,
		AccountId:          *accountID,
		Amount:             *amount,
		Currency:           *currency,
		Timestamp:          *ts,
		Channel:            *channel,
		DestinationAccount: *dest,
		Country:            *country,
	})
	if err != nil {
		log.Fatalf("error: %v", err)
	}

	fmt.Printf("Transaction: %s\n", resp.TransactionId)
	fmt.Printf("Decision:    %s\n", resp.Decision)
	fmt.Printf("Score:       %d\n", resp.Score)
	fmt.Printf("Correlation: %s\n", resp.CorrelationId)
}
