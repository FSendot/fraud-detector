package main

import (
	"context"
	"log"
	"net"
	"os"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/FSendot/fraud-detector/processor/internal/dynamo"
	"github.com/FSendot/fraud-detector/processor/internal/handler"
	"github.com/FSendot/fraud-detector/processor/internal/scoring"
	pb "github.com/FSendot/fraud-detector/processor/proto"
	"google.golang.org/grpc"
)

func main() {
	port := os.Getenv("GRPC_PORT")
	if port == "" {
		port = "50051"
	}

	// AWS SDK — credentials from env vars or IAM role
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		log.Fatalf("failed to load AWS config: %v", err)
	}

	dynamoClient := dynamo.NewClient(dynamodb.NewFromConfig(cfg))

	specPath, err := scoring.ResolveRuntimeSpecPath()
	if err != nil {
		log.Fatalf("failed to resolve fraud runtime spec: %v", err)
	}

	scoringEngine, err := scoring.NewEngine(specPath)
	if err != nil {
		log.Fatalf("failed to initialize fraud scoring engine: %v", err)
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterFraudProcessorServer(srv, handler.NewFraudHandler(dynamoClient, scoringEngine))

	log.Printf("fraud processor listening on :%s using runtime spec %s", port, specPath)
	if err := srv.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
