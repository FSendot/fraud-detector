package main

import (
	"context"
	"log"
	"net"
	"os"

	"github.com/FSendot/fraud-detector/processor/internal/dynamo"
	"github.com/FSendot/fraud-detector/processor/internal/handler"
	"github.com/FSendot/fraud-detector/processor/internal/scoring"
	"github.com/FSendot/fraud-detector/processor/internal/store"
	pb "github.com/FSendot/fraud-detector/processor/proto"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"google.golang.org/grpc"
)

func main() {
	port := os.Getenv("GRPC_PORT")
	if port == "" {
		port = "50051"
	}

	cfg, err := dynamo.LoadAWSConfig(context.Background())
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

	// RDS client — optional, logs warning if not configured
	var rdsClient *store.RDSClient
	if os.Getenv("RDS_HOST") != "" {
		rdsClient, err = store.NewRDSClient()
		if err != nil {
			log.Printf("warning: failed to connect to RDS, transactions will not be persisted: %v", err)
		} else {
			log.Printf("RDS connected at %s", os.Getenv("RDS_HOST"))
		}
	} else {
		log.Printf("RDS_HOST not set, skipping RDS connection")
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterFraudProcessorServer(srv, handler.NewFraudHandler(dynamoClient, scoringEngine, rdsClient))

	log.Printf("fraud processor listening on :%s using runtime spec %s", port, specPath)
	if err := srv.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
