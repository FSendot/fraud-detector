package main

import (
	"context"
	"log"
	"os"
	"time"

	internaldynamo "github.com/FSendot/fraud-detector/processor/internal/dynamo"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	cfg, err := internaldynamo.LoadAWSConfig(ctx)
	if err != nil {
		log.Fatalf("failed to load AWS config: %v", err)
	}

	client := dynamodb.NewFromConfig(cfg)
	if err := internaldynamo.EnsureUserProfilesTable(ctx, client); err != nil {
		log.Fatalf("failed to ensure %s table: %v", internaldynamo.TableName(), err)
	}

	log.Printf("dynamodb table ready: %s endpoint=%s region=%s",
		internaldynamo.TableName(),
		os.Getenv("DYNAMODB_ENDPOINT"),
		cfg.Region,
	)
}
