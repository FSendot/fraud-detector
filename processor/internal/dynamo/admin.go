package dynamo

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
)

const (
	defaultTableName = "user_profiles"
	envTableName     = "DYNAMODB_TABLE_NAME"
)

func TableName() string {
	if tableName := os.Getenv(envTableName); tableName != "" {
		return tableName
	}
	return defaultTableName
}

func EnsureUserProfilesTable(ctx context.Context, db *dynamodb.Client) error {
	tableName := TableName()
	fmt.Printf("checking dynamodb table %s\n", tableName)
	_, err := db.DescribeTable(ctx, &dynamodb.DescribeTableInput{
		TableName: aws.String(tableName),
	})
	if err == nil {
		fmt.Printf("dynamodb table already exists: %s\n", tableName)
		return nil
	}

	var notFound *types.ResourceNotFoundException
	if !errors.As(err, &notFound) {
		return fmt.Errorf("describe table %s: %w", tableName, err)
	}

	fmt.Printf("creating dynamodb table %s\n", tableName)
	_, err = db.CreateTable(ctx, &dynamodb.CreateTableInput{
		TableName: aws.String(tableName),
		AttributeDefinitions: []types.AttributeDefinition{
			{
				AttributeName: aws.String("user_id"),
				AttributeType: types.ScalarAttributeTypeS,
			},
		},
		KeySchema: []types.KeySchemaElement{
			{
				AttributeName: aws.String("user_id"),
				KeyType:       types.KeyTypeHash,
			},
		},
		BillingMode: types.BillingModePayPerRequest,
	})
	if err != nil {
		return fmt.Errorf("create table %s: %w", tableName, err)
	}

	waiter := dynamodb.NewTableExistsWaiter(db)
	fmt.Printf("waiting for dynamodb table %s to become active\n", tableName)
	if err := waiter.Wait(ctx, &dynamodb.DescribeTableInput{
		TableName: aws.String(tableName),
	}, 30*time.Second); err != nil {
		return fmt.Errorf("wait for table %s: %w", tableName, err)
	}

	fmt.Printf("dynamodb table is ready: %s\n", tableName)
	return nil
}
