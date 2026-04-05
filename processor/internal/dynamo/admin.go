package dynamo

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
)

const TableName = "user_profiles"

func EnsureUserProfilesTable(ctx context.Context, db *dynamodb.Client) error {
	fmt.Printf("checking dynamodb table %s\n", TableName)
	_, err := db.DescribeTable(ctx, &dynamodb.DescribeTableInput{
		TableName: aws.String(TableName),
	})
	if err == nil {
		fmt.Printf("dynamodb table already exists: %s\n", TableName)
		return nil
	}

	var notFound *types.ResourceNotFoundException
	if !errors.As(err, &notFound) {
		return fmt.Errorf("describe table %s: %w", TableName, err)
	}

	fmt.Printf("creating dynamodb table %s\n", TableName)
	_, err = db.CreateTable(ctx, &dynamodb.CreateTableInput{
		TableName: aws.String(TableName),
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
		return fmt.Errorf("create table %s: %w", TableName, err)
	}

	waiter := dynamodb.NewTableExistsWaiter(db)
	fmt.Printf("waiting for dynamodb table %s to become active\n", TableName)
	if err := waiter.Wait(ctx, &dynamodb.DescribeTableInput{
		TableName: aws.String(TableName),
	}, 30*time.Second); err != nil {
		return fmt.Errorf("wait for table %s: %w", TableName, err)
	}

	fmt.Printf("dynamodb table is ready: %s\n", TableName)
	return nil
}
