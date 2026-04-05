package dynamo

import (
	"context"
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
)

const (
	defaultRegion     = "us-east-1"
	envDynamoEndpoint = "DYNAMODB_ENDPOINT"
	envAWSRegion      = "AWS_REGION"
)

func LoadAWSConfig(ctx context.Context) (aws.Config, error) {
	endpoint := os.Getenv(envDynamoEndpoint)
	region := os.Getenv(envAWSRegion)

	loadOptions := []func(*config.LoadOptions) error{}
	if endpoint != "" {
		if region == "" {
			region = defaultRegion
		}
		loadOptions = append(loadOptions,
			config.WithRegion(region),
			config.WithEndpointResolverWithOptions(
				aws.EndpointResolverWithOptionsFunc(func(service, resolvedRegion string, _ ...interface{}) (aws.Endpoint, error) {
					if service != dynamodb.ServiceID {
						return aws.Endpoint{}, &aws.EndpointNotFoundError{}
					}
					return aws.Endpoint{
						URL:               endpoint,
						HostnameImmutable: true,
						SigningRegion:     region,
					}, nil
				}),
			),
		)
	}

	cfg, err := config.LoadDefaultConfig(ctx, loadOptions...)
	if err != nil {
		return aws.Config{}, fmt.Errorf("load aws config: %w", err)
	}
	return cfg, nil
}
