package dynamo

import (
	"context"
	"os"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
)

func TestEnsureUserProfilesTableUsesExpectedName(t *testing.T) {
	if TableName != "user_profiles" {
		t.Fatalf("TableName = %q, want %q", TableName, "user_profiles")
	}
}

func TestNewClientWrapsProvidedDynamoClient(t *testing.T) {
	client := NewClient(dynamodb.NewFromConfig(aws.Config{}))
	if client == nil {
		t.Fatal("NewClient() returned nil")
	}
}

func TestLoadAWSConfigUsesDefaultRegionForLocalDynamo(t *testing.T) {
	t.Setenv(envDynamoEndpoint, "http://localhost:8000")
	_ = os.Unsetenv(envAWSRegion)

	cfg, err := LoadAWSConfig(context.Background())
	if err != nil {
		t.Fatalf("LoadAWSConfig() error = %v", err)
	}
	if cfg.Region != defaultRegion {
		t.Fatalf("cfg.Region = %q, want %q", cfg.Region, defaultRegion)
	}
	if cfg.EndpointResolverWithOptions == nil {
		t.Fatal("expected EndpointResolverWithOptions to be configured for local dynamodb")
	}
}
