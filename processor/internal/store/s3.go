package store

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type S3Client struct {
	client *s3.Client
	bucket string
}

func NewS3Client(cfg aws.Config) (*S3Client, error) {
	bucket := os.Getenv("S3_AUDIT_BUCKET")
	if bucket == "" {
		return nil, fmt.Errorf("S3_AUDIT_BUCKET not set")
	}
	return &S3Client{
		client: s3.NewFromConfig(cfg),
		bucket: bucket,
	}, nil
}

// PutRawEvent writes the raw enriched event as JSON to S3.
// Key format: transactions/{YYYY}/{MM}/{DD}/{transaction_id}.json
func (c *S3Client) PutRawEvent(ctx context.Context, transactionID string, event any) error {
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}

	now := time.Now().UTC()
	key := fmt.Sprintf("transactions/%d/%02d/%02d/%s.json",
		now.Year(), now.Month(), now.Day(), transactionID)

	_, err = c.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(c.bucket),
		Key:         aws.String(key),
		Body:        bytes.NewReader(data),
		ContentType: aws.String("application/json"),
	})
	return err
}
