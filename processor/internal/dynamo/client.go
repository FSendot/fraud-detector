package dynamo

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
)

type Client struct {
	db *dynamodb.Client
}

func NewClient(db *dynamodb.Client) *Client {
	return &Client{db: db}
}

// GetProfile fetches a user profile by user_id.
// Returns a default empty profile if the user doesn't exist yet.
func (c *Client) GetProfile(ctx context.Context, userID string) (*UserProfile, error) {
	out, err := c.db.GetItem(ctx, &dynamodb.GetItemInput{
		TableName: aws.String(TableName),
		Key: map[string]types.AttributeValue{
			"user_id": &types.AttributeValueMemberS{Value: userID},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("dynamo GetItem: %w", err)
	}

	if out.Item == nil {
		return NewDefaultProfile(userID), nil
	}

	var profile UserProfile
	if err := attributevalue.UnmarshalMap(out.Item, &profile); err != nil {
		return nil, fmt.Errorf("dynamo unmarshal: %w", err)
	}
	return &profile, nil
}

// UpdateProfile updates the user profile with the new transaction data
// using Welford's online algorithm for incremental mean/stddev.
func (c *Client) UpdateProfile(ctx context.Context, profile *UserProfile, amount float64, country, channel, destination, timestamp string) error {
	// Welford's online algorithm for running mean and standard deviation
	newCount := profile.TxCount + 1
	delta := amount - profile.AvgAmount
	newAvg := profile.AvgAmount + delta/float64(newCount)
	delta2 := amount - newAvg

	var newStdDev float64
	if newCount < 2 {
		newStdDev = 0
	} else {
		// Incremental variance: M2(n) = M2(n-1) + delta * delta2
		// We reconstruct M2 from the old stddev, then update.
		oldM2 := profile.StdDevAmount * profile.StdDevAmount * float64(profile.TxCount)
		newM2 := oldM2 + delta*delta2
		newStdDev = math.Sqrt(newM2 / float64(newCount))
	}

	// Add country/channel/destination to known sets if new
	countries := addUnique(profile.TypicalCountries, country)
	channels := addUnique(profile.TypicalChannels, channel)
	destinations := addUnique(profile.KnownDestinations, destination)

	// Velocity: count transactions in last 10 min and last hour
	txLast10Min, txLastHour := computeVelocity(profile, timestamp)

	updated := &UserProfile{
		UserID:            profile.UserID,
		AvgAmount:         newAvg,
		StdDevAmount:      newStdDev,
		TxCount:           newCount,
		TxLastHour:        txLastHour,
		TxLast10Min:       txLast10Min,
		TypicalCountries:  countries,
		TypicalChannels:   channels,
		KnownDestinations: destinations,
		LastCountry:       country,
		LastTimestamp:     timestamp,
	}

	item, err := attributevalue.MarshalMap(updated)
	if err != nil {
		return fmt.Errorf("dynamo marshal: %w", err)
	}

	_, err = c.db.PutItem(ctx, &dynamodb.PutItemInput{
		TableName: aws.String(TableName),
		Item:      item,
	})
	if err != nil {
		return fmt.Errorf("dynamo PutItem: %w", err)
	}
	return nil
}

// computeVelocity recalculates tx_last_10min and tx_last_hour based on
// the time difference between the current and last transaction.
// This is a simplified approach — resets counters if enough time has passed.
func computeVelocity(profile *UserProfile, currentTimestamp string) (last10min int, lastHour int) {
	now, err := time.Parse(time.RFC3339, currentTimestamp)
	if err != nil {
		// If we can't parse, increment conservatively
		return profile.TxLast10Min + 1, profile.TxLastHour + 1
	}

	last, err := time.Parse(time.RFC3339, profile.LastTimestamp)
	if err != nil {
		// First transaction or unparseable — start fresh
		return 1, 1
	}

	diff := now.Sub(last)

	switch {
	case diff > time.Hour:
		return 1, 1
	case diff > 10*time.Minute:
		return 1, profile.TxLastHour + 1
	default:
		return profile.TxLast10Min + 1, profile.TxLastHour + 1
	}
}

func addUnique(slice []string, val string) []string {
	for _, v := range slice {
		if v == val {
			return slice
		}
	}
	return append(slice, val)
}
