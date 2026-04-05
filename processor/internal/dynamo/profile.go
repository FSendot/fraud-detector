package dynamo

// UserProfile represents the current statistical profile of a user in DynamoDB.
// Only aggregated state — never full transaction history.
type UserProfile struct {
	UserID            string   `dynamodbav:"user_id"`
	AvgAmount         float64  `dynamodbav:"avg_amount"`
	StdDevAmount      float64  `dynamodbav:"std_dev_amount"`
	TxCount           int64    `dynamodbav:"tx_count"`
	TxLastHour        int      `dynamodbav:"tx_last_hour"`
	TxLast10Min       int      `dynamodbav:"tx_last_10min"`
	TypicalCountries  []string `dynamodbav:"typical_countries"`
	TypicalChannels   []string `dynamodbav:"typical_channels"`
	KnownDestinations []string `dynamodbav:"known_destinations"`
	LastCountry       string   `dynamodbav:"last_country"`
	LastTimestamp     string   `dynamodbav:"last_timestamp"`
}

// NewDefaultProfile returns a blank profile for first-time users.
func NewDefaultProfile(userID string) *UserProfile {
	return &UserProfile{
		UserID:            userID,
		TypicalCountries:  []string{},
		TypicalChannels:   []string{},
		KnownDestinations: []string{},
	}
}
