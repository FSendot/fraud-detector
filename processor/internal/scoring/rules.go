package scoring

import (
	"github.com/fraud-detector/processor/internal/dynamo"
	pb "github.com/fraud-detector/processor/proto"
)

// Weights for each fraud flag.
const (
	WeightAmount      = 30
	WeightCountry     = 25
	WeightVelocity    = 35
	WeightDestination = 10
)

// RulesResult holds the output of the legacy rule-based fraud scoring engine.
type RulesResult struct {
	Score           int
	Decision        string
	FlagAmount      bool
	FlagCountry     bool
	FlagVelocity    bool
	FlagDestination bool
}

// Evaluate runs all fraud rules against the transaction and user profile.
func Evaluate(tx *pb.ProcessTransactionRequest, profile *dynamo.UserProfile) RulesResult {
	var r RulesResult
	transaction := tx.GetTransaction()
	amount := tx.GetFeatures()["amount"]

	r.FlagAmount = flagAmount(amount, profile.AvgAmount, profile.StdDevAmount)
	r.FlagCountry = flagCountry(transaction.GetCountry(), profile.TypicalCountries)
	r.FlagVelocity = flagVelocity(profile.TxLast10Min)
	r.FlagDestination = flagDestination(transaction.GetDestinationAccount(), profile.KnownDestinations)

	if r.FlagAmount {
		r.Score += WeightAmount
	}
	if r.FlagCountry {
		r.Score += WeightCountry
	}
	if r.FlagVelocity {
		r.Score += WeightVelocity
	}
	if r.FlagDestination {
		r.Score += WeightDestination
	}

	switch {
	case r.Score >= 70:
		r.Decision = "blocked"
	case r.Score >= 40:
		r.Decision = "challenged"
	default:
		r.Decision = "allowed"
	}

	return r
}

// amount > avg + 2 * std_dev
func flagAmount(amount, avg, stdDev float64) bool {
	if stdDev == 0 {
		return false
	}
	return amount > avg+2*stdDev
}

// country not in typical_countries
func flagCountry(country string, typical []string) bool {
	for _, c := range typical {
		if c == country {
			return false
		}
	}
	return len(typical) > 0
}

// tx_last_10min >= 3
func flagVelocity(txLast10Min int) bool {
	return txLast10Min >= 3
}

// destination_account not in known_destinations
func flagDestination(dest string, known []string) bool {
	for _, d := range known {
		if d == dest {
			return false
		}
	}
	return len(known) > 0
}
