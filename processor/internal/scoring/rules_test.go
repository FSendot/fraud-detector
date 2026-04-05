package scoring

import (
	"testing"

	"github.com/FSendot/fraud-detector/processor/internal/dynamo"
	pb "github.com/FSendot/fraud-detector/processor/proto"
)

func baseProfile() *dynamo.UserProfile {
	return &dynamo.UserProfile{
		UserID:            "u_123",
		AvgAmount:         4500,
		StdDevAmount:      1200,
		TxCount:           142,
		TxLastHour:        2,
		TxLast10Min:       0,
		TypicalCountries:  []string{"AR", "UY"},
		TypicalChannels:   []string{"web", "mobile"},
		KnownDestinations: []string{"acc_456", "acc_789"},
		LastCountry:       "AR",
		LastTimestamp:     "2026-04-03T10:22:00Z",
	}
}

func TestAllowNormalTransaction(t *testing.T) {
	tx := requestWithRulesInputs(5000, "AR", "acc_456")
	r := Evaluate(tx, baseProfile())

	if r.Decision != "allowed" {
		t.Errorf("expected allowed, got %s (score=%d)", r.Decision, r.Score)
	}
	if r.FlagAmount || r.FlagCountry || r.FlagDestination || r.FlagVelocity {
		t.Errorf("no flags expected, got amount=%v country=%v dest=%v velocity=%v",
			r.FlagAmount, r.FlagCountry, r.FlagDestination, r.FlagVelocity)
	}
}

func TestBlockHighRisk(t *testing.T) {
	tx := requestWithRulesInputs(15000, "BR", "acc_999") // > 4500 + 2*1200 = 6900
	r := Evaluate(tx, baseProfile())

	if !r.FlagAmount {
		t.Error("expected flag_amount=true")
	}
	if !r.FlagCountry {
		t.Error("expected flag_country=true")
	}
	if !r.FlagDestination {
		t.Error("expected flag_destination=true")
	}
	// 30 + 25 + 10 = 65 -> challenged
	if r.Decision != "challenged" {
		t.Errorf("expected challenged, got %s (score=%d)", r.Decision, r.Score)
	}
}

func TestBlockWithVelocity(t *testing.T) {
	profile := baseProfile()
	profile.TxLast10Min = 3

	tx := requestWithRulesInputs(15000, "BR", "acc_999")
	r := Evaluate(tx, profile)

	// 30 + 25 + 35 + 10 = 100
	if r.Score != 100 {
		t.Errorf("expected score=100, got %d", r.Score)
	}
	if r.Decision != "blocked" {
		t.Errorf("expected blocked, got %s", r.Decision)
	}
}

func TestChallengeMiddleScore(t *testing.T) {
	tx := requestWithRulesInputs(15000, "AR", "acc_999")
	r := Evaluate(tx, baseProfile())

	// 30 + 10 = 40 -> challenged
	if r.Score != 40 {
		t.Errorf("expected score=40, got %d", r.Score)
	}
	if r.Decision != "challenged" {
		t.Errorf("expected challenged, got %s", r.Decision)
	}
}

func TestNewUserNoFlags(t *testing.T) {
	profile := dynamo.NewDefaultProfile("u_new")
	tx := requestWithRulesInputs(99999, "JP", "acc_unknown")
	r := Evaluate(tx, profile)

	// New user: stddev=0 so no amount flag, empty lists so no country/dest flags
	if r.Decision != "allowed" {
		t.Errorf("new user should be allowed, got %s (score=%d)", r.Decision, r.Score)
	}
}

func requestWithRulesInputs(amount float64, country, destination string) *pb.ProcessTransactionRequest {
	return &pb.ProcessTransactionRequest{
		Transaction: &pb.TransactionContext{
			TransactionId:      "tx_rules",
			UserId:             "u_123",
			DestinationAccount: destination,
			Country:            country,
		},
		Features: map[string]float64{
			"amount": amount,
		},
	}
}
