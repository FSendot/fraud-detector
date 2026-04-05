package scoring

import (
	"testing"

	pb "github.com/fraud-detector/processor/proto"
)

func TestMapActionToDecision(t *testing.T) {
	testCases := map[string]string{
		"allow":   "allowed",
		"review":  "challenged",
		"block":   "blocked",
		"unknown": "challenged",
	}

	for action, want := range testCases {
		if got := mapActionToDecision(action); got != want {
			t.Fatalf("mapActionToDecision(%q) = %q, want %q", action, got, want)
		}
	}
}

func TestBuildFeaturesSetsKnownFieldsAndLeavesOthersMissing(t *testing.T) {
	engine, err := NewEngine("../../../net/outputs/go_runtime/model_v1/runtime_spec.json")
	if err != nil {
		t.Fatalf("NewEngine() error = %v", err)
	}

	features := engine.buildFeatures(&pb.ProcessTransactionRequest{
		Features: map[string]float64{
			"amount": 42.5,
		},
	})
	if got := features["amount"]; got != 42.5 {
		t.Fatalf("amount feature = %v, want 42.5", got)
	}

	if value, ok := features["card1"]; !ok {
		t.Fatalf("expected missing-backed feature to be present")
	} else if value == value {
		t.Fatalf("card1 feature = %v, want NaN placeholder", value)
	}
}
