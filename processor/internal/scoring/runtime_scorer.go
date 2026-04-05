package scoring

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/FSendot/fraud-detector/net/serving/go/pkg/fraudruntime"

	pb "github.com/FSendot/fraud-detector/processor/proto"
)

const runtimeSpecEnvVar = "FRAUD_RUNTIME_SPEC_PATH"

type Engine struct {
	scorer *fraudruntime.Scorer
}

type Result struct {
	Decision       string
	Score          int32
	Calibrated     float64
	ModelVersion   string
	RecommendedAct string
}

func NewEngine(specPath string) (*Engine, error) {
	scorer, err := fraudruntime.NewScorerFromSpecPath(specPath)
	if err != nil {
		return nil, fmt.Errorf("load fraud runtime spec %q: %w", specPath, err)
	}
	return &Engine{scorer: scorer}, nil
}

func ResolveRuntimeSpecPath() (string, error) {
	if envPath := os.Getenv(runtimeSpecEnvVar); envPath != "" {
		if _, err := os.Stat(envPath); err != nil {
			return "", fmt.Errorf("%s points to an unreadable file: %w", runtimeSpecEnvVar, err)
		}
		return envPath, nil
	}

	candidates := []string{
		"../net/outputs/go_runtime/model_v1/runtime_spec.json",
		"net/outputs/go_runtime/model_v1/runtime_spec.json",
	}

	for _, candidate := range candidates {
		absPath, err := filepath.Abs(candidate)
		if err != nil {
			continue
		}
		if _, err := os.Stat(absPath); err == nil {
			return absPath, nil
		}
	}

	return "", fmt.Errorf("runtime spec not found; set %s or place runtime_spec.json in a standard repo path", runtimeSpecEnvVar)
}

func (e *Engine) ScoreTransaction(req *pb.ProcessTransactionRequest, correlationID string) (Result, error) {
	transaction := req.GetTransaction()
	trace := req.GetTrace()

	input := fraudruntime.ScoreInput{
		TransactionID: transaction.GetTransactionId(),
		Features:      e.buildFeatures(req),
		Metadata: map[string]any{
			"correlation_id":      correlationID,
			"request_id":          trace.GetRequestId(),
			"source_system":       trace.GetSourceSystem(),
			"source_component":    trace.GetSourceComponent(),
			"source_region":       trace.GetSourceRegion(),
			"user_id":             transaction.GetUserId(),
			"person_id":           transaction.GetPersonId(),
			"account_id":          transaction.GetAccountId(),
			"currency":            transaction.GetCurrency(),
			"timestamp":           transaction.GetEventTimestamp(),
			"channel":             transaction.GetChannel(),
			"destination_account": transaction.GetDestinationAccount(),
			"country":             transaction.GetCountry(),
			"metadata_labels":     req.GetMetadataLabels(),
		},
	}

	scoreResult, err := e.scorer.ScoreOne(input)
	if err != nil {
		return Result{}, err
	}

	action := e.recommendAction(scoreResult.CalibratedScore)
	return Result{
		Decision:       mapActionToDecision(action),
		Score:          int32(math.Round(scoreResult.CalibratedScore * 100)),
		Calibrated:     scoreResult.CalibratedScore,
		ModelVersion:   scoreResult.ModelVersion,
		RecommendedAct: action,
	}, nil
}

func (e *Engine) buildFeatures(req *pb.ProcessTransactionRequest) map[string]float64 {
	features := make(map[string]float64, len(e.scorer.Spec().FeatureContract.FeatureOrder))
	for _, name := range e.scorer.Spec().FeatureContract.FeatureOrder {
		features[name] = math.NaN()
	}

	for name, value := range req.GetFeatures() {
		features[name] = value
	}

	return features
}

func (e *Engine) recommendAction(score float64) string {
	for _, action := range e.scorer.Spec().DecisionRuntime.Policy.Actions {
		if action.MinScoreInclusive != nil && score < *action.MinScoreInclusive {
			continue
		}
		if action.MaxScoreExclusive != nil && score >= *action.MaxScoreExclusive {
			continue
		}
		return action.Name
	}
	return e.scorer.Spec().DecisionRuntime.Policy.Fallback.OnMissingScore
}

func mapActionToDecision(action string) string {
	switch action {
	case "allow":
		return "allowed"
	case "block":
		return "blocked"
	case "review":
		return "challenged"
	default:
		return "challenged"
	}
}
