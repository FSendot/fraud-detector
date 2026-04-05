package runtime

import (
	"fmt"
	"math"
	"sort"
)

type ScoreInput struct {
	TransactionID string
	IsFraud       *int
	Features      map[string]float64
	Metadata      map[string]any
}

type ContractCheck struct {
	Valid          bool     `json:"valid"`
	Errors         []string `json:"errors"`
	Warnings       []string `json:"warnings"`
	RebuildApplied bool     `json:"rebuild_applied"`
	SourceStage    string   `json:"source_stage"`
}

type BranchOutput struct {
	Score              float64 `json:"score"`
	PredictedLabelAt05 int     `json:"predicted_label_at_0_5"`
}

type ScoreResult struct {
	TransactionID     string                  `json:"transaction_id"`
	IsFraud           *int                    `json:"is_fraud"`
	ModelVersion      string                  `json:"model_version"`
	ShadowMode        bool                    `json:"shadow_mode"`
	BranchOutputs     map[string]BranchOutput `json:"branch_outputs"`
	RawFusedScore     float64                 `json:"raw_fused_score"`
	CalibratedScore   float64                 `json:"calibrated_score"`
	DecisionThreshold float64                 `json:"decision_threshold"`
	PredictedLabel    int                     `json:"predicted_label"`
	Metadata          map[string]any          `json:"metadata"`
}

type ServiceResponse struct {
	RequestID       string         `json:"request_id,omitempty"`
	ShadowMode      bool           `json:"shadow_mode"`
	Mode            string         `json:"mode"`
	ModelVersion    string         `json:"model_version"`
	BundleManifest  string         `json:"bundle_manifest"`
	RuntimeDefaults map[string]any `json:"runtime_defaults"`
	ContractCheck   ContractCheck  `json:"contract_check"`
	Records         []ScoreResult  `json:"records"`
}

type Scorer struct {
	Spec *RuntimeSpec
}

func NewScorer(spec *RuntimeSpec) *Scorer {
	return &Scorer{Spec: spec}
}

func (s *Scorer) ValidateInput(input ScoreInput) ContractCheck {
	errors := []string{}
	for _, name := range s.Spec.FeatureContract.FeatureOrder {
		if _, ok := input.Features[name]; !ok {
			errors = append(errors, fmt.Sprintf("missing required feature: %s", name))
		}
	}
	return ContractCheck{
		Valid:          len(errors) == 0,
		Errors:         errors,
		Warnings:       []string{},
		RebuildApplied: false,
		SourceStage:    "contract_aligned_input",
	}
}

func (s *Scorer) ScoreOne(input ScoreInput) (ScoreResult, error) {
	features := make([]float64, len(s.Spec.FeatureContract.FeatureOrder))
	for idx, name := range s.Spec.FeatureContract.FeatureOrder {
		value, ok := input.Features[name]
		if !ok {
			return ScoreResult{}, fmt.Errorf("missing required feature: %s", name)
		}
		features[idx] = value
	}

	branchScore := s.scoreBoostedBranch(features)
	rawFusedScore := branchScore
	calibratedScore := isotonicPredict(rawFusedScore, s.Spec.Calibration)
	predictedLabel := 0
	if calibratedScore >= s.Spec.DecisionRuntime.DecisionThreshold {
		predictedLabel = 1
	}

	result := ScoreResult{
		TransactionID:     input.TransactionID,
		IsFraud:           input.IsFraud,
		ModelVersion:      s.Spec.ModelVersion,
		ShadowMode:        true,
		BranchOutputs:     map[string]BranchOutput{s.Spec.BranchRuntime.BranchName: {Score: branchScore, PredictedLabelAt05: boolToInt(branchScore >= 0.5)}},
		RawFusedScore:     rawFusedScore,
		CalibratedScore:   calibratedScore,
		DecisionThreshold: s.Spec.DecisionRuntime.DecisionThreshold,
		PredictedLabel:    predictedLabel,
		Metadata:          input.Metadata,
	}
	return result, nil
}

func (s *Scorer) ScoreMany(requestID string, inputs []ScoreInput) (ServiceResponse, error) {
	combinedErrors := []string{}
	for _, input := range inputs {
		check := s.ValidateInput(input)
		if !check.Valid {
			combinedErrors = append(combinedErrors, check.Errors...)
		}
	}
	if len(combinedErrors) > 0 {
		return ServiceResponse{
			RequestID:      requestID,
			ShadowMode:     true,
			Mode:           "shadow_only",
			ModelVersion:   s.Spec.ModelVersion,
			BundleManifest: s.Spec.BundleManifest,
			RuntimeDefaults: map[string]any{
				"decision_threshold": s.Spec.DecisionRuntime.DecisionThreshold,
			},
			ContractCheck: ContractCheck{
				Valid:          false,
				Errors:         dedupeStrings(combinedErrors),
				Warnings:       []string{},
				RebuildApplied: false,
				SourceStage:    "contract_aligned_input",
			},
			Records: []ScoreResult{},
		}, nil
	}

	results := make([]ScoreResult, 0, len(inputs))
	for _, input := range inputs {
		result, err := s.ScoreOne(input)
		if err != nil {
			return ServiceResponse{}, err
		}
		results = append(results, result)
	}

	return ServiceResponse{
		RequestID:      requestID,
		ShadowMode:     true,
		Mode:           "shadow_only",
		ModelVersion:   s.Spec.ModelVersion,
		BundleManifest: s.Spec.BundleManifest,
		RuntimeDefaults: map[string]any{
			"decision_threshold": s.Spec.DecisionRuntime.DecisionThreshold,
			"policy":             s.Spec.DecisionRuntime.Policy,
		},
		ContractCheck: ContractCheck{
			Valid:          true,
			Errors:         []string{},
			Warnings:       []string{},
			RebuildApplied: false,
			SourceStage:    "contract_aligned_input",
		},
		Records: results,
	}, nil
}

func (s *Scorer) scoreBoostedBranch(features []float64) float64 {
	raw := s.Spec.BranchRuntime.Model.BaselinePrediction
	for _, tree := range s.Spec.BranchRuntime.Model.Trees {
		raw += scoreTree(tree, features)
	}
	return logistic(raw)
}

func scoreTree(tree TreeSpec, features []float64) float64 {
	index := 0
	for {
		node := tree.Nodes[index]
		if node.IsLeaf {
			return node.Value
		}
		value := features[node.FeatureIdx]
		goLeft := false
		if math.IsNaN(value) {
			goLeft = node.MissingGoToLeft
		} else {
			goLeft = value <= node.Threshold
		}
		if goLeft {
			index = node.Left
		} else {
			index = node.Right
		}
	}
}

func isotonicPredict(score float64, calibration CalibrationRuntime) float64 {
	xs := calibration.XThresholds
	ys := calibration.YThresholds
	if len(xs) == 0 || len(xs) != len(ys) {
		return score
	}
	if score <= xs[0] {
		return ys[0]
	}
	last := len(xs) - 1
	if score >= xs[last] {
		return ys[last]
	}
	index := sort.Search(len(xs), func(i int) bool { return xs[i] >= score })
	if index == 0 {
		return ys[0]
	}
	x0 := xs[index-1]
	x1 := xs[index]
	y0 := ys[index-1]
	y1 := ys[index]
	if x1 == x0 {
		return y1
	}
	weight := (score - x0) / (x1 - x0)
	return y0 + weight*(y1-y0)
}

func logistic(value float64) float64 {
	if value >= 0 {
		exp := math.Exp(-value)
		return 1.0 / (1.0 + exp)
	}
	exp := math.Exp(value)
	return exp / (1.0 + exp)
}

func boolToInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func dedupeStrings(values []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}
