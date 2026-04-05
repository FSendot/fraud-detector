package runtime

import (
	"encoding/json"
	"fmt"
	"os"
)

type RuntimeSpec struct {
	FormatVersion   int                `json:"format_version"`
	ModelVersion    string             `json:"model_version"`
	BundleManifest  string             `json:"bundle_manifest"`
	FeatureContract FeatureContract    `json:"feature_contract"`
	BranchRuntime   BranchRuntime      `json:"branch_runtime"`
	FusionRuntime   FusionRuntime      `json:"fusion_runtime"`
	Calibration     CalibrationRuntime `json:"calibration_runtime"`
	DecisionRuntime DecisionRuntime    `json:"decision_runtime"`
}

type FeatureContract struct {
	Version            string   `json:"version"`
	TransactionIDField string   `json:"transaction_id_field"`
	LabelField         string   `json:"label_field"`
	FeatureOrder       []string `json:"feature_order"`
}

type BranchRuntime struct {
	BranchName string            `json:"branch_name"`
	Model      HistGBDTModelSpec `json:"model"`
}

type HistGBDTModelSpec struct {
	Type               string     `json:"type"`
	BaselinePrediction float64    `json:"baseline_prediction"`
	NFeatures          int        `json:"n_features"`
	Trees              []TreeSpec `json:"trees"`
}

type TreeSpec struct {
	Nodes []TreeNode `json:"nodes"`
}

type TreeNode struct {
	Value           float64 `json:"value"`
	FeatureIdx      int     `json:"feature_idx"`
	Threshold       float64 `json:"threshold"`
	MissingGoToLeft bool    `json:"missing_go_to_left"`
	Left            int     `json:"left"`
	Right           int     `json:"right"`
	IsLeaf          bool    `json:"is_leaf"`
}

type FusionRuntime struct {
	Mode                string `json:"mode"`
	SelectedBranch      string `json:"selected_branch"`
	RawFusedScoreSource string `json:"raw_fused_score_source"`
}

type CalibrationRuntime struct {
	Type        string    `json:"type"`
	XThresholds []float64 `json:"x_thresholds"`
	YThresholds []float64 `json:"y_thresholds"`
	OutOfBounds string    `json:"out_of_bounds"`
}

type DecisionRuntime struct {
	DecisionThreshold float64        `json:"decision_threshold"`
	BusinessThreshold map[string]any `json:"business_threshold"`
	Policy            PolicyRuntime  `json:"policy"`
}

type PolicyRuntime struct {
	ScoreField                     string         `json:"score_field"`
	ShadowEnabled                  bool           `json:"shadow_enabled"`
	EmitDecisionRecommendationOnly bool           `json:"emit_decision_recommendation_only"`
	Actions                        []PolicyAction `json:"actions"`
	Fallback                       PolicyFallback `json:"fallback"`
}

type PolicyAction struct {
	Name              string   `json:"name"`
	MinScoreInclusive *float64 `json:"min_score_inclusive"`
	MaxScoreExclusive *float64 `json:"max_score_exclusive"`
	Rationale         string   `json:"rationale"`
}

type PolicyFallback struct {
	OnMissingScore     string `json:"on_missing_score"`
	OnContractMismatch string `json:"on_contract_mismatch"`
}

func LoadRuntimeSpec(path string) (*RuntimeSpec, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var spec RuntimeSpec
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, err
	}
	if spec.BranchRuntime.Model.Type != "hist_gradient_boosting_binary" {
		return nil, fmt.Errorf("unsupported model type: %s", spec.BranchRuntime.Model.Type)
	}
	if spec.Calibration.Type != "isotonic_regression" {
		return nil, fmt.Errorf("unsupported calibration type: %s", spec.Calibration.Type)
	}
	return &spec, nil
}
