package fraudruntime

import (
	"encoding/json"
	"net/http"

	rt "github.com/FSendot/fraud-detector/net/serving/go/internal/runtime"
)

type RuntimeSpec = rt.RuntimeSpec
type FeatureContract = rt.FeatureContract
type BranchRuntime = rt.BranchRuntime
type CalibrationRuntime = rt.CalibrationRuntime
type DecisionRuntime = rt.DecisionRuntime
type PolicyRuntime = rt.PolicyRuntime
type PolicyAction = rt.PolicyAction
type PolicyFallback = rt.PolicyFallback

type ScoreInput = rt.ScoreInput
type ContractCheck = rt.ContractCheck
type BranchOutput = rt.BranchOutput
type ScoreResult = rt.ScoreResult
type ServiceResponse = rt.ServiceResponse

type RequestRecord struct {
	TransactionID string             `json:"transaction_id"`
	IsFraud       *int               `json:"is_fraud"`
	Features      map[string]float64 `json:"features"`
	Metadata      map[string]any     `json:"metadata"`
}

type RequestPayload struct {
	RequestID string          `json:"request_id"`
	Records   []RequestRecord `json:"records"`
}

type Scorer struct {
	inner *rt.Scorer
	spec  *rt.RuntimeSpec
}

func LoadRuntimeSpec(path string) (*RuntimeSpec, error) {
	return rt.LoadRuntimeSpec(path)
}

func NewScorer(spec *RuntimeSpec) *Scorer {
	return &Scorer{
		inner: rt.NewScorer(spec),
		spec:  spec,
	}
}

func NewScorerFromSpecPath(path string) (*Scorer, error) {
	spec, err := rt.LoadRuntimeSpec(path)
	if err != nil {
		return nil, err
	}
	return NewScorer(spec), nil
}

func (s *Scorer) Spec() *RuntimeSpec {
	return s.spec
}

func (s *Scorer) ScoreOne(input ScoreInput) (ScoreResult, error) {
	return s.inner.ScoreOne(input)
}

func (s *Scorer) ScoreMany(requestID string, inputs []ScoreInput) (ServiceResponse, error) {
	return s.inner.ScoreMany(requestID, inputs)
}

func (s *Scorer) ValidateInput(input ScoreInput) ContractCheck {
	return s.inner.ValidateInput(input)
}

func ToScoreInputs(records []RequestRecord) []ScoreInput {
	inputs := make([]ScoreInput, 0, len(records))
	for _, record := range records {
		inputs = append(inputs, ScoreInput{
			TransactionID: record.TransactionID,
			IsFraud:       record.IsFraud,
			Features:      record.Features,
			Metadata:      record.Metadata,
		})
	}
	return inputs
}

func NewShadowHTTPHandler(scorer *Scorer) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(writer http.ResponseWriter, request *http.Request) {
		writeHTTPJSON(writer, http.StatusOK, map[string]any{
			"ok":            true,
			"shadow_mode":   true,
			"model_version": scorer.spec.ModelVersion,
		})
	})
	mux.HandleFunc("/score-shadow", func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost {
			writeHTTPJSON(writer, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
			return
		}
		var payload RequestPayload
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			writeHTTPJSON(writer, http.StatusBadRequest, map[string]any{"shadow_mode": true, "error": err.Error()})
			return
		}
		response, err := scorer.ScoreMany(payload.RequestID, ToScoreInputs(payload.Records))
		if err != nil {
			writeHTTPJSON(writer, http.StatusBadRequest, map[string]any{"shadow_mode": true, "error": err.Error()})
			return
		}
		writeHTTPJSON(writer, http.StatusOK, response)
	})
	return mux
}

func writeHTTPJSON(writer http.ResponseWriter, status int, payload any) {
	body, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_, _ = writer.Write(body)
}
