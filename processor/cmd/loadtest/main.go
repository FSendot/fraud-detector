package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/FSendot/fraud-detector/net/serving/go/pkg/fraudruntime"
	"github.com/FSendot/fraud-detector/processor/internal/scoring"
	pb "github.com/FSendot/fraud-detector/processor/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type scenarioConfig struct {
	Name               string
	Weight             int
	UserMode           string
	ExpectedDecision   string
	AmountMin          float64
	AmountMax          float64
	TimeAdvanceMin     time.Duration
	TimeAdvanceMax     time.Duration
	CountryShift       bool
	DestinationShift   bool
	DeviceShift        bool
	IdentityShift      bool
	MerchantFanout     bool
	NightActivity      bool
	Currency           string
	Channel            string
	RiskProfile        string
	MissingFeatureRate float64
}

type userState struct {
	UserID             string
	PersonID           string
	AccountID          string
	HomeCountry        string
	AltCountry         string
	PrimaryDest        string
	SecondaryDest      string
	FavoriteMerchants  []string
	Card1              float64
	Card2              float64
	Card3              float64
	Card5              float64
	Addr1              float64
	Addr2              float64
	Dist1              float64
	Dist2              float64
	BaselineAmount     float64
	AmountStd          float64
	TrustScore         float64
	IdentityConfidence float64
	VelocityTolerance  float64
	DeviceStability    float64
	MerchantDiversity  float64
	LastEventTime      time.Time
	History            []txnHistory
}

type txnHistory struct {
	Amount      float64
	Country     string
	Destination string
	Merchant    string
	Channel     string
	Currency    string
	EventTime   time.Time
}

type featureContext struct {
	Scenario                scenarioConfig
	User                    *userState
	RequestIndex            int
	Amount                  float64
	EventTime               time.Time
	Country                 string
	Destination             string
	Merchant                string
	Currency                string
	Channel                 string
	PreviousAmount          float64
	SecondsSincePrevious    float64
	Prior5Count             float64
	Prior10Count            float64
	Prior5AmountSum         float64
	Prior5AmountMean        float64
	Prior5AmountStd         float64
	Prior10AmountSum        float64
	Prior10AmountMean       float64
	Prior10AmountStd        float64
	Prior5UniqueDestCount   float64
	Prior10UniqueDestCount  float64
	AmountDelta             float64
	AmountRatio             float64
	VelocityScore           float64
	GeoScore                float64
	DestinationNovelty      float64
	DeviceRisk              float64
	IdentityRisk            float64
	BehaviorRisk            float64
	NewUser                 bool
	IsCountryShift          bool
	IsDestinationShift      bool
	IsMerchantFanout        bool
	IsNightActivity         bool
	RecentBurstCount        int
	FeatureCoverageEstimate int
}

type requestPlan struct {
	Index        int
	Scenario     string
	UserMode     string
	Amount       float64
	FeatureLevel string
	Request      *pb.ProcessTransactionRequest
}

type requestRecord struct {
	Index          int                `json:"index"`
	Scenario       string             `json:"scenario"`
	UserMode       string             `json:"user_mode"`
	RequestID      string             `json:"request_id"`
	TransactionID  string             `json:"transaction_id"`
	UserID         string             `json:"user_id"`
	EventTimestamp string             `json:"event_timestamp"`
	Amount         float64            `json:"amount"`
	Country        string             `json:"country"`
	Channel        string             `json:"channel"`
	Destination    string             `json:"destination_account"`
	FeatureLevel   string             `json:"feature_level"`
	MetadataLabels map[string]string  `json:"metadata_labels"`
	Features       map[string]float64 `json:"features"`
}

type requestResult struct {
	Index           int               `json:"index"`
	Scenario        string            `json:"scenario"`
	UserMode        string            `json:"user_mode"`
	RequestID       string            `json:"request_id"`
	TransactionID   string            `json:"transaction_id"`
	UserID          string            `json:"user_id"`
	Decision        string            `json:"decision,omitempty"`
	Score           int32             `json:"score,omitempty"`
	ModelVersion    string            `json:"model_version,omitempty"`
	CalibratedScore float64           `json:"calibrated_score,omitempty"`
	ClientRTTMs     float64           `json:"client_rtt_ms"`
	ServerTotalMs   float64           `json:"server_total_ms,omitempty"`
	ProfileLookupMs float64           `json:"profile_lookup_ms,omitempty"`
	ScoringMs       float64           `json:"scoring_ms,omitempty"`
	ProfileUpdateMs float64           `json:"profile_update_ms,omitempty"`
	Error           string            `json:"error,omitempty"`
	MetadataLabels  map[string]string `json:"metadata_labels,omitempty"`
}

type durationSummary struct {
	Count int     `json:"count"`
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
	Mean  float64 `json:"mean"`
	P50   float64 `json:"p50"`
	P95   float64 `json:"p95"`
	P99   float64 `json:"p99"`
}

type scenarioAuditSummary struct {
	ExpectedDecision string          `json:"expected_decision"`
	RequestCount     int             `json:"request_count"`
	MatchedCount     int             `json:"matched_count"`
	MatchedRate      float64         `json:"matched_rate"`
	DecisionCounts   map[string]int  `json:"decision_counts"`
	CalibratedScore  durationSummary `json:"calibrated_score"`
}

type loadtestSummary struct {
	RunID               string                          `json:"run_id"`
	StartedAt           string                          `json:"started_at"`
	CompletedAt         string                          `json:"completed_at"`
	ServerAddress       string                          `json:"server_address"`
	Requests            int                             `json:"requests"`
	Concurrency         int                             `json:"concurrency"`
	StableUsers         int                             `json:"stable_users"`
	Seed                int64                           `json:"seed"`
	ModelVersion        string                          `json:"model_version,omitempty"`
	Errors              int                             `json:"errors"`
	DecisionCounts      map[string]int                  `json:"decision_counts"`
	ScenarioCounts      map[string]int                  `json:"scenario_counts"`
	DecisionByScenario  map[string]map[string]int       `json:"decision_by_scenario"`
	ScenarioAudit       map[string]scenarioAuditSummary `json:"scenario_audit"`
	ClientRTT           durationSummary                 `json:"client_rtt_ms"`
	ServerTotal         durationSummary                 `json:"server_total_ms"`
	ProfileLookup       durationSummary                 `json:"profile_lookup_ms"`
	Scoring             durationSummary                 `json:"scoring_ms"`
	ProfileUpdate       durationSummary                 `json:"profile_update_ms"`
	CalibratedScore     durationSummary                 `json:"calibrated_score"`
	RequestFeatureCount durationSummary                 `json:"request_feature_count"`
}

type generator struct {
	rng          *rand.Rand
	runID        string
	baseTime     time.Time
	featureOrder []string
	stableUsers  []*userState
	hotUsers     []*userState
	newUserSeq   int
	scenarios    []scenarioConfig
	totalWeight  int
}

func main() {
	addr := flag.String("addr", defaultServerAddress(), "gRPC server address")
	requests := flag.Int("requests", 2500, "number of mock requests to send")
	concurrency := flag.Int("concurrency", 40, "number of concurrent workers")
	stableUsers := flag.Int("stable-users", 200, "number of recurring users reused across runs")
	timeout := flag.Duration("timeout", 10*time.Second, "per-request timeout")
	userPrefix := flag.String("user-prefix", "loadtest", "stable user prefix so runs can evolve local dynamodb state over time")
	outputDir := flag.String("output-dir", "", "optional output directory; defaults to processor/output/loadtest/<run_id>")
	sourceSystem := flag.String("source-system", "loadtest", "request trace source system")
	sourceComponent := flag.String("source-component", "load-generator", "request trace source component")
	sourceRegion := flag.String("source-region", "local", "request trace source region")
	featureSpecPath := flag.String("feature-spec", "", "runtime spec path used to drive dynamic feature generation")
	seed := flag.Int64("seed", 0, "random seed; defaults to current unix nano time")
	flag.Parse()

	if *requests <= 0 {
		log.Fatal("-requests must be greater than 0")
	}
	if *concurrency <= 0 {
		log.Fatal("-concurrency must be greater than 0")
	}
	if *stableUsers <= 0 {
		log.Fatal("-stable-users must be greater than 0")
	}

	resolvedFeatureSpecPath := *featureSpecPath
	if resolvedFeatureSpecPath == "" {
		path, err := scoring.ResolveRuntimeSpecPath()
		if err != nil {
			log.Fatalf("resolve runtime spec path for load test feature generation: %v", err)
		}
		resolvedFeatureSpecPath = path
	}

	spec, err := fraudruntime.LoadRuntimeSpec(resolvedFeatureSpecPath)
	if err != nil {
		log.Fatalf("load runtime spec %q: %v", resolvedFeatureSpecPath, err)
	}
	if len(spec.FeatureContract.FeatureOrder) == 0 {
		log.Fatalf("runtime spec %q has an empty feature contract", resolvedFeatureSpecPath)
	}

	runStartedAt := time.Now().UTC()
	resolvedSeed := *seed
	if resolvedSeed == 0 {
		resolvedSeed = runStartedAt.UnixNano()
	}

	runID := runStartedAt.Format("20060102T150405Z")
	baseOutputDir := *outputDir
	if baseOutputDir == "" {
		baseOutputDir = filepath.Join("output", "loadtest")
	}
	resolvedOutputDir := filepath.Join(baseOutputDir, runID)

	gen := newGenerator(resolvedSeed, runID, *stableUsers, *userPrefix, runStartedAt, spec.FeatureContract.FeatureOrder)
	plans := make([]requestPlan, 0, *requests)
	for index := 0; index < *requests; index++ {
		plans = append(plans, gen.Next(index, *sourceSystem, *sourceComponent, *sourceRegion))
	}

	if err := os.MkdirAll(resolvedOutputDir, 0o755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	requestDatasetPath := filepath.Join(resolvedOutputDir, "requests.jsonl")
	if err := writeJSONL(requestDatasetPath, requestRecordsFromPlans(plans)); err != nil {
		log.Fatalf("write request dataset: %v", err)
	}

	conn, err := grpc.Dial(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("connect to %s: %v", *addr, err)
	}
	defer conn.Close()

	client := pb.NewFraudProcessorClient(conn)
	results := make([]requestResult, len(plans))

	jobs := make(chan int)
	var wg sync.WaitGroup
	workerCount := min(*concurrency, len(plans))
	for workerIndex := 0; workerIndex < workerCount; workerIndex++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for planIndex := range jobs {
				plan := plans[planIndex]
				requestStartedAt := time.Now()
				ctx, cancel := context.WithTimeout(context.Background(), *timeout)
				response, err := client.ProcessTransaction(ctx, plan.Request)
				cancel()

				result := requestResult{
					Index:          plan.Index,
					Scenario:       plan.Scenario,
					UserMode:       plan.UserMode,
					RequestID:      plan.Request.GetTrace().GetRequestId(),
					TransactionID:  plan.Request.GetTransaction().GetTransactionId(),
					UserID:         plan.Request.GetTransaction().GetUserId(),
					ClientRTTMs:    durationMillis(time.Since(requestStartedAt)),
					MetadataLabels: plan.Request.GetMetadataLabels(),
				}

				if err != nil {
					result.Error = err.Error()
				} else {
					result.Decision = response.GetDecision()
					result.Score = response.GetScore()
					result.ModelVersion = response.GetModelVersion()
					result.CalibratedScore = response.GetCalibratedScore()
					result.ServerTotalMs = response.GetTotalDurationMs()
					result.ProfileLookupMs = response.GetProfileLookupDurationMs()
					result.ScoringMs = response.GetScoringDurationMs()
					result.ProfileUpdateMs = response.GetProfileUpdateDurationMs()
				}

				results[planIndex] = result
			}
		}()
	}

	for planIndex := range plans {
		jobs <- planIndex
	}
	close(jobs)
	wg.Wait()

	resultsPath := filepath.Join(resolvedOutputDir, "results.jsonl")
	if err := writeJSONL(resultsPath, results); err != nil {
		log.Fatalf("write results: %v", err)
	}

	summary := summarizeResults(runID, runStartedAt, time.Now().UTC(), *addr, *requests, *concurrency, *stableUsers, resolvedSeed, spec.ModelVersion, plans, results)
	summaryPath := filepath.Join(resolvedOutputDir, "summary.json")
	if err := writeJSON(summaryPath, summary); err != nil {
		log.Fatalf("write summary: %v", err)
	}

	fmt.Printf("loadtest_run=%s output_dir=%s requests=%d concurrency=%d errors=%d\n",
		runID, resolvedOutputDir, summary.Requests, summary.Concurrency, summary.Errors)
	fmt.Printf("model_version=%s feature_count=%d\n", spec.ModelVersion, len(spec.FeatureContract.FeatureOrder))
	fmt.Printf("client_rtt_ms mean=%.2f p95=%.2f p99=%.2f\n", summary.ClientRTT.Mean, summary.ClientRTT.P95, summary.ClientRTT.P99)
	fmt.Printf("server_total_ms mean=%.2f p95=%.2f p99=%.2f\n", summary.ServerTotal.Mean, summary.ServerTotal.P95, summary.ServerTotal.P99)
	fmt.Printf("score mean=%.4f p95=%.4f p99=%.4f\n", summary.CalibratedScore.Mean, summary.CalibratedScore.P95, summary.CalibratedScore.P99)
	fmt.Printf("request_feature_count mean=%.1f p95=%.1f p99=%.1f\n", summary.RequestFeatureCount.Mean, summary.RequestFeatureCount.P95, summary.RequestFeatureCount.P99)
	fmt.Printf("decisions=%s\n", mustJSON(summary.DecisionCounts))
	fmt.Printf("scenarios=%s\n", mustJSON(summary.ScenarioCounts))
	fmt.Printf("decision_by_scenario=%s\n", mustJSON(summary.DecisionByScenario))
	fmt.Printf("scenario_audit=%s\n", mustJSON(summary.ScenarioAudit))
	fmt.Printf("requests_dataset=%s\n", requestDatasetPath)
	fmt.Printf("results_dataset=%s\n", resultsPath)
	fmt.Printf("summary_json=%s\n", summaryPath)
}

func defaultServerAddress() string {
	if envAddr := os.Getenv("GRPC_ADDRESS"); envAddr != "" {
		return envAddr
	}

	host := os.Getenv("GRPC_HOST")
	if host == "" {
		host = "127.0.0.1"
	}

	port := os.Getenv("GRPC_PORT")
	if port == "" {
		port = "50051"
	}

	return fmt.Sprintf("%s:%s", host, port)
}

func newGenerator(seed int64, runID string, stableUserCount int, userPrefix string, baseTime time.Time, featureOrder []string) *generator {
	rng := rand.New(rand.NewSource(seed))
	stableUsers := make([]*userState, 0, stableUserCount)
	hotCount := min(12, stableUserCount)
	for index := 0; index < stableUserCount; index++ {
		homeCountry, altCountry := countryPair(index)
		baselineAmount := 60 + float64(index%20)*35 + rng.Float64()*120
		amountStd := 20 + rng.Float64()*60
		stableUsers = append(stableUsers, &userState{
			UserID:             fmt.Sprintf("%s_user_%04d", userPrefix, index),
			PersonID:           fmt.Sprintf("%s_person_%04d", userPrefix, index),
			AccountID:          fmt.Sprintf("%s_account_%04d", userPrefix, index),
			HomeCountry:        homeCountry,
			AltCountry:         altCountry,
			PrimaryDest:        fmt.Sprintf("dest_%04d_primary", index),
			SecondaryDest:      fmt.Sprintf("dest_%04d_secondary", index),
			FavoriteMerchants:  favoriteMerchants(index),
			Card1:              1000 + float64(index%500),
			Card2:              100 + float64(index%80),
			Card3:              150 + float64(index%25),
			Card5:              220 + float64(index%12),
			Addr1:              200 + float64(index%60),
			Addr2:              50 + float64(index%20),
			Dist1:              float64(index % 7),
			Dist2:              float64(index % 5),
			BaselineAmount:     baselineAmount,
			AmountStd:          amountStd,
			TrustScore:         0.45 + rng.Float64()*0.5,
			IdentityConfidence: 0.50 + rng.Float64()*0.45,
			VelocityTolerance:  0.35 + rng.Float64()*0.60,
			DeviceStability:    0.35 + rng.Float64()*0.60,
			MerchantDiversity:  0.20 + rng.Float64()*0.65,
			LastEventTime:      baseTime.Add(-time.Duration(stableUserCount-index) * 20 * time.Minute),
			History:            bootstrapHistory(rng, baseTime, baselineAmount, amountStd, homeCountry, fmt.Sprintf("dest_%04d_primary", index), favoriteMerchants(index), 6+(index%7)),
		})
	}

	scenarios := []scenarioConfig{
		{Name: "returning_habitual_low_value", Weight: 15, UserMode: "stable", ExpectedDecision: "allowed", AmountMin: 15, AmountMax: 220, TimeAdvanceMin: 9 * time.Minute, TimeAdvanceMax: 40 * time.Minute, Currency: "ARS", Channel: "web", RiskProfile: "trusted", MissingFeatureRate: 0.02},
		{Name: "returning_payday_like", Weight: 7, UserMode: "stable", ExpectedDecision: "allowed", AmountMin: 400, AmountMax: 1800, TimeAdvanceMin: 6 * time.Hour, TimeAdvanceMax: 36 * time.Hour, Currency: "ARS", Channel: "api", RiskProfile: "trusted", MissingFeatureRate: 0.02},
		{Name: "trusted_large_known_beneficiary", Weight: 5, UserMode: "stable", ExpectedDecision: "allowed", AmountMin: 1800, AmountMax: 7200, TimeAdvanceMin: 18 * time.Hour, TimeAdvanceMax: 96 * time.Hour, Currency: "ARS", Channel: "web", RiskProfile: "trusted", MissingFeatureRate: 0.02},
		{Name: "salary_day_high_value", Weight: 4, UserMode: "stable", ExpectedDecision: "allowed", AmountMin: 5000, AmountMax: 12000, TimeAdvanceMin: 20 * 24 * time.Hour, TimeAdvanceMax: 35 * 24 * time.Hour, Currency: "ARS", Channel: "api", RiskProfile: "trusted", MissingFeatureRate: 0.03},
		{Name: "returning_amount_spike", Weight: 10, UserMode: "stable", ExpectedDecision: "challenged", AmountMin: 3500, AmountMax: 12000, TimeAdvanceMin: 2 * time.Minute, TimeAdvanceMax: 18 * time.Minute, Currency: "ARS", Channel: "web", RiskProfile: "review", MissingFeatureRate: 0.05},
		{Name: "returning_country_change", Weight: 6, UserMode: "stable", ExpectedDecision: "challenged", AmountMin: 120, AmountMax: 2200, TimeAdvanceMin: 2 * time.Minute, TimeAdvanceMax: 14 * time.Minute, CountryShift: true, Currency: "USD", Channel: "mobile", RiskProfile: "review", MissingFeatureRate: 0.08},
		{Name: "returning_destination_change", Weight: 6, UserMode: "stable", ExpectedDecision: "challenged", AmountMin: 75, AmountMax: 1800, TimeAdvanceMin: 90 * time.Second, TimeAdvanceMax: 10 * time.Minute, DestinationShift: true, Currency: "ARS", Channel: "mobile", RiskProfile: "review", MissingFeatureRate: 0.08},
		{Name: "returning_burst_velocity", Weight: 8, UserMode: "stable_hot", ExpectedDecision: "challenged", AmountMin: 90, AmountMax: 950, TimeAdvanceMin: 10 * time.Second, TimeAdvanceMax: 75 * time.Second, Currency: "ARS", Channel: "api", RiskProfile: "velocity", MissingFeatureRate: 0.04},
		{Name: "cross_border_travel", Weight: 5, UserMode: "stable", ExpectedDecision: "challenged", AmountMin: 200, AmountMax: 2500, TimeAdvanceMin: 40 * time.Minute, TimeAdvanceMax: 5 * time.Hour, CountryShift: true, Currency: "USD", Channel: "mobile", RiskProfile: "review", MissingFeatureRate: 0.06},
		{Name: "dormant_account_return", Weight: 5, UserMode: "stable", ExpectedDecision: "challenged", AmountMin: 120, AmountMax: 1600, TimeAdvanceMin: 45 * 24 * time.Hour, TimeAdvanceMax: 120 * 24 * time.Hour, Currency: "ARS", Channel: "web", RiskProfile: "review", MissingFeatureRate: 0.07},
		{Name: "travel_notice_like", Weight: 4, UserMode: "stable", ExpectedDecision: "challenged", AmountMin: 300, AmountMax: 2600, TimeAdvanceMin: 12 * time.Hour, TimeAdvanceMax: 3 * 24 * time.Hour, CountryShift: true, Currency: "USD", Channel: "mobile", RiskProfile: "trusted", MissingFeatureRate: 0.04},
		{Name: "merchant_fanout", Weight: 5, UserMode: "stable_hot", ExpectedDecision: "challenged", AmountMin: 180, AmountMax: 2400, TimeAdvanceMin: 30 * time.Second, TimeAdvanceMax: 5 * time.Minute, MerchantFanout: true, Currency: "ARS", Channel: "api", RiskProfile: "review", MissingFeatureRate: 0.04},
		{Name: "beneficiary_rotation_laddering", Weight: 5, UserMode: "stable_hot", ExpectedDecision: "challenged", AmountMin: 250, AmountMax: 2200, TimeAdvanceMin: 20 * time.Second, TimeAdvanceMax: 4 * time.Minute, DestinationShift: true, MerchantFanout: true, Currency: "ARS", Channel: "api", RiskProfile: "review", MissingFeatureRate: 0.04},
		{Name: "device_takeover", Weight: 6, UserMode: "stable_hot", ExpectedDecision: "blocked", AmountMin: 1100, AmountMax: 8000, TimeAdvanceMin: 20 * time.Second, TimeAdvanceMax: 3 * time.Minute, DestinationShift: true, DeviceShift: true, IdentityShift: true, Currency: "USD", Channel: "api", RiskProfile: "high", MissingFeatureRate: 0.03},
		{Name: "account_drain_attempt", Weight: 5, UserMode: "stable_hot", ExpectedDecision: "blocked", AmountMin: 12000, AmountMax: 48000, TimeAdvanceMin: 15 * time.Second, TimeAdvanceMax: 120 * time.Second, CountryShift: true, DestinationShift: true, DeviceShift: true, IdentityShift: true, MerchantFanout: true, Currency: "USD", Channel: "api", RiskProfile: "extreme", MissingFeatureRate: 0.02},
		{Name: "new_user_sparse_legit", Weight: 4, UserMode: "new", ExpectedDecision: "allowed", AmountMin: 40, AmountMax: 800, TimeAdvanceMin: 0, TimeAdvanceMax: 0, Currency: "ARS", Channel: "web", RiskProfile: "sparse", MissingFeatureRate: 0.30},
		{Name: "new_user_high_signal_risk", Weight: 2, UserMode: "new", ExpectedDecision: "blocked", AmountMin: 2500, AmountMax: 15000, TimeAdvanceMin: 0, TimeAdvanceMax: 0, CountryShift: true, DestinationShift: true, DeviceShift: true, IdentityShift: true, NightActivity: true, Currency: "USD", Channel: "api", RiskProfile: "extreme", MissingFeatureRate: 0.18},
		{Name: "micro_amount_card_testing", Weight: 4, UserMode: "new", ExpectedDecision: "blocked", AmountMin: 1, AmountMax: 35, TimeAdvanceMin: 5 * time.Second, TimeAdvanceMax: 40 * time.Second, DestinationShift: true, DeviceShift: true, MerchantFanout: true, NightActivity: true, Currency: "USD", Channel: "api", RiskProfile: "extreme", MissingFeatureRate: 0.15},
		{Name: "rapid_retry_after_decline_pattern", Weight: 4, UserMode: "stable_hot", ExpectedDecision: "blocked", AmountMin: 80, AmountMax: 900, TimeAdvanceMin: 5 * time.Second, TimeAdvanceMax: 25 * time.Second, DestinationShift: true, DeviceShift: true, MerchantFanout: true, Currency: "ARS", Channel: "api", RiskProfile: "high", MissingFeatureRate: 0.03},
	}

	totalWeight := 0
	for _, scenario := range scenarios {
		totalWeight += scenario.Weight
	}

	return &generator{
		rng:          rng,
		runID:        runID,
		baseTime:     baseTime,
		featureOrder: append([]string(nil), featureOrder...),
		stableUsers:  stableUsers,
		hotUsers:     stableUsers[:hotCount],
		scenarios:    scenarios,
		totalWeight:  totalWeight,
	}
}

func (g *generator) Next(index int, sourceSystem, sourceComponent, sourceRegion string) requestPlan {
	scenario := g.pickScenario()
	user := g.pickUser(scenario)
	amount := g.pickAmount(user, scenario)
	eventTime := g.advanceEventTime(user, scenario)
	transactionID := fmt.Sprintf("tx_%s_%06d", g.runID, index)
	requestID := fmt.Sprintf("req_%s_%06d", g.runID, index)

	country := user.HomeCountry
	if scenario.CountryShift {
		country = user.AltCountry
	}

	destination := user.PrimaryDest
	if scenario.DestinationShift {
		destination = user.SecondaryDest
	}
	if scenario.MerchantFanout && g.rng.Float64() < 0.7 {
		destination = fmt.Sprintf("%s_fanout_%03d", user.UserID, (index%17)+1)
	}

	merchant := pickMerchant(user, scenario, index)
	channel := scenario.Channel
	currency := scenario.Currency
	if currency == "" {
		currency = "ARS"
	}

	featureContext := g.buildFeatureContext(index, scenario, user, amount, eventTime, country, destination, merchant, currency, channel)
	features := g.buildFeatures(featureContext)

	metadataLabels := map[string]string{
		"scenario":          scenario.Name,
		"user_mode":         scenario.UserMode,
		"feature_level":     scenario.RiskProfile,
		"risk_profile":      scenario.RiskProfile,
		"expected_decision": scenario.ExpectedDecision,
		"run_id":            g.runID,
		"feature_count":     strconv.Itoa(len(features)),
		"merchant":          merchant,
		"currency":          currency,
		"country_shift":     strconv.FormatBool(featureContext.IsCountryShift),
		"destination_new":   strconv.FormatBool(featureContext.IsDestinationShift),
		"night_activity":    strconv.FormatBool(featureContext.IsNightActivity),
	}

	request := &pb.ProcessTransactionRequest{
		Trace: &pb.RequestTrace{
			RequestId:       requestID,
			SourceSystem:    sourceSystem,
			SourceComponent: sourceComponent,
			SourceRegion:    sourceRegion,
		},
		Transaction: &pb.TransactionContext{
			TransactionId:      transactionID,
			UserId:             user.UserID,
			PersonId:           user.PersonID,
			AccountId:          user.AccountID,
			DestinationAccount: destination,
			Currency:           currency,
			Country:            country,
			Channel:            channel,
			EventTimestamp:     eventTime.Format(time.RFC3339),
		},
		Features:       features,
		MetadataLabels: metadataLabels,
	}

	g.appendHistory(user, featureContext)

	return requestPlan{
		Index:        index,
		Scenario:     scenario.Name,
		UserMode:     scenario.UserMode,
		Amount:       amount,
		FeatureLevel: scenario.RiskProfile,
		Request:      request,
	}
}

func (g *generator) pickScenario() scenarioConfig {
	roll := g.rng.Intn(g.totalWeight)
	running := 0
	for _, scenario := range g.scenarios {
		running += scenario.Weight
		if roll < running {
			return scenario
		}
	}
	return g.scenarios[len(g.scenarios)-1]
}

func (g *generator) pickUser(scenario scenarioConfig) *userState {
	switch scenario.UserMode {
	case "new":
		g.newUserSeq++
		index := g.newUserSeq
		homeCountry, altCountry := countryPair(index + len(g.stableUsers))
		baselineAmount := 75 + g.rng.Float64()*260
		amountStd := 25 + g.rng.Float64()*90
		return &userState{
			UserID:             fmt.Sprintf("%s_new_user_%05d", g.runID, index),
			PersonID:           fmt.Sprintf("%s_new_person_%05d", g.runID, index),
			AccountID:          fmt.Sprintf("%s_new_account_%05d", g.runID, index),
			HomeCountry:        homeCountry,
			AltCountry:         altCountry,
			PrimaryDest:        fmt.Sprintf("%s_new_dest_%05d_primary", g.runID, index),
			SecondaryDest:      fmt.Sprintf("%s_new_dest_%05d_secondary", g.runID, index),
			FavoriteMerchants:  favoriteMerchants(index + 1000),
			Card1:              1700 + float64(index%400),
			Card2:              180 + float64(index%40),
			Card3:              155 + float64(index%20),
			Card5:              225 + float64(index%10),
			Addr1:              300 + float64(index%70),
			Addr2:              60 + float64(index%20),
			Dist1:              float64(index % 11),
			Dist2:              float64(index % 7),
			BaselineAmount:     baselineAmount,
			AmountStd:          amountStd,
			TrustScore:         0.18 + g.rng.Float64()*0.35,
			IdentityConfidence: 0.20 + g.rng.Float64()*0.35,
			VelocityTolerance:  0.15 + g.rng.Float64()*0.40,
			DeviceStability:    0.10 + g.rng.Float64()*0.30,
			MerchantDiversity:  0.45 + g.rng.Float64()*0.45,
			LastEventTime:      g.baseTime,
		}
	case "stable_hot":
		return g.hotUsers[g.rng.Intn(len(g.hotUsers))]
	default:
		return g.stableUsers[g.rng.Intn(len(g.stableUsers))]
	}
}

func (g *generator) pickAmount(user *userState, scenario scenarioConfig) float64 {
	amount := g.randomFloat(scenario.AmountMin, scenario.AmountMax)
	switch scenario.RiskProfile {
	case "trusted":
		center := user.BaselineAmount * (0.85 + g.rng.Float64()*0.35)
		amount = clamp(center+g.randomFloat(-user.AmountStd*0.6, user.AmountStd*0.8), scenario.AmountMin, scenario.AmountMax)
	case "velocity":
		amount = clamp(user.BaselineAmount*(0.9+g.rng.Float64()*1.4), scenario.AmountMin, scenario.AmountMax)
	case "review":
		amount = clamp(user.BaselineAmount*(1.8+g.rng.Float64()*5.5), scenario.AmountMin, scenario.AmountMax)
	case "high":
		amount = clamp(user.BaselineAmount*(4.0+g.rng.Float64()*10.0), scenario.AmountMin, scenario.AmountMax)
	case "extreme":
		amount = clamp(user.BaselineAmount*(8.0+g.rng.Float64()*24.0), scenario.AmountMin, scenario.AmountMax)
	case "sparse":
		amount = clamp(user.BaselineAmount*(0.8+g.rng.Float64()*2.5), scenario.AmountMin, scenario.AmountMax)
	}
	return round3(amount)
}

func (g *generator) advanceEventTime(user *userState, scenario scenarioConfig) time.Time {
	advance := scenario.TimeAdvanceMin
	if scenario.TimeAdvanceMax > scenario.TimeAdvanceMin {
		extra := g.rng.Int63n(int64(scenario.TimeAdvanceMax - scenario.TimeAdvanceMin))
		advance += time.Duration(extra)
	}
	if advance <= 0 {
		advance = time.Duration(g.rng.Intn(90)+1) * time.Second
	}
	nextTime := user.LastEventTime.Add(advance)
	if scenario.NightActivity {
		nextTime = time.Date(nextTime.Year(), nextTime.Month(), nextTime.Day(), 2+g.rng.Intn(3), g.rng.Intn(59), g.rng.Intn(59), 0, time.UTC)
	}
	user.LastEventTime = nextTime
	return nextTime
}

func (g *generator) buildFeatureContext(index int, scenario scenarioConfig, user *userState, amount float64, eventTime time.Time, country, destination, merchant, currency, channel string) featureContext {
	prior5 := tailHistory(user.History, 5)
	prior10 := tailHistory(user.History, 10)

	previousAmount := 0.0
	secondsSincePrevious := 86400.0
	if len(user.History) > 0 {
		last := user.History[len(user.History)-1]
		previousAmount = last.Amount
		secondsSincePrevious = math.Max(eventTime.Sub(last.EventTime).Seconds(), 1)
	}

	prior5Sum, prior5Mean, prior5Std := amountStats(prior5)
	prior10Sum, prior10Mean, prior10Std := amountStats(prior10)
	prior5UniqueDest := uniqueDestinationCount(prior5)
	prior10UniqueDest := uniqueDestinationCount(prior10)
	amountRatio := 1.0
	if previousAmount > 0 {
		amountRatio = amount / previousAmount
	} else if user.BaselineAmount > 0 {
		amountRatio = amount / user.BaselineAmount
	}
	amountDelta := amount - previousAmount

	recentBurstCount := recentCountSince(user.History, eventTime.Add(-10*time.Minute))
	velocityScore := clamp01(float64(recentBurstCount)/6.0 + 60.0/math.Max(secondsSincePrevious, 60))
	isCountryShift := country != user.HomeCountry
	isDestinationShift := len(user.History) > 0 && destination != user.History[len(user.History)-1].Destination
	newUser := len(user.History) == 0
	geoScore := ternaryFloat(isCountryShift, 0.85, 0.15)
	destinationNovelty := ternaryFloat(isDestinationShift, 0.80, 0.10)
	deviceRisk := clamp01((1-user.DeviceStability)*0.6 + ternaryFloat(scenario.DeviceShift, 0.35, 0))
	identityRisk := clamp01((1-user.IdentityConfidence)*0.5 + ternaryFloat(scenario.IdentityShift, 0.45, 0))
	behaviorRisk := clamp01(
		0.16*normalizedAmount(amount, user.BaselineAmount, user.AmountStd) +
			0.24*clamp01((amountRatio-1)/6) +
			0.20*velocityScore +
			0.14*geoScore +
			0.14*destinationNovelty +
			0.12*deviceRisk +
			0.12*identityRisk,
	)

	if scenario.RiskProfile == "trusted" {
		behaviorRisk *= 0.55
	}
	if scenario.RiskProfile == "extreme" {
		behaviorRisk = clamp01(0.65 + behaviorRisk*0.55)
	}

	return featureContext{
		Scenario:               scenario,
		User:                   user,
		RequestIndex:           index,
		Amount:                 amount,
		EventTime:              eventTime,
		Country:                country,
		Destination:            destination,
		Merchant:               merchant,
		Currency:               currency,
		Channel:                channel,
		PreviousAmount:         round3(previousAmount),
		SecondsSincePrevious:   round3(secondsSincePrevious),
		Prior5Count:            float64(len(prior5)),
		Prior10Count:           float64(len(prior10)),
		Prior5AmountSum:        round3(prior5Sum),
		Prior5AmountMean:       round3(prior5Mean),
		Prior5AmountStd:        round3(prior5Std),
		Prior10AmountSum:       round3(prior10Sum),
		Prior10AmountMean:      round3(prior10Mean),
		Prior10AmountStd:       round3(prior10Std),
		Prior5UniqueDestCount:  float64(prior5UniqueDest),
		Prior10UniqueDestCount: float64(prior10UniqueDest),
		AmountDelta:            round3(amountDelta),
		AmountRatio:            round3(amountRatio),
		VelocityScore:          round3(velocityScore),
		GeoScore:               round3(geoScore),
		DestinationNovelty:     round3(destinationNovelty),
		DeviceRisk:             round3(deviceRisk),
		IdentityRisk:           round3(identityRisk),
		BehaviorRisk:           round3(behaviorRisk),
		NewUser:                newUser,
		IsCountryShift:         isCountryShift,
		IsDestinationShift:     isDestinationShift,
		IsMerchantFanout:       scenario.MerchantFanout,
		IsNightActivity:        scenario.NightActivity,
		RecentBurstCount:       recentBurstCount,
	}
}

func (g *generator) buildFeatures(ctx featureContext) map[string]float64 {
	features := make(map[string]float64, len(g.featureOrder))
	critical := criticalFeatureNames()

	for _, name := range g.featureOrder {
		value, ok := g.featureValue(name, ctx)
		if !ok {
			continue
		}
		if !critical[name] && shouldOmitFeature(name, ctx.Scenario.MissingFeatureRate, ctx, g.rng) {
			continue
		}
		features[name] = round3(value)
	}

	ctx.FeatureCoverageEstimate = len(features)
	return features
}

func (g *generator) featureValue(name string, ctx featureContext) (float64, bool) {
	switch {
	case name == "source_file_row_number":
		return float64(ctx.RequestIndex), true
	case name == "amount":
		return ctx.Amount, true
	case name == "amount_log1p":
		return math.Log1p(ctx.Amount), true
	case name == "previous_transaction_amount":
		return ctx.PreviousAmount, true
	case name == "prior_5_transaction_count":
		return ctx.Prior5Count, true
	case name == "prior_10_transaction_count":
		return ctx.Prior10Count, true
	case name == "prior_5_amount_sum":
		return ctx.Prior5AmountSum, true
	case name == "prior_5_amount_mean":
		return ctx.Prior5AmountMean, true
	case name == "prior_5_amount_std":
		return ctx.Prior5AmountStd, true
	case name == "prior_10_amount_sum":
		return ctx.Prior10AmountSum, true
	case name == "prior_10_amount_mean":
		return ctx.Prior10AmountMean, true
	case name == "prior_10_amount_std":
		return ctx.Prior10AmountStd, true
	case name == "seconds_since_previous_transaction":
		return ctx.SecondsSincePrevious, true
	case name == "prior_5_unique_name_dest_count":
		return ctx.Prior5UniqueDestCount, true
	case name == "prior_10_unique_name_dest_count":
		return ctx.Prior10UniqueDestCount, true
	case name == "card1":
		return ctx.User.Card1, true
	case name == "card2":
		return ctx.User.Card2, true
	case name == "card3":
		return ctx.User.Card3, true
	case name == "card5":
		return ctx.User.Card5, true
	case name == "addr1":
		return ctx.User.Addr1, true
	case name == "addr2":
		return ctx.User.Addr2, true
	case name == "dist1":
		return ctx.User.Dist1 + ternaryFloat(ctx.IsCountryShift, 6, 0), true
	case name == "dist2":
		return ctx.User.Dist2 + ternaryFloat(ctx.IsDestinationShift, 4, 0), true
	}

	if index, ok := parsePrefixedIndex(name, "c"); ok {
		return featureC(index, ctx), true
	}
	if index, ok := parsePrefixedIndex(name, "d"); ok {
		return featureD(index, ctx), true
	}
	if index, ok := parsePrefixedIndex(name, "m"); ok {
		return featureM(index, ctx), true
	}
	if index, ok := parsePrefixedIndex(name, "v"); ok {
		return featureV(index, ctx), true
	}
	if index, ok := parsePrefixedIndex(name, "id_"); ok {
		return featureID(index, ctx), true
	}

	return 0, false
}

func featureC(index int, ctx featureContext) float64 {
	trust := ctx.User.TrustScore
	base := 0.18*float64(index) + ctx.BehaviorRisk*3.4 + ctx.VelocityScore*1.8 + ctx.AmountRatio*0.6 - trust*1.2
	switch index {
	case 1:
		return 1 + ctx.Prior5Count + ctx.AmountRatio*0.4
	case 2:
		return ctx.Prior10Count + ctx.VelocityScore*2.0
	case 4:
		return base + ctx.GeoScore*2.1
	case 7:
		return ctx.Prior5UniqueDestCount + ctx.DestinationNovelty*2.5
	case 10:
		return ctx.Prior10AmountMean / math.Max(ctx.User.BaselineAmount, 1)
	case 14:
		return base + ctx.IdentityRisk*2.0 + ternaryFloat(ctx.NewUser, 0.8, 0)
	default:
		return base + harmonic(index, ctx)*0.8
	}
}

func featureD(index int, ctx featureContext) float64 {
	switch index {
	case 1:
		return ctx.SecondsSincePrevious / 3600
	case 2:
		return ctx.AmountRatio * 8
	case 3:
		return ctx.Prior5AmountMean / math.Max(ctx.User.BaselineAmount, 1)
	case 4:
		return ctx.Prior5AmountStd / math.Max(ctx.User.AmountStd, 1)
	case 5:
		return float64(ctx.RecentBurstCount) * 1.4
	default:
		return 0.5*float64(index) + ctx.BehaviorRisk*5.0 + harmonic(index, ctx)*1.6
	}
}

func featureM(index int, ctx featureContext) float64 {
	flags := []bool{
		!ctx.IsCountryShift,
		!ctx.IsDestinationShift,
		!ctx.NewUser,
		ctx.User.TrustScore > 0.55,
		ctx.User.IdentityConfidence > 0.55 && !ctx.Scenario.IdentityShift,
		ctx.User.DeviceStability > 0.55 && !ctx.Scenario.DeviceShift,
		ctx.Currency == "ARS",
		ctx.Channel != "api" || ctx.User.DeviceStability > 0.6,
		!ctx.IsNightActivity,
	}
	if index <= 0 || index > len(flags) {
		return 0
	}
	if flags[index-1] {
		return 1
	}
	return 0
}

func featureV(index int, ctx featureContext) float64 {
	amountSignal := normalizedAmount(ctx.Amount, ctx.User.BaselineAmount, ctx.User.AmountStd)
	identitySignal := 1 - ctx.User.IdentityConfidence + ctx.IdentityRisk
	deviceSignal := 1 - ctx.User.DeviceStability + ctx.DeviceRisk
	trustSignal := 1 - ctx.User.TrustScore
	wave := harmonic(index, ctx)

	switch {
	case index <= 60:
		return 0.9*amountSignal + 1.5*ctx.BehaviorRisk + 0.7*ctx.VelocityScore + wave
	case index <= 120:
		return 1.2*ctx.AmountRatio + 1.3*ctx.GeoScore + 0.8*ctx.DestinationNovelty + wave*1.1
	case index <= 180:
		return 1.5*ctx.VelocityScore + 1.1*ctx.Prior10Count/10 + 0.8*amountSignal + wave*1.3
	case index <= 240:
		return 1.4*identitySignal + 0.9*deviceSignal + 0.7*trustSignal + wave*1.2
	case index <= 300:
		return 1.2*ctx.BehaviorRisk + 1.0*ctx.DestinationNovelty + 0.8*ctx.GeoScore + 0.6*identitySignal + wave*1.4
	default:
		return 1.8*ctx.BehaviorRisk + 1.2*ctx.VelocityScore + 1.1*deviceSignal + 0.9*identitySignal + wave*1.6
	}
}

func featureID(index int, ctx featureContext) float64 {
	switch index {
	case 1:
		return -7.5 + ctx.User.IdentityConfidence*6 - ctx.IdentityRisk*4
	case 2:
		return 80 + ctx.User.TrustScore*60 - ctx.DeviceRisk*25
	case 5:
		return ctx.User.DeviceStability*4 - ctx.DeviceRisk*3
	case 11:
		return ctx.User.TrustScore*3 - ctx.BehaviorRisk*1.8
	case 17:
		return ctx.BehaviorRisk*3.8 + ctx.GeoScore*2.0 + ctx.DestinationNovelty*1.5
	case 23:
		return ctx.IdentityRisk*4.2 + ctx.DeviceRisk*3.1
	case 30:
		return ctx.Prior10UniqueDestCount + ctx.DestinationNovelty*3
	case 33:
		return ctx.SecondsSincePrevious / 600
	case 38:
		return ctx.AmountRatio + ctx.BehaviorRisk*2
	default:
		return float64(index)*0.12 + ctx.BehaviorRisk*2.2 + harmonic(index+400, ctx)*0.9
	}
}

func (g *generator) appendHistory(user *userState, ctx featureContext) {
	user.History = append(user.History, txnHistory{
		Amount:      ctx.Amount,
		Country:     ctx.Country,
		Destination: ctx.Destination,
		Merchant:    ctx.Merchant,
		Channel:     ctx.Channel,
		Currency:    ctx.Currency,
		EventTime:   ctx.EventTime,
	})
	if len(user.History) > 20 {
		user.History = append([]txnHistory(nil), user.History[len(user.History)-20:]...)
	}
}

func requestRecordsFromPlans(plans []requestPlan) []requestRecord {
	records := make([]requestRecord, 0, len(plans))
	for _, plan := range plans {
		records = append(records, requestRecord{
			Index:          plan.Index,
			Scenario:       plan.Scenario,
			UserMode:       plan.UserMode,
			RequestID:      plan.Request.GetTrace().GetRequestId(),
			TransactionID:  plan.Request.GetTransaction().GetTransactionId(),
			UserID:         plan.Request.GetTransaction().GetUserId(),
			EventTimestamp: plan.Request.GetTransaction().GetEventTimestamp(),
			Amount:         plan.Amount,
			Country:        plan.Request.GetTransaction().GetCountry(),
			Channel:        plan.Request.GetTransaction().GetChannel(),
			Destination:    plan.Request.GetTransaction().GetDestinationAccount(),
			FeatureLevel:   plan.FeatureLevel,
			MetadataLabels: plan.Request.GetMetadataLabels(),
			Features:       plan.Request.GetFeatures(),
		})
	}
	return records
}

func summarizeResults(runID string, startedAt, completedAt time.Time, addr string, requests, concurrency, stableUsers int, seed int64, modelVersion string, plans []requestPlan, results []requestResult) loadtestSummary {
	decisionCounts := map[string]int{}
	scenarioCounts := map[string]int{}
	decisionByScenario := map[string]map[string]int{}
	scenarioExpected := map[string]string{}
	scenarioScores := map[string][]float64{}
	scenarioMatched := map[string]int{}
	clientRTTs := make([]float64, 0, len(results))
	serverTotals := make([]float64, 0, len(results))
	profileLookup := make([]float64, 0, len(results))
	scoringDurations := make([]float64, 0, len(results))
	profileUpdate := make([]float64, 0, len(results))
	calibratedScores := make([]float64, 0, len(results))
	featureCounts := make([]float64, 0, len(plans))
	errors := 0

	for _, plan := range plans {
		scenarioCounts[plan.Scenario]++
		featureCounts = append(featureCounts, float64(len(plan.Request.GetFeatures())))
		if decisionByScenario[plan.Scenario] == nil {
			decisionByScenario[plan.Scenario] = map[string]int{}
		}
		if scenarioExpected[plan.Scenario] == "" {
			scenarioExpected[plan.Scenario] = plan.Request.GetMetadataLabels()["expected_decision"]
		}
	}

	for _, result := range results {
		clientRTTs = append(clientRTTs, result.ClientRTTMs)
		if result.Error != "" {
			errors++
			continue
		}
		decisionCounts[result.Decision]++
		decisionByScenario[result.Scenario][result.Decision]++
		if result.Decision == scenarioExpected[result.Scenario] {
			scenarioMatched[result.Scenario]++
		}
		serverTotals = append(serverTotals, result.ServerTotalMs)
		profileLookup = append(profileLookup, result.ProfileLookupMs)
		scoringDurations = append(scoringDurations, result.ScoringMs)
		profileUpdate = append(profileUpdate, result.ProfileUpdateMs)
		calibratedScores = append(calibratedScores, result.CalibratedScore)
		scenarioScores[result.Scenario] = append(scenarioScores[result.Scenario], result.CalibratedScore)
	}

	scenarioAudit := make(map[string]scenarioAuditSummary, len(scenarioCounts))
	for scenario, count := range scenarioCounts {
		matched := scenarioMatched[scenario]
		matchedRate := 0.0
		if count > 0 {
			matchedRate = float64(matched) / float64(count)
		}
		scenarioAudit[scenario] = scenarioAuditSummary{
			ExpectedDecision: scenarioExpected[scenario],
			RequestCount:     count,
			MatchedCount:     matched,
			MatchedRate:      matchedRate,
			DecisionCounts:   decisionByScenario[scenario],
			CalibratedScore:  summarizeDurations(scenarioScores[scenario]),
		}
	}

	return loadtestSummary{
		RunID:               runID,
		StartedAt:           startedAt.Format(time.RFC3339),
		CompletedAt:         completedAt.Format(time.RFC3339),
		ServerAddress:       addr,
		Requests:            requests,
		Concurrency:         concurrency,
		StableUsers:         stableUsers,
		Seed:                seed,
		ModelVersion:        modelVersion,
		Errors:              errors,
		DecisionCounts:      decisionCounts,
		ScenarioCounts:      scenarioCounts,
		DecisionByScenario:  decisionByScenario,
		ScenarioAudit:       scenarioAudit,
		ClientRTT:           summarizeDurations(clientRTTs),
		ServerTotal:         summarizeDurations(serverTotals),
		ProfileLookup:       summarizeDurations(profileLookup),
		Scoring:             summarizeDurations(scoringDurations),
		ProfileUpdate:       summarizeDurations(profileUpdate),
		CalibratedScore:     summarizeDurations(calibratedScores),
		RequestFeatureCount: summarizeDurations(featureCounts),
	}
}

func summarizeDurations(values []float64) durationSummary {
	if len(values) == 0 {
		return durationSummary{}
	}
	sortedValues := append([]float64(nil), values...)
	sort.Float64s(sortedValues)

	sum := 0.0
	for _, value := range sortedValues {
		sum += value
	}

	return durationSummary{
		Count: len(sortedValues),
		Min:   sortedValues[0],
		Max:   sortedValues[len(sortedValues)-1],
		Mean:  sum / float64(len(sortedValues)),
		P50:   percentile(sortedValues, 0.50),
		P95:   percentile(sortedValues, 0.95),
		P99:   percentile(sortedValues, 0.99),
	}
}

func percentile(sortedValues []float64, p float64) float64 {
	if len(sortedValues) == 0 {
		return 0
	}
	if len(sortedValues) == 1 {
		return sortedValues[0]
	}

	index := p * float64(len(sortedValues)-1)
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))
	if lower == upper {
		return sortedValues[lower]
	}

	weight := index - float64(lower)
	return sortedValues[lower] + (sortedValues[upper]-sortedValues[lower])*weight
}

func writeJSONL(path string, rows any) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	switch values := rows.(type) {
	case []requestRecord:
		for _, row := range values {
			if err := encoder.Encode(row); err != nil {
				return err
			}
		}
	case []requestResult:
		for _, row := range values {
			if err := encoder.Encode(row); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported rows type %T", rows)
	}
	return nil
}

func writeJSON(path string, payload any) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(payload)
}

func mustJSON(payload any) string {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Sprintf(`{"error":"%v"}`, err)
	}
	return string(body)
}

func durationMillis(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

func countryPair(index int) (string, string) {
	pairs := [][2]string{
		{"AR", "BR"},
		{"UY", "AR"},
		{"CL", "PY"},
		{"BR", "AR"},
		{"AR", "US"},
		{"AR", "CL"},
		{"PY", "AR"},
	}
	pair := pairs[index%len(pairs)]
	return pair[0], pair[1]
}

func favoriteMerchants(index int) []string {
	return []string{
		fmt.Sprintf("merchant_%02d_grocery", index%17),
		fmt.Sprintf("merchant_%02d_utilities", index%13),
		fmt.Sprintf("merchant_%02d_transport", index%11),
	}
}

func bootstrapHistory(rng *rand.Rand, baseTime time.Time, baselineAmount, amountStd float64, country, destination string, merchants []string, count int) []txnHistory {
	history := make([]txnHistory, 0, count)
	eventTime := baseTime.Add(-time.Duration(count*8) * time.Hour)
	for index := 0; index < count; index++ {
		eventTime = eventTime.Add(time.Duration(45+rng.Intn(360)) * time.Minute)
		amount := clamp(baselineAmount+rng.Float64()*amountStd-rng.Float64()*amountStd*0.4, 10, baselineAmount+amountStd*1.8)
		history = append(history, txnHistory{
			Amount:      round3(amount),
			Country:     country,
			Destination: destination,
			Merchant:    merchants[index%len(merchants)],
			Channel:     []string{"web", "mobile", "api"}[index%3],
			Currency:    "ARS",
			EventTime:   eventTime,
		})
	}
	return history
}

func pickMerchant(user *userState, scenario scenarioConfig, index int) string {
	if scenario.MerchantFanout {
		return fmt.Sprintf("merchant_fanout_%s_%02d", user.UserID, index%23)
	}
	if scenario.DestinationShift && len(user.FavoriteMerchants) > 0 {
		return fmt.Sprintf("%s_alt", user.FavoriteMerchants[index%len(user.FavoriteMerchants)])
	}
	return user.FavoriteMerchants[index%len(user.FavoriteMerchants)]
}

func tailHistory(history []txnHistory, n int) []txnHistory {
	if len(history) <= n {
		return append([]txnHistory(nil), history...)
	}
	return append([]txnHistory(nil), history[len(history)-n:]...)
}

func amountStats(history []txnHistory) (sum, mean, std float64) {
	if len(history) == 0 {
		return 0, 0, 0
	}
	for _, item := range history {
		sum += item.Amount
	}
	mean = sum / float64(len(history))
	if len(history) == 1 {
		return sum, mean, 0
	}
	variance := 0.0
	for _, item := range history {
		delta := item.Amount - mean
		variance += delta * delta
	}
	std = math.Sqrt(variance / float64(len(history)))
	return sum, mean, std
}

func uniqueDestinationCount(history []txnHistory) int {
	seen := make(map[string]struct{}, len(history))
	for _, item := range history {
		seen[item.Destination] = struct{}{}
	}
	return len(seen)
}

func recentCountSince(history []txnHistory, threshold time.Time) int {
	count := 0
	for _, item := range history {
		if item.EventTime.After(threshold) {
			count++
		}
	}
	return count
}

func criticalFeatureNames() map[string]bool {
	return map[string]bool{
		"source_file_row_number":             true,
		"amount":                             true,
		"amount_log1p":                       true,
		"previous_transaction_amount":        true,
		"prior_5_transaction_count":          true,
		"prior_10_transaction_count":         true,
		"prior_5_amount_sum":                 true,
		"prior_5_amount_mean":                true,
		"prior_5_amount_std":                 true,
		"prior_10_amount_sum":                true,
		"prior_10_amount_mean":               true,
		"prior_10_amount_std":                true,
		"seconds_since_previous_transaction": true,
		"prior_5_unique_name_dest_count":     true,
		"prior_10_unique_name_dest_count":    true,
		"card1":                              true,
		"card2":                              true,
		"card3":                              true,
		"card5":                              true,
		"addr1":                              true,
		"addr2":                              true,
		"dist1":                              true,
		"dist2":                              true,
	}
}

func shouldOmitFeature(name string, missingRate float64, ctx featureContext, rng *rand.Rand) bool {
	if missingRate <= 0 {
		return false
	}
	if strings.HasPrefix(name, "m") {
		return false
	}
	familyBias := 1.0
	switch {
	case strings.HasPrefix(name, "v"):
		familyBias = 1.2
	case strings.HasPrefix(name, "id_"):
		familyBias = 0.8
	case strings.HasPrefix(name, "c"), strings.HasPrefix(name, "d"):
		familyBias = 0.6
	}
	if ctx.Scenario.RiskProfile == "sparse" {
		familyBias *= 1.35
	}
	return rng.Float64() < missingRate*familyBias
}

func parsePrefixedIndex(name, prefix string) (int, bool) {
	if !strings.HasPrefix(name, prefix) {
		return 0, false
	}
	value, err := strconv.Atoi(strings.TrimPrefix(name, prefix))
	if err != nil {
		return 0, false
	}
	return value, true
}

func harmonic(index int, ctx featureContext) float64 {
	seed := float64(index)*0.37 + float64(ctx.RequestIndex)*0.03 + ctx.User.TrustScore*3.7 + ctx.User.DeviceStability*2.1
	return math.Sin(seed) + math.Cos(seed*0.7)*0.6 + math.Sin(seed*0.17)*0.4
}

func normalizedAmount(amount, baseline, std float64) float64 {
	if baseline <= 0 {
		baseline = 1
	}
	if std <= 0 {
		std = math.Max(baseline*0.2, 1)
	}
	return clamp01((amount - baseline + std) / (baseline + 3*std))
}

func round3(value float64) float64 {
	return math.Round(value*1000) / 1000
}

func clamp(value, minValue, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func clamp01(value float64) float64 {
	return clamp(value, 0, 1)
}

func ternaryFloat(condition bool, whenTrue, whenFalse float64) float64 {
	if condition {
		return whenTrue
	}
	return whenFalse
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (g *generator) randomFloat(minValue, maxValue float64) float64 {
	if maxValue <= minValue {
		return minValue
	}
	return minValue + g.rng.Float64()*(maxValue-minValue)
}
