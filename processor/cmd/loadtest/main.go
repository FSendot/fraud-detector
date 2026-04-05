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
	"sync"
	"time"

	pb "github.com/FSendot/fraud-detector/processor/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type scenarioConfig struct {
	Name             string
	Weight           int
	UserMode         string
	AmountMin        float64
	AmountMax        float64
	TimeAdvanceMin   time.Duration
	TimeAdvanceMax   time.Duration
	CountryShift     bool
	DestinationShift bool
	Channel          string
	FeatureProfile   string
}

type userState struct {
	UserID        string
	PersonID      string
	AccountID     string
	HomeCountry   string
	AltCountry    string
	PrimaryDest   string
	SecondaryDest string
	Card1         float64
	Card2         float64
	Card3         float64
	Card5         float64
	Addr1         float64
	Addr2         float64
	Dist1         float64
	Dist2         float64
	LastEventTime time.Time
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

type loadtestSummary struct {
	RunID          string          `json:"run_id"`
	StartedAt      string          `json:"started_at"`
	CompletedAt    string          `json:"completed_at"`
	ServerAddress  string          `json:"server_address"`
	Requests       int             `json:"requests"`
	Concurrency    int             `json:"concurrency"`
	StableUsers    int             `json:"stable_users"`
	Seed           int64           `json:"seed"`
	Errors         int             `json:"errors"`
	DecisionCounts map[string]int  `json:"decision_counts"`
	ScenarioCounts map[string]int  `json:"scenario_counts"`
	ClientRTT      durationSummary `json:"client_rtt_ms"`
	ServerTotal    durationSummary `json:"server_total_ms"`
	ProfileLookup  durationSummary `json:"profile_lookup_ms"`
	Scoring        durationSummary `json:"scoring_ms"`
	ProfileUpdate  durationSummary `json:"profile_update_ms"`
}

type generator struct {
	rng         *rand.Rand
	runID       string
	baseTime    time.Time
	stableUsers []*userState
	hotUsers    []*userState
	newUserSeq  int
	scenarios   []scenarioConfig
	totalWeight int
}

func main() {
	addr := flag.String("addr", "localhost:50051", "gRPC server address")
	requests := flag.Int("requests", 2500, "number of mock requests to send")
	concurrency := flag.Int("concurrency", 40, "number of concurrent workers")
	stableUsers := flag.Int("stable-users", 200, "number of recurring users reused across runs")
	timeout := flag.Duration("timeout", 10*time.Second, "per-request timeout")
	userPrefix := flag.String("user-prefix", "loadtest", "stable user prefix so runs can evolve local dynamodb state over time")
	outputDir := flag.String("output-dir", "", "optional output directory; defaults to processor/output/loadtest/<run_id>")
	sourceSystem := flag.String("source-system", "loadtest", "request trace source system")
	sourceComponent := flag.String("source-component", "load-generator", "request trace source component")
	sourceRegion := flag.String("source-region", "local", "request trace source region")
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

	gen := newGenerator(resolvedSeed, runID, *stableUsers, *userPrefix, runStartedAt)
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

	summary := summarizeResults(runID, runStartedAt, time.Now().UTC(), *addr, *requests, *concurrency, *stableUsers, resolvedSeed, plans, results)
	summaryPath := filepath.Join(resolvedOutputDir, "summary.json")
	if err := writeJSON(summaryPath, summary); err != nil {
		log.Fatalf("write summary: %v", err)
	}

	fmt.Printf("loadtest_run=%s output_dir=%s requests=%d concurrency=%d errors=%d\n",
		runID, resolvedOutputDir, summary.Requests, summary.Concurrency, summary.Errors)
	fmt.Printf("client_rtt_ms mean=%.2f p95=%.2f p99=%.2f\n", summary.ClientRTT.Mean, summary.ClientRTT.P95, summary.ClientRTT.P99)
	fmt.Printf("server_total_ms mean=%.2f p95=%.2f p99=%.2f\n", summary.ServerTotal.Mean, summary.ServerTotal.P95, summary.ServerTotal.P99)
	fmt.Printf("decisions=%s\n", mustJSON(summary.DecisionCounts))
	fmt.Printf("scenarios=%s\n", mustJSON(summary.ScenarioCounts))
	fmt.Printf("requests_dataset=%s\n", requestDatasetPath)
	fmt.Printf("results_dataset=%s\n", resultsPath)
	fmt.Printf("summary_json=%s\n", summaryPath)
}

func newGenerator(seed int64, runID string, stableUserCount int, userPrefix string, baseTime time.Time) *generator {
	rng := rand.New(rand.NewSource(seed))
	stableUsers := make([]*userState, 0, stableUserCount)
	hotCount := min(10, stableUserCount)
	for index := 0; index < stableUserCount; index++ {
		homeCountry, altCountry := countryPair(index)
		stableUsers = append(stableUsers, &userState{
			UserID:        fmt.Sprintf("%s_user_%04d", userPrefix, index),
			PersonID:      fmt.Sprintf("%s_person_%04d", userPrefix, index),
			AccountID:     fmt.Sprintf("%s_account_%04d", userPrefix, index),
			HomeCountry:   homeCountry,
			AltCountry:    altCountry,
			PrimaryDest:   fmt.Sprintf("dest_%04d_primary", index),
			SecondaryDest: fmt.Sprintf("dest_%04d_secondary", index),
			Card1:         1000 + float64(index%500),
			Card2:         100 + float64(index%80),
			Card3:         150 + float64(index%25),
			Card5:         220 + float64(index%12),
			Addr1:         200 + float64(index%60),
			Addr2:         50 + float64(index%20),
			Dist1:         float64(index % 7),
			Dist2:         float64(index % 5),
			LastEventTime: baseTime.Add(-time.Duration(stableUserCount-index) * 20 * time.Minute),
		})
	}

	scenarios := []scenarioConfig{
		{Name: "returning_normal", Weight: 30, UserMode: "stable", AmountMin: 25, AmountMax: 300, TimeAdvanceMin: 8 * time.Minute, TimeAdvanceMax: 35 * time.Minute, Channel: "web", FeatureProfile: "low"},
		{Name: "returning_amount_spike", Weight: 15, UserMode: "stable", AmountMin: 3500, AmountMax: 12000, TimeAdvanceMin: 3 * time.Minute, TimeAdvanceMax: 12 * time.Minute, Channel: "web", FeatureProfile: "review"},
		{Name: "returning_country_change", Weight: 10, UserMode: "stable", AmountMin: 120, AmountMax: 1800, TimeAdvanceMin: 2 * time.Minute, TimeAdvanceMax: 10 * time.Minute, CountryShift: true, Channel: "mobile", FeatureProfile: "review"},
		{Name: "returning_destination_change", Weight: 10, UserMode: "stable", AmountMin: 75, AmountMax: 1400, TimeAdvanceMin: 2 * time.Minute, TimeAdvanceMax: 10 * time.Minute, DestinationShift: true, Channel: "mobile", FeatureProfile: "review"},
		{Name: "returning_burst", Weight: 15, UserMode: "stable_hot", AmountMin: 90, AmountMax: 900, TimeAdvanceMin: 15 * time.Second, TimeAdvanceMax: 90 * time.Second, Channel: "api", FeatureProfile: "low"},
		{Name: "synthetic_review_risk", Weight: 10, UserMode: "stable", AmountMin: 1500, AmountMax: 6000, TimeAdvanceMin: 1 * time.Minute, TimeAdvanceMax: 6 * time.Minute, CountryShift: true, DestinationShift: true, Channel: "web", FeatureProfile: "review"},
		{Name: "synthetic_high_risk", Weight: 5, UserMode: "stable_hot", AmountMin: 9000, AmountMax: 40000, TimeAdvanceMin: 30 * time.Second, TimeAdvanceMax: 3 * time.Minute, CountryShift: true, DestinationShift: true, Channel: "api", FeatureProfile: "high"},
		{Name: "brand_new_user", Weight: 5, UserMode: "new", AmountMin: 50, AmountMax: 2500, TimeAdvanceMin: 0, TimeAdvanceMax: 0, Channel: "web", FeatureProfile: "low"},
	}

	totalWeight := 0
	for _, scenario := range scenarios {
		totalWeight += scenario.Weight
	}

	return &generator{
		rng:         rng,
		runID:       runID,
		baseTime:    baseTime,
		stableUsers: stableUsers,
		hotUsers:    stableUsers[:hotCount],
		scenarios:   scenarios,
		totalWeight: totalWeight,
	}
}

func (g *generator) Next(index int, sourceSystem, sourceComponent, sourceRegion string) requestPlan {
	scenario := g.pickScenario()
	user := g.pickUser(scenario)
	amount := g.randomFloat(scenario.AmountMin, scenario.AmountMax)
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

	features := baseFeaturesForScenario(scenario.FeatureProfile, amount, user, float64(index))
	metadataLabels := map[string]string{
		"scenario":      scenario.Name,
		"user_mode":     scenario.UserMode,
		"feature_level": scenario.FeatureProfile,
		"run_id":        g.runID,
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
			Currency:           "ARS",
			Country:            country,
			Channel:            scenario.Channel,
			EventTimestamp:     eventTime.Format(time.RFC3339),
		},
		Features:       features,
		MetadataLabels: metadataLabels,
	}

	return requestPlan{
		Index:        index,
		Scenario:     scenario.Name,
		UserMode:     scenario.UserMode,
		Amount:       amount,
		FeatureLevel: scenario.FeatureProfile,
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
		return &userState{
			UserID:        fmt.Sprintf("%s_new_user_%05d", g.runID, index),
			PersonID:      fmt.Sprintf("%s_new_person_%05d", g.runID, index),
			AccountID:     fmt.Sprintf("%s_new_account_%05d", g.runID, index),
			HomeCountry:   homeCountry,
			AltCountry:    altCountry,
			PrimaryDest:   fmt.Sprintf("%s_new_dest_%05d_primary", g.runID, index),
			SecondaryDest: fmt.Sprintf("%s_new_dest_%05d_secondary", g.runID, index),
			Card1:         1700 + float64(index%400),
			Card2:         180 + float64(index%40),
			Card3:         155 + float64(index%20),
			Card5:         225 + float64(index%10),
			Addr1:         300 + float64(index%70),
			Addr2:         60 + float64(index%20),
			Dist1:         float64(index % 11),
			Dist2:         float64(index % 7),
			LastEventTime: g.baseTime,
		}
	case "stable_hot":
		return g.hotUsers[g.rng.Intn(len(g.hotUsers))]
	default:
		return g.stableUsers[g.rng.Intn(len(g.stableUsers))]
	}
}

func (g *generator) advanceEventTime(user *userState, scenario scenarioConfig) time.Time {
	advance := scenario.TimeAdvanceMin
	if scenario.TimeAdvanceMax > scenario.TimeAdvanceMin {
		extra := g.rng.Int63n(int64(scenario.TimeAdvanceMax - scenario.TimeAdvanceMin))
		advance += time.Duration(extra)
	}
	if advance <= 0 {
		advance = time.Duration(g.rng.Intn(60)+1) * time.Second
	}
	user.LastEventTime = user.LastEventTime.Add(advance)
	return user.LastEventTime
}

func baseFeaturesForScenario(profile string, amount float64, user *userState, sourceRow float64) map[string]float64 {
	features := map[string]float64{
		"source_file_row_number": sourceRow,
		"amount":                 amount,
		"card1":                  user.Card1,
		"card2":                  user.Card2,
		"card3":                  user.Card3,
		"card5":                  user.Card5,
		"addr1":                  user.Addr1,
		"addr2":                  user.Addr2,
		"dist1":                  user.Dist1,
		"dist2":                  user.Dist2,
	}

	switch profile {
	case "high":
		features["c1"] = 0.45
		features["c4"] = 0.32
		features["c14"] = 0.35
		features["v12"] = 3.5
		features["v55"] = 1.8
		features["v62"] = 4.4
		features["v187"] = 0.45
		features["v252"] = 3.1
		features["v258"] = 2.8
		features["v281"] = 1.4
		features["v294"] = 0.25
		features["id_01"] = -1.5
		features["id_17"] = 3.2
	case "review":
		features["c1"] = 0.10
		features["c4"] = 0.08
		features["c14"] = 0.12
		features["v12"] = 1.8
		features["v55"] = 1.0
		features["v62"] = 2.6
		features["v187"] = 0.20
		features["v252"] = 2.1
		features["v258"] = 1.9
		features["v281"] = 0.75
		features["v294"] = 0.05
		features["id_01"] = -3.0
		features["id_17"] = 2.1
	default:
		features["c1"] = -0.20
		features["c4"] = -0.10
		features["c14"] = -0.18
		features["v12"] = 0.4
		features["v55"] = 0.6
		features["v62"] = 0.8
		features["v187"] = 0.05
		features["v252"] = 0.7
		features["v258"] = 0.9
		features["v281"] = 0.2
		features["v294"] = -0.10
		features["id_01"] = -6.0
		features["id_17"] = 0.4
	}

	return features
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

func summarizeResults(runID string, startedAt, completedAt time.Time, addr string, requests, concurrency, stableUsers int, seed int64, plans []requestPlan, results []requestResult) loadtestSummary {
	decisionCounts := map[string]int{}
	scenarioCounts := map[string]int{}
	clientRTTs := make([]float64, 0, len(results))
	serverTotals := make([]float64, 0, len(results))
	profileLookup := make([]float64, 0, len(results))
	scoring := make([]float64, 0, len(results))
	profileUpdate := make([]float64, 0, len(results))
	errors := 0

	for _, plan := range plans {
		scenarioCounts[plan.Scenario]++
	}

	for _, result := range results {
		clientRTTs = append(clientRTTs, result.ClientRTTMs)
		if result.Error != "" {
			errors++
			continue
		}
		decisionCounts[result.Decision]++
		serverTotals = append(serverTotals, result.ServerTotalMs)
		profileLookup = append(profileLookup, result.ProfileLookupMs)
		scoring = append(scoring, result.ScoringMs)
		profileUpdate = append(profileUpdate, result.ProfileUpdateMs)
	}

	return loadtestSummary{
		RunID:          runID,
		StartedAt:      startedAt.Format(time.RFC3339),
		CompletedAt:    completedAt.Format(time.RFC3339),
		ServerAddress:  addr,
		Requests:       requests,
		Concurrency:    concurrency,
		StableUsers:    stableUsers,
		Seed:           seed,
		Errors:         errors,
		DecisionCounts: decisionCounts,
		ScenarioCounts: scenarioCounts,
		ClientRTT:      summarizeDurations(clientRTTs),
		ServerTotal:    summarizeDurations(serverTotals),
		ProfileLookup:  summarizeDurations(profileLookup),
		Scoring:        summarizeDurations(scoring),
		ProfileUpdate:  summarizeDurations(profileUpdate),
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
	}
	pair := pairs[index%len(pairs)]
	return pair[0], pair[1]
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
