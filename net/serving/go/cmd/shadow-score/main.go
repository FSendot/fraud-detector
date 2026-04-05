package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/FSendot/fraud-detector/net/serving/go/pkg/fraudruntime"
)

func main() {
	specPath := flag.String("spec", "../../../outputs/go_runtime/model_v1/runtime_spec.json", "Path to Go runtime spec JSON")
	requestPath := flag.String("request-file", "", "Optional request JSON path for one-shot scoring")
	responsePath := flag.String("response-output", "", "Optional response JSON path for one-shot scoring")
	host := flag.String("host", "127.0.0.1", "HTTP host")
	port := flag.Int("port", 8081, "HTTP port")
	flag.Parse()

	spec, err := fraudruntime.LoadRuntimeSpec(*specPath)
	if err != nil {
		log.Fatalf("load runtime spec: %v", err)
	}
	scorer := fraudruntime.NewScorer(spec)

	if *requestPath != "" {
		payload, err := loadRequestFile(*requestPath)
		if err != nil {
			log.Fatalf("load request file: %v", err)
		}
		response, err := scorer.ScoreMany(payload.RequestID, fraudruntime.ToScoreInputs(payload.Records))
		if err != nil {
			log.Fatalf("score request: %v", err)
		}
		if *responsePath == "" {
			writeJSON(os.Stdout, response)
			return
		}
		file, err := os.Create(*responsePath)
		if err != nil {
			log.Fatalf("create response output: %v", err)
		}
		defer file.Close()
		writeJSON(file, response)
		fmt.Printf("response=%s\n", *responsePath)
		return
	}

	http.Handle("/", fraudruntime.NewShadowHTTPHandler(scorer))

	addr := fmt.Sprintf("%s:%d", *host, *port)
	fmt.Printf("go_shadow_scoring_service=http://%s model_version=%s\n", addr, spec.ModelVersion)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func loadRequestFile(path string) (fraudruntime.RequestPayload, error) {
	file, err := os.Open(path)
	if err != nil {
		return fraudruntime.RequestPayload{}, err
	}
	defer file.Close()
	var payload fraudruntime.RequestPayload
	if err := json.NewDecoder(file).Decode(&payload); err != nil {
		return fraudruntime.RequestPayload{}, err
	}
	return payload, nil
}

func writeResponse(writer http.ResponseWriter, status int, payload any) {
	body, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_, _ = writer.Write(body)
}

func writeJSON(writer io.Writer, payload any) {
	body, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		log.Fatalf("marshal json: %v", err)
	}
	if _, err := writer.Write(body); err != nil {
		log.Fatalf("write json: %v", err)
	}
	if _, err := writer.Write([]byte("\n")); err != nil {
		log.Fatalf("write newline: %v", err)
	}
}
