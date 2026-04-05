package handler

import (
	"testing"

	pb "github.com/fraud-detector/processor/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestValidateRequestRequiresTraceableScoringInput(t *testing.T) {
	testCases := []struct {
		name string
		req  *pb.ProcessTransactionRequest
	}{
		{name: "nil request", req: nil},
		{name: "missing trace", req: &pb.ProcessTransactionRequest{}},
		{
			name: "missing request id",
			req: &pb.ProcessTransactionRequest{
				Trace:       &pb.RequestTrace{SourceSystem: "payments-api"},
				Transaction: &pb.TransactionContext{TransactionId: "tx_1"},
				Features:    map[string]float64{"amount": 10},
			},
		},
		{
			name: "missing source system",
			req: &pb.ProcessTransactionRequest{
				Trace:       &pb.RequestTrace{RequestId: "req_1"},
				Transaction: &pb.TransactionContext{TransactionId: "tx_1"},
				Features:    map[string]float64{"amount": 10},
			},
		},
		{
			name: "missing transaction id",
			req: &pb.ProcessTransactionRequest{
				Trace:       &pb.RequestTrace{RequestId: "req_1", SourceSystem: "payments-api"},
				Transaction: &pb.TransactionContext{},
				Features:    map[string]float64{"amount": 10},
			},
		},
		{
			name: "missing features",
			req: &pb.ProcessTransactionRequest{
				Trace:       &pb.RequestTrace{RequestId: "req_1", SourceSystem: "payments-api"},
				Transaction: &pb.TransactionContext{TransactionId: "tx_1"},
			},
		},
	}

	for _, testCase := range testCases {
		if err := validateRequest(testCase.req); status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%s: code = %v, want %v", testCase.name, status.Code(err), codes.InvalidArgument)
		}
	}
}
