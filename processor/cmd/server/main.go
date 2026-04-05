package main

import (
	"log"
	"net"
	"os"

	"google.golang.org/grpc"

	pb "github.com/fraud-detector/processor/proto"
	"github.com/fraud-detector/processor/internal/handler"
)

func main() {
	port := os.Getenv("GRPC_PORT")
	if port == "" {
		port = "50051"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterFraudProcessorServer(srv, handler.NewFraudHandler())

	log.Printf("fraud processor listening on :%s", port)
	if err := srv.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
