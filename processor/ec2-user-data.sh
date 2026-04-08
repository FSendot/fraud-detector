#!/bin/bash
set -e

yum update -y
yum install -y aws-cli

# Bajar bundle de S3
aws s3 cp s3://amzn-s3-fraud-detector-artifacts/processor/fraud-processor-bundle-linux-amd64-20260407T230454Z.tar.gz /opt/bundle.tar.gz

# Descomprimir
tar -xzf /opt/bundle.tar.gz -C /opt/
cd /opt/fraud-processor-bundle-linux-amd64-20260407T230454Z

# Configurar .env
cat > .env <<EOF
GRPC_PORT=50051
FRAUD_RUNTIME_SPEC_PATH=./ml/model_v1/runtime_spec.json
AWS_REGION=us-east-1
DYNAMODB_TABLE_NAME=user_profiles
RDS_HOST=rds-fraude-detector.cls0lky8mvkm.us-east-1.rds.amazonaws.com
RDS_PORT=5432
RDS_USER=postgres
RDS_PASSWORD=REEMPLAZAR_CON_PASSWORD
RDS_DBNAME=fraud
EOF

# Configurar systemd para que arranque automaticamente
cat > /etc/systemd/system/fraud-processor.service <<EOF
[Unit]
Description=Fraud Processor gRPC Service
After=network.target

[Service]
WorkingDirectory=/opt/fraud-processor-bundle-linux-amd64-20260407T230454Z
EnvironmentFile=/opt/fraud-processor-bundle-linux-amd64-20260407T230454Z/.env
ExecStart=/opt/fraud-processor-bundle-linux-amd64-20260407T230454Z/bin/fraud-processor
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable fraud-processor
systemctl start fraud-processor
