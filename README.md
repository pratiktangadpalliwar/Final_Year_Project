# Resilient Federated Learning — Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes (EKS)                         │
│                                                              │
│  FL Server Pods (×2, multi-AZ)    Bank Node Pods (×7)        │
│  ┌─────────────────────┐          ┌────────┐ ┌────────┐      │
│  │  REST API :8080     │◄────────►│ Bank 1 │ │ Bank 2 │ ...  │
│  │  FedAvg + Krum      │          │ Docker │ │ Docker │      │
│  │  Checkpointing      │          └────────┘ └────────┘      │
│  │  Rollback           │                ↕                    │
│  └─────────────────────┘         AWS S3 / SQS               │
└─────────────────────────────────────────────────────────────┘
```

**Bank operator experience:**
1. Drop `bank_XX.csv` into `/data/input/`
2. Done — container detects, trains, uploads, notifies server automatically

---

## Prerequisites

| Tool        | Version  | Install |
|-------------|----------|---------|
| Docker      | ≥ 24     | https://docs.docker.com/get-docker/ |
| kubectl     | ≥ 1.28   | https://kubernetes.io/docs/tasks/tools/ |
| Helm        | ≥ 3.13   | https://helm.sh/docs/intro/install/ |
| Terraform   | ≥ 1.5    | https://developer.hashicorp.com/terraform/install |
| AWS CLI     | ≥ 2.0    | https://aws.amazon.com/cli/ |

---

## Option A — Local Testing (Docker Compose, no AWS)

**Fastest way to test the full system locally.**

```bash
# 1. Clone and build
git clone https://github.com/YOUR_USERNAME/fl-project
cd fl-project
docker-compose build

# 2. Start everything
docker-compose up -d

# 3. Check server is healthy
curl http://localhost:8080/health

# 4. Trigger training on Bank 01
#    Copy a bank CSV into the input volume
docker cp data/bank_01_retail_urban.csv fl-bank-01:/data/input/

# 5. Watch training logs
docker-compose logs -f fl-bank-01

# 6. Check FL server metrics
curl http://localhost:8080/metrics | python -m json.tool

# 7. Repeat for other banks
docker cp data/bank_02_corporate.csv fl-bank-02:/data/input/
docker cp data/bank_03_regional_rural.csv fl-bank-03:/data/input/
# ... etc

# 8. Stop
docker-compose down
```

---

## Option B — Production on AWS EKS

### Step 1 — AWS Infrastructure (Terraform)

```bash
cd terraform

# Configure your variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your AWS account details

# Deploy infrastructure (~10 minutes)
terraform init
terraform plan
terraform apply

# Save outputs for next steps
terraform output
```

### Step 2 — Build & Push Docker Images

```bash
# Set your Docker Hub username
export DOCKERHUB_USERNAME=your_username

# Build and push server
docker build -t $DOCKERHUB_USERNAME/fl-server:latest ./server
docker push $DOCKERHUB_USERNAME/fl-server:latest

# Build and push client
docker build -t $DOCKERHUB_USERNAME/fl-client:latest ./client
docker push $DOCKERHUB_USERNAME/fl-client:latest
```

### Step 3 — Configure kubectl

```bash
aws eks update-kubeconfig \
  --name fl-eks-cluster \
  --region us-east-1
```

### Step 4 — Create Secrets

```bash
kubectl create namespace federated-learning

kubectl create secret generic fl-secrets \
  --from-literal=AWS_ACCESS_KEY_ID=YOUR_KEY \
  --from-literal=AWS_SECRET_ACCESS_KEY=YOUR_SECRET \
  --from-literal=AWS_REGION=us-east-1 \
  -n federated-learning
```

### Step 5 — Deploy with Helm

```bash
cd helm

# Edit values.yaml — set your Docker Hub username, AWS account ID, domain
helm install fl-project . \
  --namespace federated-learning \
  --create-namespace \
  --set global.dockerUsername=$DOCKERHUB_USERNAME \
  --set global.awsAccountId=$(aws sts get-caller-identity --query Account --output text) \
  --wait

# Verify all pods are running
kubectl get pods -n federated-learning
```

### Step 6 — Trigger Training (Bank Operator)

```bash
# Get the PVC name for bank 01
kubectl get pvc -n federated-learning

# Copy CSV to bank node input volume
kubectl cp data/bank_01_retail_urban.csv \
  federated-learning/$(kubectl get pod -n federated-learning -l bank=bank-01-retail-urban -o name):/data/input/

# Watch training
kubectl logs -f \
  -n federated-learning \
  -l bank=bank-01-retail-urban
```

---

## Adding a New Bank Node

```bash
# 1. Edit helm/values.yaml — add entry to clients list:
#    - id:    bank_08_new_bank
#      name:  "New Bank Name"
#      fault: none
#      resources: ...

# 2. Upgrade Helm release — new deployment created automatically
helm upgrade fl-project ./helm -n federated-learning

# 3. Copy CSV to trigger training
kubectl cp data/bank_08_new_bank.csv \
  federated-learning/POD_NAME:/data/input/
```

## Removing a Bank Node

```bash
# Remove from values.yaml clients list, then:
helm upgrade fl-project ./helm -n federated-learning
# Kubernetes removes the deployment and PVC automatically
```

---

## Monitoring

```bash
# FL Server metrics
curl http://localhost:8080/metrics

# Round status
curl http://localhost:8080/round/status

# Manual rollback (if needed)
curl -X POST http://localhost:8080/admin/rollback \
  -H "Content-Type: application/json" \
  -d '{"target_round": 10}'

# Kubernetes dashboard
kubectl get all -n federated-learning

# Pod logs
kubectl logs -f deployment/fl-server -n federated-learning
```

---

## GitHub Actions Secrets Required

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `AWS_REGION` | e.g. us-east-1 |
| `AWS_ACCOUNT_ID` | 12-digit AWS account ID |
| `EKS_CLUSTER_NAME` | fl-eks-cluster |
| `S3_BUCKET` | fl-models-prod |

---

## Fault Tolerance Reference

| Bank | Fault Scenario | Round | System Response |
|------|---------------|-------|-----------------|
| Bank 01 | Node crash | 25 | Server continues with 6 banks |
| Bank 02 | Straggler | 15+ | Timeout policy skips late update |
| Bank 03 | Network partition | 30–38 | Excluded, re-syncs on recovery |
| Bank 04 | Byzantine | 20+ | Krum aggregation rejects update |
| Bank 05 | Dropout & rejoin | 18–28 | Excluded, re-admits at round 28 |
| Bank 06 | None | — | Stable baseline |
| Bank 07 | None | — | Stable baseline |
