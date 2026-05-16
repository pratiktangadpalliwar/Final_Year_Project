#!/usr/bin/env bash
# deploy.sh — end-to-end stand-up of fl-demo on AWS.
# Prereqs on laptop: aws cli v2 (logged in), terraform >= 1.6, kubectl, helm, docker, python3, npm.
# Usage:
#   ./deploy.sh                  # full deploy
#   ./deploy.sh --apps-only      # skip terraform, only rebuild images + helm upgrade
#   ./deploy.sh --datasets-only  # only re-upload CSVs to S3
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PROJECT="fl-demo"
CHART_DIR="k8s/fl-chart"

step() { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }

mode="${1:-full}"

# ---------- 0. Account discovery ----------
step "0/6 Discovering AWS account"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${PROJECT}-${ACCOUNT_ID}-${REGION}"
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
TAG="$(git rev-parse --short HEAD)"
echo "  account=$ACCOUNT_ID  region=$REGION  bucket=$BUCKET  tag=$TAG"

# ---------- 1. Terraform ----------
if [[ "$mode" == "full" ]]; then
  step "1/6 Terraform — VPC + EKS + S3 + ECR + IAM (IRSA)"
  pushd infra >/dev/null
  terraform init -input=false
  terraform apply -auto-approve \
      -var "project=${PROJECT}" \
      -var "region=${REGION}"
  CLUSTER="$(terraform output -raw cluster_name)"
  SERVER_ROLE_ARN="$(terraform output -raw server_role_arn)"
  CLIENT_ROLE_ARN="$(terraform output -raw client_role_arn)"
  popd >/dev/null
  aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"
else
  CLUSTER="$PROJECT"
  SERVER_ROLE_ARN="$(cd infra && terraform output -raw server_role_arn 2>/dev/null || echo "")"
  CLIENT_ROLE_ARN="$(cd infra && terraform output -raw client_role_arn 2>/dev/null || echo "")"
fi

# ---------- 2. ECR + Docker push ----------
if [[ "$mode" != "--datasets-only" ]]; then
  step "2/6 Docker build + push (server + client) → ECR"
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_BASE"

  step "2a/6 Building dashboard (vite)"
  pushd dashboard >/dev/null
  npm ci
  npm run build
  popd >/dev/null

  step "2b/6 Building + pushing server image"
  docker buildx build --platform linux/amd64 \
      -t "${ECR_BASE}/fl-server:${TAG}" \
      -t "${ECR_BASE}/fl-server:latest" \
      --push \
      -f server/Dockerfile .

  step "2c/6 Building + pushing client image"
  docker buildx build --platform linux/amd64 \
      -t "${ECR_BASE}/fl-client:${TAG}" \
      -t "${ECR_BASE}/fl-client:latest" \
      --push \
      -f client/Dockerfile .
fi

# ---------- 3. Dataset seed ----------
step "3/6 Uploading 7 bank CSVs to s3://${BUCKET}/datasets/"
for f in dataset/bank_*.csv; do
  bank="$(basename "$f" .csv)"
  aws s3 cp "$f" "s3://${BUCKET}/datasets/${bank}.csv" --no-progress
done

[[ "$mode" == "--datasets-only" ]] && { echo "Done (datasets only)."; exit 0; }

# ---------- 4. Validation set ----------
step "4/6 Building + uploading held-out validation set"
python3 dataset/build_val_set.py \
  --inputs dataset/bank_*.csv \
  --frac 0.05 \
  --out /tmp/val_set.pkl
aws s3 cp /tmp/val_set.pkl "s3://${BUCKET}/validation/val_set.pkl"

# ---------- 5. Helm install / upgrade ----------
step "5/6 helm upgrade --install fl-demo"
ADMIN_PWD="${FL_ADMIN_PASSWORD:-$(openssl rand -hex 12)}"
JWT_SECRET="$(openssl rand -hex 32)"
PWD_HASH="$(python3 -c "import bcrypt; print(bcrypt.hashpw('${ADMIN_PWD}'.encode(), bcrypt.gensalt()).decode())")"

helm upgrade --install fl-demo "$CHART_DIR" \
  --namespace fl --create-namespace \
  --set global.region="$REGION" \
  --set global.accountId="$ACCOUNT_ID" \
  --set global.s3Bucket="$BUCKET" \
  --set global.imageTag="$TAG" \
  --set global.ecrBase="$ECR_BASE" \
  --set admin.passwordHash="$PWD_HASH" \
  --set admin.jwtSecret="$JWT_SECRET" \
  --set server.serviceAccountRoleArn="$SERVER_ROLE_ARN" \
  --set clients.serviceAccountRoleArn="$CLIENT_ROLE_ARN" \
  --wait --timeout 8m

# ---------- 6. ALB DNS ----------
step "6/6 Waiting for ALB DNS"
ALB=""
for _ in $(seq 1 40); do
  ALB="$(kubectl -n fl get ingress fl-server -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)"
  [[ -n "$ALB" ]] && break
  sleep 5
done

cat <<EOF

==============================================================
  fl-demo deployed.
  Dashboard:   http://${ALB:-<pending>}/
  Admin pwd:   ${ADMIN_PWD}

  S3 bucket:   s3://${BUCKET}
  EKS:         $(kubectl config current-context)

  Tail server: kubectl -n fl logs -f deploy/fl-server
  Tail bank:   kubectl -n fl logs -f deploy/fl-client-bank-04-neobank-digital
  Teardown:    ./teardown.sh
==============================================================
EOF
