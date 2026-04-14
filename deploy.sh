#!/usr/bin/env bash
# deploy.sh — one-shot bootstrap for FL demo stack.
# Run after `terraform apply`. Idempotent; safe to re-run.
#
# Usage:
#   ./deploy.sh                       # full install
#   ./deploy.sh --upgrade-only        # skip kubeconfig + access entry
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${EKS_CLUSTER:-fl-eks-cluster}"
NAMESPACE="${FL_NAMESPACE:-federated-learning}"
RELEASE="${HELM_RELEASE:-fl}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
CALLER_ARN="$(aws sts get-caller-identity --query Arn --output text)"

echo "==> Account: $ACCOUNT_ID | Cluster: $CLUSTER | Region: $REGION"

if [[ "${1:-}" != "--upgrade-only" ]]; then
  echo "==> Updating kubeconfig"
  aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"

  echo "==> Ensuring EKS access entry for $CALLER_ARN"
  aws eks create-access-entry \
    --cluster-name "$CLUSTER" --region "$REGION" \
    --principal-arn "$CALLER_ARN" 2>/dev/null || true
  aws eks associate-access-policy \
    --cluster-name "$CLUSTER" --region "$REGION" \
    --principal-arn "$CALLER_ARN" \
    --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
    --access-scope type=cluster 2>/dev/null || true
fi

echo "==> Creating namespace (if missing)"
kubectl create ns "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo "==> Helm deploy (install or upgrade)"
helm upgrade --install "$RELEASE" ./helm \
  -n "$NAMESPACE" \
  --set global.awsAccountId="$ACCOUNT_ID" \
  --set global.awsRegion="$REGION" \
  --wait --timeout 10m

echo "==> Verify S3_BUCKET env"
kubectl exec -n "$NAMESPACE" deploy/fl-server -- sh -c 'env | grep S3_BUCKET'

echo "==> Done. Rollout status:"
kubectl rollout status deployment/fl-server -n "$NAMESPACE"
