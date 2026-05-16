#!/usr/bin/env bash
# teardown.sh — reverse of deploy.sh. Removes all AWS resources.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PROJECT="fl-demo"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${PROJECT}-${ACCOUNT_ID}-${REGION}"

step() { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }

step "1/4 helm uninstall fl-demo"
helm uninstall fl-demo -n fl 2>/dev/null || true

step "2/4 delete namespace fl"
kubectl delete ns fl --ignore-not-found

step "3/4 empty S3 bucket"
aws s3 rm "s3://${BUCKET}" --recursive 2>/dev/null || true

step "4/4 terraform destroy"
pushd infra >/dev/null
terraform destroy -auto-approve \
    -var "project=${PROJECT}" \
    -var "region=${REGION}"
popd >/dev/null

echo "Teardown complete."
