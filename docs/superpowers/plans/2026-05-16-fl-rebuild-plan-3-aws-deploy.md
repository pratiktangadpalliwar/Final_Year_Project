# FL Rebuild — Plan 3: AWS Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Single `./deploy.sh` end-to-end stands up the entire FL demo on AWS (VPC + EKS + S3 + ECR + IAM + dashboard) and prints the dashboard URL + admin password. Single `./teardown.sh` reverses it.

**Architecture:** Terraform provisions VPC + EKS (3× t3.large) + ECR (fl-server + fl-client) + S3 + IRSA roles. Helm chart deploys 1 server pod (FastAPI + bundled React dashboard) + 7 bank client pods (each with an init container that fetches its CSV from S3). ALB ingress fronts the server. `deploy.sh` chains terraform → docker buildx push → dataset upload → helm install → URL print.

**Tech Stack:** Terraform 1.6+, AWS EKS module (terraform-aws-modules/eks v20), Helm 3, kubectl, AWS CLI v2, Docker Buildx, bash.

**Predecessors:** Plan 1 (FL core) + Plan 2 (dashboard) — `claude/plan-2-dashboard` branch merged into main or layered.
**Successors:** None — Plan 3 closes the rebuild.

**Reference design:** `docs/superpowers/specs/2026-05-15-fl-rebuild-design.md` sections 2 (topology), 4 (dataset flow), 7 (deploy), 10 (cost/security), 11 (acceptance #1, #2, #3, #5, #8).

---

## Verification posture

**Cannot run without AWS account.** All Plan 3 tasks validate locally via:
- `terraform init -backend=false && terraform validate` — checks HCL syntax + provider compat
- `helm lint k8s/fl-chart` — checks template syntax
- `helm template k8s/fl-chart --values <test-values>` — renders to stdout to inspect manifests
- `bash -n deploy.sh` — bash syntax check
- `shellcheck deploy.sh` (if available) — bash lint

Real `terraform apply` + `helm install` against EKS must be done by the operator on a real AWS account. Spec acceptance criteria #1–#8 verify only there.

---

## File structure (locked in this plan)

```
deploy.sh                           NEW — single-shot deploy entrypoint
teardown.sh                         NEW — single-shot nuke
README.md                           MODIFY — replace v1 instructions with Plan 3 flow

infra/                              NEW (replaces v1 terraform/)
├── versions.tf                     terraform + provider versions
├── main.tf                         VPC, EKS, S3, ALB controller IAM, OIDC provider
├── iam.tf                          IRSA roles: fl-server-sa, fl-client-sa
├── ecr.tf                          ECR repos for fl-server + fl-client
├── variables.tf                    project, region, node_size, capacity_type
└── outputs.tf                      cluster_name, ecr_base, s3_bucket, region

k8s/fl-chart/                       NEW (replaces v1 helm/)
├── Chart.yaml
├── values.yaml                     banks list, resources, secrets, fault scenarios
└── templates/
    ├── namespace.yaml
    ├── server-sa.yaml              ServiceAccount + IRSA annotation
    ├── server-deployment.yaml      1 pod, FastAPI + dashboard
    ├── server-service.yaml         ClusterIP
    ├── server-ingress.yaml         ALB ingress
    ├── server-secret.yaml          ADMIN_PASSWORD_HASH + JWT_SECRET
    ├── client-sa.yaml              ServiceAccount + IRSA
    ├── client-deployment.yaml      range over .Values.banks with init container
    └── networkpolicy.yaml          egress to fl-server svc + S3 only

dataset/
├── build_val_set.py                NEW — generates val_set.pkl from all 7 bank CSVs

scripts/                            NEW
└── cleanup_v1.sh                   one-shot script to delete v1 legacy files
```

**Files explicitly deleted by Plan 3:**
- `server/*.py` (top-level v1 files like `server/aggregator.py`, `server/round_manager.py`, etc.) — superseded by `server/app/`
- `client/*.py` (top-level v1 files) — superseded by `client/app/`
- `helm/` (v1 helm chart) — superseded by `k8s/fl-chart/`
- `terraform/` (v1 terraform) — superseded by `infra/`
- `docker-compose.yml` at repo root — superseded by `tests/e2e/compose.yml`
- `.github/workflows/build-server.yml` + `build-client.yml` — superseded by `.github/workflows/ci.yml`

---

## Phase 10 — Cleanup + val_set generator

### Task 10.1: Delete v1 legacy files

**Files:**
- Delete: `server/aggregator.py`, `server/dp_engine.py`, `server/model.py`, `server/utils.py`, `server/app.py`, `server/round_manager.py`, `server/storage.py`, `server/requirements.txt` (top-level; the new one is the same path — verify carefully), any other top-level `server/*.py`
- Delete: `client/preprocessor.py`, `client/model.py`, `client/utils.py`, `client/watcher.py`, `client/trainer.py`, `client/fl_client.py`, `client/storage_client.py`, any other top-level `client/*.py`
- Delete: `helm/` (entire dir, except don't touch `k8s/fl-chart/`)
- Delete: `terraform/` (entire dir, except don't touch `infra/`)
- Delete: `docker-compose.yml` at repo root
- Delete: `.github/workflows/build-server.yml`, `.github/workflows/build-client.yml`

**Caveat:** `server/requirements.txt` and `client/requirements.txt` were rewritten by Plan 1 Task 0.2 and ARE the new files we want. Do NOT delete them. Same with `server/Dockerfile` and `client/Dockerfile` (rewritten in Plan 1 Task 6.1/6.2).

- [ ] **Step 1: Audit which top-level server/*.py exist before deleting**

Run: `ls server/*.py 2>/dev/null && ls client/*.py 2>/dev/null`
Capture output. Anything NOT in {`server/app/...`, `server/__init__.py`, `server/Dockerfile`, `server/pyproject.toml`, `server/requirements.txt`} is fair game to delete.

- [ ] **Step 2: Run the cleanup**

```bash
# v1 server files (top-level, NOT under server/app/)
git rm -r helm/ terraform/ docker-compose.yml \
  .github/workflows/build-server.yml .github/workflows/build-client.yml 2>/dev/null || true

# v1 module files — list them by name (DO NOT use `rm server/*.py` because that
# would also remove __init__.py which we need)
for f in server/aggregator.py server/dp_engine.py server/model.py \
         server/utils.py server/app.py server/round_manager.py server/storage.py \
         server/auth.py server/health.py; do
  [[ -f "$f" ]] && git rm "$f" || true
done

for f in client/preprocessor.py client/model.py client/utils.py client/watcher.py \
         client/trainer.py client/fl_client.py client/storage_client.py; do
  [[ -f "$f" ]] && git rm "$f" || true
done
```

- [ ] **Step 3: Verify the new layout is intact**

Run: `ls server/ client/`
Expected: `__init__.py`, `Dockerfile`, `app/`, `pyproject.toml`, `requirements.txt` in each.

Run: `python -m pytest tests/unit tests/integration --tb=short`
Expected: 86 passed (no regressions; the v1 files were not imported by Plan 1/2 code).

Run: `python -m ruff check server client tests`
Expected: clean. (Remove the now-stale `server/*.py` + `client/*.py` exclusions from `pyproject.toml` ruff config in Step 4.)

- [ ] **Step 4: Remove ruff excludes for v1 files**

Edit `pyproject.toml`:

```toml
[tool.ruff]
line-length = 110
target-version = "py311"
extend-exclude = [
    "dataset/*.csv",
    "helm/**",
    "terraform/**",
]
```

(Drop `"server/*.py"` and `"client/*.py"` — they're gone now. Keep `helm/**` + `terraform/**` excluded for safety; those directories shouldn't exist post-cleanup but ruff exclude doesn't error on missing paths.)

Re-run `ruff check` → clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete v1 legacy files (server/*.py, client/*.py, helm/, terraform/, docker-compose.yml, old workflows)"
```

---

### Task 10.2: `dataset/build_val_set.py`

**Files:**
- Create: `dataset/build_val_set.py`

This script reads all 7 bank CSVs, runs the same preprocessor as the client, takes a stratified 5% sample, and pickles `{X, y}` for upload to `s3://<bucket>/validation/val_set.pkl` (deploy.sh uploads it).

- [ ] **Step 1: Write `dataset/build_val_set.py`**

```python
"""Build the held-out global validation set.

Reads all bank CSVs in dataset/, runs the same preprocessor as the client,
stratified-samples 5% of each, concatenates, pickles {X, y} to disk.

Usage:
    python dataset/build_val_set.py \
        --inputs dataset/bank_*.csv \
        --frac 0.05 \
        --out /tmp/val_set.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Use the same feature pipeline as the client preprocessor.
from client.app.preprocessor import preprocess


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="bank CSV paths (glob expands ok)")
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--out", type=str, default="/tmp/val_set.pkl")
    args = p.parse_args()

    xs: list[torch.Tensor] = []
    ys: list[np.ndarray] = []

    for path in args.inputs:
        # preprocess yields train + val splits; we ignore the internal split and
        # just use the union of features. For the held-out global set we then
        # re-sample to `frac` of THIS bank.
        x_tr, y_tr, x_v, y_v, _ = preprocess(path, val_frac=0.15)
        x_full = torch.cat([x_tr, x_v], dim=0).numpy()
        y_full = torch.cat([y_tr, y_v], dim=0).numpy()
        if y_full.sum() > 1 and y_full.sum() < len(y_full):
            x_s, _, y_s, _ = train_test_split(
                x_full, y_full, train_size=args.frac, stratify=y_full, random_state=42,
            )
        else:  # degenerate: only one class — take a non-stratified sample
            n = max(1, int(len(y_full) * args.frac))
            x_s, y_s = x_full[:n], y_full[:n]
        xs.append(torch.from_numpy(x_s))
        ys.append(y_s)

    X = torch.cat(xs, dim=0)
    y = np.concatenate(ys, axis=0)
    print(f"validation set: {len(X)} rows, {int(y.sum())} positives ({y.mean()*100:.2f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"X": X, "y": y}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test against the tiny golden CSV**

Run:
```bash
python dataset/build_val_set.py \
  --inputs tests/shared/golden_inputs/tiny_bank.csv \
  --frac 0.5 \
  --out /tmp/val_set_test.pkl
```

Expected output: `validation set: ~100 rows, N positives (X%)` and `wrote /tmp/val_set_test.pkl`.

- [ ] **Step 3: Verify the pickle is loadable**

```bash
python -c "
import pickle
with open('/tmp/val_set_test.pkl', 'rb') as f:
    d = pickle.load(f)
print(type(d['X']), d['X'].shape, type(d['y']), d['y'].shape)
"
```

Expected: `<class 'torch.Tensor'> torch.Size([N, 19]) <class 'numpy.ndarray'> (N,)`.

- [ ] **Step 4: Commit**

```bash
git add dataset/build_val_set.py
git commit -m "feat(dataset): build_val_set.py — stratified-sample 7 bank CSVs into val_set.pkl"
```

---

## Phase 11 — Terraform (infra/)

### Task 11.1: `infra/versions.tf` + providers

**Files:**
- Create: `infra/versions.tf`
- Create: `infra/variables.tf`

- [ ] **Step 1: Write `infra/versions.tf`**

```hcl
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws        = { source = "hashicorp/aws",        version = "~> 5.70" }
    kubernetes = { source = "hashicorp/kubernetes", version = "~> 2.32" }
    helm       = { source = "hashicorp/helm",       version = "~> 2.15" }
  }
}

provider "aws" {
  region = var.region
}

# Kubernetes provider configured against the EKS cluster created below.
# Helm provider used for the AWS Load Balancer Controller installation only.
data "aws_eks_cluster_auth" "this" {
  name = module.eks.cluster_name
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.this.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.this.token
  }
}
```

- [ ] **Step 2: Write `infra/variables.tf`**

```hcl
variable "project" {
  description = "Project name prefix (e.g. fl-demo) used for cluster, bucket, etc."
  type        = string
  default     = "fl-demo"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "node_instance_type" {
  description = "EKS node instance type"
  type        = string
  default     = "t3.large"
}

variable "node_count" {
  description = "Desired EKS node count"
  type        = number
  default     = 3
}

variable "capacity_type" {
  description = "ON_DEMAND or SPOT"
  type        = string
  default     = "ON_DEMAND"
}
```

- [ ] **Step 3: Validate**

```bash
cd infra
terraform init -backend=false
terraform validate
cd -
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 4: Commit**

```bash
git add infra/versions.tf infra/variables.tf
git commit -m "feat(infra): terraform versions + variables"
```

---

### Task 11.2: `infra/main.tf` — VPC + EKS + S3

**Files:**
- Create: `infra/main.tf`

- [ ] **Step 1: Write `infra/main.tf`**

```hcl
data "aws_caller_identity" "current" {}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  bucket_name = "${var.project}-${local.account_id}-${var.region}"
  cluster     = var.project
}

# --- VPC ---
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.13"

  name = "${var.project}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.region}a", "${var.region}b", "${var.region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = true  # cost-saving for demo
  enable_dns_hostnames   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb"                        = "1"
    "kubernetes.io/cluster/${local.cluster}"        = "shared"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"               = "1"
    "kubernetes.io/cluster/${local.cluster}"        = "shared"
  }
}

# --- EKS ---
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.24"

  cluster_name    = local.cluster
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    main = {
      instance_types = [var.node_instance_type]
      capacity_type  = var.capacity_type
      min_size       = var.node_count
      desired_size   = var.node_count
      max_size       = var.node_count + 2
    }
  }

  # OIDC is needed for IRSA (iam.tf consumes module.eks.oidc_provider_arn)
  enable_irsa = true
}

# --- S3 bucket (single bucket for datasets, models, checkpoints, control state) ---
resource "aws_s3_bucket" "fl" {
  bucket        = local.bucket_name
  force_destroy = true  # teardown.sh needs this to empty the bucket
}

resource "aws_s3_bucket_versioning" "fl" {
  bucket = aws_s3_bucket.fl.id
  versioning_configuration { status = "Disabled" }
}

resource "aws_s3_bucket_public_access_block" "fl" {
  bucket                  = aws_s3_bucket.fl.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

- [ ] **Step 2: Validate**

```bash
cd infra && terraform init -backend=false && terraform validate && cd -
```

Expected: success. (`terraform init` downloads providers — may take 30s.)

- [ ] **Step 3: Commit**

```bash
git add infra/main.tf
git commit -m "feat(infra): VPC + EKS (1.30, 3× node group) + S3 bucket"
```

---

### Task 11.3: `infra/iam.tf` — IRSA roles

**Files:**
- Create: `infra/iam.tf`

- [ ] **Step 1: Write `infra/iam.tf`**

```hcl
# IRSA = IAM Roles for Service Accounts. Two roles:
#   fl-server-sa: full S3 on this bucket
#   fl-client-sa: read datasets/* + read/write own updates/* + read models/*

data "aws_iam_policy_document" "server_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub"
      values   = ["system:serviceaccount:fl:fl-server-sa"]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "server_s3" {
  statement {
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.fl.arn}/*"]
  }
  statement {
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = [aws_s3_bucket.fl.arn]
  }
}

resource "aws_iam_role" "server" {
  name               = "${var.project}-server"
  assume_role_policy = data.aws_iam_policy_document.server_assume.json
}

resource "aws_iam_role_policy" "server_s3" {
  role   = aws_iam_role.server.id
  policy = data.aws_iam_policy_document.server_s3.json
}

# --- client role ---
data "aws_iam_policy_document" "client_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub"
      values   = ["system:serviceaccount:fl:fl-client-sa"]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "client_s3" {
  statement {
    effect  = "Allow"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.fl.arn}/datasets/*",
      "${aws_s3_bucket.fl.arn}/models/*",
    ]
  }
  statement {
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.fl.arn}/updates/*"]
  }
  statement {
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = [aws_s3_bucket.fl.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["datasets/*", "models/*", "updates/*"]
    }
  }
}

resource "aws_iam_role" "client" {
  name               = "${var.project}-client"
  assume_role_policy = data.aws_iam_policy_document.client_assume.json
}

resource "aws_iam_role_policy" "client_s3" {
  role   = aws_iam_role.client.id
  policy = data.aws_iam_policy_document.client_s3.json
}
```

- [ ] **Step 2: Validate**

```bash
cd infra && terraform validate && cd -
```

- [ ] **Step 3: Commit**

```bash
git add infra/iam.tf
git commit -m "feat(infra): IRSA roles (fl-server-sa full S3, fl-client-sa scoped prefixes)"
```

---

### Task 11.4: `infra/ecr.tf` + `infra/outputs.tf`

**Files:**
- Create: `infra/ecr.tf`
- Create: `infra/outputs.tf`

- [ ] **Step 1: Write `infra/ecr.tf`**

```hcl
resource "aws_ecr_repository" "server" {
  name                 = "fl-server"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_repository" "client" {
  name                 = "fl-client"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
}
```

- [ ] **Step 2: Write `infra/outputs.tf`**

```hcl
output "cluster_name"        { value = module.eks.cluster_name }
output "cluster_endpoint"    { value = module.eks.cluster_endpoint }
output "region"              { value = var.region }
output "account_id"          { value = local.account_id }
output "s3_bucket"           { value = aws_s3_bucket.fl.bucket }
output "ecr_base"            { value = "${local.account_id}.dkr.ecr.${var.region}.amazonaws.com" }
output "server_role_arn"     { value = aws_iam_role.server.arn }
output "client_role_arn"     { value = aws_iam_role.client.arn }
```

- [ ] **Step 3: Validate**

```bash
cd infra && terraform validate && cd -
```

Expected: success.

- [ ] **Step 4: Commit**

```bash
git add infra/ecr.tf infra/outputs.tf
git commit -m "feat(infra): ECR repos + terraform outputs (cluster, ecr, bucket, roles)"
```

---

## Phase 12 — Helm chart (k8s/fl-chart/)

### Task 12.1: Chart skeleton + values

**Files:**
- Create: `k8s/fl-chart/Chart.yaml`
- Create: `k8s/fl-chart/values.yaml`
- Create: `k8s/fl-chart/.helmignore`

- [ ] **Step 1: Write `Chart.yaml`**

```yaml
apiVersion: v2
name: fl-demo
description: Federated learning demo — server + 7 banks
type: application
version: 0.1.0
appVersion: "0.2.0"
```

- [ ] **Step 2: Write `values.yaml`**

```yaml
global:
  namespace: fl
  region: us-east-1
  accountId: ""              # set at install time
  s3Bucket: ""               # set at install time
  imageTag: latest           # overridden by deploy.sh to git SHA
  ecrBase: ""                # set at install time

admin:
  password: ""               # set at install time (plain text — chart hashes it)

server:
  replicaCount: 1
  resources:
    requests: { cpu: "500m", memory: "512Mi" }
    limits:   { cpu: "2000m", memory: "2Gi" }
  serviceAccountName: fl-server-sa
  serviceAccountRoleArn: ""  # set at install time from terraform output
  config:
    minNodes: "3"
    maxRounds: "50"
    quorumPct: "0.6"
    roundTimeoutS: "300"
    interRoundDelayS: "2"
    rollbackThreshold: "0.05"
    dpEpsilon: "5.0"
    dpDelta: "1e-5"
    dpClipNorm: "0.5"
    jwtTtlMinutes: "480"

ingress:
  enabled: true
  className: alb
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP":80}]'
    alb.ingress.kubernetes.io/healthcheck-path: /health

clients:
  serviceAccountName: fl-client-sa
  serviceAccountRoleArn: ""  # set at install time
  resources:
    requests: { cpu: "500m", memory: "1Gi" }
    limits:   { cpu: "2000m", memory: "4Gi" }
  pollIntervalS: "2"
  localEpochs: "10"
  batchSize: "512"
  banks:
    - id: bank_01_retail_urban
      name: Retail Urban
    - id: bank_02_corporate
      name: Corporate
    - id: bank_03_regional_rural
      name: Regional Rural
    - id: bank_04_neobank_digital
      name: Neobank Digital
    - id: bank_05_international
      name: International
    - id: bank_06_credit_union
      name: Credit Union
    - id: bank_07_investment_premium
      name: Investment Premium
```

- [ ] **Step 3: Write `.helmignore`**

```
.DS_Store
.git/
*.bak
*.tmp
```

- [ ] **Step 4: Lint**

```bash
helm lint k8s/fl-chart \
  --set global.s3Bucket=fl-test \
  --set global.accountId=123456789012 \
  --set global.ecrBase=123456789012.dkr.ecr.us-east-1.amazonaws.com \
  --set admin.password=hunter2 \
  --set server.serviceAccountRoleArn=arn:aws:iam::123456789012:role/fl-demo-server \
  --set clients.serviceAccountRoleArn=arn:aws:iam::123456789012:role/fl-demo-client
```

Expected: `1 chart(s) linted, 0 chart(s) failed`. (Templates don't exist yet, so just chart metadata lints.)

- [ ] **Step 5: Commit**

```bash
git add k8s/fl-chart/Chart.yaml k8s/fl-chart/values.yaml k8s/fl-chart/.helmignore
git commit -m "feat(helm): chart skeleton + values.yaml (banks list, resources, ingress)"
```

---

### Task 12.2: Namespace + ServiceAccounts

**Files:**
- Create: `k8s/fl-chart/templates/namespace.yaml`
- Create: `k8s/fl-chart/templates/server-sa.yaml`
- Create: `k8s/fl-chart/templates/client-sa.yaml`

- [ ] **Step 1: Write `namespace.yaml`**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: {{ .Values.global.namespace }}
```

- [ ] **Step 2: Write `server-sa.yaml`**

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ .Values.server.serviceAccountName }}
  namespace: {{ .Values.global.namespace }}
  annotations:
    eks.amazonaws.com/role-arn: {{ .Values.server.serviceAccountRoleArn | quote }}
```

- [ ] **Step 3: Write `client-sa.yaml`**

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ .Values.clients.serviceAccountName }}
  namespace: {{ .Values.global.namespace }}
  annotations:
    eks.amazonaws.com/role-arn: {{ .Values.clients.serviceAccountRoleArn | quote }}
```

- [ ] **Step 4: Render + spot-check**

```bash
helm template k8s/fl-chart \
  --set global.s3Bucket=fl-test \
  --set global.accountId=123456789012 \
  --set global.ecrBase=123456789012.dkr.ecr.us-east-1.amazonaws.com \
  --set admin.password=hunter2 \
  --set server.serviceAccountRoleArn=arn:aws:iam::123456789012:role/fl-demo-server \
  --set clients.serviceAccountRoleArn=arn:aws:iam::123456789012:role/fl-demo-client \
  | grep -E "Namespace|ServiceAccount" -A 5
```

Expected: 1 Namespace + 2 ServiceAccount entries with the right IRSA annotations.

- [ ] **Step 5: Commit**

```bash
git add k8s/fl-chart/templates/namespace.yaml k8s/fl-chart/templates/server-sa.yaml k8s/fl-chart/templates/client-sa.yaml
git commit -m "feat(helm): namespace + IRSA-annotated service accounts"
```

---

### Task 12.3: Server Deployment + Service + Ingress + Secret

**Files:**
- Create: `k8s/fl-chart/templates/server-deployment.yaml`
- Create: `k8s/fl-chart/templates/server-service.yaml`
- Create: `k8s/fl-chart/templates/server-ingress.yaml`
- Create: `k8s/fl-chart/templates/server-secret.yaml`

- [ ] **Step 1: Write `server-secret.yaml`**

The chart hashes the plaintext admin password using bcrypt at install time via a helper. Helm doesn't have bcrypt built-in, so we use Go's `htpasswd` template function — but the simplest portable approach is to expect the password is ALREADY hashed by deploy.sh (which has bcrypt via Python). Pass the pre-hashed value as `admin.passwordHash` instead.

Update `values.yaml` (modify Task 12.1's file):
Change `admin: { password: "" }` to:
```yaml
admin:
  passwordHash: ""    # bcrypt hash; deploy.sh generates with python -c 'import bcrypt;...'
  jwtSecret: ""       # random hex; deploy.sh generates
```

Now `server-secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: fl-server-auth
  namespace: {{ .Values.global.namespace }}
type: Opaque
stringData:
  ADMIN_PASSWORD_HASH: {{ .Values.admin.passwordHash | quote }}
  JWT_SECRET:          {{ .Values.admin.jwtSecret    | quote }}
```

- [ ] **Step 2: Write `server-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fl-server
  namespace: {{ .Values.global.namespace }}
spec:
  replicas: {{ .Values.server.replicaCount }}
  selector:
    matchLabels: { app: fl-server }
  template:
    metadata:
      labels: { app: fl-server }
    spec:
      serviceAccountName: {{ .Values.server.serviceAccountName }}
      containers:
        - name: fl-server
          image: "{{ .Values.global.ecrBase }}/fl-server:{{ .Values.global.imageTag }}"
          imagePullPolicy: IfNotPresent
          ports: [{ containerPort: 8080 }]
          env:
            - name: S3_BUCKET
              value: {{ .Values.global.s3Bucket | quote }}
            - name: AWS_REGION
              value: {{ .Values.global.region | quote }}
            - name: MIN_NODES
              value: {{ .Values.server.config.minNodes | quote }}
            - name: MAX_ROUNDS
              value: {{ .Values.server.config.maxRounds | quote }}
            - name: QUORUM_PCT
              value: {{ .Values.server.config.quorumPct | quote }}
            - name: ROUND_TIMEOUT_S
              value: {{ .Values.server.config.roundTimeoutS | quote }}
            - name: INTER_ROUND_DELAY_S
              value: {{ .Values.server.config.interRoundDelayS | quote }}
            - name: ROLLBACK_THRESHOLD
              value: {{ .Values.server.config.rollbackThreshold | quote }}
            - name: DP_EPSILON
              value: {{ .Values.server.config.dpEpsilon | quote }}
            - name: DP_DELTA
              value: {{ .Values.server.config.dpDelta | quote }}
            - name: DP_CLIP_NORM
              value: {{ .Values.server.config.dpClipNorm | quote }}
            - name: JWT_TTL_MINUTES
              value: {{ .Values.server.config.jwtTtlMinutes | quote }}
            - name: ADMIN_PASSWORD_HASH
              valueFrom: { secretKeyRef: { name: fl-server-auth, key: ADMIN_PASSWORD_HASH } }
            - name: JWT_SECRET
              valueFrom: { secretKeyRef: { name: fl-server-auth, key: JWT_SECRET } }
          resources:
            {{- toYaml .Values.server.resources | nindent 12 }}
          readinessProbe:
            httpGet: { path: /health, port: 8080 }
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /health, port: 8080 }
            initialDelaySeconds: 30
            periodSeconds: 10
```

- [ ] **Step 3: Write `server-service.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: fl-server
  namespace: {{ .Values.global.namespace }}
spec:
  type: ClusterIP
  selector: { app: fl-server }
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
```

- [ ] **Step 4: Write `server-ingress.yaml`**

```yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fl-server
  namespace: {{ .Values.global.namespace }}
  annotations:
    {{- range $k, $v := .Values.ingress.annotations }}
    {{ $k }}: {{ $v | quote }}
    {{- end }}
spec:
  ingressClassName: {{ .Values.ingress.className }}
  rules:
    - http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: fl-server
                port: { number: 8080 }
{{- end }}
```

- [ ] **Step 5: Render + spot-check**

```bash
helm template k8s/fl-chart \
  --set global.s3Bucket=fl-test \
  --set global.accountId=123456789012 \
  --set global.ecrBase=123456789012.dkr.ecr.us-east-1.amazonaws.com \
  --set admin.passwordHash='$2b$12$abc...' \
  --set admin.jwtSecret=test \
  --set server.serviceAccountRoleArn=arn:aws:iam::123456789012:role/fl-demo-server \
  --set clients.serviceAccountRoleArn=arn:aws:iam::123456789012:role/fl-demo-client \
  > /tmp/rendered.yaml
grep -c "kind:" /tmp/rendered.yaml
```

Expected: at least 6 kinds (Namespace, 2× SA, Deployment, Service, Ingress, Secret).

- [ ] **Step 6: Commit**

```bash
git add k8s/fl-chart/values.yaml k8s/fl-chart/templates/server-deployment.yaml k8s/fl-chart/templates/server-service.yaml k8s/fl-chart/templates/server-ingress.yaml k8s/fl-chart/templates/server-secret.yaml
git commit -m "feat(helm): fl-server deployment + service + ALB ingress + auth secret"
```

---

### Task 12.4: Client Deployment template (range over banks) + init container

**Files:**
- Create: `k8s/fl-chart/templates/client-deployment.yaml`

- [ ] **Step 1: Write the template**

```yaml
{{- $g := .Values.global -}}
{{- $c := .Values.clients -}}
{{- range $bank := $c.banks }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fl-client-{{ $bank.id | replace "_" "-" }}
  namespace: {{ $g.namespace }}
  labels: { app: fl-client, bank: {{ $bank.id }} }
spec:
  replicas: 1
  strategy: { type: Recreate }
  selector:
    matchLabels: { app: fl-client, bank: {{ $bank.id }} }
  template:
    metadata:
      labels: { app: fl-client, bank: {{ $bank.id }} }
    spec:
      serviceAccountName: {{ $c.serviceAccountName }}
      terminationGracePeriodSeconds: 60
      volumes:
        - name: work-data
          emptyDir: { sizeLimit: 2Gi }
      initContainers:
        - name: fetch-dataset
          image: amazon/aws-cli:2
          command:
            - sh
            - -c
            - |
              set -eu
              aws s3 cp s3://${S3_BUCKET}/datasets/${BANK_ID}.csv /work/data/bank.csv
              echo "fetched dataset $(wc -c </work/data/bank.csv) bytes"
          env:
            - name: S3_BUCKET
              value: {{ $g.s3Bucket | quote }}
            - name: BANK_ID
              value: {{ $bank.id | quote }}
            - name: AWS_REGION
              value: {{ $g.region | quote }}
          volumeMounts:
            - { name: work-data, mountPath: /work/data }
      containers:
        - name: fl-client
          image: "{{ $g.ecrBase }}/fl-client:{{ $g.imageTag }}"
          imagePullPolicy: IfNotPresent
          env:
            - name: BANK_ID
              value: {{ $bank.id | quote }}
            - name: BANK_NAME
              value: {{ $bank.name | quote }}
            - name: S3_BUCKET
              value: {{ $g.s3Bucket | quote }}
            - name: FL_SERVER_URL
              value: "http://fl-server.{{ $g.namespace }}.svc.cluster.local:8080"
            - name: AWS_REGION
              value: {{ $g.region | quote }}
            - name: DATASET_PATH
              value: /work/data/bank.csv
            - name: LOCAL_EPOCHS
              value: {{ $c.localEpochs | quote }}
            - name: BATCH_SIZE
              value: {{ $c.batchSize | quote }}
            - name: POLL_INTERVAL_S
              value: {{ $c.pollIntervalS | quote }}
          volumeMounts:
            - { name: work-data, mountPath: /work/data, readOnly: true }
          resources:
            {{- toYaml $c.resources | nindent 12 }}
{{- end }}
```

- [ ] **Step 2: Render + count Deployments**

```bash
helm template k8s/fl-chart \
  --set global.s3Bucket=fl-test \
  --set global.accountId=123456789012 \
  --set global.ecrBase=123456789012.dkr.ecr.us-east-1.amazonaws.com \
  --set admin.passwordHash='$2b$12$abc' --set admin.jwtSecret=test \
  --set server.serviceAccountRoleArn=arn:aws:iam::123456789012:role/x \
  --set clients.serviceAccountRoleArn=arn:aws:iam::123456789012:role/y \
  | grep -c "^kind: Deployment$"
```

Expected: `8` (1 server + 7 banks).

- [ ] **Step 3: Commit**

```bash
git add k8s/fl-chart/templates/client-deployment.yaml
git commit -m "feat(helm): client deployment (range over 7 banks) + S3 init container"
```

---

### Task 12.5: NetworkPolicy

**Files:**
- Create: `k8s/fl-chart/templates/networkpolicy.yaml`

- [ ] **Step 1: Write**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fl-default-deny
  namespace: {{ .Values.global.namespace }}
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
  ingress:
    # Allow ALB → fl-server, fl-server ← bank clients (same namespace)
    - from:
        - podSelector: {}
        - namespaceSelector:
            matchLabels: { kubernetes.io/metadata.name: ingress-nginx }
  egress:
    # Allow DNS + S3 (via VPC endpoint or NAT) + in-namespace
    - to:
        - podSelector: {}
    - to:
        - namespaceSelector: {}
      ports:
        - { protocol: UDP, port: 53 }
        - { protocol: TCP, port: 53 }
    # Egress to internet for S3 + ECR (NAT gateway routes this)
    - to:
        - ipBlock: { cidr: 0.0.0.0/0 }
      ports:
        - { protocol: TCP, port: 443 }
```

- [ ] **Step 2: Render**

Run the same `helm template` command from Task 12.4 → confirm NetworkPolicy in output.

- [ ] **Step 3: Commit**

```bash
git add k8s/fl-chart/templates/networkpolicy.yaml
git commit -m "feat(helm): NetworkPolicy (default-deny + DNS + S3 egress)"
```

---

## Phase 13 — deploy.sh + teardown.sh + README

### Task 13.1: `deploy.sh`

**Files:**
- Create: `deploy.sh` (executable)

- [ ] **Step 1: Write the script**

```bash
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
PWD_HASH="$(python3 -c "import bcrypt,sys; print(bcrypt.hashpw('${ADMIN_PWD}'.encode(), bcrypt.gensalt()).decode())")"

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
```

- [ ] **Step 2: Make executable + bash syntax check**

```bash
chmod +x deploy.sh
bash -n deploy.sh
```

Expected: no output (syntax OK).

- [ ] **Step 3: Optional shellcheck**

```bash
command -v shellcheck >/dev/null && shellcheck deploy.sh || echo "shellcheck not installed; skip"
```

Expected: warnings OK; any error must be fixed.

- [ ] **Step 4: Commit**

```bash
git add deploy.sh
git commit -m "feat(deploy): single-shot deploy.sh (terraform → docker → helm → URL)"
```

---

### Task 13.2: `teardown.sh`

**Files:**
- Create: `teardown.sh`

- [ ] **Step 1: Write**

```bash
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
```

- [ ] **Step 2: Make executable + syntax check**

```bash
chmod +x teardown.sh
bash -n teardown.sh
```

- [ ] **Step 3: Commit**

```bash
git add teardown.sh
git commit -m "feat(teardown): single-shot teardown.sh (helm + ns + s3 + terraform destroy)"
```

---

### Task 13.3: README

**Files:**
- Modify: `README.md` (replace v1 content)

- [ ] **Step 1: Write the new README**

`README.md`:

```markdown
# FL Project — Federated Fraud Detection Demo

7-bank federated learning demo with operator dashboard. FedAvg + Trimmed-Mean +
Krum aggregation, Gaussian DP, byzantine fault injection. Single `./deploy.sh`
stands up the entire stack on AWS.

## Prerequisites

On your laptop:
- AWS CLI v2 (logged in: `aws sts get-caller-identity` must succeed)
- terraform >= 1.6
- kubectl + helm 3
- docker (with buildx)
- python3 + pip
- node 20+ + npm

## Quick start

```bash
# 1. Clone + cd
git clone https://github.com/pratiktangadpalliwar/Final_Year_Project.git
cd Final_Year_Project

# 2. Full deploy (~25 min cold)
./deploy.sh

# 3. Open the URL it printed, log in with the printed admin password
```

## Modes

| Command | Time | Use |
|---|---|---|
| `./deploy.sh` | ~25 min | Full cold deploy (terraform + images + helm) |
| `./deploy.sh --apps-only` | ~5 min | Rebuild images + helm upgrade (no infra change) |
| `./deploy.sh --datasets-only` | ~2 min | Re-upload 7 bank CSVs to S3 |
| `./teardown.sh` | ~15 min | Remove everything |

## Cost

Roughly $9/day running (3× t3.large + EKS control plane + ALB + S3).
Teardown drops it to $0. Run `./teardown.sh` after each session.

## Demo flow

1. `./deploy.sh` → wait for URL.
2. Open the URL, log in. 7 BankCards appear; rounds tick every ~2s.
3. Show global AUC sparkline trending up.
4. Click **⚠ fault** on `bank_04` → choose `byzantine`. Watch the EventLog
   show "flagged bank_04" within 1-2 rounds; its trust score drops.
5. Drag a different CSV onto `bank_03` (📂 swap). Dashboard shows
   `dataset_version` increment; the bank trains on new data next round.
6. **⏸ Pause** for explanation; **▶ Resume** to continue.
7. `./teardown.sh` when done.

## Architecture

See [`docs/superpowers/specs/2026-05-15-fl-rebuild-design.md`](docs/superpowers/specs/2026-05-15-fl-rebuild-design.md)
for the full design. Plans (implementation): [`docs/superpowers/plans/`](docs/superpowers/plans/).

Components:
- **server** (`server/app/`): FastAPI + Uvicorn. FedAvg aggregator. Differential
  privacy. Boot-time S3 checkpoint restore. WebSocket fan-out. Bundles the
  React dashboard at `/static`.
- **client** (`client/app/`): one container per bank. Init container fetches
  the bank's CSV from S3 to an emptyDir. Main container preprocesses + trains
  + uploads weights via boto3 + IRSA.
- **dashboard** (`dashboard/`): React + Vite + TypeScript. Built into
  `server/app/static/` at deploy time. Bcrypt + JWT cookie auth.
- **infra** (`infra/`): Terraform. VPC + EKS + S3 + ECR + IRSA.
- **k8s/fl-chart** (`k8s/fl-chart/`): Helm chart. 1 server pod + 7 client pods
  + ALB ingress + NetworkPolicy.

## Local development (no AWS)

```bash
# Run the FL loop locally with minio + 3 banks via docker-compose
docker compose -f tests/e2e/compose.yml up --build

# Run all unit + integration tests
pip install -e .[dev] -e ./server -e ./client
pytest tests/unit tests/integration

# Run dashboard dev server (proxies API to localhost:8080)
cd dashboard && npm install && npm run dev
```

## Tests

86 unit + integration tests pass without AWS (uses moto for S3).
2 e2e tests via docker-compose + minio. CI runs all of the above on every push.

## Security posture (demo)

- Single ALB, HTTPS optional (Plan 3 ships HTTP by default for the demo URL;
  add ACM cert ARN to `values.yaml ingress.annotations` to enable HTTPS).
- Single ADMIN_PASSWORD_HASH (bcrypt) + JWT cookie. HttpOnly, Secure, SameSite=Strict.
- IRSA — zero static AWS keys in the cluster.
- Least-privilege IAM: client SA can read datasets/* + models/*, write updates/* only.
- Client→Server traffic is ClusterIP-only (no public route). NetworkPolicy
  restricts pod egress to in-namespace + DNS + HTTPS (S3/ECR).
- Differential privacy: Gaussian clip+noise at both client and server.

## Project context

Final-year project rebuild of an earlier v1 system. The rebuild solves the v1
operational issue: bank datasets are now S3-distributed via init containers
instead of `kubectl cp` into per-pod PVCs. Plans 1/2/3 document the entire
rebuild. See [`docs/superpowers/`](docs/superpowers/).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README — Plan 3 deploy flow + demo script"
```

---

## Phase 14 — Final verify + tag

### Task 14.1: Local verification of all artifacts

- [ ] **Step 1: Lint everything**

```bash
python -m pytest tests/unit tests/integration --tb=short
python -m ruff check server client tests
npm --prefix dashboard run build
npm --prefix dashboard run lint
helm lint k8s/fl-chart \
  --set global.s3Bucket=fl-test \
  --set global.accountId=123456789012 \
  --set global.ecrBase=123456789012.dkr.ecr.us-east-1.amazonaws.com \
  --set admin.passwordHash='$2b$12$abc' --set admin.jwtSecret=test \
  --set server.serviceAccountRoleArn=arn:aws:iam::123456789012:role/x \
  --set clients.serviceAccountRoleArn=arn:aws:iam::123456789012:role/y
cd infra && terraform validate && cd -
bash -n deploy.sh teardown.sh
```

Expected: all green.

- [ ] **Step 2: Render full helm output sanity check**

```bash
helm template k8s/fl-chart \
  --set global.s3Bucket=fl-test \
  --set global.accountId=123456789012 \
  --set global.ecrBase=123456789012.dkr.ecr.us-east-1.amazonaws.com \
  --set admin.passwordHash='$2b$12$abc' --set admin.jwtSecret=test \
  --set server.serviceAccountRoleArn=arn:aws:iam::123456789012:role/x \
  --set clients.serviceAccountRoleArn=arn:aws:iam::123456789012:role/y \
  > /tmp/rendered.yaml
grep "^kind:" /tmp/rendered.yaml | sort | uniq -c
```

Expected counts: 8 Deployment, 2 ServiceAccount, 1 Service, 1 Ingress, 1 Secret, 1 Namespace, 1 NetworkPolicy.

- [ ] **Step 3: Tag**

```bash
git tag plan-3-complete
git push origin claude/plan-3-aws-deploy
git push origin plan-3-complete
```

---

## What's NOT done in Plan 3 (intentionally deferred)

- **HTTPS on ALB**: requires a Route53 domain + ACM cert. The chart accepts an
  `ingress.annotations` override; documented in README. Plan 3 ships HTTP by
  default for the unlisted-URL demo.
- **AWS Load Balancer Controller installation**: the ALB ingress class
  `alb` requires the controller to be installed in the cluster. Plan 3
  assumes the operator runs `eksctl utils associate-iam-oidc-provider` + the
  controller helm install once per cluster, or extends `infra/main.tf` with a
  `helm_release` resource for the controller. Documented as a Phase 14 README
  follow-up if needed.
- **Cluster Autoscaler / Karpenter**: not installed. Fixed-size node group of 3.
- **Real CloudWatch dashboard / alarms**: only structured-json log shipping
  via fluent-bit DaemonSet (terraform addon enabled by default in EKS module).
  Custom CloudWatch dashboards are out of scope.
- **Multi-region failover**: single region.
- **Spot fallback policy**: `capacity_type` variable supports SPOT but no
  on-demand fallback policy. Acceptable for demo.
- **Cosign / image signature verification**: images pushed unsigned.
- **Secrets via AWS Secrets Manager / External Secrets Operator**: admin
  password lives in a K8s Secret (etcd KMS-encrypted on EKS by default).

---

*End of Plan 3.*
