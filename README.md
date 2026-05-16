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
git clone https://github.com/pratiktangadpalliwar/Final_Year_Project.git
cd Final_Year_Project

# Linux / mac / git-bash on Windows:
./deploy.sh

# Native PowerShell (Windows):
.\deploy.ps1

# wait ~25 min, open the URL it printed, log in with the printed admin password
```

## Modes

bash:

| Command | Time | Use |
|---|---|---|
| `./deploy.sh` | ~25 min | Full cold deploy (terraform + images + helm) |
| `./deploy.sh --apps-only` | ~5 min | Rebuild images + helm upgrade (no infra change) |
| `./deploy.sh --datasets-only` | ~2 min | Re-upload 7 bank CSVs to S3 |
| `./teardown.sh` | ~15 min | Remove everything |

PowerShell (Windows equivalents):

| Command | Equivalent |
|---|---|
| `.\deploy.ps1` | full deploy |
| `.\deploy.ps1 -Mode AppsOnly` | rebuild + helm upgrade |
| `.\deploy.ps1 -Mode DatasetsOnly` | re-upload CSVs |
| `.\teardown.ps1` | nuke everything |

First time on Windows you may need to allow script execution:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Cost

Roughly $9/day running (3× t3.large + EKS control plane + ALB + S3).
Teardown drops it to $0. Run `./teardown.sh` after each session.

## Demo flow

1. `./deploy.sh` → wait for URL.
2. Open the URL, log in. 7 BankCards appear; rounds tick every ~2s.
3. Global AUC sparkline trends up.
4. Click **⚠ fault** on `bank_04` → choose `byzantine`. EventLog shows
   "flagged bank_04" within 1–2 rounds; trust score drops.
5. Drag a different CSV onto `bank_03` (📂 swap). Dashboard shows
   `dataset_version` increment; bank trains on new data next round.
6. **⏸ Pause** for explanation; **▶ Resume** to continue.
7. `./teardown.sh` when done.

## Architecture

See [`docs/superpowers/specs/2026-05-15-fl-rebuild-design.md`](docs/superpowers/specs/2026-05-15-fl-rebuild-design.md)
for full design. Plans: [`docs/superpowers/plans/`](docs/superpowers/plans/).

Components:
- **server** (`server/app/`): FastAPI + Uvicorn. FedAvg aggregator. Differential
  privacy. Boot-time S3 checkpoint restore. WebSocket fan-out. Bundles React
  dashboard at `/static`.
- **client** (`client/app/`): one container per bank. Init container fetches
  bank's CSV from S3 to emptyDir. Main container preprocesses + trains
  + uploads weights via boto3 + IRSA.
- **dashboard** (`dashboard/`): React + Vite + TypeScript. Built into
  `server/app/static/` at deploy time. Bcrypt + JWT cookie auth.
- **infra** (`infra/`): Terraform. VPC + EKS + S3 + ECR + IRSA.
- **k8s/fl-chart** (`k8s/fl-chart/`): Helm chart. 1 server pod + 7 client pods
  + ALB ingress + NetworkPolicy.

## Local development (no AWS)

```bash
# Run FL loop locally with minio + 3 banks via docker-compose
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

- Single ALB. HTTPS optional (add ACM cert ARN to
  `values.yaml ingress.annotations` to enable).
- Single `ADMIN_PASSWORD_HASH` (bcrypt) + JWT cookie. HttpOnly, Secure,
  SameSite=Strict.
- IRSA — zero static AWS keys in the cluster.
- Least-privilege IAM: client SA can read `datasets/*` + `models/*`, write
  `updates/*` only.
- Client→Server traffic is ClusterIP-only (no public route). NetworkPolicy
  restricts pod egress to in-namespace + DNS + HTTPS (S3/ECR).
- Differential privacy: Gaussian clip+noise at both client and server.

## Project context

Final-year project rebuild of an earlier v1 system. The rebuild solves the v1
operational issue: bank datasets are now S3-distributed via init containers
instead of `kubectl cp` into per-pod PVCs. Plans 1/2/3 document the entire
rebuild. See [`docs/superpowers/`](docs/superpowers/).

## Cluster add-on prerequisite

After `./deploy.sh` runs the first time, you may need to install the AWS Load
Balancer Controller into the cluster so the ALB Ingress class resolves. Run
once per cluster:

```bash
# Replace <CLUSTER> with the cluster name from terraform output
eksctl utils associate-iam-oidc-provider --cluster <CLUSTER> --approve
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system --set clusterName=<CLUSTER> --set serviceAccount.create=true
```

(Future iteration: bundle this into `infra/main.tf` as a `helm_release`.)
