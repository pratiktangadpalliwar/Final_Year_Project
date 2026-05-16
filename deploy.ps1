# deploy.ps1 — PowerShell equivalent of deploy.sh.
# Prereqs: aws CLI v2, terraform >= 1.6, kubectl, helm, docker, python, npm, openssl.
# Usage:
#   .\deploy.ps1                  # full deploy
#   .\deploy.ps1 -Mode AppsOnly   # skip terraform, rebuild images + helm upgrade
#   .\deploy.ps1 -Mode DatasetsOnly # only re-upload CSVs to S3
param(
    [ValidateSet("Full", "AppsOnly", "DatasetsOnly")]
    [string]$Mode = "Full"
)

$ErrorActionPreference = "Stop"

$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$Project = "fl-demo"
$ChartDir = "k8s/fl-chart"

function Step($msg) { Write-Host "`n▶ $msg" -ForegroundColor Cyan }

# ---------- 0. Account discovery ----------
Step "0/6 Discovering AWS account"
$AccountId = (aws sts get-caller-identity --query Account --output text)
$Bucket = "$Project-$AccountId-$Region"
$EcrBase = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$Tag = (git rev-parse --short HEAD)
Write-Host "  account=$AccountId  region=$Region  bucket=$Bucket  tag=$Tag"

# ---------- 1. Terraform ----------
if ($Mode -eq "Full") {
    Step "1/6 Terraform — VPC + EKS + S3 + ECR + IAM (IRSA)"
    Push-Location infra
    terraform init -input=false
    terraform apply -auto-approve -var "project=$Project" -var "region=$Region"
    $Cluster = (terraform output -raw cluster_name)
    $ServerRoleArn = (terraform output -raw server_role_arn)
    $ClientRoleArn = (terraform output -raw client_role_arn)
    Pop-Location
    aws eks update-kubeconfig --name $Cluster --region $Region
} else {
    $Cluster = $Project
    Push-Location infra
    $ServerRoleArn = (terraform output -raw server_role_arn 2>$null)
    $ClientRoleArn = (terraform output -raw client_role_arn 2>$null)
    Pop-Location
}

# ---------- 2. ECR + Docker push ----------
if ($Mode -ne "DatasetsOnly") {
    Step "2/6 Docker build + push (server + client) → ECR"
    aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $EcrBase

    Step "2a/6 Building dashboard (vite)"
    Push-Location dashboard
    npm ci
    npm run build
    Pop-Location

    Step "2b/6 Building + pushing server image"
    docker buildx build --platform linux/amd64 `
        -t "$EcrBase/fl-server:$Tag" `
        -t "$EcrBase/fl-server:latest" `
        --push `
        -f server/Dockerfile .

    Step "2c/6 Building + pushing client image"
    docker buildx build --platform linux/amd64 `
        -t "$EcrBase/fl-client:$Tag" `
        -t "$EcrBase/fl-client:latest" `
        --push `
        -f client/Dockerfile .
}

# ---------- 3. Dataset seed ----------
Step "3/6 Uploading 7 bank CSVs to s3://$Bucket/datasets/"
Get-ChildItem dataset/bank_*.csv | ForEach-Object {
    $bank = $_.BaseName
    aws s3 cp $_.FullName "s3://$Bucket/datasets/$bank.csv" --no-progress
}

if ($Mode -eq "DatasetsOnly") {
    Write-Host "Done (datasets only)."
    exit 0
}

# ---------- 4. Validation set ----------
Step "4/6 Building + uploading held-out validation set"
$ValSet = "$env:TEMP\val_set.pkl"
python dataset/build_val_set.py --inputs dataset/bank_*.csv --frac 0.05 --out $ValSet
aws s3 cp $ValSet "s3://$Bucket/validation/val_set.pkl"

# ---------- 5. Helm install / upgrade ----------
Step "5/6 helm upgrade --install fl-demo"
$AdminPwd = if ($env:FL_ADMIN_PASSWORD) { $env:FL_ADMIN_PASSWORD } else { (openssl rand -hex 12) }
$JwtSecret = (openssl rand -hex 32)
$PwdHash = (python -c "import bcrypt; print(bcrypt.hashpw('$AdminPwd'.encode(), bcrypt.gensalt()).decode())")

helm upgrade --install fl-demo $ChartDir `
    --namespace fl --create-namespace `
    --set global.region=$Region `
    --set global.accountId=$AccountId `
    --set global.s3Bucket=$Bucket `
    --set global.imageTag=$Tag `
    --set global.ecrBase=$EcrBase `
    --set admin.passwordHash=$PwdHash `
    --set admin.jwtSecret=$JwtSecret `
    --set server.serviceAccountRoleArn=$ServerRoleArn `
    --set clients.serviceAccountRoleArn=$ClientRoleArn `
    --wait --timeout 8m

# ---------- 6. ALB DNS ----------
Step "6/6 Waiting for ALB DNS"
$Alb = ""
for ($i = 1; $i -le 40; $i++) {
    $Alb = (kubectl -n fl get ingress fl-server -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null)
    if ($Alb) { break }
    Start-Sleep -Seconds 5
}

$Context = (kubectl config current-context)
@"

==============================================================
  fl-demo deployed.
  Dashboard:   http://$(if ($Alb) { $Alb } else { '<pending>' })/
  Admin pwd:   $AdminPwd

  S3 bucket:   s3://$Bucket
  EKS:         $Context

  Tail server: kubectl -n fl logs -f deploy/fl-server
  Tail bank:   kubectl -n fl logs -f deploy/fl-client-bank-04-neobank-digital
  Teardown:    .\teardown.ps1
==============================================================
"@
