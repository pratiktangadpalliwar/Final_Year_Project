# teardown.ps1 — PowerShell equivalent of teardown.sh. Removes all AWS resources.
$ErrorActionPreference = "Stop"

$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$Project = "fl-demo"
$AccountId = (aws sts get-caller-identity --query Account --output text)
$Bucket = "$Project-$AccountId-$Region"

function Step($msg) { Write-Host "`n▶ $msg" -ForegroundColor Cyan }

Step "1/4 helm uninstall fl-demo"
helm uninstall fl-demo -n fl 2>$null

Step "2/4 delete namespace fl"
kubectl delete ns fl --ignore-not-found

Step "3/4 empty S3 bucket"
aws s3 rm "s3://$Bucket" --recursive 2>$null

Step "4/4 terraform destroy"
Push-Location infra
terraform destroy -auto-approve -var "project=$Project" -var "region=$Region"
Pop-Location

Write-Host "Teardown complete."
