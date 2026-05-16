output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "region" {
  value = var.region
}

output "account_id" {
  value = local.account_id
}

output "s3_bucket" {
  value = aws_s3_bucket.fl.bucket
}

output "ecr_base" {
  value = "${local.account_id}.dkr.ecr.${var.region}.amazonaws.com"
}

output "server_role_arn" {
  value = aws_iam_role.server.arn
}

output "client_role_arn" {
  value = aws_iam_role.client.arn
}
