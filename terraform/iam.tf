################################################################################
# terraform/iam.tf  —  IAM roles for FL Server and Bank Node pods (IRSA)
################################################################################

locals {
  oidc_provider     = module.eks.oidc_provider
  oidc_provider_arn = module.eks.oidc_provider_arn
  account_id        = data.aws_caller_identity.current.account_id
  namespace         = "federated-learning"
}

# ══════════════════════════════════════════════════════════════════
# FL SERVER ROLE
# Permissions: S3 full access to fl-models, DynamoDB read/write
# ══════════════════════════════════════════════════════════════════
resource "aws_iam_role" "fl_server" {
  name = "fl-server-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = {
        Federated = local.oidc_provider_arn
      }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_provider}:sub" = "system:serviceaccount:${local.namespace}:fl-server-sa"
          "${local.oidc_provider}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "fl_server_policy" {
  name   = "fl-server-policy"
  role   = aws_iam_role.fl_server.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 — full access to model bucket
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject",
                    "s3:ListBucket", "s3:GetObjectVersion"]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}",
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      },
      # DynamoDB — round state
      {
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan"]
        Resource = "arn:aws:dynamodb:${var.aws_region}:${local.account_id}:table/${var.dynamodb_table_name}"
      },
      # SQS — send notifications
      {
        Effect   = "Allow"
        Action   = ["sqs:SendMessage", "sqs:GetQueueAttributes"]
        Resource = aws_sqs_queue.fl_updates.arn
      },
      # CloudWatch Logs
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/fl-project/*"
      }
    ]
  })
}

# ══════════════════════════════════════════════════════════════════
# FL CLIENT ROLE
# Permissions: S3 put (upload updates), S3 get (download global model)
# Minimal permissions — clients cannot read other clients' updates
# ══════════════════════════════════════════════════════════════════
resource "aws_iam_role" "fl_client" {
  name = "fl-client-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = {
        Federated = local.oidc_provider_arn
      }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${local.oidc_provider}:sub" = "system:serviceaccount:${local.namespace}:fl-client-sa"
          "${local.oidc_provider}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "fl_client_policy" {
  name   = "fl-client-policy"
  role   = aws_iam_role.fl_client.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3 — download global model
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "arn:aws:s3:::${var.s3_bucket_name}/models/*"
      },
      # S3 — upload own updates only (path-restricted by bank_id)
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "arn:aws:s3:::${var.s3_bucket_name}/updates/*"
      },
      # SQS — receive task messages
      {
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes"]
        Resource = aws_sqs_queue.fl_updates.arn
      }
    ]
  })
}

# ── Outputs ───────────────────────────────────────────────────────
output "fl_server_role_arn" {
  value = aws_iam_role.fl_server.arn
}

output "fl_client_role_arn" {
  value = aws_iam_role.fl_client.arn
}
