################################################################################
# terraform/storage.tf  —  S3 + DynamoDB + SQS
################################################################################

# ── S3 Bucket for FL model weights and checkpoints ────────────────
resource "aws_s3_bucket" "fl_models" {
  bucket        = var.s3_bucket_name
  force_destroy = true

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "fl_models" {
  bucket = aws_s3_bucket.fl_models.id
  versioning_configuration {
    status = "Enabled"     # keep version history of every global model
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "fl_models" {
  bucket = aws_s3_bucket.fl_models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "fl_models" {
  bucket                  = aws_s3_bucket.fl_models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle: auto-delete client update files after 7 days
resource "aws_s3_bucket_lifecycle_configuration" "fl_models" {
  bucket = aws_s3_bucket.fl_models.id

  rule {
    id     = "expire-client-updates"
    status = "Enabled"
    filter { prefix = "updates/" }
    expiration { days = 7 }
  }

  rule {
    id     = "keep-global-models-90-days"
    status = "Enabled"
    filter { prefix = "models/" }
    expiration { days = 90 }
  }
}

# ── DynamoDB for FL round state ───────────────────────────────────
resource "aws_dynamodb_table" "fl_rounds" {
  name           = var.dynamodb_table_name
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "round_id"

  attribute {
    name = "round_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

# ── SQS Queue for FL update notifications ─────────────────────────
resource "aws_sqs_queue" "fl_updates" {
  name                       = var.sqs_queue_name
  visibility_timeout_seconds = 300
  message_retention_seconds  = 86400    # 1 day
  receive_wait_time_seconds  = 20       # long polling

  # Dead-letter queue for failed messages
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.fl_dlq.arn
    maxReceiveCount     = 3
  })
}

resource "aws_sqs_queue" "fl_dlq" {
  name                      = "${var.sqs_queue_name}-dlq"
  message_retention_seconds = 1209600   # 14 days
}

# ── Outputs ───────────────────────────────────────────────────────
output "s3_bucket_name" {
  value = aws_s3_bucket.fl_models.bucket
}

output "dynamodb_table_name" {
  value = aws_dynamodb_table.fl_rounds.name
}

output "sqs_queue_url" {
  value = aws_sqs_queue.fl_updates.url
}
