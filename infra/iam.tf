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
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.fl.arn}/*"]
  }
  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
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
    effect = "Allow"
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
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
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
