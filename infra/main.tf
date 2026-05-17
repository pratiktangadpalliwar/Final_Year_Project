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

  enable_nat_gateway   = true
  single_nat_gateway   = true # cost-saving for demo
  enable_dns_hostnames = true

  public_subnet_tags = {
    "kubernetes.io/role/elb"                 = "1"
    "kubernetes.io/cluster/${local.cluster}" = "shared"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"        = "1"
    "kubernetes.io/cluster/${local.cluster}" = "shared"
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

  enable_irsa = true
}

# --- S3 bucket (datasets, models, checkpoints, control state) ---
resource "aws_s3_bucket" "fl" {
  bucket        = local.bucket_name
  force_destroy = true # teardown.sh needs this to empty the bucket
}

resource "aws_s3_bucket_versioning" "fl" {
  bucket = aws_s3_bucket.fl.id
  versioning_configuration {
    status = "Disabled"
  }
}

resource "aws_s3_bucket_public_access_block" "fl" {
  bucket                  = aws_s3_bucket.fl.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
