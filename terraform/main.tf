################################################################################
# terraform/main.tf
# Resilient FL on AWS — EKS + S3 + SQS + DynamoDB + IAM
################################################################################

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }

  # Remote state — replace with your S3 bucket + DynamoDB table for locking
  backend "s3" {
    bucket         = "fl-terraform-state"
    key            = "fl-project/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "fl-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "fl-project"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Data sources ───────────────────────────────────────────────────
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ── VPC ───────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets

  enable_nat_gateway     = true
  single_nat_gateway     = false   # one per AZ for HA
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # Required tags for EKS load balancers
  public_subnet_tags = {
    "kubernetes.io/role/elb"                    = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# ── EKS Cluster ───────────────────────────────────────────────────
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.29"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # Enable IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Cluster add-ons
  cluster_addons = {
    coredns                = { most_recent = true }
    kube-proxy             = { most_recent = true }
    vpc-cni                = { most_recent = true }
    aws-ebs-csi-driver     = { most_recent = true }
  }

  # ── Node Groups ────────────────────────────────────────────────
  eks_managed_node_groups = {

    # FL Server node group — always on, multi-AZ
    fl_server_nodes = {
      name           = "fl-server-nodes"
      instance_types = [var.server_instance_type]
      min_size       = 2
      max_size       = 6
      desired_size   = 2
      disk_size      = 50

      labels = {
        role       = "fl-server"
        "bank-node"= "false"
      }

      subnet_ids = module.vpc.private_subnets
    }

    # Bank node group — one per bank (scalable)
    fl_bank_nodes = {
      name           = "fl-bank-nodes"
      instance_types = [var.bank_instance_type]
      min_size       = 0
      max_size       = 15        # supports up to 15 bank nodes
      desired_size   = var.num_banks
      disk_size      = 100       # large disk for CSV + model storage

      labels = {
        role       = "fl-client"
        "bank-node"= "true"
      }

      taints = [{
        key    = "bank-node"
        value  = "true"
        effect = "NO_SCHEDULE"    # only bank pods scheduled here
      }]

      subnet_ids = module.vpc.private_subnets
    }
  }
}

# ── Outputs ───────────────────────────────────────────────────────
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "vpc_id" {
  value = module.vpc.vpc_id
}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}
