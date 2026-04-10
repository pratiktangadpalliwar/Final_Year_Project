################################################################################
# terraform/variables.tf
################################################################################

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "fl-project"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "fl-eks-cluster"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDRs (one per AZ)"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDRs (one per AZ)"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "server_instance_type" {
  description = "EC2 instance type for FL server nodes"
  type        = string
  default     = "t3.large"
}

variable "bank_instance_type" {
  description = "EC2 instance type for bank nodes (needs more RAM for PyTorch)"
  type        = string
  default     = "t3.xlarge"
}

variable "num_banks" {
  description = "Initial number of bank nodes to provision"
  type        = number
  default     = 7
}

variable "s3_bucket_name" {
  description = "S3 bucket name for FL model storage"
  type        = string
  default     = "fl-models-prod"
}

variable "dynamodb_table_name" {
  description = "DynamoDB table for FL round state"
  type        = string
  default     = "fl-rounds"
}

variable "sqs_queue_name" {
  description = "SQS queue for FL update notifications"
  type        = string
  default     = "fl-updates"
}
