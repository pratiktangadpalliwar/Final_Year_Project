variable "project" {
  description = "Project name prefix (e.g. fl-demo) used for cluster, bucket, etc."
  type        = string
  default     = "fl-demo"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "node_instance_type" {
  description = "EKS node instance type"
  type        = string
  default     = "t3.large"
}

variable "node_count" {
  description = "Desired EKS node count"
  type        = number
  default     = 3
}

variable "capacity_type" {
  description = "ON_DEMAND or SPOT"
  type        = string
  default     = "ON_DEMAND"
}
