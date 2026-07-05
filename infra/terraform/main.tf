# Skeleton Terraform for the vLLM serving GPU instance on AWS.
# Fill in variables per-environment (dev/staging/prod) via *.tfvars.
# This provisions compute only -- pair with your own VPC/subnet/ALB
# modules for a full production network stack.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "region" {
  default = "us-east-1"
}

variable "instance_type" {
  description = "g5.12xlarge for 13B, g5.48xlarge or p4d.24xlarge for 70B"
  default     = "g5.12xlarge"
}

variable "lora_adapter_s3_uri" {
  description = "S3 path to the trained LoRA adapter to load at boot"
  type        = string
}

provider "aws" {
  region = var.region
}

resource "aws_instance" "vllm_server" {
  ami           = "ami-0abcd1234examplegpu" # replace with current Deep Learning AMI
  instance_type = var.instance_type

  root_block_device {
    volume_size = 200
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e
    aws s3 sync ${var.lora_adapter_s3_uri} /opt/checkpoints/medical-lora
    docker run -d --gpus all -p 8001:8001 \
      -e LORA_PATH=/opt/checkpoints/medical-lora \
      -v /opt/checkpoints:/opt/checkpoints \
      medassist-vllm:latest
  EOF

  tags = {
    Name = "medassist-vllm-server"
  }
}

output "vllm_server_public_ip" {
  value = aws_instance.vllm_server.public_ip
}
