# AWS Deployment Guide

## Training (QLoRA fine-tune)
- **Instance**: `p4d.24xlarge` (8x A100 40GB) for 70B QLoRA, or
  `g5.12xlarge` (4x A10G 24GB) for 13B QLoRA.
- Use **EC2 Spot** for training runs (checkpoint every 200 steps so spot
  interruption only loses partial progress) — typically 60-70% cost savings.
- Store checkpoints and datasets in **S3**; mount via `s3fs` or sync
  periodically with `aws s3 sync`.
- Alternative: **SageMaker Training Jobs** with a custom container built
  from `requirements.txt` — gives you managed spot training + automatic
  checkpoint sync to S3 out of the box.

## Serving (vLLM)
- **Instance**: `g5.2xlarge`/`g5.12xlarge` for 13B (1-2 GPUs), `p4d.24xlarge`
  or `g5.48xlarge` for 70B (4-8 GPUs with tensor parallelism).
- Deploy vLLM inside a Docker container behind an **Application Load
  Balancer**; run on **ECS with GPU-enabled EC2 launch type** or **EKS**
  with the NVIDIA device plugin.
- Autoscaling: scale on **GPU utilization** (via CloudWatch custom metric
  from `nvidia-smi`) rather than CPU/request-count, since GPU saturation is
  the actual bottleneck for vLLM throughput.
- Use a **Reserved Instance or Savings Plan** for the baseline serving
  capacity, and Spot/On-Demand for burst scaling.

## Gateway (FastAPI)
- Deploy on **ECS Fargate** (CPU-only, cheap) — it's a thin orchestration
  layer, no GPU needed here since inference happens on the vLLM instances.
- Put **API Gateway + WAF** in front for auth, rate limiting at the edge,
  and basic bot/abuse protection before requests even reach FastAPI.

## Data / Vector store
- Chroma persistence directory → **EBS volume** (io2) for low-latency
  reads, or migrate to **pgvector on RDS Postgres** for a managed,
  HA-capable option as the corpus grows.

## Cost control checklist
- [ ] Spot instances for training
- [ ] Autoscaling floor of 0-1 vLLM replicas outside business hours if
      usage is bursty (accept cold-start latency trade-off)
- [ ] CloudWatch billing alarms per environment (dev/staging/prod)
- [ ] S3 lifecycle policy to move old checkpoints to Glacier
