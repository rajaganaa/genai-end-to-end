# GCP Deployment Guide

## Training (QLoRA fine-tune)
- **Instance**: `a2-ultragpu-8g` (8x A100 80GB) for 70B, or
  `a2-highgpu-4g` (4x A100 40GB) for 13B.
- Use **Spot VMs** with a Compute Engine managed instance group +
  `gcloud compute instances create --provisioning-model=SPOT` and
  checkpoint-to-GCS every N steps to survive preemption.
- Alternative: **Vertex AI Custom Training Jobs** — managed spot/preemptible
  support, automatic GCS checkpoint sync, no cluster babysitting.

## Serving (vLLM)
- **Instance**: `g2-standard-*` (L4 GPUs) for 13B at lower cost, or
  `a2-highgpu-*`/`a3-highgpu-*` (A100/H100) for 70B tensor-parallel serving.
- Deploy via **GKE with the NVIDIA GPU device plugin**; use a
  `HorizontalPodAutoscaler` driven by a custom GPU-utilization metric
  (exported via `dcgm-exporter` + Prometheus adapter).
- **Cloud Load Balancing** in front of the GKE service for TLS termination
  and multi-region failover if needed.

## Gateway (FastAPI)
- **Cloud Run** is a good fit — scales to zero, no GPU needed, and it's
  just proxying to the vLLM backend + running the (CPU-only) agent logic.
- Put **Cloud Armor** in front for WAF/rate-limiting at the edge.

## Data / Vector store
- Chroma persistence → **Persistent Disk (SSD)** attached to the GKE node,
  or migrate to **AlloyDB/Cloud SQL for Postgres with pgvector** as the
  corpus scales past what a single-node embedded store handles well.

## Cost control checklist
- [ ] Spot VMs / Preemptible VMs for training
- [ ] Cloud Run min-instances=0 for the gateway in low-traffic periods
- [ ] Committed Use Discounts for baseline GPU serving capacity
- [ ] Budget alerts per project/environment
