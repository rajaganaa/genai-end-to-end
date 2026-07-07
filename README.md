# MedAssist-GenAI: Medical Diagnosis Support Assistant

A production-oriented GenAI system combining **RAG + Agents + LoRA/QLoRA
fine-tuning + vLLM serving**, built as a clinical **decision-support** tool
(not an autonomous diagnostician).

> ⚠️ **Scope & Safety**: This system assists clinicians / triages patients
> toward appropriate care. It never outputs a final diagnosis without
> disclaimers, always cites retrieved sources, and escalates
> emergency-pattern symptoms to "seek immediate care" regardless of model
> output. See `agents/prompts.py` and `eval/test_cases.jsonl` for the
> guardrails and safety test suite.

---

## 1. Architecture

```
                                   ┌─────────────────────────┐
                                   │        Client(s)         │
                                   │  (Web app / EHR plugin)  │
                                   └────────────┬─────────────┘
                                                │ HTTPS
                                                ▼
                                   ┌─────────────────────────┐
                                   │      FastAPI Gateway      │
                                   │  (auth, rate limit, PII   │
                                   │   scrub, streaming, logs) │
                                   └────────────┬─────────────┘
                                                │
                              ┌─────────────────┼──────────────────┐
                              ▼                                    ▼
                 ┌─────────────────────┐                ┌──────────────────────┐
                 │   Agent Orchestrator │                │   Safety Guardrail    │
                 │  (LangChain Agent    │◄──────────────►│   Layer (pre/post)    │
                 │   Executor + Tools)  │                │  - emergency detector │
                 └─────────┬───────────┘                │  - PII redaction      │
                           │                             │  - refusal policy     │
             ┌─────────────┼──────────────┐              └──────────────────────┘
             ▼             ▼              ▼
   ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐
   │  RAG Tool     │ │ Drug         │ │ Triage Calculator │
   │ (retriever +  │ │ Interaction  │ │ Tool (rule-based) │
   │  reranker)    │ │ Lookup Tool  │ │                   │
   └──────┬────────┘ └─────────────┘ └──────────────────┘
          │
          ▼
   ┌──────────────────────┐
   │  Vector Store         │
   │  (Chroma / pgvector)  │
   │  medical corpus       │
   │  (guidelines, drug DB,│
   │   de-identified notes)│
   └──────────────────────┘

                              All LLM calls route to:
                                                │
                                                ▼
                          ┌───────────────────────────────────────┐
                          │            vLLM Inference Server        │
                          │  OpenAI-compatible API, tensor-parallel │
                          │  base model: Llama-3-13B/70B (4-bit)    │
                          │  + hot-loaded LoRA adapter (medical SFT)│
                          │  Deployed on AWS g5/p4d or GCP A2/A3    │
                          └───────────────────────────────────────┘

Offline pipeline (separate from the online path above):

  Raw medical corpora ─► data/prepare_dataset.py ─► instruction dataset
        │                                                  │
        ▼                                                  ▼
  rag/ingest.py (chunk+embed)                 finetune/train_qlora.py (QLoRA SFT)
        │                                                  │
        ▼                                                  ▼
  Vector Store (used by RAG tool)             finetune/merge_adapter.py
                                                            │
                                                            ▼
                                             LoRA adapter → served by vLLM
```

**Key design decisions**

| Concern | Choice | Why |
|---|---|---|
| Fine-tuning | QLoRA (4-bit NF4 base + LoRA adapters) on 13B–70B model | Full fine-tuning of 70B needs 8x A100-80GB minimum; QLoRA gets comparable instruction-following at ~1/10th the GPU-memory cost, and adapters are swappable/versionable independent of the base model |
| Serving | vLLM with PagedAttention + continuous batching | Needed for concurrent multi-user throughput; supports LoRA adapter hot-loading so we don't need to merge-and-redeploy for every fine-tune iteration |
| RAG | Hybrid (BM25 + dense) retrieval + cross-encoder reranker | Medical queries need both lexical precision (drug names, ICD codes) and semantic recall |
| Agent framework | LangChain AgentExecutor with typed tools | Explicit tool boundaries (RAG / drug DB / triage rules) are easier to audit than a single monolithic prompt for a safety-critical domain |
| Guardrails | Rule-based emergency detector runs *outside* the LLM, pre- and post- | LLMs must never be the sole safety mechanism for "call 911" style detection |

---

## 2. Repository layout

```
medical-genai-assistant/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── config/settings.py
├── data/prepare_dataset.py
├── finetune/
│   ├── train_qlora.py
│   ├── merge_adapter.py
│   └── ds_config.json
├── rag/
│   ├── ingest.py
│   ├── retriever.py
│   └── vector_store.py
├── agents/
│   ├── prompts.py
│   ├── tools.py
│   └── agent.py
├── serving/
│   ├── vllm_server.sh
│   ├── api.py
│   └── schemas.py
├── eval/
│   ├── evaluate.py
│   └── test_cases.jsonl
├── monitoring/
│   ├── tracing.py          # OpenTelemetry spans for RAG + LLM calls
│   ├── metrics.py          # Prometheus metric definitions
│   └── grafana_dashboard.json
├── mlops/
│   ├── model_registry.py   # MLflow registry: register/promote/rollback
│   ├── ab_testing.py       # base vs. fine-tuned traffic split + analysis
│   └── retrain_trigger.py  # auto-trigger retraining on eval score drop
├── infra/
│   ├── aws_deploy.md
│   ├── gcp_deploy.md
│   └── terraform/main.tf
└── tests/test_api.py
```

---

## 3. Step-by-step setup

### 3.1 Prerequisites
- Python 3.10+
- CUDA 12.1+ GPU node for training (recommend A100/H100 x4-8 for 70B QLoRA, x1-2 for 13B)
- Docker + docker-compose (for local RAG stack)
- AWS or GCP account with GPU quota (see `infra/`)

### 3.2 Local environment
```bash
git clone <this-repo>
cd medical-genai-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN, DB creds, etc.
pip install -r requirements.txt
> Use `requirements.txt` for the full stack (RAG, agents, fine-tuning, eval). Use `requirements-gateway.txt` instead if you're only deploying the lightweight FastAPI gateway (e.g. a separate container that just proxies to an already-running vLLM server).
```

### 3.3 Build the RAG knowledge base
```bash
# Place source docs (guidelines PDFs, drug formulary CSV, de-identified notes)
# under data/raw/, then:
python rag/ingest.py --source data/raw --collection medical_kb
```

### 3.4 Prepare fine-tuning dataset
```bash
python data/prepare_dataset.py \
    --output data/processed/medical_sft.jsonl \
    --sources medqa,pubmedqa,internal_notes
```

### 3.5 QLoRA fine-tune (multi-GPU)
```bash
# Single node, 4 GPUs, 13B model:
accelerate launch --config_file finetune/ds_config.json \
    finetune/train_qlora.py \
    --base_model meta-llama/Llama-3-13b \
    --dataset data/processed/medical_sft.jsonl \
    --output_dir checkpoints/medical-lora-13b

# For 70B, add --deepspeed finetune/ds_config.json (ZeRO-3) and use 8 GPUs.
```

### 3.6 Merge / prep adapter for serving
```bash
python finetune/merge_adapter.py \
    --base_model meta-llama/Llama-3-13b \
    --adapter checkpoints/medical-lora-13b \
    --output_dir checkpoints/medical-lora-13b-vllm-ready
# (We keep adapter unmerged and serve via vLLM's --enable-lora instead of
#  merging, so we can hot-swap adapters. merge_adapter.py also supports
#  a fully-merged export if you need a single-artifact deployment.)
```

### 3.7 Launch vLLM server
```bash
bash serving/vllm_server.sh
# Starts an OpenAI-compatible server on :8001 with LoRA adapter registered
```

### 3.8 Launch FastAPI gateway + agent
```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3.9 Run evaluation suite
```bash
python eval/evaluate.py --endpoint http://localhost:8000
```

### 3.10 Cloud deployment
See `infra/aws_deploy.md` or `infra/gcp_deploy.md` for GPU instance sizing,
autoscaling, and cost-control notes.

---

## 4. Observability

- **LangSmith**: set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT` in
  `.env` -- every agent step (tool selection, LLM calls, scratchpad) traces
  automatically since LangChain reads these env vars, no code changes needed.
- **Prometheus**: scrape `GET /metrics` on the gateway (port 8000). Tracks
  request latency/count, emergency-trigger rate, insufficient-evidence rate,
  retrieval relevance distribution, tool-call breakdown, and active LoRA
  adapter version (`monitoring/metrics.py`).
- **OpenTelemetry**: spans wrap retrieval and can wrap LLM/tool calls
  (`monitoring/tracing.py`), exported via OTLP to your collector of choice
  (Tempo/Jaeger/Honeycomb), independent of the LangSmith trace.
- **Grafana**: import `monitoring/grafana_dashboard.json` against a
  Prometheus datasource. Includes suggested alert thresholds (e.g. p95
  latency SLO breach, insufficient-evidence rate spike).

## 5. MLOps pipeline

- **MLflow experiment tracking**: `finetune/train_qlora.py` now logs
  hyperparameters, per-step metrics, and the final adapter as an artifact
  to MLflow (set `MLFLOW_TRACKING_URI` in `.env`).
- **Model registry**: `mlops/model_registry.py` wraps MLflow's registry --
  `register`, `promote` (auto-archives the previous Production version),
  `rollback`, `list`. `resolve_production_adapter_path()` is what a
  deployment script should call instead of hardcoding a checkpoint path.
- **A/B testing**: `mlops/ab_testing.py` deterministically splits traffic
  between `base` and the fine-tuned adapter (both served by the same vLLM
  process, see `serving/vllm_server.sh`), and `ABTestAnalyzer` computes
  safety-pass-rate / grounded-answer-rate per variant to gate promotion.
- **Automated retraining trigger**: `mlops/retrain_trigger.py` compares the
  latest eval score against a rolling baseline and fires a retraining
  pipeline call when the drop exceeds `RETRAIN_TRIGGER_DROP` -- wire this
  into a scheduled job that runs `eval/evaluate.py` first.

## 6. Advanced RAG

- **HyDE** (`rag/hyde.py`): generates a hypothetical answer passage and
  searches with that instead of the raw question, closing the
  question/answer phrasing gap common in clinical queries.
- **Query decomposition** (`rag/query_decomposition.py`): breaks multi-hop
  questions ("is X safe with Y given condition Z?") into sub-questions,
  each retrieved independently, then merged (`retriever.retrieve_multi_hop`).
- **Self-RAG confidence grading** (`rag/self_rag.py`): after retrieval, the
  LLM grades whether passages actually answer the question (not just
  "topically related") before the agent is allowed to answer; fails closed
  (treats grading errors as insufficient evidence).
- **Parent-child chunking** (`rag/parent_child_store.py` +
  updated `rag/ingest.py`/`rag/retriever.py`): embeds small child chunks
  (~512 tokens) for precise matching, but returns the larger parent chunk
  (~2048 tokens) as LLM context, so answers aren't generated from a
  fragment that's missing an adjacent caveat.

## 7. Edge cases handled

- **Emergency symptoms** (chest pain + shortness of breath, stroke signs, etc.) →
  rule-based detector short-circuits the LLM/agent path and returns an
  immediate "seek emergency care" response (see `agents/tools.py::EmergencyTriageTool`).
- **Out-of-scope queries** (non-medical) → agent's router refuses and redirects.
- **Low retrieval confidence** → RAG tool returns "insufficient evidence,
  consult a clinician" instead of letting the LLM guess.
- **Conflicting drug interactions** → deterministic lookup table wins over
  LLM free text; LLM only explains, never overrides.
- **PII in input** → gateway-level scrub/redaction before logging or
  forwarding to the LLM provider.
- **vLLM adapter mismatch / OOM** → `serving/api.py` catches inference
  errors, retries once without LoRA (base model fallback), then fails
  gracefully with a safe error message.
- **Concurrent load** → vLLM continuous batching + FastAPI async workers;
  see `infra/` for autoscaling triggers on GPU utilization.

## 8. CI/CD, model card, and cost estimation

- **CI/CD** (`.github/workflows/ci-cd.yml`): lint (ruff + black) → unit
  tests (`pytest tests/`) → eval harness as a hard quality/safety gate
  (`eval/evaluate.py --mock`, runs the adversarial suite in
  `eval/test_cases.jsonl` against a stubbed agent) → Docker build → deploy
  (only on `main`, only if every prior stage is green). If you don't have
  cloud credentials wired into repo secrets yet, the `deploy` job will
  simply fail at the AWS/ECR steps — that's expected until you add
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `ECR_REGISTRY` as GitHub
  Actions secrets, or you can comment those steps out and keep the
  lint/test/eval gate for a portfolio repo without live infra.
- **Model card** (`MODEL_CARD.md`): intended use, out-of-scope uses, the
  four safety layers and where each can fail, known limitations, and a
  bias/fairness section. Fill in the bracketed fields (base model,
  training data source, eval numbers) once you've run a real fine-tune —
  this is meant to be edited, not left as a template.
- **Cost estimation** (`infra/cost_estimate.py`): rough $/1000-requests
  and $/month for serving on AWS or GCP, given an instance's hourly price
  and assumed throughput. Example:
  ```bash
  python infra/cost_estimate.py --cloud aws --model-size 13b --requests-per-day 5000
  python infra/cost_estimate.py --cloud gcp --model-size 70b --requests-per-day 50000 --replicas 2 --json
  ```
  The built-in prices are placeholders — pass `--gpu-hourly-usd` and
  `--throughput-req-per-hour` with numbers from your own pricing page and
  load test before using this for a real budget decision.
