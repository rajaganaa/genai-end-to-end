# Model Card: MedAssist-GenAI

This card describes the fine-tuned model *and* the system it's embedded
in (RAG + agent + safety gates), because for a system like this the model
weights alone are not the right unit of evaluation — the surrounding
retrieval, tool-calling, and hard-coded safety layer materially change
what it's safe to claim about behavior. Fill in the bracketed fields with
your actual run's values before publishing.

## Model details

| | |
|---|---|
| Base model | `[e.g. meta-llama/Llama-3.1-13B-Instruct]` |
| Fine-tuning method | QLoRA (4-bit NF4 base, LoRA adapters on attention + MLP projections) |
| Training data | `[dataset name/source, size, license]` |
| Adapter size | `[e.g. ~250M trainable params]` |
| Serving | vLLM, OpenAI-compatible endpoint, adapter hot-swappable via `VLLM_MODEL_NAME` |
| Retrieval corpus | `[corpus name — e.g. de-identified clinical guideline excerpts + drug interaction table]`, see `data/raw/` |
| Developed by | `[your name / team]`, `[date]` |
| License | `[repo license]` — note base model license terms apply separately and may restrict commercial use |

## Intended use

- **Intended**: a clinical-decision-support *aid* for licensed clinicians,
  and a general medical-information assistant for patients, that always
  grounds answers in retrieved source documents and defers to a human for
  final diagnosis/treatment decisions.
- **Intended users**: clinicians, students, and developers evaluating a
  reference RAG + agent + fine-tuning architecture. This repo is a
  **portfolio/teaching project**, not a validated clinical product.
- **Out of scope**: autonomous diagnosis, prescribing, triage without
  human review, or any use where the system's output is the sole basis
  for a clinical decision. Not evaluated or cleared as a medical device.
  Do not deploy this against real patients without independent clinical
  validation, IRB/regulatory review as applicable in your jurisdiction,
  and a licensed clinician in the loop.

## How the system tries to stay safe (and where that can fail)

1. **Hard-coded emergency keyword gate** (`agents/prompts.py`,
   `agents/tools.py`) runs before the LLM sees the query. It catches
   known phrasings of chest pain, stroke signs, suicidal ideation, etc.,
   and short-circuits straight to "seek emergency care."
   *Failure mode*: keyword/pattern lists cannot cover every phrasing,
   especially typos, non-English input, or oblique descriptions of
   symptoms. This is a safety net, not a guarantee.
2. **Retrieval grounding + "insufficient evidence" fallback**: the system
   prompt instructs the model to answer only from retrieved passages and
   to say so explicitly when retrieval confidence is low (Self-RAG
   scoring in `rag/self_rag.py`, threshold in `config/settings.py`).
   *Failure mode*: LLMs can still hallucinate plausible-sounding claims
   that aren't in the retrieved context, particularly under adversarial
   prompting; this is a mitigation, not an elimination, of that risk.
3. **Prompt-level refusal instructions** for out-of-scope or
   instruction-override attempts (`AGENT_SYSTEM_PROMPT`).
   *Failure mode*: prompt injection and jailbreak techniques evolve
   continuously; treat this layer as best-effort, not a security
   boundary, and pair it with the eval harness's adversarial test suite
   (`eval/test_cases.jsonl`) plus your own red-teaming before any
   higher-stakes deployment.
4. **PII/PHI log redaction** (`serving/api.py::redact_for_logging`) is a
   regex-based pattern for logs only — explicitly *not* a HIPAA-grade
   de-identification pipeline for the data path itself. Do not treat this
   as sufficient for handling real patient data.

## Known limitations

- Fine-tuned on `[describe scale/scope of training data]` — performance
  outside that distribution (rare conditions, pediatric-specific
  guidance, non-English queries, regions with different standards of
  care) is unvalidated and likely worse.
- No clinical validation study, no comparison against clinician baseline
  accuracy, and no regulatory review has been performed. Eval numbers in
  this repo (`eval/evaluate.py`) measure safety-gate behavior and
  retrieval-groundedness on a small curated test set — they are not a
  substitute for a proper clinical accuracy benchmark (e.g. MedQA,
  clinician-adjudicated review).
- Drug interaction data (`data/raw/drug_interactions.csv`) is a small
  illustrative sample, not a complete or currently-maintained interaction
  database — do not rely on it for real dosing decisions.
- Like any LLM-based system, it can be confidently wrong. The
  "insufficient evidence" fallback reduces but does not eliminate this.
- Latency and cost scale with retrieval depth (`retrieval_top_k`,
  reranking) and base-model size — see `infra/cost_estimate.py` for
  back-of-envelope figures at your actual configuration.

## Bias and fairness considerations

- Training data composition drives what the fine-tune has and hasn't
  seen; if `[dataset]` under-represents certain conditions, demographics,
  presentation styles, or languages, expect degraded performance there
  without it necessarily being visible in aggregate eval scores.
- Medical training corpora frequently reflect historical documentation
  biases (e.g. under-diagnosis of pain in some populations, uneven
  research coverage across conditions). Fine-tuning on such data can
  reproduce those patterns rather than correct them.
- No formal fairness/subgroup evaluation has been run against this build.
  Before any real-world use, evaluate accuracy and safety-gate trigger
  rates broken out by relevant subgroups, not just in aggregate.

## Evaluation summary

Run `python eval/evaluate.py --endpoint <your deployed endpoint>` and
record results here, e.g.:

| Metric | Result |
|---|---|
| Safety suite pass rate | `[X/Y]` |
| Task accuracy (held-out QA) | `[fill in]` |
| Eval date / model version | `[fill in]` |

## Contact / reporting issues

`[maintainer contact]` — flag any incorrect, unsafe, or biased outputs
here so the eval suite and training data can be updated.
