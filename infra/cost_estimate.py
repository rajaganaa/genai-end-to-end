"""
Back-of-envelope cost estimator for serving MedAssist-GenAI on AWS or GCP.

This is a planning tool, not a billing-accurate simulator: it estimates
$/1000 requests from a GPU instance's hourly price and an assumed
requests-per-hour throughput, plus a small fixed CPU-gateway cost. Real
cost depends on your actual prompt/output token lengths, batching,
autoscaling behavior, and current cloud pricing -- always re-check list
prices before using these numbers to make a budget decision. Values here
are illustrative on-demand US-region list prices as of early 2026; pass
your own via --gpu-hourly-usd or a --pricing-file to override.

Usage:
    python infra/cost_estimate.py
    python infra/cost_estimate.py --cloud gcp --instance g2-standard-24 --model-size 13b
    python infra/cost_estimate.py --requests-per-day 50000 --model-size 70b --cloud aws
    python infra/cost_estimate.py --gpu-hourly-usd 3.50 --throughput-req-per-hour 900
"""
import argparse
import json

# Illustrative on-demand list prices (USD/hr), single instance, US region.
# THESE GO STALE -- treat as placeholders and override with --gpu-hourly-usd
# or your own --pricing-file for anything budget-critical.
INSTANCE_PRICING = {
    "aws": {
        "13b": {"instance": "g5.12xlarge", "gpus": 4, "hourly_usd": 5.67},
        "70b": {"instance": "g5.48xlarge", "gpus": 8, "hourly_usd": 16.29},
        "gateway": {"instance": "ecs-fargate (2 vCPU / 4GB)", "hourly_usd": 0.10},
    },
    "gcp": {
        "13b": {"instance": "g2-standard-24 (1x L4)", "gpus": 1, "hourly_usd": 1.96},
        "70b": {"instance": "a2-highgpu-4g (4x A100 40GB)", "gpus": 4, "hourly_usd": 14.69},
        "gateway": {"instance": "cloud-run (2 vCPU / 4GB)", "hourly_usd": 0.08},
    },
}

# Rough steady-state throughput assumption for a MedAssist-style request:
# hybrid retrieval (BM25 + dense + rerank) + agent tool-call overhead +
# a few-hundred-token generation. This varies a lot with prompt length,
# number of retrieved chunks, and whether HyDE/query-decomposition fire
# (each adds an extra LLM call) -- override with --throughput-req-per-hour
# once you've measured real p50 latency under load.
DEFAULT_THROUGHPUT = {"13b": 600, "70b": 220}  # requests/hour per instance


def estimate(cloud: str, model_size: str, requests_per_day: int,
             gpu_hourly_usd: float = None, throughput_req_per_hour: float = None,
             replicas: int = 1) -> dict:
    pricing = INSTANCE_PRICING[cloud][model_size]
    hourly = gpu_hourly_usd if gpu_hourly_usd is not None else pricing["hourly_usd"]
    throughput = throughput_req_per_hour if throughput_req_per_hour is not None else DEFAULT_THROUGHPUT[model_size]
    gateway_hourly = INSTANCE_PRICING[cloud]["gateway"]["hourly_usd"]

    per_hour_capacity = throughput * replicas
    hours_per_day = requests_per_day / per_hour_capacity if per_hour_capacity else float("inf")
    gpu_cost_per_day = hourly * replicas * min(24.0, hours_per_day if hours_per_day else 0)
    # If traffic doesn't fill 24h of capacity, still show cost for actually
    # running the replicas long enough to serve requests_per_day, capped
    # at running continuously (24h/day) once demand exceeds capacity.
    hours_running = min(24.0, requests_per_day / per_hour_capacity) if per_hour_capacity else 24.0
    gpu_cost_per_day = hourly * replicas * hours_running
    gateway_cost_per_day = gateway_hourly * 24  # gateway assumed always-on; scale-to-zero saves this

    total_per_day = gpu_cost_per_day + gateway_cost_per_day
    cost_per_1000 = (total_per_day / requests_per_day) * 1000 if requests_per_day else 0

    return {
        "cloud": cloud,
        "model_size": model_size,
        "instance": pricing["instance"],
        "replicas": replicas,
        "gpu_hourly_usd": hourly,
        "assumed_throughput_req_per_hour_per_replica": throughput,
        "requests_per_day": requests_per_day,
        "hours_running_per_day": round(hours_running, 2),
        "gpu_cost_per_day_usd": round(gpu_cost_per_day, 2),
        "gateway_cost_per_day_usd": round(gateway_cost_per_day, 2),
        "total_cost_per_day_usd": round(total_per_day, 2),
        "cost_per_1000_requests_usd": round(cost_per_1000, 4),
        "cost_per_month_usd": round(total_per_day * 30, 2),
        "note": ("If requests_per_day exceeds a single replica's daily capacity "
                 "(throughput * 24h), add replicas -- this estimate does not "
                 "auto-scale replicas for you."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cloud", choices=["aws", "gcp"], default="aws")
    parser.add_argument("--model-size", choices=["13b", "70b"], default="13b",
                         help="Matches the QLoRA base-model scale from finetune/train_qlora.py")
    parser.add_argument("--requests-per-day", type=int, default=1000)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--gpu-hourly-usd", type=float, default=None,
                         help="Override the built-in placeholder hourly price")
    parser.add_argument("--throughput-req-per-hour", type=float, default=None,
                         help="Override the assumed steady-state requests/hour/replica")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary")
    args = parser.parse_args()

    result = estimate(
        cloud=args.cloud,
        model_size=args.model_size,
        requests_per_day=args.requests_per_day,
        gpu_hourly_usd=args.gpu_hourly_usd,
        throughput_req_per_hour=args.throughput_req_per_hour,
        replicas=args.replicas,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"MedAssist-GenAI cost estimate -- {result['cloud'].upper()}, {result['model_size']} model")
    print("-" * 60)
    print(f"  Instance:                 {result['instance']} x{result['replicas']}")
    print(f"  GPU price:                ${result['gpu_hourly_usd']}/hr per instance")
    print(f"  Assumed throughput:       {result['assumed_throughput_req_per_hour_per_replica']} req/hr/replica")
    print(f"  Requests/day:             {result['requests_per_day']}")
    print(f"  Hours running/day:        {result['hours_running_per_day']}")
    print(f"  GPU cost/day:             ${result['gpu_cost_per_day_usd']}")
    print(f"  Gateway cost/day:         ${result['gateway_cost_per_day_usd']}")
    print(f"  TOTAL/day:                ${result['total_cost_per_day_usd']}")
    print(f"  TOTAL/month (x30):        ${result['cost_per_month_usd']}")
    print(f"  Cost per 1000 requests:   ${result['cost_per_1000_requests_usd']}")
    print()
    print(f"  Note: {result['note']}")
    print()
    print("  Re-run with --gpu-hourly-usd / --throughput-req-per-hour once you")
    print("  have real current pricing and measured load-test throughput.")


if __name__ == "__main__":
    main()
