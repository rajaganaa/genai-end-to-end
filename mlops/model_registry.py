"""
Model registry operations on top of MLflow's built-in Model Registry.
Wraps the parts we actually use (register, promote, rollback, list) so the
rest of the codebase (e.g. serving/vllm_server.sh via a resolved path)
doesn't need to know MLflow's API directly.

Lifecycle stages used: "Staging" -> "Production" -> "Archived".
Promotion to "Production" is what serving/vllm_server.sh should read from
(via `resolve_production_adapter_path`) rather than a hardcoded checkpoint
directory, so a promotion is a one-command operation, not a manual file copy.

Usage:
    python mlops/model_registry.py register --run_id <mlflow_run_id>
    python mlops/model_registry.py promote --version 3
    python mlops/model_registry.py rollback
    python mlops/model_registry.py list
"""

import argparse
import logging

import mlflow
from mlflow.tracking import MlflowClient

from config.settings import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def register_model(run_id: str) -> str:
    """Registers the LoRA adapter artifact from a completed training run
    as a new version in the model registry (stage: None -> will need
    explicit promotion)."""
    client = _client()
    model_uri = f"runs:/{run_id}/lora_adapter"
    result = mlflow.register_model(model_uri, settings.model_registry_name)
    log.info(
        "Registered version %s of '%s' from run %s",
        result.version,
        settings.model_registry_name,
        run_id,
    )
    return result.version


def promote_to_production(version: str) -> None:
    """Promotes a specific version to Production, automatically archiving
    whatever was previously in Production (so exactly one version is ever
    live at a time -- important for reproducibility of "what served this
    response" questions during incident review)."""
    client = _client()

    current_prod = client.get_latest_versions(
        settings.model_registry_name, stages=["Production"]
    )
    for mv in current_prod:
        client.transition_model_version_stage(
            settings.model_registry_name, mv.version, stage="Archived"
        )
        log.info("Archived previous production version %s", mv.version)

    client.transition_model_version_stage(
        settings.model_registry_name, version, stage="Production"
    )
    log.info("Promoted version %s to Production", version)


def rollback_to_previous() -> str:
    """Rolls back to the most recently archived version. Useful as a fast
    incident-response action when a newly promoted model regresses."""
    client = _client()
    archived = client.get_latest_versions(
        settings.model_registry_name, stages=["Archived"]
    )
    if not archived:
        raise RuntimeError("No archived version available to roll back to")
    # Most recent archived version (highest version number that isn't current prod)
    rollback_version = max(archived, key=lambda mv: int(mv.version)).version
    promote_to_production(rollback_version)
    log.warning("Rolled back to version %s", rollback_version)
    return rollback_version


def resolve_production_adapter_path() -> str:
    """Returns the local/S3 artifact path for whatever is currently in
    Production. serving/vllm_server.sh (or its Terraform user_data) should
    call this at boot rather than hardcoding LORA_PATH."""
    client = _client()
    prod = client.get_latest_versions(
        settings.model_registry_name, stages=["Production"]
    )
    if not prod:
        raise RuntimeError(
            f"No Production version of '{settings.model_registry_name}' found"
        )
    return client.get_model_version_download_uri(
        settings.model_registry_name, prod[0].version
    )


def list_versions() -> None:
    client = _client()
    versions = client.search_model_versions(f"name='{settings.model_registry_name}'")
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"v{v.version:>3}  stage={v.current_stage:<10}  run_id={v.run_id}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_register = sub.add_parser("register")
    p_register.add_argument("--run_id", required=True)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("--version", required=True)

    sub.add_parser("rollback")
    sub.add_parser("list")

    args = parser.parse_args()

    if args.command == "register":
        register_model(args.run_id)
    elif args.command == "promote":
        promote_to_production(args.version)
    elif args.command == "rollback":
        rollback_to_previous()
    elif args.command == "list":
        list_versions()


if __name__ == "__main__":
    main()
