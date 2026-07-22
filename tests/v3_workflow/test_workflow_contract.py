from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRADE_WORKFLOW = ROOT / ".github" / "workflows" / "trade.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _load_yaml(path: Path) -> dict:
    # BaseLoader preserves the GitHub Actions key `on` instead of treating it
    # as the YAML 1.1 boolean true.
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def _uses_references(document: object) -> list[str]:
    references: list[str] = []
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "uses":
                references.append(str(value))
            references.extend(_uses_references(value))
    elif isinstance(document, list):
        for value in document:
            references.extend(_uses_references(value))
    return references


def test_trade_workflow_preserves_and_classifies_all_triggers():
    workflow = _load_yaml(TRADE_WORKFLOW)
    triggers = workflow["on"]

    assert set(triggers) == {"repository_dispatch", "schedule", "workflow_dispatch"}
    assert {item["cron"] for item in triggers["schedule"]} == {
        "17 14,15 * * 1-5",
        "47 20,21 * * 1-5",
    }

    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["purpose"]["default"] == "health"
    assert inputs["purpose"]["options"] == [
        "health",
        "eod",
        "rebalance",
        "reconcile",
        "bootstrap",
    ]
    assert inputs["mode"]["default"] == "shadow"
    assert inputs["mode"]["options"] == ["shadow", "paper"]
    assert inputs["force_recompute"]["default"] == "false"

    text = TRADE_WORKFLOW.read_text(encoding="utf-8")
    assert "github.event.client_payload.purpose || 'health'" in text
    assert "github.event.client_payload.mode || 'shadow'" in text
    assert 'purpose = "rebalance" if schedule.startswith("17 ") else "eod"' in text


def test_trade_workflow_has_locked_safety_and_environment_contracts():
    workflow = _load_yaml(TRADE_WORKFLOW)
    run_job = workflow["jobs"]["run"]

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "aegisquant-us-v3-paper",
        "cancel-in-progress": "false",
    }
    assert run_job["runs-on"] == "ubuntu-24.04"
    assert run_job["env"]["ALPACA_BASE_URL"] == "https://paper-api.alpaca.markets"
    assert run_job["env"]["RL_ENABLED"] == "false"

    text = TRADE_WORKFLOW.read_text(encoding="utf-8")
    assert "alpaca-paper-shadow" in text
    assert "alpaca-paper-execution" in text
    assert "Automatic paper is deliberately disabled" in text
    assert "needs.classify.outputs.mode == 'paper' && secrets.ALPACA_API_KEY" in text
    assert "AEGISQUANT_ACCOUNT_KEY" in text
    assert "V3_INPUT_BUNDLE_SHA256" in text
    assert "scripts/fetch_v3_runtime_input.py" in text
    assert '"$RUN_PURPOSE" == "bootstrap" && "$TRADING_MODE" == "paper"' in text
    assert "main_us_v3.py" in text
    assert "main_us_v2.py" not in text
    assert "OPENAI_API_KEY" not in text
    assert "api.alpaca.markets" not in text.replace("paper-api.alpaca.markets", "")

    # Secrets are never job-scoped; they appear only on the DB preflight and
    # the single runtime step.
    assert "secrets." not in repr(run_job.get("env", {}))
    secret_steps = [
        step["name"]
        for step in run_job["steps"]
        if "secrets." in repr(step.get("env", {}))
    ]
    assert secret_steps == [
        "Validate durable PostgreSQL for paper",
        "Run one AegisQuant v3 cycle",
    ]
    step_names = [step["name"] for step in run_job["steps"]]
    assert step_names.index("Validate durable PostgreSQL for paper") < step_names.index(
        "Run one AegisQuant v3 cycle"
    )


def test_trade_workflow_requires_exact_complete_artifact_directory():
    workflow = _load_yaml(TRADE_WORKFLOW)
    run_job = workflow["jobs"]["run"]
    text = TRADE_WORKFLOW.read_text(encoding="utf-8")

    for filename in (
        "manifest.json",
        "preflight.json",
        "targets.json",
        "orders.json",
        "reconciliation.json",
        "performance.json",
        "run.log",
    ):
        assert filename in text

    upload = next(step for step in run_job["steps"] if step["name"] == "Upload audit artifacts")
    assert upload["with"]["path"] == "artifacts/${{ env.AEGISQUANT_RUN_ID }}/"
    assert upload["with"]["if-no-files-found"] == "error"
    assert "90" in upload["with"]["retention-days"]
    assert "30" in upload["with"]["retention-days"]


def test_all_actions_are_pinned_to_full_reviewable_commits():
    for workflow_path in (TRADE_WORKFLOW, CI_WORKFLOW):
        workflow = _load_yaml(workflow_path)
        references = _uses_references(workflow)
        assert references
        for reference in references:
            assert re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}", reference), reference


def test_ci_is_secret_free_locked_and_runs_all_required_checks():
    workflow = _load_yaml(CI_WORKFLOW)
    triggers = workflow["on"]
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert set(triggers) == {"pull_request", "push"}
    assert triggers["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "secrets." not in text
    assert "ubuntu-24.04" in text
    assert 'python-version: "3.11"' in text
    assert 'RUN_NETWORK_TESTS: "0"' in text
    assert "-r requirements.lock" in text
    assert "python -m compileall" in text
    assert 'python -c "import main_us_v3"' in text
    assert "PYTHONPATH: tests" in text
    assert "python -m pytest -p v3_workflow.no_network_order_post" in text
    assert "actionlint@v1.7.7" in text
    assert "docker build" in text
    assert "docker compose --env-file .env.example --profile manual-shadow config --quiet" in text
