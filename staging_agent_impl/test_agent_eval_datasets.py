from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fundermatch.evals.schema import AgentReleaseCase, SendBackRoutingCase

ROOT = Path(__file__).parents[1]


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_agent_release_dataset_has_required_n24_cohorts_and_expectations() -> None:
    rows = [
        AgentReleaseCase.model_validate(item)
        for item in _jsonl(ROOT / "evals" / "datasets" / "agent_release_cases.jsonl")
    ]
    assert len(rows) == 24
    assert len({item.case_id for item in rows}) == 24
    assert len({item.application.application_id for item in rows}) == 24
    assert Counter(item.cohort for item in rows) == {
        "eligible_aligned": 8,
        "hard_rule": 8,
        "no_close_precedent": 4,
        "adversarial": 4,
    }
    assert all(item.forbidden_funders for item in rows)
    assert all(
        item.expected_citation_metrics or item.cohort == "adversarial" for item in rows
    )


def test_sendback_dataset_forces_all_ambiguous_routes_to_stop_safely() -> None:
    rows = [
        SendBackRoutingCase.model_validate(item)
        for item in _jsonl(ROOT / "evals" / "datasets" / "sendback_routing.jsonl")
    ]
    assert {item.mode for item in rows} == {
        "reviewer_selected",
        "supervisor_clear",
        "supervisor_ambiguous",
    }
    ambiguous = [item for item in rows if item.mode == "supervisor_ambiguous"]
    assert ambiguous
    assert all(item.expected_stage is None for item in ambiguous)
    assert all(item.expected_stop_state.value == "needs_attention" for item in ambiguous)
