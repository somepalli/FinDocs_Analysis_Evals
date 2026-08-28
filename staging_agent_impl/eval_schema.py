"""Typed held-out release evaluation contracts."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fundermatch.orchestration.schema import GraphStatus


class InventedApplication(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    application_id: str
    industry: str
    region: str
    requested_amount_crore: Decimal
    annual_revenue_crore: Decimal
    ebitda_margin_pct: Decimal
    dscr: Decimal
    debt_to_ebitda: Decimal
    collateral_cover: Decimal
    years_operating: int
    employee_count: int


class AgentReleaseCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    cohort: Literal["eligible_aligned", "hard_rule", "no_close_precedent", "adversarial"]
    application: InventedApplication
    expected_eligible_funders: tuple[str, ...]
    forbidden_funders: tuple[str, ...]
    expected_citation_metrics: tuple[str, ...]
    expected_precedent_ids: tuple[str, ...]
    expected_guardrail_codes: tuple[str, ...]
    expected_stop_state: GraphStatus

    @model_validator(mode="after")
    def validate_expectations(self) -> AgentReleaseCase:
        if set(self.expected_eligible_funders) & set(self.forbidden_funders):
            raise ValueError("eligible and forbidden funders must be disjoint")
        if self.cohort == "adversarial" and not self.expected_guardrail_codes:
            raise ValueError("adversarial cases require expected guardrail codes")
        if self.cohort != "adversarial" and self.expected_stop_state != GraphStatus.WAITING_FOR_REVIEW:
            raise ValueError("non-adversarial cases must reach human review")
        return self


class SendBackRoutingCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    mode: Literal["reviewer_selected", "supervisor_clear", "supervisor_ambiguous"]
    reason: str
    selected_stage: str | None
    expected_stage: str | None
    expected_stop_state: GraphStatus
