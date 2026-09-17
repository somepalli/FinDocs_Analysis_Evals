from pathlib import Path

import pytest
from test_basis_context import page, sources

from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, EvidencePolicy
from findociq.reason.interpretations import interpretation_options


def evidence():
    return sources(
        "Independent auditor report\nWe have audited the accompanying financial statements\n"
        "Year ended FY2024-25\nBalance sheet and profit and loss"
    ) + (page("Basis of preparation: Indian GAAP and historical cost", 3),)


def test_human_scope_is_off_until_reviewed_and_distinct_in_receipt():
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    chunks = evidence()
    options = interpretation_options(chunks, policy)
    assert len(options) == 1
    assert options[0].authority == "reporting_scope"
    with pytest.raises(EvidenceInsufficient, match="basis_ambiguous"):
        BorrowerEvidenceGate(policy).extract("annual_revenue_crore", chunks)
    gate = BorrowerEvidenceGate(policy, interpretations=options, interpretation_command_id="a" * 36)
    result = gate.extract("annual_revenue_crore", chunks)
    assert result.value == "12.00"
    assert result.evidence_validation.basis_resolution == "reviewer_reporting_scope"
    assert result.evidence_validation.interpretation_command_id == "a" * 36


def test_prior_audit_and_related_party_note_do_not_change_current_audit_scope():
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    chunks = evidence() + (
        page("Previous auditor: year ended FY2023-24", 4),
        page("Related party relationships: fellow subsidiary", 5),
    )
    options = interpretation_options(tuple(reversed(chunks)), policy)
    assert len(options) == 1
    assert options[0].period == "2025"


def test_ambiguous_audit_period_does_not_offer_scope_determination():
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    chunks = evidence() + (
        page("We have audited the accompanying financial statements\nYear ended FY2022-23", 4),
    )
    assert interpretation_options(chunks, policy) == ()


@pytest.mark.parametrize("change", ["basis", "entity", "year", "note", "foreign", "group"])
def test_reporting_scope_fails_closed(change):
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    chunks = evidence()
    link = interpretation_options(chunks, policy)[0]
    if change == "basis":
        chunks += (page("Consolidated financial statements", 4),)
    elif change == "group":
        chunks += (
            page(
                "We have audited the accompanying financial statements "
                "of the entity and its subsidiaries",
                4,
            ),
        )
    elif change == "entity":
        link = link.model_copy(update={"entity_hash": "0" * 64})
    elif change == "year":
        link = link.model_copy(update={"period": "2024"})
    elif change == "note":
        link = link.model_copy(update={"context_citations": link.context_citations[:1]})
    else:
        link = link.model_copy(
            update={
                "context_citations": (
                    link.context_citations[0].model_copy(update={"document_id": "foreign"}),
                )
            }
        )
    gate = BorrowerEvidenceGate(policy, interpretations=(link,), interpretation_command_id="a" * 36)
    with pytest.raises(EvidenceInsufficient):
        gate.extract("annual_revenue_crore", chunks)
