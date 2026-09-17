from pathlib import Path

import pytest
from test_basis_context import page, sources

from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, EvidencePolicy
from findociq.reason.interpretations import interpretation_options
from findociq.reason.schema import BasisInterpretation, citation_from_provenance


def configured(declaration="Standalone financial statements\nYear ended FY2024-25"):
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    chunks = sources(declaration) + (
        page("Consolidated financial statements\nYear ended FY2024-25", 3),
    )
    link = BasisInterpretation(
        document_id="report",
        target_chunk_id="report-2",
        target_citation=citation_from_provenance(chunks[1].provenance[0]),
        basis="standalone",
        period="2025",
        policy_hash=policy.policy_hash,
        context_citations=(citation_from_provenance(chunks[0].provenance[0]),),
    )
    gate = BorrowerEvidenceGate(policy, interpretations=(link,), interpretation_command_id="a" * 36)
    return gate, chunks, link


def test_reviewed_context_link_resolves_mixed_report_without_changing_numbers():
    gate, chunks, link = configured()
    with pytest.raises(EvidenceInsufficient, match="basis_ambiguous"):
        BorrowerEvidenceGate(gate.policy).extract("annual_revenue_crore", chunks)
    result = gate.extract("annual_revenue_crore", chunks)
    assert result.value == "12.00"
    assert result.evidence_validation.basis_resolution == "reviewer_context_link"
    assert result.evidence_validation.context_citations == link.context_citations
    assert len(interpretation_options(chunks, gate.policy)) == 2


@pytest.mark.parametrize(
    "declaration",
    [
        "Year ended FY2024-25",
        "Standalone financial statements\nYear ended FY2021-22",
        "This agreement requires standalone accounts.\nYear ended FY2024-25",
        "Consolidated financial statements\nYear ended FY2024-25",
    ],
)
def test_review_cannot_invent_or_override_source_context(declaration):
    gate, chunks, _ = configured(declaration)
    with pytest.raises(EvidenceInsufficient):
        gate.extract("annual_revenue_crore", chunks)


def test_foreign_citation_and_stale_policy_fail():
    gate, chunks, link = configured()
    for changed in (
        link.model_copy(update={"policy_hash": "0" * 64}),
        link.model_copy(
            update={
                "context_citations": (
                    link.context_citations[0].model_copy(update={"document_id": "foreign"}),
                )
            }
        ),
    ):
        reviewed = BorrowerEvidenceGate(
            gate.policy, interpretations=(changed,), interpretation_command_id="a" * 36
        )
        with pytest.raises(EvidenceInsufficient):
            reviewed.extract("annual_revenue_crore", chunks)
