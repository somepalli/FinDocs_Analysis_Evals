from hashlib import sha256

import pytest
from test_evidence_gate import chunk, gate  # noqa: F401

from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, SourceApproval

DOC = sha256(b"invented third party report").hexdigest()


def test_third_party_approval_is_document_and_application_scoped(gate):  # noqa: F811
    source = chunk(
        "Third-party financial information report\nLegal name: Example Manufacturing LLP\n"
        "Standalone\nAmounts in lakhs\n| Particulars | FY2024-25 | FY2023-24 |\n"
        "|---|---|---|\n| Revenue from operations | 1200 | 900 |",
        DOC,
    )
    approval = SourceApproval(
        application_id="APP-APPROVED",
        document_id=DOC,
        provider="probe42",
        approval_id="approval-test-1",
        approved_by="test-reviewer",
        reason="Synthetic source acceptance",
    )
    policy = gate.policy.model_copy(update={"source_approvals": (approval,)})
    result = BorrowerEvidenceGate(policy, application_id="APP-APPROVED").extract(
        "annual_revenue_crore", (source,)
    )
    assert result.value == "12.00"
    assert result.evidence_validation.document_type == "financial_information_report"
    assert result.evidence_validation.source_approval_id == approval.approval_id
    for app in (None, "APP-FOREIGN"):
        with pytest.raises(EvidenceInsufficient, match="document_type_missing"):
            BorrowerEvidenceGate(policy, application_id=app).extract(
                "annual_revenue_crore", (source,)
            )
    wrong = approval.model_copy(update={"document_id": sha256(b"other").hexdigest()})
    with pytest.raises(EvidenceInsufficient, match="document_type_missing"):
        BorrowerEvidenceGate(
            policy.model_copy(update={"source_approvals": (wrong,)}), application_id="APP-APPROVED"
        ).extract("annual_revenue_crore", (source,))


def test_borrower_heading_requires_matching_document_entity(gate):  # noqa: F811
    source = chunk(
        "EXAMPLE MANUFACTURING LIMITED\nAudited financial statements\n"
        "Standalone financial statements\nYear ended FY2024-25"
    )
    result = gate.extract("borrower_name", (source,))
    assert result.value == "EXAMPLE MANUFACTURING LIMITED"
    assert result.citation.document_id == source.provenance[0].document_id
