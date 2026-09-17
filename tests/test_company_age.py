import pytest
from test_evidence_gate import chunk, gate  # noqa: F401

from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient


def test_incorporation_age_is_pinned_and_cited(gate):  # noqa: F811
    source = chunk(
        "Udyam registration certificate\nLegal name: Example Manufacturing LLP\n"
        "Date of incorporation: 16/09/2016",
        "registration",
    )
    selected = BorrowerEvidenceGate(gate.policy, assessment_date="2026-09-15")
    result = selected.extract("years_operating", (source,))
    assert result.value == "9"
    assert result.evidence_validation.date_calculation.assessment_date == "2026-09-15"
    assert result.evidence_validation.date_calculation.source_citation == result.citation
    assert result.evidence_validation.date_calculation.source_chunk_id == source.chunk_id


def test_commencement_is_not_used_as_incorporation(gate):  # noqa: F811
    source = chunk(
        "Udyam registration certificate\nLegal name: Example Manufacturing LLP\n"
        "Date of commencement: 16/09/2016"
    )
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        BorrowerEvidenceGate(gate.policy, assessment_date="2026-09-15").extract(
            "years_operating", (source,)
        )


def test_future_incorporation_is_rejected(gate):  # noqa: F811
    source = chunk(
        "Udyam registration certificate\nLegal name: Example Manufacturing LLP\n"
        "Date of incorporation: 16/09/2027"
    )
    with pytest.raises(EvidenceInsufficient, match="value_ambiguous"):
        BorrowerEvidenceGate(gate.policy, assessment_date="2026-09-15").extract(
            "years_operating", (source,)
        )
