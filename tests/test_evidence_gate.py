from pathlib import Path

import pytest

from findociq.ingest.schema import BoundingBox, Provenance, TableChunk, TextChunk
from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidenceInsufficient, EvidencePolicy


@pytest.fixture
def gate():
    return BorrowerEvidenceGate(EvidencePolicy.load(Path("configs/evidence/borrower.yaml")))


def chunk(text, doc="financials", page=1):
    return TextChunk(
        chunk_id=f"{doc}-{page}",
        text=text,
        provenance=(
            Provenance(
                document_id=doc,
                source_path="untrusted-filename.pdf",
                page_number=page,
                bbox=BoundingBox(x0=10, y0=10, x1=500, y1=700),
                page_width=600,
                page_height=800,
            ),
        ),
    )


def statement(
    *,
    entity="Example Manufacturing LLP",
    basis="Standalone",
    units="Amounts in lakhs",
    row="| Revenue from operations | 1200 | 900 |",
):
    return chunk(
        f"Audited financial statements\nLegal name: {entity}\n{basis}\n{units}\n"
        f"| Particulars | FY2024-25 | FY2023-24 |\n|---|---|---|\n{row}"
    )


def test_latest_year_and_unit_conversion_have_receipt(gate):
    result = gate.extract("annual_revenue_crore", (statement(),))
    assert result.value == "12.00"
    assert result.unit == "INR crore"
    assert result.period == "2025"
    assert result.evidence_validation.source_value == "1200"
    assert result.evidence_validation.conversion_factor == "0.01"
    assert result.citation.document_id == "financials"


def test_explicit_rupee_heading(gate):
    result = gate.extract("annual_revenue_crore", (statement(units="Amount in Rs."),))
    assert result.value == "0.0001200"
    assert result.evidence_validation.source_unit == "rupee"


def test_utility_bill_never_supplies_revenue(gate):
    identity = chunk(
        "Udyam registration certificate\nLegal name: Example Manufacturing LLP", "identity"
    )
    bill = chunk(
        "Electricity bill\nLegal name: Example Manufacturing LLP\nAnnual revenue: 1200", "bill"
    )
    with pytest.raises(EvidenceInsufficient, match="document_type_missing"):
        gate.extract("annual_revenue_crore", (identity, bill))


def test_vendor_only_report_requests_authoritative_source(gate):
    report = chunk(
        "Third-party financial information report\nLegal name: Example Manufacturing LLP\n"
        "Auditor's report\nStandalone financial data\nNet Revenue: 1200"
    )
    with pytest.raises(EvidenceInsufficient, match="evidence_document_type_missing"):
        gate.extract("annual_revenue_crore", (report,))


def test_matching_proof_can_supply_address(gate):
    identity = chunk(
        "Udyam registration certificate\nLegal name: Example Manufacturing LLP", "identity"
    )
    bill = chunk(
        "Electricity bill\nLegal name: Example Manufacturing LLP\n"
        "Service address: Pune Maharashtra",
        "bill",
    )
    result = gate.extract("region", (identity, bill))
    assert result.value == "West"
    assert result.evidence_validation.document_type == "utility_bill"


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"basis": ""}, "basis_ambiguous"),
        ({"basis": "Standalone and consolidated"}, "basis_ambiguous"),
        ({"units": ""}, "unit_ambiguous"),
        ({"units": "Amounts in lakhs and amounts in crore"}, "unit_ambiguous"),
        ({"row": "| Revenue from operations | approximately 1200 | 900 |"}, "value_ambiguous"),
    ],
)
def test_uncertain_financial_context_stops(gate, kwargs, code):
    with pytest.raises(EvidenceInsufficient, match=code):
        gate.extract("annual_revenue_crore", (statement(**kwargs),))


def test_other_company_proof_stops(gate):
    foreign = chunk(
        "Electricity bill\nLegal name: Foreign Borrower LLP\nService address: Delhi", "proof"
    )
    with pytest.raises(EvidenceInsufficient, match="entity_ambiguous"):
        gate.extract("region", (statement(), foreign))


def test_serial_and_note_columns_are_not_amounts(gate):
    source = statement().model_copy(
        update={
            "text": "Audited financial statements\nLegal name: Example Manufacturing LLP\n"
            "Standalone\nAmounts in lakhs\n"
            "| | Particulars | Note No | 31-03-2025 | 31-03-2024 |\n"
            "| 1 | Revenue from operations | 18 | 1200 | 900 |"
        }
    )
    assert gate.extract("annual_revenue_crore", (source,)).value == "12.00"


def test_registration_field_on_following_line(gate):
    source = chunk("Udyam registration\nNAME OF ENTERPRISE\nExample Manufacturing LLP")
    assert gate.extract("borrower_name", (source,)).value == "Example Manufacturing LLP"


def test_table_context_cannot_borrow_table_bbox_for_another_field(gate):
    identity = chunk("Udyam registration\nLegal name: Example Manufacturing LLP", "registration")
    table = TableChunk(
        chunk_id="table", text=identity.text + "\n| Assets | 10 |",
        table_text="| Assets | 10 |", provenance=identity.provenance,
    )
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract("borrower_name", (table,))


def test_conflicting_latest_figures_stop(gate):
    other = statement(row="| Revenue from operations | 1300 | 900 |").model_copy(
        update={"chunk_id": "conflict"}
    )
    with pytest.raises(EvidenceInsufficient, match="conflicting_values"):
        gate.extract("annual_revenue_crore", (statement(), other))


def test_missing_newer_year_metric_does_not_select_old_value(gate):
    later = chunk(
        "Audited financial statements\nLegal name: Example Manufacturing LLP\n"
        "Standalone\nAmounts in lakhs\n| Particulars | FY2025-26 |\n| Assets | 4000 |",
        "new-financials",
    )
    with pytest.raises(EvidenceInsufficient, match="latest_period_missing"):
        gate.extract("annual_revenue_crore", (statement(), later))


def test_unlabelled_number_is_not_evidence(gate):
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract(
            "annual_revenue_crore", (statement(row="| Electricity expense | 1200 | 900 |"),)
        )


def test_reversed_year_columns_select_latest(gate):
    source = statement().model_copy(
        update={"text": statement().text.replace("FY2024-25 | FY2023-24", "FY2023-24 | FY2024-25")}
    )
    assert gate.extract("annual_revenue_crore", (source,)).value == "9.00"


def test_missing_identity_and_unknown_document_stop(gate):
    with pytest.raises(EvidenceInsufficient, match="entity_ambiguous"):
        gate.extract("annual_revenue_crore", (chunk("Revenue: 12"),))


def test_proof_cannot_substitute_for_missing_dscr(gate):
    proof = chunk("No dues certificate\nLegal name: Example Manufacturing LLP\nDSCR: 10", "proof")
    with pytest.raises(EvidenceInsufficient, match="metric_missing"):
        gate.extract("dscr", (statement(), proof))


def test_unrelated_maturity_date_does_not_change_reporting_year(gate):
    source = statement().model_copy(update={"text": statement().text + "\nLoan matures in 2030"})
    assert gate.extract("annual_revenue_crore", (source,)).period == "2025"


def test_bill_with_audited_keyword_does_not_become_financial_evidence(gate):
    source = statement().model_copy(update={"text": statement().text + "\nElectricity bill"})
    with pytest.raises(EvidenceInsufficient, match="document_ambiguous"):
        gate.extract("annual_revenue_crore", (source,))


def test_percentage_scale_must_be_explicit(gate):
    with pytest.raises(EvidenceInsufficient, match="unit_ambiguous"):
        gate.extract("ebitda_margin_pct", (statement(row="| EBITDA margin | 0.12 | 0.10 |"),))
    result = gate.extract("ebitda_margin_pct", (statement(row="| EBITDA margin (%) | 12 | 10 |"),))
    assert result.value == "12"
    with pytest.raises(EvidenceInsufficient, match="value_ambiguous"):
        gate.extract("ebitda_margin_pct", (statement(row="| EBITDA margin | 1%2 | 10% |"),))


def test_region_does_not_confuse_west_bengal_with_west(gate):
    proof = chunk(
        "Udyam registration\nLegal name: Example Manufacturing LLP\n"
        "Registered address: Kolkata West Bengal"
    )
    assert gate.extract("region", (proof,)).value == "East"


def test_counterparty_on_later_page_is_not_document_owner(gate):
    title = chunk("Audited financial statements\nEXAMPLE MANUFACTURING LLP", page=1)
    table = statement().model_copy(
        update={
            "text": statement().text.replace("Legal name: Example Manufacturing LLP\n", ""),
            "provenance": (title.provenance[0].model_copy(update={"page_number": 2}),),
            "chunk_id": "table-page-2",
        }
    )
    counterparty = chunk("Counterparty\nFOREIGN CUSTOMER LLP", page=3)
    assert gate.extract("annual_revenue_crore", (title, table, counterparty)).value == "12.00"


def test_legal_salutation_and_audit_addressee_resolve_same_owner(gate):
    registration = chunk(
        "GST registration\nLegal name: M/S Example Manufacturing LLP", "registration"
    )
    source = statement().model_copy(
        update={
            "text": statement().text.replace("Legal name:", "To the Partners of"),
        }
    )
    assert gate.extract("annual_revenue_crore", (registration, source)).value == "12.00"
