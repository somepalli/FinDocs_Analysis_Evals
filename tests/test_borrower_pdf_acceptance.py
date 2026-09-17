from pathlib import Path

from findociq.ingest.chunker import LayoutAwareChunker
from findociq.ingest.docling_parser import DocumentParser, ParserConfig
from findociq.reason.classification import ClassificationPolicy, classify_sections
from findociq.reason.evidence_gate import BorrowerEvidenceGate, EvidencePolicy


def test_five_real_synthetic_pdfs_parse_and_assess_without_mocked_chunks():
    files = sorted((Path(__file__).parent / "fixtures" / "borrower").glob("*.pdf"))
    assert len(files) == 5
    parser = DocumentParser(ParserConfig(prefer_docling=False))
    chunker = LayoutAwareChunker()
    chunks = tuple(c for file in files for c in chunker.chunk(parser.parse(file)))
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    fields = BorrowerEvidenceGate(policy).assess(tuple(policy.metrics), chunks)
    # This unchanged fixture reports a DSCR but supplies no CFADS/debt-service
    # operands. It must no longer masquerade as calculated-DSCR acceptance.
    assert {f.metric_id: f.error_code for f in fields if f.status != "supported"} == {
        "dscr": "evidence_metric_missing"
    }
    dscr = next(f for f in fields if f.metric_id == "dscr")
    assert dscr.figure is None
    assert any(d.error_code == "evidence_metric_missing" for d in dscr.operand_diagnostics)
    assert next(f.figure.value for f in fields if f.metric_id == "annual_revenue_crore") == "12.00"
    for field in fields:
        if field.figure is None:
            continue
        assert field.figure.citation.page_number >= 1
        if field.figure.evidence_validation.basis:
            assert field.figure.evidence_validation.context_citations
    kinds = {
        kind
        for section in classify_sections(chunks, ClassificationPolicy.load())
        for kind in section.types
    }
    assert {"udyam", "gst", "agreement", "audited_financials", "financial_statement"} <= kinds
