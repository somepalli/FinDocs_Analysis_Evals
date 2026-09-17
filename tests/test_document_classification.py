import pytest
from test_basis_context import page

from findociq.reason.classification import ClassificationPolicy, classify_sections


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Permanent Account Number", "pan"),
        ("Unique Identification Authority", "aadhaar"),
        ("GST REG-06", "gst"),
        ("Udyam registration", "udyam"),
        ("Certificate of incorporation", "incorporation"),
        ("LLP agreement", "agreement"),
        ("Statement of account", "bank_statement"),
        ("Electricity bill", "utility_bill"),
        ("Audited financial statements", "audited_financials"),
        ("Statement of profit and loss", "financial_statement"),
        ("Payroll register", "payroll"),
        ("Valuation report", "valuation"),
        ("Income tax return", "tax_return"),
        ("No dues certificate", "no_dues"),
        ("Unrecognised material", "unknown"),
    ],
)
def test_content_classification_has_citations(text, expected):
    result = classify_sections((page(text),), ClassificationPolicy.load())[0]
    assert result.types == (expected,)
    assert bool(result.citations) == (expected != "unknown")
    assert result.classifier_version


def test_mixed_bundle_preserves_separate_page_scopes():
    result = classify_sections(
        (page("Statement of account"), page("Audited financial statements", 2)),
        ClassificationPolicy.load(),
    )
    assert len(result) == 2
    assert result[0].page_end == 1
    assert result[1].types == ("audited_financials",)


def test_ambiguous_page_is_not_confirmed():
    result = classify_sections(
        (page("Statement of account\nElectricity bill"),), ClassificationPolicy.load()
    )[0]
    assert result.status == "ambiguous"


def test_udyam_legacy_memorandum_is_not_personal_aadhaar():
    result = classify_sections(
        (page("Udyam Registration Certificate\nUdyoga Aadhaar Memorandum"),),
        ClassificationPolicy.load(),
    )[0]
    assert result.types == ("udyam",)
    assert result.status == "confirmed"
