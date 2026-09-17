import pytest
from test_evidence_gate import chunk, gate  # noqa: F401

from findociq.ingest.schema import TableChunk
from findociq.reason.evidence_gate import EvidenceInsufficient


def source():
    identity = chunk(
        "Udyam registration certificate\nLegal name: Example Manufacturing LLP", "registration"
    )
    text = (
        "| NIC 2 Digit | NIC 4 Digit | NIC 5 Digit | Activity |\n|---|---|---|---|\n"
        "|22 - Plastics|2220 - Plastic products|22209 - Other plastic products|Manufacturing|\n"
        "|27 - Electrical equipment|2740 - Lighting|27400 - Lighting equipment|Manufacturing|"
    )
    table = TableChunk(
        chunk_id="nic",
        text=text,
        table_text=text,
        caption=None,
        preceding_context=None,
        table_id="nic-table",
        provenance=identity.provenance,
    )
    return identity, table


def test_preserve_all_nic_activities_with_citation(gate):  # noqa: F811
    result = gate.extract("industry", source())
    assert result.value == "Plastics; Electrical equipment"
    assert result.evidence_validation.nic_mapping_version == "udyam-nic-hierarchy-v1"
    assert result.evidence_validation.chunk_id == "nic"
    assert [a.subclass for a in result.evidence_validation.nic_activities] == ["22209", "27400"]
    assert all(a.activity == "Manufacturing" for a in result.evidence_validation.nic_activities)
    assert (
        gate.extract("sub_industry", source()).value == "Other plastic products; Lighting equipment"
    )


def test_conflicting_nic_hierarchy_stops(gate):  # noqa: F811
    identity, table = source()
    table = table.model_copy(update={"table_text": table.table_text.replace("22209", "27409")})
    with pytest.raises(EvidenceInsufficient, match="conflicting_values"):
        gate.extract("industry", (identity, table))


def test_activity_descriptions_are_selected_from_explicit_columns(gate):  # noqa: F811
    identity, table = source()
    text = (
        "| Main Activity Group Code | Description of Main Activity Group | "
        "Business Activity Code | Description of Business Activity | % of Turnover |\n"
        "|---|---|---|---|---|\n|Q|Health services|86.00|Health activities|100.0|"
    )
    table = table.model_copy(update={"text": text, "table_text": text})
    assert gate.extract("industry", (identity, table)).value == "Health services"
    assert gate.extract("sub_industry", (identity, table)).value == "Health activities"
