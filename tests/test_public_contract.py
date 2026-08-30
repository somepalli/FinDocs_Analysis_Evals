from scripts.export_public_contract import CONTRACT_PATH, encoded_contract


def test_committed_public_contract_matches_producer_models() -> None:
    assert CONTRACT_PATH.read_text(encoding="utf-8") == encoded_contract()
