from pathlib import Path

from findociq.reason.evidence_gate import EvidencePolicy


def test_classification_rules_are_pinned_and_part_of_evidence_policy_hash():
    policy = EvidencePolicy.load(Path("configs/evidence/borrower.yaml"))
    classifier = policy.classification_policy
    changed = classifier.model_copy(
        update={"patterns": {**classifier.patterns, "pan": (r"new detector",)}}
    )
    assert (
        policy.model_copy(update={"classification_policy": changed}).policy_hash
        != policy.policy_hash
    )
    assert policy.classification_policy == classifier
