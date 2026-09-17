"""Conservative, cited accounting-basis resolution within one financial report."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from findociq.ingest.schema import Chunk
from findociq.reason.schema import SourceCitation, citation_from_provenance

if TYPE_CHECKING:
    from findociq.reason.evidence_gate import EvidencePolicy


def resolve_basis(
    target: Chunk, source: tuple[Chunk, ...], period: str | None, policy: EvidencePolicy
) -> tuple[str | None, str | None, tuple[SourceCitation, ...]]:
    # Import lazily to keep policy ownership in the evidence module.
    from findociq.reason.evidence_gate import EvidenceInsufficient, _matches, _report_years

    document_ids = {p.document_id for p in target.provenance}
    if len(document_ids) != 1 or any(
        p.document_id not in document_ids for c in source for p in c.provenance
    ):
        raise EvidenceInsufficient("evidence_scope_violation")
    pages = {p.page_number for p in target.provenance}
    local = tuple(c for c in source if any(p.page_number in pages for p in c.provenance))
    local_bases = _matches("\n".join(c.text for c in local), policy.basis_patterns)
    if len(local_bases) > 1:
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    if len(local_bases) == 1:
        basis = next(iter(local_bases))
        citations = tuple(
            citation_from_provenance(p)
            for c in local
            if basis in _matches(c.text, policy.basis_patterns)
            for p in c.provenance
        )
        return basis, "same_page", tuple({c.model_dump_json(): c for c in citations}.values())

    # A mention elsewhere (e.g. in an LLP agreement, notes or a disclaimer) is
    # not a declaration. Require an explicit heading and its reporting year.
    declarations = []
    for chunk in source:
        if not any(
            re.search(p, chunk.text, re.I | re.M) for p in policy.basis_declaration_patterns
        ):
            continue
        bases = _matches(chunk.text, policy.basis_patterns)
        if len(bases) != 1:
            raise EvidenceInsufficient("evidence_basis_ambiguous")
        declaration_pages = {p.page_number for p in chunk.provenance}
        declaration_context = tuple(
            c for c in source if any(p.page_number in declaration_pages for p in c.provenance)
        )
        years = set(_report_years("\n".join(c.text for c in declaration_context)))
        if period is None or int(period) not in years:
            continue
        # Carry both the declaration and its period heading into the receipt.
        basis = next(iter(bases))
        declarations.extend(
            (basis, c)
            for c in declaration_context
            if c == chunk or int(period) in _report_years(c.text)
        )
    # Never borrow context between mixed standalone/consolidated reports, even
    # if a conveniently nearby heading would otherwise provide a match.
    all_bases = _matches("\n".join(c.text for c in source), policy.basis_patterns)
    if len(all_bases) > 1 or len({basis for basis, _ in declarations}) != 1:
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    basis = declarations[0][0]
    citations = tuple(citation_from_provenance(p) for _, c in declarations for p in c.provenance)
    return basis, "report_declaration", tuple({c.model_dump_json(): c for c in citations}.values())
