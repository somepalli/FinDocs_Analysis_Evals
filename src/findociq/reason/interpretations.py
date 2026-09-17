"""Verify reviewer context links against owned source chunks, not free-form reasons."""

import re

from findociq.reason.schema import BasisInterpretation, citation_from_provenance, ground_citation


def resolve_interpretation(link, target, source, period, policy):
    from findociq.reason.evidence_gate import EvidenceInsufficient, _matches, _report_years

    if link.policy_hash != policy.policy_hash or link.period != period:
        raise EvidenceInsufficient("evidence_context_conflict")
    if any(p.document_id != link.document_id for c in source for p in c.provenance):
        raise EvidenceInsufficient("evidence_scope_violation")
    if target.chunk_id != link.target_chunk_id:
        raise EvidenceInsufficient("evidence_citation_ambiguous")
    try:
        ground_citation(
            link.target_citation, (citation_from_provenance(p) for p in target.provenance)
        )
    except ValueError as error:
        raise EvidenceInsufficient("evidence_citation_ambiguous") from error
    selected = []
    for citation in link.context_citations:
        if citation.document_id != link.document_id:
            raise EvidenceInsufficient("evidence_scope_violation")
        try:
            canonical = ground_citation(
                citation, (citation_from_provenance(p) for c in source for p in c.provenance)
            )
        except ValueError as error:
            raise EvidenceInsufficient("evidence_citation_ambiguous") from error
        selected.extend(
            c for c in source if any(citation_from_provenance(p) == canonical for p in c.provenance)
        )
    text = "\n".join(c.text for c in selected)
    if link.authority == "reporting_scope":
        return resolve_reporting_scope(link, target, source, selected, period, policy)
    if not any(
        re.search(pattern, text, re.I | re.M) for pattern in policy.basis_declaration_patterns
    ):
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    # A reviewer may link explicit context that the automatic resolver cannot
    # unambiguously associate, but cannot manufacture a declaration or a year.
    if _matches(text, policy.basis_patterns) != {link.basis} or set(_report_years(text)) != {
        int(period)
    }:
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    pages = {p.page_number for p in target.provenance}
    local = "\n".join(c.text for c in source if any(p.page_number in pages for p in c.provenance))
    bases = _matches(local, policy.basis_patterns)
    if bases and bases != {link.basis}:
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    return link.basis, "reviewer_context_link", link.context_citations


def resolve_reporting_scope(link, target, source, selected, period, policy):
    """Validate evidence for a human determination, never infer scope from silence."""
    from findociq.reason.evidence_gate import (
        EvidenceInsufficient,
        _matches,
        _report_years,
        profile,
    )

    owner = profile(link.document_id, tuple(source), policy)
    if owner.entity_ambiguous or not owner.entity_hash or link.entity_hash != owner.entity_hash:
        raise EvidenceInsufficient("evidence_entity_ambiguous")
    # This authority is deliberately unavailable for explicit/mixed-scope reports.
    # Those reports must use their actual declarations, not a reviewer override.
    source_text = "\n".join(c.text for c in source)
    if _matches(source_text, policy.basis_patterns):
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    if link.basis != "standalone":
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    groups = policy.reporting_scope_patterns
    if set(groups) != {"audit", "scope", "statements", "preparation"}:
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    matched = {
        name: [c for c in selected if any(re.search(p, c.text, re.I) for p in patterns)]
        for name, patterns in groups.items()
    }
    if not all(matched.values()):
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    scope_text = "\n".join(
        c.text for c in source if any(re.search(p, c.text, re.I) for p in groups["scope"])
    )
    # Related-party lists do not establish consolidation scope. The audit's own
    # description of what was audited does; never ignore a conflicting scope.
    if any(re.search(p, scope_text, re.I) for p in policy.reporting_scope_conflicts):
        raise EvidenceInsufficient("evidence_basis_ambiguous")
    if set(_report_years(scope_text)) != {int(period)}:
        raise EvidenceInsufficient("evidence_period_ambiguous")
    audit_pages = {p.page_number for c in matched["audit"] for p in c.provenance}
    audit_context = [c for c in selected if any(p.page_number in audit_pages for p in c.provenance)]
    if not all(any(p.page_number in audit_pages for p in c.provenance) for c in matched["scope"]):
        raise EvidenceInsufficient("evidence_period_ambiguous")
    # Require the owner's identity in the cited audit-page context itself.
    cited_owner = profile(link.document_id, tuple(audit_context), policy)
    if cited_owner.entity_ambiguous or cited_owner.entity_hash != owner.entity_hash:
        raise EvidenceInsufficient("evidence_entity_ambiguous")
    return link.basis, "reviewer_reporting_scope", link.context_citations


def interpretation_options(chunks, policy):
    """Offer bounded links for review, not pre-approved interpretations."""
    from findociq.reason.evidence_gate import EvidenceInsufficient, _matches, _report_years, _rows

    options = []
    declarations = []
    for chunk in chunks:
        if len(chunk.provenance) != 1 or not any(
            re.search(p, chunk.text, re.I | re.M) for p in policy.basis_declaration_patterns
        ):
            continue
        bases, years = _matches(chunk.text, policy.basis_patterns), set(_report_years(chunk.text))
        if len(bases) == len(years) == 1:
            declarations.append((chunk, next(iter(bases)), str(next(iter(years)))))
    for target in chunks:
        provenance = (
            (target.table_provenance or target.provenance)
            if target.kind == "table"
            else target.provenance
        )
        if len(provenance) != 1:
            continue
        try:
            periods = {
                period
                for rule in (*policy.metrics.values(), *policy.operands.values())
                if rule.require_basis
                for _, period in _rows(target, rule)
            }
        except EvidenceInsufficient:
            continue
        for declaration, basis, period in declarations:
            if (
                declaration.provenance[0].document_id != provenance[0].document_id
                or declaration == target
                or period not in periods
            ):
                continue
            options.append(
                BasisInterpretation(
                    document_id=provenance[0].document_id,
                    target_chunk_id=target.chunk_id,
                    target_citation=citation_from_provenance(provenance[0]),
                    basis=basis,
                    period=period,
                    policy_hash=policy.policy_hash,
                    context_citations=(citation_from_provenance(declaration.provenance[0]),),
                )
            )
            if len(options) >= 100:
                return tuple(options)
    return tuple(options) + reporting_scope_options(chunks, policy, 100 - len(options))


def reporting_scope_options(chunks, policy, budget):
    """Offer unselected human attestations only when all supporting anchors resolve."""
    from findociq.reason.evidence_gate import EvidenceInsufficient, _report_years, _rows, profile

    result = []
    if budget <= 0:
        return ()
    for document in sorted({p.document_id for c in chunks for p in c.provenance}):
        source = tuple(
            sorted(
                (c for c in chunks if {p.document_id for p in c.provenance} == {document}),
                key=lambda c: (c.provenance[0].page_number, c.provenance[0].bbox.y0, c.chunk_id),
            )
        )
        owner = profile(document, source, policy)
        selected = []
        for patterns in policy.reporting_scope_patterns.values():
            match = next(
                (c for c in source if any(re.search(p, c.text, re.I) for p in patterns)), None
            )
            if match is not None:
                selected.append(match)
        # Preserve a bounded evidence packet, not every statement/table in the PDF.
        first_page = [c for c in source if any(p.page_number == 1 for p in c.provenance)]
        for predicate in (
            lambda c, doc=document, entity=owner.entity_hash: profile(doc, (c,), policy).entity_hash
            == entity,
            lambda c: bool(_report_years(c.text)),
        ):
            match = next((c for c in first_page if predicate(c)), None)
            if match is not None:
                selected.append(match)
        citations = tuple(
            {
                citation_from_provenance(p).model_dump_json(): citation_from_provenance(p)
                for c in selected
                for p in c.provenance
            }.values()
        )
        if not owner.entity_hash or not 1 <= len(citations) <= 8:
            continue
        for target in source:
            provenance = (
                target.table_provenance or target.provenance
                if target.kind == "table"
                else target.provenance
            )
            if len(provenance) != 1:
                continue
            try:
                periods = {
                    period
                    for rule in (*policy.metrics.values(), *policy.operands.values())
                    if rule.require_basis
                    for _, period in _rows(target, rule)
                }
                if len(periods) != 1 or None in periods:
                    continue
                period = next(iter(periods))
                link = BasisInterpretation(
                    document_id=document,
                    target_chunk_id=target.chunk_id,
                    target_citation=citation_from_provenance(provenance[0]),
                    basis="standalone",
                    period=period,
                    policy_hash=policy.policy_hash,
                    context_citations=citations,
                    authority="reporting_scope",
                    entity_hash=owner.entity_hash,
                )
                resolve_interpretation(link, target, source, period, policy)
            except EvidenceInsufficient:
                continue
            result.append(link)
            if len(result) >= budget:
                return tuple(result)
    return tuple(result)
