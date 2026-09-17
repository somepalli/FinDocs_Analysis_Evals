"""Conservative, source-specific borrower fact selection before generation.

Unknown layouts abstain. Semantic similarity and filenames are never evidence
of document authority, a financial amount, an entity, or a fiscal period.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from decimal import Decimal, localcontext
from hashlib import sha256
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from findociq.ingest.schema import Chunk
from findociq.reason.classification import ClassificationPolicy, classify_sections
from findociq.reason.nic import activities
from findociq.reason.schema import EvidenceValidation, ExtractedFigure, citation_from_provenance


class EvidenceInsufficient(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class MetricRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sources: tuple[str, ...]
    labels: tuple[str, ...]
    kind: Literal["money", "percent", "ratio", "count", "text", "date"]
    require_period: bool
    require_basis: bool
    nic_level: Literal[2, 4, 5] | None = None
    table_column: str | None = None
    entity_heading_pattern: str | None = None
    required_context_labels: tuple[str, ...] = ()


class SourceApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    application_id: str = Field(min_length=3)
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: Literal["probe42"]
    approval_id: str = Field(min_length=8)
    approved_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class EvidencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str
    classification_policy: ClassificationPolicy
    max_chunks: int
    region_patterns: dict[str, tuple[str, ...]]
    document_types: dict[str, tuple[str, ...]]
    entity_patterns: tuple[str, ...]
    basis_patterns: dict[str, tuple[str, ...]]
    basis_declaration_patterns: tuple[str, ...] = ()
    reporting_scope_patterns: dict[str, tuple[str, ...]] = {}
    reporting_scope_conflicts: tuple[str, ...] = ()
    unit_patterns: dict[str, tuple[str, ...]]
    unit_to_crore: dict[str, Decimal]
    metrics: dict[str, MetricRule]
    calculations: dict[str, str] = {}
    calculation_alternatives: dict[str, tuple[str, ...]] = {}
    operands: dict[str, MetricRule] = {}
    source_approvals: tuple[SourceApproval, ...] = ()

    @model_validator(mode="after")
    def validate_formula_policy(self):
        from findociq.reason.accounting_formulas import FORMULAS

        scopes = [(a.application_id, a.document_id) for a in self.source_approvals]
        if len(scopes) != len(set(scopes)):
            raise ValueError("duplicate source approval scope")

        for metric, primary in self.calculations.items():
            choices = (primary, *self.calculation_alternatives.get(metric, ()))
            if any(f not in FORMULAS or FORMULAS[f].metric != metric for f in choices):
                raise ValueError("formula does not match metric")
            if (
                metric == "dscr"
                and len(
                    {
                        "noi"
                        if f.startswith("dscr_noi_")
                        else "cash_accrual"
                        if "cash_accrual" in f
                        else "cfads"
                        for f in choices
                    }
                )
                != 1
            ):
                raise ValueError("DSCR definitions cannot be mixed as fallbacks")
        if set(self.calculation_alternatives) - set(self.calculations):
            raise ValueError("calculation alternatives require a primary formula")
        return self

    @classmethod
    def load(cls, path: Path) -> EvidencePolicy:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["classification_policy"] = ClassificationPolicy.load(
            path.parent / "classification.yaml"
        )
        if approval_file := os.getenv("FINDOCIQ_SOURCE_APPROVALS_FILE"):
            approvals = yaml.safe_load(Path(approval_file).read_text(encoding="utf-8"))
            payload["source_approvals"] = approvals["source_approvals"]
        return cls.model_validate(payload)

    @property
    def policy_hash(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()


class DocumentProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: str
    types: frozenset[str]
    entity_hash: str | None
    entity_ambiguous: bool


def _matches(text: str, rules: dict[str, tuple[str, ...]]) -> set[str]:
    return {
        key for key, patterns in rules.items() if any(re.search(p, text, re.I) for p in patterns)
    }


def _identity(value: str) -> str:
    value = re.sub(r"^\s*(?:m\s*/\s*s\.?|messrs\.?)\s+", "", value, flags=re.I)
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold().replace("limited", "ltd"))
    return sha256(normalized.encode()).hexdigest()


def profile(document_id: str, chunks: tuple[Chunk, ...], policy: EvidencePolicy) -> DocumentProfile:
    text = "\n".join(chunk.text for chunk in chunks)
    # The first pattern is an explicit owner field. Otherwise use only first-page
    # title lines, not counterparties and related parties throughout the report.
    explicit = {_identity(m.group(1)) for m in re.finditer(policy.entity_patterns[0], text)}
    heading = "\n".join(c.text for c in chunks if any(p.page_number == 1 for p in c.provenance))
    identities = explicit or {
        _identity(m.group(1)) for p in policy.entity_patterns[1:] for m in re.finditer(p, heading)
    }
    types = _matches(text, policy.document_types)
    if "financial_information_report" in types:
        # A vendor's references to auditors/registrations do not make its summary
        # an original audit report or a registration certificate.
        types -= {"audited_financials", "registration", "agreement", "utility_bill", "no_dues"}
    return DocumentProfile(
        document_id=document_id,
        types=frozenset(types),
        entity_hash=next(iter(identities)) if len(identities) == 1 else None,
        entity_ambiguous=len(identities) > 1,
    )


def _years(text: str) -> tuple[int, ...]:
    # Preserve column order; convert FY2023-24 to its ending year.
    pattern = r"\b(?:FY\s*)?(20\d{2})(?:\s*[-–/]\s*(20\d{2}|\d{2}))?\b"
    result = []
    for match in re.finditer(pattern, text, re.I):
        end = match.group(2)
        year = int(end if end and len(end) == 4 else "20" + end) if end else int(match.group(1))
        if year not in result:
            result.append(year)
    return tuple(result)


def _label_pattern(rule: MetricRule) -> str:
    return r"(?:" + "|".join(re.escape(label) for label in rule.labels) + r")"


def _report_years(text: str) -> tuple[int, ...]:
    """Do not mistake loan maturity dates or unrelated amounts for reporting years."""
    return tuple(
        year
        for line in text.splitlines()
        if re.search(
            r"\b(?:FY\s*20|financial year|year ended|period end|as at|particulars|metric)",
            line,
            re.I,
        )
        or re.search(
            r"^\s*\|\s*(?:balance sheet|profit (?:and|&) loss|income statement)"
            r"[^|]*\|\s*\d{1,2}\s+[A-Za-z]{3,9},?\s+20\d{2}\s*\|",
            line,
            re.I,
        )
        for year in _years(line)
    )


def _numeric_cell(text: str) -> Decimal | None:
    value = re.sub(r"\s*%$", "", text.strip().replace(",", "")).strip()
    value = re.sub(r"\s*(?:times|x)\s*$", "", value, flags=re.I)
    if re.fullmatch(r"\(\d+(?:\.\d+)?\)", value):
        value = "-" + value[1:-1]
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return None
    return Decimal(value)


def _rows(
    chunk: Chunk, rule: MetricRule, *, expected_entity: str | None = None
) -> list[tuple[str, str | None]]:
    """Only label/value rows with an explicit, resolvable year column qualify."""
    rows = []
    if rule.required_context_labels and not any(
        re.search(re.escape(label), chunk.table_text if chunk.kind == "table" else chunk.text, re.I)
        for label in rule.required_context_labels
    ):
        return []
    if rule.entity_heading_pattern and expected_entity:
        names = [
            line.strip()
            for line in chunk.text.splitlines()
            if re.fullmatch(rule.entity_heading_pattern, line.strip())
            and _identity(line.strip()) == expected_entity
        ]
        if names:
            return [(name, None) for name in dict.fromkeys(names)]
    if rule.nic_level and chunk.kind == "table":
        from findociq.reason.nic import activities

        try:
            nic = activities(chunk.table_text)
        except ValueError as error:
            raise EvidenceInsufficient("evidence_conflicting_values") from error
        if nic:
            attribute = {
                2: "division_description",
                4: "class_description",
                5: "subclass_description",
            }[rule.nic_level]
            return [("; ".join(dict.fromkeys(getattr(item, attribute) for item in nic)), None)]
    if rule.table_column and chunk.kind == "table":
        header = None
        selected = []
        for line in chunk.table_text.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            lowered = [c.casefold() for c in cells]
            if rule.table_column.casefold() in lowered:
                header = lowered
                continue
            if (
                header
                and len(cells) == len(header)
                and not all(re.fullmatch(r"[-: ]*", c) for c in cells)
            ):
                value = cells[header.index(rule.table_column.casefold())]
                if value:
                    selected.append(value)
        if selected:
            return [("; ".join(dict.fromkeys(selected)), None)]
    headers: list[str] = []
    # Captions/preceding context are useful for interpretation, but their field
    # values cannot inherit the table's bounding box.
    lines = (chunk.table_text if chunk.kind == "table" else chunk.text).splitlines()
    for line_index, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if (
            len(cells) > 1
            and any(
                re.fullmatch(
                    r"(?:particulars|metric|description|financial year|year|"
                    r"balance sheet.*|profit [& ]+loss.*|ratios.*)",
                    c,
                    re.I,
                )
                for c in cells
            )
            and any(_years(c) for c in cells[1:])
        ):
            headers = cells
        match = re.match(
            r"^\s*" + _label_pattern(rule) + r"\s*(?:\([^)]*\))?\s*(?:[:|]|$)",
            line.strip().strip("|"),
            re.I,
        )
        if len(cells) > 1:
            positions = [
                i
                for i, c in enumerate(cells)
                if re.fullmatch(
                    r"(?:(?:\d+|[a-z]|[ivx]+)[ .)-]+)?"
                    + _label_pattern(rule)
                    + r"\s*(?:\([^)]*\))?\s*\*?",
                    c,
                    re.I,
                )
            ]
            if len(positions) != 1:
                continue
            label_index = positions[0]
            if any(
                c and not re.fullmatch(r"(?:\d+|[a-z]|[ivx]+)[ .)-]*", c, re.I)
                for c in cells[:label_index]
            ):
                continue
            value_cells = [
                (i, c) for i, c in enumerate(cells) if i > label_index and c and c not in {"-", "—"}
            ]
            # A note-section title with date columns is not an amount row.
            if value_cells and all(_years(c) and _numeric_cell(c) is None for _, c in value_cells):
                continue
            if rule.require_period:
                if len(headers) != len(cells):
                    raise EvidenceInsufficient("evidence_period_ambiguous")
                dated = [(i, c, _years(headers[i])) for i, c in value_cells]
                dated = [(i, c, y[0]) for i, c, y in dated if len(y) == 1]
                if not dated:
                    raise EvidenceInsufficient("evidence_period_ambiguous")
                newest = max(y for _, _, y in dated)
                selected = [(c, str(y)) for _, c, y in dated if y == newest]
                if len(selected) != 1:
                    raise EvidenceInsufficient("evidence_period_ambiguous")
                rows.extend(selected)
            elif len(value_cells) == 1:
                rows.append((value_cells[0][1], None))
            else:
                raise EvidenceInsufficient("evidence_value_ambiguous")
        else:
            if not match:
                continue
            value = line.strip()[match.end() :].strip()
            if not value and line_index + 1 < len(lines):
                value = lines[line_index + 1].strip()
                if not value or value.startswith("|"):
                    continue
            years = tuple(dict.fromkeys(_report_years(chunk.text)))
            if rule.require_period and len(years) != 1:
                raise EvidenceInsufficient("evidence_period_ambiguous")
            rows.append((value, str(years[0]) if rule.require_period else None))
    return rows


class BorrowerEvidenceGate:
    def __init__(
        self,
        policy: EvidencePolicy,
        *,
        interpretations=(),
        interpretation_command_id=None,
        assessment_date=None,
        application_id=None,
    ) -> None:
        self.policy = policy
        self.assessment_date = assessment_date
        self.application_id = application_id
        self.interpretations = interpretations
        self.interpretation_command_id = interpretation_command_id
        if bool(interpretations) != bool(interpretation_command_id):
            raise EvidenceInsufficient("evidence_receipt_invalid")
        if len({(i.document_id, i.target_chunk_id) for i in interpretations}) != len(
            interpretations
        ):
            raise EvidenceInsufficient("evidence_context_conflict")

    def assess(self, metrics: tuple[str, ...], chunks: tuple[Chunk, ...]):
        from findociq.reason.accounting_formulas import FORMULAS
        from findociq.reason.schema import FieldAssessment, OperandDiagnostic

        outcomes = []
        for metric in metrics:
            rule = self.policy.metrics.get(metric) or self.policy.operands.get(metric)
            try:
                figure = self.extract(metric, chunks)
                outcomes.append(
                    FieldAssessment(metric_id=metric, status="supported", figure=figure)
                )
            except EvidenceInsufficient as error:
                if error.code == "evidence_scope_violation":
                    raise PermissionError("evidence_scope_violation") from error
                diagnostics = []
                formula_ids = (
                    (
                        self.policy.calculations[metric],
                        *self.policy.calculation_alternatives.get(metric, ()),
                    )
                    if metric in self.policy.calculations
                    else ()
                )
                for formula_id in formula_ids:
                    formula = FORMULAS[formula_id]
                    for index, operand in enumerate(formula.operands):
                        try:
                            self._direct(operand, chunks)
                            code = None
                        except EvidenceInsufficient as operand_error:
                            if operand_error.code == "evidence_scope_violation":
                                raise PermissionError("evidence_scope_violation") from operand_error
                            code = operand_error.code
                        diagnostics.append(
                            OperandDiagnostic(
                                formula_id=formula_id,
                                metric_id=operand,
                                error_code=code,
                                stage=(
                                    "numerator_and_debt_service"
                                    if formula.numerator[index] and formula.denominator[index]
                                    else "numerator"
                                    if formula.numerator[index]
                                    else "debt_service"
                                )
                                if metric == "dscr"
                                else None,
                            )
                        )
                candidate_citations = []
                if rule:
                    for chunk in chunks:
                        try:
                            matched = bool(_rows(chunk, rule))
                        except EvidenceInsufficient:
                            matched = False
                        if matched:
                            provenance = (
                                (chunk.table_provenance or chunk.provenance)
                                if chunk.kind == "table"
                                else chunk.provenance
                            )
                            candidate_citations.extend(
                                citation_from_provenance(p) for p in provenance
                            )
                status = (
                    "conflicting"
                    if error.code == "evidence_conflicting_values"
                    else "missing"
                    if error.code.endswith("missing")
                    else "ambiguous"
                )
                outcomes.append(
                    FieldAssessment(
                        metric_id=metric,
                        status=status,
                        error_code=error.code,
                        acceptable_sources=rule.sources if rule else (),
                        candidate_citations=tuple(candidate_citations[:20]),
                        operand_diagnostics=tuple(diagnostics),
                    )
                )
        return tuple(outcomes)

    def extract(self, metric_id: str, chunks: tuple[Chunk, ...]) -> ExtractedFigure:
        if metric_id == "years_operating" and self.assessment_date is not None:
            from datetime import date

            from findociq.reason.company_age import completed_years
            from findociq.reason.schema import CompanyAgeReceipt

            source = self._direct("incorporation_date", chunks)
            try:
                value = str(
                    completed_years(
                        date.fromisoformat(source.value),
                        date.fromisoformat(str(self.assessment_date)),
                    )
                )
            except ValueError as error:
                raise EvidenceInsufficient("evidence_value_ambiguous") from error
            receipt = source.evidence_validation.model_copy(
                update={
                    "metric_id": metric_id,
                    "source_value": value,
                    "source_unit": "count",
                    "date_calculation": CompanyAgeReceipt(
                        incorporation_date=source.value,
                        assessment_date=str(self.assessment_date),
                        source_chunk_id=source.evidence_validation.chunk_id,
                        source_citation=source.citation,
                        policy_hash=self.policy.policy_hash,
                    ),
                }
            )
            return source.model_copy(
                update={
                    "label": metric_id,
                    "value": value,
                    "unit": "count",
                    "evidence_validation": receipt,
                }
            )
        if metric_id == "dscr":
            from findociq.reason.dscr import calculate_dscr

            return calculate_dscr(self, chunks)
        else:
            try:
                return self._direct(metric_id, chunks)
            except EvidenceInsufficient as error:
                # Never calculate around conflicting, ambiguous, or stale evidence.
                if (
                    error.code != "evidence_metric_missing"
                    or metric_id not in self.policy.calculations
                ):
                    raise
        from findociq.reason.calculations import calculate

        formulas = (
            self.policy.calculations[metric_id],
            *self.policy.calculation_alternatives.get(metric_id, ()),
        )
        for formula_id in formulas:
            try:
                return calculate(
                    formula_id,
                    lambda operand: self._direct(operand, chunks),
                    self.policy.policy_hash,
                )
            except EvidenceInsufficient as error:
                # Alternatives cannot bypass a present but conflicting/ambiguous input.
                if error.code != "evidence_metric_missing":
                    raise
        raise EvidenceInsufficient("evidence_metric_missing")

    def _direct(self, metric_id: str, chunks: tuple[Chunk, ...]) -> ExtractedFigure:
        rule = self.policy.metrics.get(metric_id) or self.policy.operands.get(metric_id)
        if rule is None:
            raise EvidenceInsufficient("metric_not_allowed")
        documents: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            ids = {p.document_id for p in chunk.provenance}
            if len(ids) != 1:
                raise EvidenceInsufficient("evidence_document_ambiguous")
            documents[next(iter(ids))].append(chunk)
        profiles = {
            key: profile(key, tuple(items), self.policy) for key, items in documents.items()
        }
        approvals = {
            a.document_id: a
            for a in self.policy.source_approvals
            if a.application_id == self.application_id
        }
        allowed_types = set(rule.sources)
        if "audited_financials" in allowed_types:
            allowed_types.add("financial_information_report")
        sections = classify_sections(chunks, self.policy.classification_policy)
        sources = [
            p
            for p in profiles.values()
            if p.types.intersection(allowed_types)
            and ("financial_information_report" not in p.types or p.document_id in approvals)
        ]
        if not sources and any(
            "financial_information_report" in p.types for p in profiles.values()
        ):
            raise EvidenceInsufficient("evidence_document_type_missing")
        # Identity-bearing documents anchor the borrower. Proofs must match that identity.
        registrations = [
            p
            for p in profiles.values()
            if "registration" in p.types and "agreement" not in p.types and p.entity_hash
        ]
        anchors = registrations or [
            p
            for p in profiles.values()
            if p.types & {"registration", "agreement", "audited_financials", "pan"}
            or ("financial_information_report" in p.types and p.document_id in approvals)
        ]
        if any(p.entity_ambiguous for p in anchors):
            raise EvidenceInsufficient("evidence_entity_ambiguous")
        entities = {p.entity_hash for p in anchors if p.entity_hash}
        if len(entities) != 1:
            raise EvidenceInsufficient("evidence_entity_ambiguous")
        entity = next(iter(entities))
        if not sources:
            raise EvidenceInsufficient("evidence_document_type_missing")
        candidates = []
        latest_source_year = None
        if rule.require_period:
            source_years = [
                year
                for source in sources
                for c in documents[source.document_id]
                for year in _report_years(c.text)
            ]
            if not source_years:
                raise EvidenceInsufficient("evidence_period_ambiguous")
            latest_source_year = max(source_years)
        older_rows = False
        for source in sources:
            if "audited_financials" in source.types and source.types & {"utility_bill", "no_dues"}:
                raise EvidenceInsufficient("evidence_document_ambiguous")
            source_chunks = documents[source.document_id]
            for chunk in source_chunks:
                rows = _rows(chunk, rule, expected_entity=entity)
                if rule.require_period and rows:
                    older_rows |= any(
                        p is not None and int(p) < latest_source_year for _, p in rows
                    )
                    # Select the requested latest report period before resolving
                    # context. An older annexure cannot veto a current-year row.
                    # Undated rows still fail normal ambiguity checks below.
                    rows = [(v, p) for v, p in rows if p is None or int(p) >= latest_source_year]
                if not rows:
                    continue
                # An audit elsewhere in a mixed PDF cannot make a bank/identity
                # page authoritative for a financial metric.
                if rule.require_basis:
                    candidate_pages = {p.page_number for p in chunk.provenance}
                    incompatible = {
                        "pan",
                        "aadhaar",
                        "gst",
                        "udyam",
                        "incorporation",
                        "agreement",
                        "bank_statement",
                        "utility_bill",
                        "no_dues",
                    }
                    if any(
                        s.document_id == source.document_id
                        and set(s.types) & incompatible
                        and any(s.page_start <= p <= s.page_end for p in candidate_pages)
                        for s in sections
                    ):
                        raise EvidenceInsufficient("evidence_document_ambiguous")
                if source.entity_ambiguous or source.entity_hash != entity:
                    raise EvidenceInsufficient("evidence_entity_ambiguous")
                page_numbers = {p.page_number for p in chunk.provenance}
                page_context = "\n".join(
                    c.text
                    for c in source_chunks
                    if any(p.page_number in page_numbers for p in c.provenance)
                )
                basis, basis_resolution, context_citations = None, None, ()
                if rule.require_basis:
                    from findociq.reason.basis import resolve_basis

                    periods = {period for _, period in rows}
                    if len(periods) != 1:
                        raise EvidenceInsufficient("evidence_period_ambiguous")
                    link = next(
                        (
                            i
                            for i in self.interpretations
                            if i.document_id == source.document_id
                            and i.target_chunk_id == chunk.chunk_id
                        ),
                        None,
                    )
                    if link is not None:
                        from findociq.reason.interpretations import resolve_interpretation

                        basis, basis_resolution, context_citations = resolve_interpretation(
                            link, chunk, tuple(source_chunks), next(iter(periods)), self.policy
                        )
                    else:
                        basis, basis_resolution, context_citations = resolve_basis(
                            chunk, tuple(source_chunks), next(iter(periods)), self.policy
                        )
                if rule.kind == "percent" and not re.search(
                    _label_pattern(rule) + r"[^\n]*%", chunk.text, re.I
                ):
                    raise EvidenceInsufficient("evidence_unit_ambiguous")
                units = (
                    _matches(page_context, self.policy.unit_patterns)
                    if rule.kind == "money"
                    else set()
                )
                if rule.kind == "money" and len(units) != 1:
                    raise EvidenceInsufficient("evidence_unit_ambiguous")
                source_unit = (
                    next(iter(units))
                    if units
                    else {
                        "ratio": "x",
                        "percent": "%",
                        "count": "count",
                        "text": "text",
                        "date": "date",
                    }.get(rule.kind)
                )
                factor = (
                    self.policy.unit_to_crore[source_unit] if rule.kind == "money" else Decimal(1)
                )
                for raw_value, period in rows:
                    if rule.kind == "date":
                        from findociq.reason.company_age import incorporation_date

                        try:
                            value = incorporation_date(raw_value).isoformat()
                        except ValueError as error:
                            raise EvidenceInsufficient("evidence_value_ambiguous") from error
                    elif rule.kind == "text":
                        value = raw_value.strip()
                        if not value or len(value) > 300:
                            raise EvidenceInsufficient("evidence_value_ambiguous")
                    else:
                        number = _numeric_cell(raw_value)
                        if number is None:
                            raise EvidenceInsufficient("evidence_value_ambiguous")
                        if rule.kind == "count" and (
                            number < 0 or number != number.to_integral_value()
                        ):
                            raise EvidenceInsufficient("evidence_value_ambiguous")
                        with localcontext() as ctx:
                            ctx.prec = 28
                            value = format(number * factor, "f")
                    if metric_id == "borrower_name" and _identity(value) != entity:
                        raise EvidenceInsufficient("evidence_entity_ambiguous")
                    if metric_id == "region":
                        regions = (
                            {value.title()}
                            if value.title() in self.policy.region_patterns
                            else _matches(
                                value,
                                {
                                    name: patterns[1:]
                                    for name, patterns in self.policy.region_patterns.items()
                                },
                            )
                        )
                        if len(regions) != 1:
                            raise EvidenceInsufficient("evidence_region_ambiguous")
                        value = next(iter(regions))
                        source_unit = "address"
                    provenance = (
                        (chunk.table_provenance or chunk.provenance)
                        if chunk.kind == "table"
                        else chunk.provenance
                    )
                    if any(p not in chunk.provenance for p in provenance):
                        raise EvidenceInsufficient("evidence_citation_ambiguous")
                    if len(provenance) != 1:
                        raise EvidenceInsufficient("evidence_citation_ambiguous")
                    citation = citation_from_provenance(provenance[0])
                    receipt = EvidenceValidation(
                        policy_hash=self.policy.policy_hash,
                        metric_id=metric_id,
                        document_type=sorted(source.types.intersection(allowed_types))[0],
                        source_provider=approvals[source.document_id].provider
                        if source.document_id in approvals
                        else None,
                        source_approval_id=approvals[source.document_id].approval_id
                        if source.document_id in approvals
                        else None,
                        entity_hash=entity,
                        chunk_id=chunk.chunk_id,
                        source_value=raw_value,
                        source_unit=source_unit or "text",
                        conversion_factor=str(factor),
                        period=period,
                        basis=basis,
                        basis_resolution=basis_resolution,
                        context_citations=context_citations,
                        interpretation_command_id=(
                            self.interpretation_command_id
                            if basis_resolution
                            in {"reviewer_context_link", "reviewer_reporting_scope"}
                            else None
                        ),
                        nic_mapping_version="udyam-nic-hierarchy-v1"
                        if rule.nic_level and chunk.kind == "table" and activities(chunk.table_text)
                        else None,
                        nic_activities=activities(chunk.table_text)
                        if rule.nic_level and chunk.kind == "table"
                        else (),
                    )
                    candidates.append(
                        ExtractedFigure(
                            label=metric_id,
                            value=value,
                            unit="INR crore" if rule.kind == "money" else source_unit,
                            period=period,
                            citation=citation,
                            evidence_validation=receipt,
                        )
                    )
        if not candidates:
            if older_rows:
                raise EvidenceInsufficient("evidence_latest_period_missing")
            raise EvidenceInsufficient("evidence_metric_missing")
        nic_candidates = [c for c in candidates if c.evidence_validation.nic_mapping_version]
        if nic_candidates:
            candidates = (
                nic_candidates  # Specific NIC descriptions outrank generic "Manufacturing".
            )
        if rule.require_period:
            latest = max(c.period or "" for c in candidates)
            if latest_source_year is not None and int(latest) != latest_source_year:
                raise EvidenceInsufficient("evidence_latest_period_missing")
            candidates = [c for c in candidates if c.period == latest]
        if (
            len(
                {
                    (
                        _identity(c.value) if metric_id == "borrower_name" else c.value,
                        c.unit,
                        c.period,
                        c.evidence_validation.basis,
                    )
                    for c in candidates
                }
            )
            != 1
        ):
            raise EvidenceInsufficient("evidence_conflicting_values")
        return sorted(candidates, key=lambda c: (c.citation.document_id, c.citation.page_number))[0]
