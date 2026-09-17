"""Fixed flat accounting formula definitions; never execute document expressions."""

from dataclasses import dataclass
from decimal import Decimal, localcontext


@dataclass(frozen=True)
class Formula:
    metric: str
    operands: tuple[str, ...]
    numerator: tuple[int, ...]
    denominator: tuple[int, ...] = ()
    scale: int = 1
    unit: str = "INR crore"

    def evaluate(self, values: tuple[Decimal, ...]) -> Decimal:
        if len(values) != len(self.operands) or not all(v.is_finite() for v in values):
            raise ValueError("invalid calculation operands")
        for name, value in zip(self.operands, values, strict=True):
            if value < 0 and (
                "borrowings" in name
                or name
                in {
                    "all_loan_principal_due_crore",
                    "all_loan_interest_due_crore",
                    "additional_lease_service_crore",
                    "operating_expenses_crore",
                    "pnl_total_expenses_crore",
                    "finance_cost_crore",
                }
                or name in {"da_crore", "term_interest_due_crore", "principal_due_crore"}
            ):
                raise ValueError("negative calculation component")
        with localcontext() as context:
            context.prec = 28
            top = sum((v * c for v, c in zip(values, self.numerator, strict=True)), Decimal(0))
            bottom = (
                sum((v * c for v, c in zip(values, self.denominator, strict=True)), Decimal(0))
                if self.denominator
                else Decimal(1)
            )
            if bottom <= 0:
                raise ValueError("nonpositive calculation denominator")
            return top / bottom * self.scale


FORMULAS = {
    # NOI: operating revenue less operating expenses, including depreciation,
    # excluding finance costs and tax. Never substitute EBITDA or CFADS.
    "dscr_noi_operating_components_total_v1": Formula(
        "dscr",
        ("operating_revenue_crore", "operating_expenses_crore", "debt_service_crore"),
        (1, -1, 0),
        (0, 0, 1),
        1,
        "x",
    ),
    "dscr_noi_operating_components_schedule_v1": Formula(
        "dscr",
        (
            "operating_revenue_crore",
            "operating_expenses_crore",
            "all_loan_principal_due_crore",
            "all_loan_interest_due_crore",
            "additional_lease_service_crore",
        ),
        (1, -1, 0, 0, 0),
        (0, 0, 1, 1, 1),
        1,
        "x",
    ),
    "dscr_noi_pnl_components_total_v1": Formula(
        "dscr",
        (
            "operating_revenue_crore",
            "pnl_total_expenses_crore",
            "finance_cost_crore",
            "debt_service_crore",
        ),
        (1, -1, 1, 0),
        (0, 0, 0, 1),
        1,
        "x",
    ),
    "dscr_noi_pnl_components_schedule_v1": Formula(
        "dscr",
        (
            "operating_revenue_crore",
            "pnl_total_expenses_crore",
            "finance_cost_crore",
            "all_loan_principal_due_crore",
            "all_loan_interest_due_crore",
            "additional_lease_service_crore",
        ),
        (1, -1, 1, 0, 0, 0),
        (0, 0, 0, 1, 1, 1),
        1,
        "x",
    ),
    "dscr_noi_reported_total_v1": Formula(
        "dscr",
        ("noi_crore", "debt_service_crore"),
        (1, 0),
        (0, 1),
        1,
        "x",
    ),
    "dscr_noi_reported_schedule_v1": Formula(
        "dscr",
        (
            "noi_crore",
            "all_loan_principal_due_crore",
            "all_loan_interest_due_crore",
            "additional_lease_service_crore",
        ),
        (1, 0, 0, 0),
        (0, 1, 1, 1),
        1,
        "x",
    ),
    "debt_ebitda_borrowings_v1": Formula(
        "debt_to_ebitda",
        ("long_term_borrowings_crore", "short_term_borrowings_crore", "ebitda_crore"),
        (1, 1, 0),
        (0, 0, 1),
        1,
        "x",
    ),
    "dscr_cfads_all_obligations_v1": Formula(
        "dscr",
        (
            "cfads_crore",
            "all_loan_principal_due_crore",
            "all_loan_interest_due_crore",
            "additional_lease_service_crore",
        ),
        (1, 0, 0, 0),
        (0, 1, 1, 1),
        1,
        "x",
    ),
    "dscr_cash_accrual_pbt_v1": Formula(
        "dscr",
        (
            "pbt_crore",
            "net_tax_crore",
            "da_crore",
            "term_interest_due_crore",
            "principal_due_crore",
        ),
        (1, -1, 1, 1, 0),
        (0, 0, 0, 1, 1),
        1,
        "x",
    ),
    # Borrowings-based policy: current maturities already included in short-term
    # borrowings must not be added again; trade payables are not loan debt.
    "debt_equity_llp_borrowings_v1": Formula(
        "debt_to_equity",
        (
            "long_term_borrowings_crore",
            "short_term_borrowings_crore",
            "partners_contribution_crore",
            "partners_current_account_crore",
        ),
        (1, 1, 0, 0),
        (0, 0, 1, 1),
        1,
        "x",
    ),
    "debt_equity_borrowings_v1": Formula(
        "debt_to_equity",
        ("long_term_borrowings_crore", "short_term_borrowings_crore", "total_equity_crore"),
        (1, 1, 0),
        (0, 0, 1),
        1,
        "x",
    ),
    "debt_ebitda_borrowings_pbt_v1": Formula(
        "debt_to_ebitda",
        (
            "long_term_borrowings_crore",
            "short_term_borrowings_crore",
            "pbt_crore",
            "finance_cost_crore",
            "da_crore",
        ),
        (1, 1, 0, 0, 0),
        (0, 0, 1, 1, 1),
        1,
        "x",
    ),
    "ebitda_margin_v1": Formula(
        "ebitda_margin_pct", ("ebitda_crore", "annual_revenue_crore"), (1, 0), (0, 1), 100, "%"
    ),
    "dscr_cfads_v1": Formula("dscr", ("cfads_crore", "debt_service_crore"), (1, 0), (0, 1), 1, "x"),
    "debt_equity_v1": Formula(
        "debt_to_equity", ("total_debt_crore", "total_equity_crore"), (1, 0), (0, 1), 1, "x"
    ),
    "debt_ebitda_v1": Formula(
        "debt_to_ebitda", ("total_debt_crore", "ebitda_crore"), (1, 0), (0, 1), 1, "x"
    ),
    "pat_pbt_net_tax_v1": Formula("pat_crore", ("pbt_crore", "net_tax_crore"), (1, -1)),
    "ebitda_pbt_finance_da_v1": Formula(
        "ebitda_crore", ("pbt_crore", "finance_cost_crore", "da_crore"), (1, 1, 1)
    ),
    "ebitda_margin_pbt_finance_da_v1": Formula(
        "ebitda_margin_pct",
        ("pbt_crore", "finance_cost_crore", "da_crore", "annual_revenue_crore"),
        (1, 1, 1, 0),
        (0, 0, 0, 1),
        100,
        "%",
    ),
    "debt_ebitda_pbt_finance_da_v1": Formula(
        "debt_to_ebitda",
        ("total_debt_crore", "pbt_crore", "finance_cost_crore", "da_crore"),
        (1, 0, 0, 0),
        (0, 1, 1, 1),
        1,
        "x",
    ),
    "dscr_cfads_schedule_v1": Formula(
        "dscr",
        ("cfads_crore", "term_interest_due_crore", "principal_due_crore"),
        (1, 0, 0),
        (0, 1, 1),
        1,
        "x",
    ),
    # This alternative requires an explicitly selected lender cash-accrual policy.
    # It is NOT a default substitute for CFADS and does not treat all finance costs as interest.
    "dscr_cash_accrual_v1": Formula(
        "dscr",
        ("pat_crore", "da_crore", "term_interest_due_crore", "principal_due_crore"),
        (1, 1, 1, 0),
        (0, 0, 1, 1),
        1,
        "x",
    ),
}
