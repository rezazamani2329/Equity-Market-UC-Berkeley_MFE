"""
Data-quality vocabulary: Issue, Severity, QualityCheck, QualityAudit,
QualityReport — plus the concrete checks this project's daily panel needs.

Provenance: the five vocabulary classes are copied from
`asset_embeddings/data/quality_checks.py` in the 230ZA project (itself adapted
from `quantdata/core/quality.py` in eliasroub/quant-data-toolkit).  The
concrete checks at the bottom are new and specific to a daily CRSP panel.

Why one vocabulary
------------------
A duplicated `(date, permno)` row, a stock whose price never moves, and a
return of +4,000% are different problems found by different code — but they
should be *reported* in one shape, so that a single table tells the reader
what was wrong and how bad it was.  For a five-person project that table is
also the handoff: parts 2-5 need to know what part 1 removed and why, and
"here is the audit frame" answers that better than a paragraph.

Detection and validation are the same code run twice:
  * `QualityAudit.run(dataset)` -> a `QualityReport` listing every finding;
  * `QualityAudit.assert_clean(dataset)` re-runs the checks and raises if any
    finding remains — used after cleaning to PROVE the cleaning did its job.

Vocabulary
----------
* **Issue** — one finding: which entity (a permno, a date, an industry), what
  kind of problem, when, a human-readable description, and how severe.
* **QualityCheck** — one detector.  Subclasses implement `run(dataset)` and
  return a list of Issues (possibly empty).
* **QualityAudit** — an ordered list of checks, run against one dataset.
* **QualityReport** — the audit's output, with tabular views.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum

import pandas as pd

from lead_lag.data.typed_dataset import Dataset


class Severity(Enum):
    """How bad a finding is.  Cleaning steps and the blocking audit key off it:
    HIGH findings must be gone after cleaning; LOW ones are informational."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)  # frozen: an Issue is a record, never edited after creation
class Issue:
    """A single data-quality finding, in a shape shared by every check.

    Attributes:
        entity:      what the finding is about — a permno, a date, an industry
                     label — as a string.
        issue_type:  short machine-readable tag, e.g. "duplicate_key".
        date:        the date the finding refers to (None if global).
        description: one human-readable sentence with the numbers.
        severity:    Severity.LOW / MEDIUM / HIGH.
        field:       the column involved, when there is one.
    """

    entity: str
    issue_type: str
    date: date | pd.Timestamp | None
    description: str
    severity: Severity
    field: str | None = None


class QualityCheck(ABC):
    """One detector.  Subclasses set `issue_type` and implement `run`."""

    issue_type: str = ""

    @abstractmethod
    def run(self, dataset: Dataset) -> list[Issue]:
        """Inspect `dataset` and return every finding (an empty list = clean)."""
        ...


class QualityReport:
    """The result of an audit: a list of Issues with tabular views."""

    # Column order of the tabular view — fixed so reports are comparable
    # across runs and across data types.
    COLUMNS = ("entity", "issue_type", "issue_date", "description", "severity")

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = list(issues)  # defensive copy: the caller's list stays theirs

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0

    def __len__(self) -> int:
        return len(self.issues)

    def to_frame(self) -> pd.DataFrame:
        """Tabular report: entity | issue_type | issue_date | description | severity.

        Severity is stored as its string value ("high"), not the Enum member,
        so the frame can be written to parquet / CSV as-is.
        """
        rows = [
            {
                "entity": i.entity,
                "issue_type": i.issue_type,
                "issue_date": i.date,
                "description": i.description,
                "severity": i.severity.value,
            }
            for i in self.issues
        ]
        # Passing `columns=` keeps the column order (and yields an empty frame
        # with the right columns when there are no issues).
        return pd.DataFrame(rows, columns=list(self.COLUMNS))

    def by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    def summary(self) -> pd.Series:
        """Issue counts by severity, as an int Series indexed by "low"/"medium"/"high".

        Always returns all three levels (zeros included) so callers can index
        `summary()["high"]` without a KeyError on a clean dataset.
        """
        counts = self.to_frame()["severity"].value_counts()
        # `reindex` inserts the missing levels as NaN; fill and cast back to int.
        return counts.reindex([s.value for s in Severity]).fillna(0).astype(int)


class QualityAudit:
    """Runs an ordered suite of checks against one dataset."""

    def __init__(self, checks: list[QualityCheck]) -> None:
        self.checks = list(checks)

    def run(self, dataset: Dataset) -> QualityReport:
        issues: list[Issue] = []
        for check in self.checks:
            issues.extend(check.run(dataset))  # each check contributes 0..n findings
        return QualityReport(issues)

    def assert_clean(self, dataset: Dataset) -> None:
        """Re-run the checks and raise AssertionError if any finding remains.

        Used as post-cleaning validation: an audit built from the blocking
        checks (the HIGH-severity ones) proves the cleaning pipeline did its
        job.  The message carries the counts and the first finding, so a
        failing test says what is wrong without a debugger.
        """
        report = self.run(dataset)
        if not report.is_clean:
            counts = report.summary().to_dict()
            raise AssertionError(
                f"{type(dataset).__name__} failed validation: {len(report)} issue(s) "
                f"remain ({counts}). First: {report.issues[0]}"
            )


# ===========================================================================
# Concrete checks for the daily panel
# ===========================================================================
#
# These are the ones that actually fire on a CRSP daily pull.  Each is small
# on purpose: one question, one issue_type, so the report's `issue_type`
# column is a usable group-by.


class DuplicateKeyCheck(QualityCheck):
    """HIGH: the dataset's natural key is not unique.

    Every downstream merge in this project is a join on `(date, permno)` or
    `(date, ff49)`.  A duplicated key silently FANS OUT the join — the panel
    grows, and every cross-sectional mean is then a weighted average with
    weights nobody chose.  This is the check that must never fire after
    cleaning.
    """

    issue_type = "duplicate_key"

    def run(self, dataset: Dataset) -> list[Issue]:
        key = list(dataset.KEY)
        if not key:
            return []
        df = dataset.frame
        # `duplicated(keep=False)` marks EVERY member of a duplicated group,
        # not just the repeats, so the count below is the number of rows
        # involved rather than the number of excess rows.
        dup = df.duplicated(subset=key, keep=False)
        if not dup.any():
            return []
        n_groups = df.loc[dup, key].drop_duplicates().shape[0]
        return [
            Issue(
                entity=type(dataset).__name__,
                issue_type=self.issue_type,
                date=None,
                description=(
                    f"{int(dup.sum())} rows in {n_groups} duplicated key group(s) "
                    f"on {key}"
                ),
                severity=Severity.HIGH,
                field=",".join(key),
            )
        ]


class MissingReturnCheck(QualityCheck):
    """MEDIUM: stocks whose return column is missing on more than `max_share`
    of their own observed days.

    A daily lead-lag regression lines up day t-1 against day t.  A stock with
    scattered missing returns still has rows, so the alignment silently
    compares non-adjacent days.  Flagging the worst offenders lets the
    universe filter drop them by rule rather than by accident.
    """

    issue_type = "missing_return"

    def __init__(self, column: str = "ret", max_share: float = 0.10) -> None:
        self.column = column
        self.max_share = max_share

    def run(self, dataset: Dataset) -> list[Issue]:
        df = dataset.frame
        if self.column not in df.columns or "permno" not in df.columns:
            return []
        share = df.groupby("permno")[self.column].apply(lambda s: s.isna().mean())
        bad = share[share > self.max_share]
        return [
            Issue(
                entity=str(permno),
                issue_type=self.issue_type,
                date=None,
                description=(
                    f"{self.column} missing on {value:.1%} of this stock's rows "
                    f"(threshold {self.max_share:.0%})"
                ),
                severity=Severity.MEDIUM,
                field=self.column,
            )
            for permno, value in bad.items()
        ]


class ExtremeReturnCheck(QualityCheck):
    """MEDIUM: single-day returns outside [-`bound`, +`bound`].

    Not automatically an error — a daily return of +200% happens on real
    biotech news — but it is the signature of an unadjusted split or a stale
    price followed by a catch-up print, both of which would dominate an
    equal-weighted lead-lag coefficient.  Reported, never dropped here; the
    decision to winsorise belongs to the analysis, not to the loader.
    """

    issue_type = "extreme_return"

    def __init__(self, column: str = "ret", bound: float = 1.0) -> None:
        self.column = column
        self.bound = bound

    def run(self, dataset: Dataset) -> list[Issue]:
        df = dataset.frame
        if self.column not in df.columns:
            return []
        hit = df[self.column].abs() > self.bound
        if not hit.any():
            return []
        worst = df.loc[hit, self.column].abs().max()
        return [
            Issue(
                entity=type(dataset).__name__,
                issue_type=self.issue_type,
                date=None,
                description=(
                    f"{int(hit.sum())} row(s) with |{self.column}| > {self.bound:.0%}; "
                    f"largest {worst:.1%}"
                ),
                severity=Severity.MEDIUM,
                field=self.column,
            )
        ]


class CalendarGapCheck(QualityCheck):
    """LOW: trading days on which the panel has implausibly few stocks.

    A day with a tenth of the usual cross-section is normally a half-session
    (the day after Thanksgiving, Christmas Eve) or a partial WRDS pull.  Both
    matter for a lead-lag design, which treats consecutive rows as
    consecutive sessions, so they are surfaced rather than assumed away.
    """

    issue_type = "thin_cross_section"

    def __init__(self, min_share_of_median: float = 0.5) -> None:
        self.min_share_of_median = min_share_of_median

    def run(self, dataset: Dataset) -> list[Issue]:
        df = dataset.frame
        if "date" not in df.columns or "permno" not in df.columns:
            return []
        counts = df.groupby("date")["permno"].nunique()
        if counts.empty:
            return []
        median = counts.median()
        thin = counts[counts < self.min_share_of_median * median]
        return [
            Issue(
                entity=str(day.date() if hasattr(day, "date") else day),
                issue_type=self.issue_type,
                date=day,
                description=(
                    f"{int(n)} stocks on this date vs a median of {int(median)} "
                    f"({n / median:.0%} of typical)"
                ),
                severity=Severity.LOW,
                field="permno",
            )
            for day, n in thin.items()
        ]


def daily_panel_audit(
    max_missing_share: float = 0.10, extreme_bound: float = 1.0
) -> QualityAudit:
    """The standard audit for a daily CRSP panel: all four checks, in order.

    Kept as a function rather than a module-level constant so the thresholds
    are visible at the call site in the notebook — the cleaning report should
    say "missing-return threshold 10%", not "the usual one".
    """
    return QualityAudit(
        [
            DuplicateKeyCheck(),
            MissingReturnCheck(max_share=max_missing_share),
            ExtremeReturnCheck(bound=extreme_bound),
            CalendarGapCheck(),
        ]
    )


def blocking_audit() -> QualityAudit:
    """Only the HIGH-severity checks — the ones that must be clean after the
    universe filter runs.  Used by `assert_clean` in the tests."""
    return QualityAudit([DuplicateKeyCheck()])
