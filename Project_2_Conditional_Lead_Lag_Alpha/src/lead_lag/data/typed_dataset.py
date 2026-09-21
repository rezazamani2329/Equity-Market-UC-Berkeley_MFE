"""
Typed dataset contract: a pandas DataFrame whose schema is guaranteed on
construction.

Provenance: copied from `asset_embeddings/data/typed_dataset.py` in the 230ZA
project, which itself adapted `quantdata/core/dataset.py` from
eliasroub/quant-data-toolkit.  Copied, not imported — this repo stays
self-contained so a teammate can clone and run it without any of my other
work on their machine.

Why a wrapper at all
--------------------
A raw WRDS extract is "just a DataFrame", but every downstream function makes
silent assumptions about it: which columns exist, what they are called, that
dates are parsed, that ids are integers.  Putting those assumptions in ONE
class means they are checked once, at the boundary, and every function that
receives a `CRSPDaily` or a `Universe` can rely on them.  The pattern is: any
dataset object that exists is schema-valid.

That matters more here than usual, because this project is split five ways.
The datasets defined in `data/` are the *interface* between my part and the
other four: when the signal lead hands `DailyReturns` to a regression, the
column names and dtypes are a promise, not a convention.

Vocabulary
----------
* **key** — the columns that identify one row (the "natural key"), e.g.
  `("date", "permno")` for a daily return.  Duplicate keys are a data-quality
  issue, detected by a `QualityCheck`, not by this class.
* **required** — columns that must be present for the dataset to be usable.
  Extra columns are allowed and pass through untouched.
* **synonyms** — vendor spellings mapped to our canonical names, so the same
  loader code works whether the raw frame says `ncusip` or `cusip`.
* **coerce** — the subclass hook that parses dates, casts dtypes and drops
  structurally broken rows (e.g. a null key).

Subclasses declare the three class attributes and, if needed, override
`_coerce`.  They never override `__init__`.
"""

from __future__ import annotations

from typing import ClassVar, Self  # Self: "the concrete subclass", for with_frame/from_raw

import pandas as pd


class Dataset:
    """Base class for all typed datasets in this project.

    Class attributes (declared by each subclass):
        KEY:       natural-key column names, e.g. ("date", "permno").
        REQUIRED:  columns that must exist; a frame missing one is rejected.
        SYNONYMS:  {raw column name -> canonical name}, applied in `from_raw`.

    The constructor only *validates*; normalisation lives in `from_raw`.
    Construct with `from_raw` for vendor frames and with the plain constructor
    only for frames that are already canonical (e.g. the output of a
    transform).
    """

    # ClassVar tells pyright these are per-class declarations, not per-instance
    # fields, so a subclass may override them with different tuples.
    KEY: ClassVar[tuple[str, ...]] = ()
    REQUIRED: ClassVar[tuple[str, ...]] = ()
    SYNONYMS: ClassVar[dict[str, str]] = {}

    def __init__(self, frame: pd.DataFrame) -> None:
        # Set difference: every REQUIRED name that is not a column of `frame`.
        # An empty set means the frame is acceptable.
        missing = set(self.REQUIRED) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{type(self).__name__} missing required columns: {sorted(missing)}. "
                f"Found: {sorted(frame.columns)}"
            )
        # The wrapped frame.  Single underscore = "internal"; read it through
        # the `frame` property, replace it through `with_frame`.
        self._frame = frame

    # ------------------------------------------------------------------ views
    @property
    def frame(self) -> pd.DataFrame:
        """The underlying DataFrame — always available as an escape hatch."""
        return self._frame

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rows={len(self._frame)})"

    # --------------------------------------------------------- construction
    @classmethod
    def from_raw(cls, df: pd.DataFrame) -> Self:
        """Normalise a vendor frame into this dataset's schema.

        Steps, in order:
          1. lowercase + strip every column name (WRDS is lowercase already,
             hand-built CSVs often are not);
          2. rename synonyms to canonical names;
          3. verify REQUIRED columns exist (fail loudly, listing what was found);
          4. subclass-specific coercion (`_coerce`);
          5. wrap.

        The input frame is copied first, so the caller's object is untouched.
        """
        df = df.copy()
        # Step 1: canonical column spelling.  `str(c)` guards against integer
        # column labels that a bare `.strip()` would reject.
        df.columns = [str(c).strip().lower() for c in df.columns]
        # Step 2: dict comprehension builds {old: new} for EVERY column; names
        # without a synonym map to themselves, which `rename` treats as a no-op.
        df = df.rename(columns={c: cls.SYNONYMS.get(c, c) for c in df.columns})

        # Step 3: same check as the constructor, but done here too so the error
        # message points at `from_raw` (the boundary) rather than at `__init__`.
        missing = set(cls.REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(
                f"{cls.__name__}.from_raw: missing required columns {sorted(missing)}. "
                f"Found after synonym mapping: {sorted(df.columns)}"
            )

        # Step 4 + 5.
        df = cls._coerce(df)
        return cls(df)

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Subclass hook: parse dates, cast dtypes, drop structurally invalid rows.

        The base implementation is the identity.  Subclasses return a NEW
        frame (or the same one after in-place column assignment — `from_raw`
        already copied it, so in-place edits here are safe).
        """
        return df

    def with_frame(self, frame: pd.DataFrame) -> Self:
        """Return a new dataset of the same concrete type wrapping `frame`.

        Transforms and cleaning steps use this so they never mutate their
        input: `return dataset.with_frame(new_frame)`.  Validation runs again
        through the constructor, so a transform that drops a REQUIRED column
        fails immediately rather than three functions later.
        """
        return type(self)(frame)
