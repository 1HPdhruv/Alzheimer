"""Explicit schema mapping: scripts never assume dataset column names."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

CANONICAL = ("participant_id", "visit_code", "visit_date", "diagnosis")


class SchemaError(ValueError):
    pass


@dataclass
class ColumnMap:
    """Maps canonical names -> dataset-specific columns (set in config, not guessed)."""
    participant_id: str
    visit_date: str
    diagnosis: str
    visit_code: str | None = None
    extra: dict = field(default_factory=dict)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"participant_id": self.participant_id, "visit_date": self.visit_date,
                    "diagnosis": self.diagnosis}
        missing = [src for src in required.values() if src not in df.columns]
        if missing:
            raise SchemaError(f"columns missing from data: {missing}; available: {list(df.columns)[:30]}")
        ren = {v: k for k, v in required.items()}
        if self.visit_code:
            if self.visit_code not in df.columns:
                raise SchemaError(f"visit_code column {self.visit_code!r} missing")
            ren[self.visit_code] = "visit_code"
        out = df.rename(columns=ren).copy()
        out["visit_date_raw"] = out["visit_date"]
        out["visit_date"] = pd.to_datetime(out["visit_date"], errors="coerce")
        bad = int(out["visit_date"].isna().sum())
        if bad:
            log.warning("%d/%d rows have unparseable/missing visit_date (kept, flagged)", bad, len(out))
        return out


def schema_diff(a: list[str], b: list[str]) -> dict:
    """Variable-name comparison between two data dictionaries / headers."""
    sa, sb = {c.lower() for c in a}, {c.lower() for c in b}
    return {"only_a": sorted(sa - sb), "only_b": sorted(sb - sa), "shared": sorted(sa & sb)}
