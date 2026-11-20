"""Model-freeze manifest and version-control checks (synthetic-tested; never used on real participants)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from dataclasses import dataclass, asdict, field


class FrozenConfigError(RuntimeError):
    """A frozen model's configuration or artifact differs from its manifest without a new version."""


class IntegrityError(FrozenConfigError):
    pass


def software_versions() -> dict:
    import numpy, pandas, scipy, sklearn
    v = {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
         "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    try:
        import xgboost; v["xgboost"] = xgboost.__version__
    except Exception:  # pragma: no cover
        v["xgboost"] = None
    return v


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=lambda o: list(o) if isinstance(o, (tuple, set)) else str(o))


@dataclass(frozen=True)
class FreezeManifest:
    model_id: str
    version: str
    training_cohort_id: str
    outcome: str
    horizon_months: int
    features: tuple                    # exact names AND order
    preprocessing: dict                # imputer/scaler type, domains/units, preprocessing_version
    artifact_sha256: str               # checksum of the fitted parameters
    software: dict
    selection_result: dict             # source-only model-selection outcome (may be {} for a pre-selection freeze)
    validation_plan: dict
    freeze_date: str
    parent_manifest_sha256: str | None = None
    manifest_sha256: str = field(default="")

    def compute_hash(self) -> str:
        d = asdict(self); d.pop("manifest_sha256")
        return hashlib.sha256(_canon(d).encode()).hexdigest()

    def verify(self) -> None:
        if self.manifest_sha256 != self.compute_hash():
            raise IntegrityError("manifest content does not match its recorded checksum")


def make_manifest(**kw) -> FreezeManifest:
    kw.setdefault("freeze_date", _dt.datetime.now(_dt.timezone.utc).date().isoformat())
    kw.setdefault("software", software_versions())
    kw.setdefault("selection_result", {}); kw.setdefault("validation_plan", {})
    m = FreezeManifest(**kw)
    return FreezeManifest(**{**asdict(m), "manifest_sha256": m.compute_hash()})


def assert_same_configuration(manifest: FreezeManifest, *, features, preprocessing, horizon_months, outcome, version) -> None:
    """Raise unless the proposed configuration equals the frozen one OR it carries a different version."""
    same = (tuple(features) == tuple(manifest.features) and preprocessing == manifest.preprocessing
            and horizon_months == manifest.horizon_months and outcome == manifest.outcome)
    if not same and version == manifest.version:
        raise FrozenConfigError("configuration changed under the same version; create a new model version explicitly")


def register_new_version(old: FreezeManifest, new: FreezeManifest, reason: str) -> dict:
    old.verify(); new.verify()
    if not reason.strip():
        raise FrozenConfigError("a new version requires a written reason")
    if new.version == old.version:
        raise FrozenConfigError("new manifest must carry a new version identifier")
    if new.parent_manifest_sha256 != old.manifest_sha256:
        raise FrozenConfigError("new manifest must reference the old manifest checksum as its parent")
    return {"from": old.version, "to": new.version, "reason": reason, "parent": old.manifest_sha256}
