"""Save/load a FrozenModel with checksum verification. Pickle is used ONLY for artefacts produced by this pipeline: never load a
frozen-model directory from an untrusted source (unpickling executes code)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle

from yoad.models.freeze import FreezeManifest, IntegrityError
from yoad.models.frozen_pipeline import FrozenModel


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def save_frozen(model: FrozenModel, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=False)
    model.verify_integrity()
    pk = os.path.join(out_dir, "model.pkl")
    with open(pk, "wb") as f:
        pickle.dump(dict(spec=model._spec, params=model._params, imp=model._imp, sc=model._sc, est=model._est, platt=model._platt,
                         cv_table=model.cv_table, source_n=model.source_n), f)
    mf = os.path.join(out_dir, "manifest.json")
    with open(mf, "w") as f:
        json.dump(dataclasses.asdict(model.manifest), f, indent=2, default=list)
    with open(os.path.join(out_dir, "files.sha256"), "w") as f:
        f.write(f"{_sha(pk)}  model.pkl\n{_sha(mf)}  manifest.json\n")
    return {"manifest_sha256": model.manifest.manifest_sha256, "model_pkl_sha256": _sha(pk)}


def load_frozen(directory: str, expected_manifest_sha256: str | None = None) -> FrozenModel:
    sums = dict(line.split()[::-1] for line in open(os.path.join(directory, "files.sha256")).read().strip().splitlines())
    for name in ("model.pkl", "manifest.json"):
        if _sha(os.path.join(directory, name)) != sums[name]:
            raise IntegrityError(f"{name} does not match files.sha256")
    d = json.load(open(os.path.join(directory, "manifest.json")))
    d["features"] = tuple(d["features"])
    manifest = FreezeManifest(**d); manifest.verify()
    if expected_manifest_sha256 is not None and manifest.manifest_sha256 != expected_manifest_sha256:
        raise IntegrityError("not the pre-registered manifest")
    b = pickle.load(open(os.path.join(directory, "model.pkl"), "rb"))
    m = FrozenModel(b["spec"], b["params"], b["imp"], b["sc"], b["est"], b["platt"], b["cv_table"], b["source_n"], manifest)
    m.verify_integrity()
    return m
