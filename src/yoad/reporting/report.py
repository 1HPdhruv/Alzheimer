"""Markdown/JSON report assembly (deterministic, aggregate-only)."""
from __future__ import annotations

import json
from pathlib import Path


def _fmt(v, depth=0):
    if isinstance(v, dict):
        return "\n".join(f"{'  '*depth}- **{k}**: {_fmt(x, depth+1) if isinstance(x,(dict,list)) else x}" for k, x in v.items())
    if isinstance(v, list):
        return "[" + ", ".join(map(str, v[:20])) + (" …" if len(v) > 20 else "") + "]"
    return str(v)


def write_report(sections: dict, md_path: str | Path, json_path: str | Path | None = None) -> None:
    lines = ["# Audit report", ""]
    for name, body in sections.items():
        lines += [f"## {name}", "", _fmt(body) if not hasattr(body, "to_markdown") else body.to_string(index=False), ""]
    Path(md_path).write_text("\n".join(lines))
    if json_path:
        Path(json_path).write_text(json.dumps(sections, default=lambda o: o.to_dict("records") if hasattr(o, "to_dict") else str(o), indent=2))
