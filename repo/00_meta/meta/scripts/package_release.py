#!/usr/bin/env python3
"""Create a deterministic, checksummed release archive for control-plane artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_ARTIFACTS = [
    ".index/README.md",
    ".index/interfaces/emit_index.py",
    ".index/main_index.yaml",
    ".index/main_index_payload.json",
    ".index/bridge_index.json",
    ".index/bridge_candidates.csv",
    ".index/hyperedges.json",
    ".index/clustered_graph.json",
    ".index/control_ontology.json",
    ".index/control_ontology.schema.json",
    ".index/skill_graph.json",
    ".index/quality_report.md",
    ".index/processing_report.md",
    ".index/issue_log.json",
    ".index/bd_ledger.json",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_release(base: Path, output: Path, strict: bool = False, include_missing: bool = False) -> dict:
    entries = []
    missing = []
    for rel in DEFAULT_ARTIFACTS:
        path = base / rel
        if path.exists():
            entries.append((rel, path))
        else:
            missing.append(rel)

    if missing and strict:
        raise SystemExit(f"Missing required release artifacts: {', '.join(sorted(missing))}")

    manifest = {
        "generated_at": _now_iso(),
        "source_root": str(base),
        "output": str(output),
        "artifact_count": len(entries),
        "artifacts": [],
        "missing": sorted(missing) if include_missing else sorted(missing),
    }

    total_bytes = 0
    for rel, path in entries:
        size = path.stat().st_size
        checksum = _sha256(path)
        total_bytes += size
        manifest["artifacts"].append({"path": rel, "size": size, "sha256": checksum})

    manifest["total_bytes"] = total_bytes

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as zf:
        for rel, path in entries:
            zf.write(path, rel)
        manifest_path = base / ".index" / "meta_release_manifest.json"
        _write_json(manifest_path, manifest)
        zf.write(manifest_path, ".index/meta_release_manifest.json")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic meta skill release archive")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=".index/meta_release.zip")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--include-missing", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    manifest = build_release(base=base, output=(base / args.output), strict=args.strict, include_missing=args.include_missing)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
