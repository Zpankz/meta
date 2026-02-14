#!/usr/bin/env python3
"""Create a deterministic bundle for current index artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_INPUTS = [
    ".index/bridge_candidates.csv",
    ".index/bridge_index.json",
    ".index/control_ontology.json",
    ".index/control_ontology.schema.json",
    ".index/hyperedges.json",
    ".index/main_index_payload.json",
    ".index/main_index.yaml",
    ".index/processing_report.md",
    ".index/quality_report.md",
    ".index/skill_graph.json",
    ".index/clustered_graph.json",
]


def _find_project_root(script_path: Path) -> Path:
    """Find the repository root for deterministic invocation.

    Preference order:
    1) A directory in the ancestor chain that already contains `.index`
    2) A directory in the ancestor chain that contains `.claude-plugin/marketplace.json`
    3) The ancestor of the script package (`skills/<skill>/scripts` -> `skills/<skill>/..`)
    4) The immediate parent of the script location
    """
    for ancestor in [script_path] + list(script_path.parents):
        if (ancestor / ".index").is_dir():
            return ancestor

    for ancestor in [script_path] + list(script_path.parents):
        if (ancestor / ".claude-plugin" / "marketplace.json").is_file():
            return ancestor

    # Preserve deterministic behavior for both legacy and new layouts.
    if len(script_path.parents) >= 2:
        return script_path.parents[2]

    return script_path.parent


_DEFAULT_BASE = _find_project_root(Path(__file__).resolve().parent)


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_existing_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_bundle(
    base_dir: Path,
    source_paths: list[str],
    output_path: Path,
    overwrite: bool = True,
    include_previous_manifest: bool = True,
    strict: bool = False,
) -> Path:
    input_paths = []
    missing = []
    base = base_dir.resolve()
    for rel in source_paths:
        candidate = Path(rel)
        if candidate.is_absolute():
            raise SystemExit(f"Input path must be repository-relative: {rel}")
        file_path = base_dir / rel
        resolved = file_path.resolve()
        if not resolved.is_relative_to(base):
            raise SystemExit(f"Rejected path traversal input: {rel}")
        if file_path.exists():
            input_paths.append(file_path)
        else:
            missing.append(rel)
    if missing and strict:
        raise SystemExit(f"Missing required input artifacts: {', '.join(missing)}")
    if missing:
        # Keep a permissive mode for recovery and post-check auditing.
        print(f"[warn] missing artifacts ignored: {', '.join(missing)}")

    manifest_path = base_dir / ".index" / "skill_bundle_manifest.json"
    manifest_payload = {
        "generated_at": "",
        "source_root": str(base_dir),
        "inputs": [],
        "output": str(output_path.relative_to(base_dir)),
        "counts": {
            "files": 0,
            "total_bytes": 0,
        },
    }

    manifest_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    for file_path in sorted(input_paths):
        rel = str(file_path.relative_to(base_dir))
        size = file_path.stat().st_size
        checksum = _sha256_hex(file_path)
        manifest_payload["inputs"].append(
            {
                "path": rel,
                "size": size,
                "sha256": checksum,
            }
        )
        manifest_payload["counts"]["files"] += 1
        manifest_payload["counts"]["total_bytes"] += size

    previous = _load_existing_manifest(manifest_path)
    manifest_payload["previous"] = previous if previous else None

    if output_path.exists() and not overwrite:
        raise SystemExit(
            f"Bundle already exists: {output_path}. Re-run with --overwrite to replace it."
        )
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zip_out:
        for file_path in sorted(input_paths):
            rel = str(file_path.relative_to(base_dir))
            zip_out.write(file_path, rel)

        manifest_temp = base_dir / ".index" / "skill_bundle_manifest.json"
        manifest_temp.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        zip_out.write(manifest_temp, ".index/skill_bundle_manifest.json")

        if include_previous_manifest and previous:
            zip_out.writestr(".index/skill_bundle_previous_manifest.json", json.dumps(previous, indent=2, sort_keys=True) + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build a deterministic skill-bundle archive for governance outputs.",
    )
    parser.add_argument(
        "--base",
        default=str(_DEFAULT_BASE),
        help="Project root containing the .index directory.",
    )
    parser.add_argument(
        "--output",
        default=".index/skill_bundle.zip",
        help="Destination archive path, relative to base.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing bundle archive.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any configured input artifact is missing.",
    )
    parser.add_argument(
        "--inputs",
        action="append",
        default=[],
        help="Additional input paths (relative to base). Can be passed multiple times.",
    )
    args = parser.parse_args()

    base = Path(args.base)
    seen = set()
    paths = []
    for item in list(DEFAULT_INPUTS) + list(args.inputs):
        if item in seen:
            continue
        seen.add(item)
        paths.append(item)
    output = Path(base / args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_bundle(
        base,
        paths,
        output,
        overwrite=bool(args.overwrite),
        strict=bool(args.strict),
    )


if __name__ == "__main__":
    main()
