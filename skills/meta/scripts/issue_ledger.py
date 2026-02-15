#!/usr/bin/env python3
"""Generate BD-style issue ledger from emitted control-plane artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _issue(severity: str, code: str, message: str, details: dict) -> dict:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "details": details,
    }


def _build_ledger(base: Path) -> dict[str, Any]:
    issue_log = _read_json(base / ".index" / "issue_log.json")
    payload = _read_json(base / ".index" / "main_index_payload.json")
    bridge = _read_json(base / ".index" / "bridge_index.json")
    graph = _read_json(base / ".index" / "skill_graph.json")
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}

    unresolved = issue_log.get("unresolved_references", []) if isinstance(issue_log, dict) else []
    mctsr = metrics.get("mctsr", 0.0)
    hard_dag_ratio = metrics.get("acyclic_ratio_hard_dag", 0.0)
    soft_edge_count = metrics.get("soft_edges", 0)
    hard_edge_count = metrics.get("hard_edges", 0)
    node_count = payload.get("node_count", 0)

    findings: list[dict] = []

    if unresolved:
        findings.append(
            _issue(
                "high",
                "UNRESOLVED_DEPENDENCY",
                "Unresolved hard/soft dependencies were detected in manifest references.",
                {"count": len(unresolved), "sample": unresolved[:10]},
            )
        )

    if mctsr < 95.0:
        findings.append(
            _issue(
                "critical",
                "MCTSR_THRESHOLD",
                "MCTSR target was not reached.",
                {"mctsr": mctsr},
            )
        )

    if hard_dag_ratio < 1.0:
        findings.append(
            _issue(
                "high",
                "HARD_DAG_CYCLE",
                "Hard dependency DAG is not acyclic after cycle demotion/transitive reduction.",
                {"acyclic_ratio_hard_dag": hard_dag_ratio},
            )
        )

    if soft_edge_count < 0:
        findings.append(
            _issue(
                "info",
                "SOFT_EDGE_NEGATIVE",
                "Soft edge count is invalid.",
                {"soft_edges": soft_edge_count},
            )
        )

    if hard_edge_count == 0 and node_count > 0:
        findings.append(
            _issue(
                "medium",
                "NO_HARD_DEPS",
                "No hard dependencies were detected; check manifests for missed edges.",
                {"hard_edges": hard_edge_count, "node_count": node_count},
            )
        )

    if not (graph.get("scc") and graph.get("scc").get("components")):
        findings.append(
            _issue(
                "medium",
                "SCC_MISSING",
                "SCC metadata missing; dependency structure could be under-scoped.",
                {},
            )
        )

    skills = bridge.get("skills", {}) if isinstance(bridge, dict) else {}
    missing_bridges = []
    for skill_id, entry in skills.items() if isinstance(skills, dict) else []:
        if not entry.get("bridge_path"):
            missing_bridges.append(skill_id)

    if missing_bridges:
        findings.append(
            _issue(
                "info",
                "MISSING_BRIDGE_PATH",
                "Some skills are missing bridge paths in bridge_index.",
                {"skills": missing_bridges[:10]},
            )
        )

    if not payload:
        findings.append(
            _issue(
                "critical",
                "MISSING_INDEX_PAYLOAD",
                "main_index_payload.json is missing or unreadable.",
                {},
            )
        )

    if not issue_log:
        findings.append(
            _issue(
                "high",
                "MISSING_ISSUE_LOG",
                "issue_log.json is missing or unreadable.",
                {},
            )
        )

    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
    findings.sort(key=lambda item: severity_rank.get(item.get("severity", "info"), 0), reverse=True)

    return {
        "generated_at": _now_iso(),
        "project_root": str(base),
        "summary": {
            "node_count": node_count,
            "findings": len(findings),
            "critical": sum(1 for f in findings if f["severity"] == "critical"),
            "high": sum(1 for f in findings if f["severity"] == "high"),
            "medium": sum(1 for f in findings if f["severity"] == "medium"),
            "low": sum(1 for f in findings if f["severity"] == "low"),
            "info": sum(1 for f in findings if f["severity"] == "info"),
            "mctsr": mctsr,
            "acyclic_ratio_hard_dag": hard_dag_ratio,
        },
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate issue ledger from generated index artifacts.")
    parser.add_argument("--base", default=".", help="Repository root")
    parser.add_argument("--output", default=".index/bd_ledger.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any critical findings are present.",
    )
    args = parser.parse_args()

    base = Path(args.base).resolve()
    ledger = _build_ledger(base)
    output = base / args.output
    _write_json(output, ledger)
    print(json.dumps(ledger, indent=2, sort_keys=True))

    if args.strict and ledger["summary"]["critical"] > 0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
