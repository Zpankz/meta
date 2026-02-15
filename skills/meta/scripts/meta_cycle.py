#!/usr/bin/env python3
"""One-shot meta execution loop: emit index, build issue ledger, optionally package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_ITERATIONS = 5


def _run(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    return proc.returncode


def run_cycle(base: Path, iterations: int, strict: bool, package: bool, output: str) -> list[dict]:
    payloads: list[dict] = []

    for i in range(1, iterations + 1):
        print(f"== meta cycle iteration {i}/{iterations} ==")
        code = _run([sys.executable, str(base / ".index/interfaces/emit_index.py"), "--base", str(base)], base)
        if code != 0:
            raise SystemExit(f"emit_index failed at iteration {i}")

        ledger_args = [
            sys.executable,
            str(base / "repo/00_meta/meta/scripts/issue_ledger.py"),
            "--base",
            str(base),
            "--output",
            ".index/bd_ledger.json",
        ]
        if strict:
            ledger_args.append("--strict")
        code = _run(ledger_args, base)
        if strict and code != 0:
            raise SystemExit(f"issue ledger strict mode failed at iteration {i}")

        payload_path = base / ".index/main_index_payload.json"
        if payload_path.exists():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payloads.append(payload.get("metrics", {}))

    if package:
        cmd = [
            sys.executable,
            str(base / "repo/00_meta/meta/scripts/package_release.py"),
            "--base",
            str(base),
            "--output",
            output,
        ]
        if strict:
            cmd.append("--strict")
        code = _run(cmd, base)
        if code != 0:
            raise SystemExit("package_release failed")

    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recursive governance loop and emit deterministic artifacts")
    parser.add_argument("--base", default=".")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--output", default=".index/meta_release.zip")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    payloads = run_cycle(base=base, iterations=args.iterations, strict=args.strict, package=args.package, output=args.output)
    if payloads:
        last = payloads[-1]
        print(
            "FINAL METRICS",
            json.dumps(
                {
                    "mctsr": last.get("mctsr"),
                    "mctsr_passed": last.get("mctsr_passed"),
                    "hard_edges_dag": last.get("hard_edges_dag"),
                    "acyclic_ratio_hard_dag": last.get("acyclic_ratio_hard_dag"),
                    "acyclic_ratio_raw": last.get("acyclic_ratio_raw"),
                },
                indent=2,
            ),
        )


if __name__ == "__main__":
    main()
