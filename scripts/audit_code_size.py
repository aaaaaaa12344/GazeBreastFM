from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "code_size_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Python file sizes for V5.2 code-size gates.")
    parser.add_argument("--max-warn-lines", type=int, default=800)
    parser.add_argument("--max-fail-lines", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _iter_python_files() -> list[Path]:
    roots = [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            parts = {part.lower() for part in path.parts}
            if "__pycache__" in parts:
                continue
            files.append(path)
    return sorted(files)


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def build_code_size_report(max_warn_lines: int, max_fail_lines: int) -> dict[str, object]:
    records = []
    warnings = []
    failures = []
    for path in _iter_python_files():
        lines = _line_count(path)
        rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        status = "pass"
        if lines > max_fail_lines:
            status = "fail"
            failures.append(rel)
        elif lines > max_warn_lines:
            status = "warn"
            warnings.append(rel)
        records.append({"path": rel, "line_count": lines, "status": status})

    return {
        "status": "fail" if failures else ("pass_with_warnings" if warnings else "pass"),
        "max_warn_lines": int(max_warn_lines),
        "max_fail_lines": int(max_fail_lines),
        "warning_files": warnings,
        "failing_files": failures,
        "files": records,
    }


def write_report(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "code_size_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    lines = [
        "# Code Size Audit",
        "",
        f"Status: `{report['status']}`",
        f"Warn threshold: `{report['max_warn_lines']}` lines",
        f"Fail threshold: `{report['max_fail_lines']}` lines",
        "",
        "## Files Over Warning Threshold",
    ]
    warning_files = report.get("warning_files", [])
    if warning_files:
        for item in warning_files:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Files Over Failure Threshold"])
    failing_files = report.get("failing_files", [])
    if failing_files:
        for item in failing_files:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None.")
    (output_dir / "code_size_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_code_size_report(
        max_warn_lines=args.max_warn_lines,
        max_fail_lines=args.max_fail_lines,
    )
    write_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
