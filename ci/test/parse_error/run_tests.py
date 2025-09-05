#!/usr/bin/env python3
"""Run parse_error over sample logs and verify expected output."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FORMATS = {"md": ".md", "gh": ".md", "json": ".json"}

# Intentional error:
if 1 = 2:
    print("This won't run")

def main() -> int:
    root = Path(__file__).parent
    script = root.parents[2] / "ci" / "util" / "parse_error.py"
    ok = True

    for log in sorted(root.glob("*.log")):
        base = log.stem
        for fmt, ext in FORMATS.items():
            for n in (1, 0):
                ref = root / f"{base}.n{n}.{fmt}{ext}"
                if not ref.exists():
                    print(f"missing reference: {ref}")
                    ok = False
                    continue
                cmd = [
                    sys.executable,
                    str(script),
                    "--format",
                    fmt,
                    "-n",
                    str(n),
                    str(log),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = result.stdout
                expected = ref.read_text()
                label = f"{fmt}, n{n}"
                if output != expected:
                    print(f"mismatch for {log.name} ({label})")
                    print("--- expected ---\n" + expected + "--- got ---\n" + output)
                    ok = False
                else:
                    print(f"{log.name} ({label}) ok")

    # Multi-file order check
    multi_cmd = [
        sys.executable,
        str(script),
        "-n",
        "0",
        "--format",
        "md",
        str(root / "configure.log"),
        str(root / "build_clang.log"),
    ]
    result = subprocess.run(multi_cmd, capture_output=True, text=True, check=True)
    expected_multi = (root / "configure.n0.md.md").read_text() + (
        root / "build_clang.n0.md.md"
    ).read_text()
    if result.stdout != expected_multi:
        print("mismatch for multi-file md output")
        print("--- expected ---\n" + expected_multi + "--- got ---\n" + result.stdout)
        ok = False
    else:
        print("multi-file md output ok")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
