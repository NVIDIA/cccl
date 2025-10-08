#!/usr/bin/env python3
"""Run parse_error over sample logs and verify expected output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FORMATS = {"md": "", "json": ""}

# Intentional error (kept commented to avoid breaking tests):
# if 1 == 2:
#     print("This won't run")


def main() -> int:
    root = Path(__file__).parent
    script = root.parents[2] / "ci" / "util" / "parse_error.py"
    ok = True

    for log in sorted(root.glob("*.log")):
        base = log.stem
        for fmt, ext in FORMATS.items():
            for n in (1, 0):
                # New filenames drop the duplicated extension: use '.md' or '.json' only once
                ref = root / f"{base}.n{n}.{fmt}"
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
                if fmt == "json":
                    # Lenient JSON compare: allow extra fields; enforce all expected keys/values
                    try:
                        got = json.loads(output)
                        want = json.loads(expected)
                        if len(got) != len(want):
                            raise AssertionError("length mismatch")
                        for g, w in zip(got, want):
                            for k, v in w.items():
                                if g.get(k) != v:
                                    raise AssertionError(
                                        f"key {k} mismatch: {g.get(k)} != {v}"
                                    )
                        print(f"{log.name} ({label}) ok")
                    except Exception as e:
                        print(f"mismatch for {log.name} ({label})")
                        print(f"json compare failed: {e}")
                        print("--- expected ---\n" + expected + "--- got ---\n" + output)
                        ok = False
                    continue
                # Markdown: raw compare only
                if output != expected:
                    print(f"mismatch for {log.name} ({label})")
                    print("--- expected ---\n" + expected + "--- got ---\n" + output)
                    ok = False
                else:
                    print(f"{log.name} ({label}) ok")

    # Multi-file order check for Markdown output
    multi_cmd = [
        sys.executable,
        str(script),
        "-n",
        "0",
        "--format",
        "md",
        str(root / "ctest.log"),
        str(root / "build.clang.log"),
    ]
    result = subprocess.run(multi_cmd, capture_output=True, text=True, check=True)
    expected_multi = (root / "ctest.n0.md").read_text() + (
        root / "build.clang.n0.md"
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
