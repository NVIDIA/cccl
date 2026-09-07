# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dependency-light launcher for the optional CUTLASS AOT command."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

_LINUX_ONLY_ERROR = "CUTLASS provider AOT packs currently require Linux."


def main(argv: Sequence[str] | None = None) -> int:
    """Load the optional CUTLASS command with concise dependency diagnostics."""

    if sys.platform != "linux":
        print(f"cuda-coop-aot: error: {_LINUX_ONLY_ERROR}", file=sys.stderr)
        return 2
    try:
        cli = importlib.import_module("cuda.coop.cutlass._aot_cli")
    except ImportError as error:
        if getattr(error, "backend", None) != "cutlass":
            raise
        print(f"cuda-coop-aot: error: {error}", file=sys.stderr)
        return 2
    return int(cli.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
