# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import sys

import pytest

from ....support.paths import PACKAGE_ROOT, REPO_ROOT

GENERATOR = REPO_ROOT / "tile" / "tools" / "generate_tile_declarations.py"
INTERFACE = REPO_ROOT / "tile/cpp/tilelib/interface/cuda_coop_tile.mlir"


def test_generated_frontend_declarations_are_current():
    subprocess.run(
        [sys.executable, "-S", str(GENERATOR), "--check"], check=True
    )


@pytest.mark.parametrize(
    "output_option", ["--output", "--python-output", "--native-output"]
)
def test_unsupported_interface_does_not_overwrite_declarations(
    tmp_path, output_option
):
    interface = tmp_path / "interface.mlir"
    output = tmp_path / "declarations.cuh"
    # A new dtype requires an explicit frontend binding; silently dropping or
    # guessing it would produce a C++ ABI inconsistent with the TileLib.
    interface.write_text(
        INTERFACE.read_text().replace(
            "[N]x!cuda_tile.preview$sym_t<T>", "[N]xi16", 1
        )
    )
    output.write_text("existing output\n")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(GENERATOR),
            "--interface",
            str(interface),
            output_option,
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unsupported TileIR type" in result.stderr
    assert output.read_text() == "existing output\n"


def test_generated_python_schema_imports_without_optional_dependencies():
    schema = PACKAGE_ROOT / "cuda" / "coop" / "tile" / "_declarations.py"
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import runpy, sys; "
                "runpy.run_path(sys.argv[1]); "
                "assert not any(name == 'cuda' or name.startswith('cuda.') "
                "for name in sys.modules)"
            ),
            str(schema),
        ],
        check=True,
    )
