# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for the CUTLASS CUB LTO-IR wrappers."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("cuda.coop.cutlass", exc_type=ImportError)
pytest.importorskip("cutlass.cute")

from ._compile_support import assert_compiled, compile_example  # noqa: E402


def _nvdisasm() -> str | None:
    if path := shutil.which("nvdisasm"):
        return path
    candidates = sorted(Path("/usr/local").glob("cuda*/bin/nvdisasm"))
    return str(candidates[-1]) if candidates else None


@pytest.mark.parametrize("import_form", ("root", "qualified"))
def test_ltoir_is_linked_into_the_final_cubin(
    import_form: str,
    tmp_path: Path,
) -> None:
    nvdisasm = _nvdisasm()
    if nvdisasm is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")

    result = compile_example(import_form=import_form, dump_dir=tmp_path)
    assert_compiled(result)

    clean_mlir_paths = sorted(tmp_path.rglob("*_clean.mlir"))
    cubin_paths = sorted(tmp_path.rglob("*.cubin"))
    assert len(clean_mlir_paths) == 1
    assert len(cubin_paths) == 1
    clean_mlir = clean_mlir_paths[0].read_text(encoding="utf-8")
    assert '"link-libraries"' in clean_mlir

    symbols = tuple(
        sorted(
            set(
                re.findall(
                    r"func\.func private @(cuda_coop_cutlass_cub_"
                    r"(?:load|store)_block_[A-Za-z0-9_]+)",
                    clean_mlir,
                )
            )
        )
    )
    assert len(symbols) == 2
    assert sum("_load_block_" in symbol for symbol in symbols) == 1
    assert sum("_store_block_" in symbol for symbol in symbols) == 1
    for symbol in symbols:
        assert clean_mlir.count(f"func.call @{symbol}") == 1

    match = re.search(r'"link-libraries" = \["([^\"]+\.ltoir)"\]', clean_mlir)
    assert match is not None
    assert not Path(match.group(1)).exists(), (
        "the subprocess must clean ephemeral LTO-IR"
    )

    disassembly = subprocess.run(
        [nvdisasm, str(cubin_paths[0])],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for symbol in symbols:
        assert symbol not in disassembly
    assert (
        re.search(
            r"(?m)^\s*(?:\.section\s+\.text\.|\.text\.)"
            r"cuda_coop_cutlass_cub_(?:load|store)_block_",
            disassembly,
        )
        is None
    )
