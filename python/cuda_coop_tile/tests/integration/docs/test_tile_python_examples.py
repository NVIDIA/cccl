# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the complete Python snippets from the primitive reference."""

import importlib.util
import sys
from textwrap import indent

import pytest

from ...backends.tile_python.support.e2e import (
    configure_e2e_environment,
    import_e2e_modules,
)
from ...support.paths import REPO_ROOT
from ._tile_examples import code_blocks

pytestmark = [
    pytest.mark.backend_tile_python,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

PRIMITIVES = (
    "scan",
    "reduce",
    "exchange_shuffle",
    "adjacent_discontinuity",
    "histogram",
    "run_length_decode",
    "merge_sort",
    "radix_sort_rank",
    "topk",
)


def _load_example(source, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_tile_python_documentation_examples(primitive, tmp_path, monkeypatch):
    configure_e2e_environment(monkeypatch)
    torch, _, _ = import_e2e_modules()
    page = REPO_ROOT / "docs/tile/python/primitives" / f"{primitive}.rst"
    examples = code_blocks(page, "python")
    assert len(examples) == (3 if primitive == "reduce" else 1), page
    complete = [block for block in examples if "@ct.kernel" in block["code"]]
    assert len(complete) == 1, page
    source = tmp_path / f"{primitive}_example.py"
    source.write_text(complete[0]["code"], encoding="utf-8")
    name = f"_coop_tile_documentation_{primitive}"
    _load_example(source, name, monkeypatch)
    torch.cuda.synchronize()


def test_native_reduction_shapes(tmp_path, monkeypatch):
    configure_e2e_environment(monkeypatch)
    torch, ct, _ = import_e2e_modules()
    page = REPO_ROOT / "docs/tile/python/primitives/reduce.rst"
    examples = [
        block
        for block in code_blocks(page, "python")
        if "total_1d = ct.sum(values, keepdims=True)" in block["code"]
    ]
    assert len(examples) == 1, page
    lines = examples[0]["code"].splitlines()
    imports = "\n".join(line for line in lines if line.startswith("import "))
    body = "\n".join(line for line in lines if not line.startswith("import "))
    source = tmp_path / "native_reduce.py"
    source.write_text(
        "# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. "
        "ALL RIGHTS RESERVED.\n"
        "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
        + imports
        + "\n\n@ct.kernel(num_worker_warps=4)\n"
        "def native_reduce(source, total_out, total_1d_out, repeated_out, "
        "shifted_out):\n"
        "    values = ct.load(source, (0,), shape=(256,))\n"
        + indent(body, "    ")
        + "\n    ct.store(total_out, (0,), ct.reshape(total, (1,)))\n"
        "    ct.store(total_1d_out, (0,), total_1d)\n"
        "    ct.store(repeated_out, (0,), repeated)\n"
        "    ct.store(shifted_out, (0,), shifted)\n",
        encoding="utf-8",
    )
    module = _load_example(source, "_coop_tile_native_reduce", monkeypatch)
    data = [(index // 3) % 17 - 8 for index in range(256)]
    values = torch.tensor(data, dtype=torch.int32, device="cuda")
    outputs = [
        torch.empty(size, dtype=torch.int32, device="cuda")
        for size in (1, 1, 256, 256)
    ]
    ct.launch(
        torch.cuda.current_stream(),
        (1, 1, 1),
        module.native_reduce,
        (values, *outputs),
    )
    torch.cuda.synchronize()
    assert [output.cpu().tolist() for output in outputs] == [
        [sum(data)],
        [sum(data)],
        [sum(data)] * 256,
        [value - max(data) for value in data],
    ]


def test_tile_python_guide_scan_controls(tmp_path, monkeypatch):
    configure_e2e_environment(monkeypatch)
    torch, ct, _ = import_e2e_modules()
    page = REPO_ROOT / "docs/tile/python/programming.rst"
    examples = [
        block
        for block in code_blocks(page, "python")
        if "exclusive = coop.exclusive_sum" in block["code"]
    ]
    assert len(examples) == 1, page
    source = tmp_path / "guide_scan_controls.py"
    source.write_text(
        "import cuda.coop.tile as coop\nimport cuda.tile as ct\n\n"
        "@ct.kernel\n"
        "def scan_controls(input, exclusive_out, maximum_out, product_out, "
        "block_prefix):\n"
        "    values = ct.load(input, (0,), shape=(256,))\n"
        + indent(examples[0]["code"], "    ")
        + "\n    ct.store(exclusive_out, (0,), exclusive)\n"
        "    ct.store(maximum_out, (0,), maximum)\n"
        "    ct.store(product_out, (0,), product)\n",
        encoding="utf-8",
    )
    module = _load_example(source, "_coop_tile_guide_controls", monkeypatch)
    # Keep products bounded while making each scan's result distinct.
    data = torch.ones(256, device="cuda", dtype=torch.int32)
    data[:4] = torch.tensor([2, 3, 1, -1], device="cuda", dtype=torch.int32)
    outputs = [torch.empty_like(data) for _ in range(3)]
    for prefix in (-7, 19):
        ct.launch(
            torch.cuda.current_stream(),
            (1, 1, 1),
            module.scan_controls,
            (data, *outputs, prefix),
        )
        torch.cuda.synchronize()
        expected_product = torch.cat(
            (torch.ones_like(data[:1]), torch.cumprod(data[:-1], 0))
        )
        assert torch.equal(outputs[0], torch.cumsum(data, 0) - data + prefix)
        assert torch.equal(outputs[1], torch.cummax(data, 0).values)
        assert torch.equal(outputs[2], expected_product)
