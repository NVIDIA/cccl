# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Link CuTe caller and provider LTO-IR before host lowering."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from cutlass._mlir import ir, passmanager

from ._nvjitlink import link_lto


def lower_and_link(
    module: ir.Module,
    pipeline: str,
    paths: tuple[str, ...],
    arch: str,
    *,
    cubin_path: str | Path | None = None,
    ptxas_options: str = "",
) -> ir.Module:
    """Lower a CuTe launch and embed its fully linked device image."""
    # Use the device and host passes for this invocation, preserving the
    # configured optimization level, host ABI and CUDA compiler options.
    expanded = str(passmanager.PassManager.parse(pipeline))
    _, binary_pass, tail = expanded.partition(",cuda-to-binary{")
    device_options, host_pass, host_tail = tail.partition("},cuda-to-llvm")
    head, options_end, closing = pipeline.rpartition("}")
    if not (binary_pass and host_pass and options_end):
        raise RuntimeError(
            f"Unsupported CuTe device compilation pipeline: {pipeline}"
        )

    for gpu in module.body.operations:
        if (
            gpu.operation.name == "gpu.module"
            and "link-libraries" in gpu.attributes
        ):
            del gpu.attributes["link-libraries"]
    # CuTe's llvm-only pass keeps the host launcher and its caller in the
    # same IR module so their symbol and launch metadata stay together.
    passmanager.PassManager.parse(head + " llvm-only=true}" + closing).run(
        module.operation
    )
    gpu_modules = [
        op for op in module.body.operations if op.operation.name == "gpu.module"
    ]
    if len(gpu_modules) != 1:
        raise RuntimeError(
            f"Expected one CuTe GPU module, found {len(gpu_modules)}"
        )
    gpu = gpu_modules[0]
    targets = gpu.attributes["compute_targets"]
    lto_targets = str(targets).replace("<sass,", "<ltoir,")
    if lto_targets == str(targets):
        raise RuntimeError(f"Expected a CuTe SASS target, found {targets}")
    gpu.attributes["compute_targets"] = ir.Attribute.parse(lto_targets)
    with TemporaryDirectory(prefix="cuda-coop-lto-") as scratch:
        prefix = Path(scratch, "caller").as_posix()
        # The pass appends the architecture and .ltoir to this prefix.
        emit_caller = (
            "builtin.module(cuda-to-binary{"
            + device_options
            + f' dump-cubin-path= dump-fatbin-path= dump-ltoir-path="{prefix}"'
            + "})"
        )
        passmanager.PassManager.parse(emit_caller).run(module.operation)
        caller = Path(scratch, f"caller.{arch}.ltoir").read_bytes()

    cubin = link_lto(caller, paths, arch, ptxas_options=ptxas_options)
    binary = next(
        op
        for op in module.body.operations
        if op.operation.name == "cuda.binary"
    )
    image_type = ir.VectorType.get([len(cubin)], ir.IntegerType.get_signless(8))
    binary.attributes["binary"] = ir.DenseResourceElementsAttr.get_from_buffer(
        cubin, "linked_cubin", image_type, alignment=8
    )
    binary.attributes["compute_targets"] = targets
    passmanager.PassManager.parse(
        "builtin.module(cuda-to-llvm" + host_tail
    ).run(module.operation)
    if cubin_path is not None:
        Path(cubin_path).write_bytes(cubin)
    return module
