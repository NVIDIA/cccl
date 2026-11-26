# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Select the Windows device linker for CuTe modules containing CCCL calls."""

from __future__ import annotations

import hashlib

from cutlass._mlir import ir
from cutlass.base_dsl.compiler import LinkLibraries, PtxasOptions

from ._windows_lto import lower_and_link

_LINK_PLAN = "cuda_coop.windows_lto_v1"


class _DeviceLinkPipeline:
    def __init__(self, native, pipeline, dsl):
        self.native = native
        self.pipeline = pipeline
        self.dsl = dsl

    def __getattr__(self, name):
        return getattr(self.native, name)

    def run(self, operation, *args, **kwargs):
        if _LINK_PLAN not in operation.attributes:
            return self.native.run(operation, *args, **kwargs)

        plan = ir.DictAttr(operation.attributes[_LINK_PLAN])
        paths = tuple(path.value for path in plan["libraries"])
        options = self.dsl.compile_options
        # CuTe merges module libraries into the option after tracing. Use its
        # own serialization to retain Windows quoting and paths with spaces.
        native_option = options.options[LinkLibraries].serialize()
        bitcode = ",".join(path.value for path in plan["bitcode"])
        pipeline = self.pipeline
        if native_option:
            pipeline = pipeline.replace(
                native_option, LinkLibraries(bitcode).serialize(), 1
            )
        lower_and_link(
            operation.opview,
            pipeline,
            paths,
            plan["arch"].value,
            cubin_path=options.full_cubin_path,
            ptxas_options=options.options[PtxasOptions].value,
        )


class _PassManagers:
    """Keep the compiler's diagnostics, hooks and JIT around the device link."""

    def __init__(self, native, dsl):
        self.native = native
        self.dsl = dsl

    @property
    def PassManager(self):
        # CuTe takes a passmanager module in its Compiler constructor.
        return self

    def __getattr__(self, name):
        return getattr(self.native, name)

    def parse(self, pipeline, *args, **kwargs):
        manager = self.native.PassManager.parse(pipeline, *args, **kwargs)
        if "cute-to-nvvm{" in pipeline and "llvm-only=true" not in pipeline:
            return _DeviceLinkPipeline(manager, pipeline, self.dsl)
        return manager


def configure_device_link(dsl, module, arch, paths):
    """Record this trace's inputs before CuTe computes the module cache key."""
    options = dsl.compile_options.options[LinkLibraries]
    inputs = [path for path in options.value.split(",") if path]
    for op in module.body.operations:
        if op.operation.name == "gpu.module":
            inputs.extend(
                path.value for path in op.attributes.get("link-libraries", ())
            )
    # Raw LLVM bitcode is linked before CuTe emits caller LTO-IR. All
    # other device inputs belong to nvJitLink. Include user file contents
    # in the cache key even when the caller reuses the same filename.
    bitcode = []
    device_files = []
    hashes = []
    for path in dict.fromkeys(inputs):
        with open(path, "rb") as library:
            magic = library.read(4)
            digest = hashlib.sha256(magic)
            for chunk in iter(lambda: library.read(1 << 20), b""):
                digest.update(chunk)
        hashes.append(ir.StringAttr.get(digest.hexdigest()))
        if magic in (b"BC\xc0\xde", b"\xde\xc0\x17\x0b"):
            bitcode.append(ir.StringAttr.get(path))
        else:
            device_files.append(path)
    device_files.extend(paths)
    module.operation.attributes[_LINK_PLAN] = ir.DictAttr.get(
        {
            "arch": ir.StringAttr.get("sm_" + arch.removeprefix("compute_")),
            "bitcode": ir.ArrayAttr.get(bitcode),
            "user_input_hashes": ir.ArrayAttr.get(hashes),
            "libraries": ir.ArrayAttr.get(
                [
                    ir.StringAttr.get(path)
                    for path in dict.fromkeys(device_files)
                ]
            ),
        }
    )
    compiler = dsl.compiler_provider
    if not isinstance(compiler.passmanager, _PassManagers):
        compiler.passmanager = _PassManagers(compiler.passmanager, dsl)
