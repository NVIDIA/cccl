# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build matched source kernels and launch their final cubins identically."""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _cases import _CORE_OPERATIONS


def _checked(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def capture(stream, launch, count=1):
    """Capture callbacks and leave capture mode after a callback error."""
    from cuda.bindings import driver

    if count < 1:
        raise ValueError("Capture requires at least one launch")
    _checked(
        driver.cuStreamBeginCapture(
            stream,
            driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
    )
    try:
        for _ in range(count):
            launch()
    except BaseException:
        # EndCapture resets both active and invalidated captures. Preserve the
        # original callback exception; destroy a graph if capture stayed valid.
        ended = driver.cuStreamEndCapture(stream)
        if int(ended[0]) == 0 and int(ended[1]) != 0:
            _checked(driver.cuGraphDestroy(ended[1]))
        raise
    return _checked(driver.cuStreamEndCapture(stream))


def _captured_parameters(parameters, pointers):
    """Bind the optimized kernel ABI to the original live device allocations.

    CuTe may remove unused parameters. Read the Driver's actual parameter
    layout and captured values instead of assuming the source signature.
    """
    from cuda.bindings import driver

    by_address = {int(pointer): name for name, pointer in pointers.items()}
    if len(by_address) != len(pointers) or not parameters.kernelParams:
        raise RuntimeError("Kernel parameters need unique pointer allocations")
    arguments = ctypes.cast(
        int(parameters.kernelParams), ctypes.POINTER(ctypes.c_void_p)
    )
    layout = []
    end = 0
    for index in range(len(pointers) + 1):
        info = driver.cuFuncGetParamInfo(parameters.func, index)
        if info[0] == driver.CUresult.CUDA_ERROR_INVALID_VALUE:
            break
        offset, size = _checked(info)
        if index == len(pointers) or size != 8 or offset < end or offset % 8:
            raise RuntimeError("Unsupported captured kernel parameter layout")
        value = ctypes.c_uint64.from_address(arguments[index]).value
        name = by_address.get(value)
        if name is None or any(item["binding"] == name for item in layout):
            raise RuntimeError(
                "Captured kernel argument is not a unique pointer"
            )
        layout.append(
            {"index": index, "offset": offset, "size": size, "binding": name}
        )
        end = offset + size
    if not layout:
        raise RuntimeError("Captured kernel has no pointer parameters")
    return layout


def native_source(case):
    """CUB baseline with the same per-thread loads and output ownership."""
    if case.operation not in _CORE_OPERATIONS:
        from _advanced import native_source as advanced_source

        return advanced_source(case)
    n, width, items = case.threads, case.group_size, case.items
    dtype = {"int32": "int", "float32": "float"}[case.dtype]
    block = case.group == "block"
    operation = case.operation
    algorithm = case.options.get("algorithm", "raking")
    base = "Block" if block else "Warp"
    shared = "scratch" if block else "scratch[threadIdx.x / W]"
    args = ""
    body = ""
    loads = (
        "cub::LoadDirectBlockedVectorized(threadIdx.x, "
        "const_cast<T*>(input) + blockIdx.x * N * I, values);"
        if items % 4 == 0
        else "for (int i = 0; i < I; ++i) values[i] = input[offset + i];"
    )
    stores = (
        "cub::StoreDirectBlockedVectorized(threadIdx.x, "
        "output + blockIdx.x * N * I, result);"
        if items % 4 == 0
        else "for (int i = 0; i < I; ++i) output[offset + i] = result[i];"
    )
    if operation in ("load", "store"):
        enum = f"cub::{base.upper()}_{operation.upper()}_{algorithm.upper()}"
        args = (
            f"cub::{base}{operation.title()}<T, {n if block else items}, "
            f"{items if block else enum}, {enum if block else width}>"
        )
        if operation == "load":
            loads = ""
            body = f"Collective({shared}).Load(input + group_offset, result);"
        else:
            body = f"Collective({shared}).Store(output + group_offset, values);"
            stores = ""
    elif operation in ("sum", "reduce"):
        args = (
            f"cub::BlockReduce<T, N, cub::BLOCK_REDUCE_{algorithm.upper()}>"
            if block
            else "cub::WarpReduce<T, W>"
        )
        operand = "values" if block else "values[0]"
        method = (
            f"Sum({operand})"
            if operation == "sum"
            else f"Reduce({operand}, cuda::minimum<>())"
        )
        body = f"T aggregate = Collective({shared}).{method};"
        stores = "if (lane == 0) output[group_index] = aggregate;"
    elif operation in (
        "inclusive_sum",
        "exclusive_sum",
        "inclusive_scan",
        "exclusive_scan",
    ):
        args = (
            f"cub::BlockScan<T, N, cub::BLOCK_SCAN_{algorithm.upper()}>"
            if block
            else "cub::WarpScan<T, W>"
        )
        operand = "values, result" if block else "values[0], result[0]"
        method = "".join(word.title() for word in operation.split("_"))
        extra = ""
        if operation == "inclusive_scan":
            extra = ", cuda::maximum<>()"
        elif operation == "exclusive_scan":
            extra = ", T(-2147483647 - 1), cuda::maximum<>()"
        body = f"Collective({shared}).{method}({operand}{extra});"
    elif operation == "reduce_batched":
        args = "cub::WarpReduceBatched<T, I, W, false>"
        layout = case.options["output_layout"]
        method = "ReduceTo" + layout.title()
        index = "lane + i * W" if layout == "striped" else "lane * O + i"
        body = (
            "constexpr int O = (I + W - 1) / W; T reduced[O]; "
            f"Collective({shared}).{method}"
            "(values, reduced, cuda::std::plus<>()); "
            f"for (int i = 0; i < O; ++i) {{ const int batch = {index}; "
            "if (batch < I) output[group_index * I + batch] = reduced[i]; }"
        )
        stores = ""
    elif operation == "exchange":
        args = (
            "cub::BlockExchange<T, N, I>"
            if block
            else "cub::WarpExchange<T, I, W>"
        )
        method = "".join(
            word.title() for word in case.options["mode"].split("_")
        )
        body = f"Collective({shared}).{method}(values, result);"
    elif operation == "shuffle":
        args = "cub::BlockShuffle<T, N>"
        mode = case.options["mode"]
        if mode in ("up", "down"):
            body = f"Collective(scratch).{mode.title()}(values, result);"
            body += (
                "if (lane == 0) result[0] = 0;"
                if mode == "up"
                else "if (lane == N - 1) result[I - 1] = 0;"
            )
        else:
            body = (
                f"Collective(scratch).{mode.title()}(values[0], result[0], 3);"
            )
            if mode == "offset":
                body += "if (lane + 3 >= N) result[0] = 0;"
    elif operation in ("merge_sort_keys", "merge_sort_pairs"):
        pairs = operation.endswith("pairs")
        value_type = "int" if pairs else "cub::NullType"
        args = (
            f"cub::BlockMergeSort<T, N, I, {value_type}>"
            if block
            else f"cub::WarpMergeSort<T, I, W, {value_type}>"
        )
        compare = "greater" if case.options["descending"] else "less"
        body = "for (int i = 0; i < I; ++i) result[i] = values[i];"
        if pairs:
            body += (
                "int associated[I]; "
                "cub::LoadDirectBlockedVectorized(threadIdx.x, "
                "const_cast<int*>(input_aux) + blockIdx.x * N * I, associated);"
            )
        body += (
            f"Collective({shared}).Sort(result, "
            f"{'associated, ' if pairs else ''}cuda::std::{compare}<T>());"
        )
        if pairs:
            stores += (
                "cub::StoreDirectBlockedVectorized(threadIdx.x, "
                "output_aux + blockIdx.x * N * I, associated);"
            )
    else:
        raise ValueError(operation)
    family = operation
    if operation in ("sum", "reduce"):
        family = "reduce"
    elif operation in (
        "inclusive_sum",
        "exclusive_sum",
        "inclusive_scan",
        "exclusive_scan",
    ):
        family = "scan"
    elif operation.startswith("merge_sort"):
        family = "merge_sort"
    scope = "block" if block else "warp"
    header = f"cub/{scope}/{scope}_{family}.cuh"
    # Match the default API storage-reuse guarantees, including narrow masks.
    # Direct memory operations and batched reduction have no shared scratch.
    barrier = ""
    if operation != "reduce_batched" and not (
        operation in ("load", "store") and algorithm == "direct"
    ):
        if block:
            barrier = "__syncthreads();"
        else:
            mask = (
                "0xffffffffu"
                if width == 32
                else f"{(1 << width) - 1}u << ((threadIdx.x % 32) / W * W)"
            )
            barrier = f"__syncwarp({mask});"
    scratch = "" if block else "[N / W]"
    dummy = (
        ""
        if operation.endswith("pairs")
        else "if (blockIdx.x == 0 && threadIdx.x == 0) output_aux[0] = 0;"
    )
    return f"""// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <{header}>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/functional>
#include <cuda/functional>
using T = {dtype};
inline constexpr int N = {n}, W = {width}, I = {items};
using Collective = {args};
extern "C" __global__ void benchmark(
  const T* input, const int* input_aux, T* output, int* output_aux) {{
  input = cuda::std::assume_aligned<16>(input);
  input_aux = cuda::std::assume_aligned<16>(input_aux);
  output = cuda::std::assume_aligned<16>(output);
  output_aux = cuda::std::assume_aligned<16>(output_aux);
  __shared__ typename Collective::TempStorage scratch{scratch};
  const int lane = threadIdx.x % W;
  const int group_index = blockIdx.x * (N / W) + threadIdx.x / W;
  const int group_offset = group_index * W * I;
  const int offset = blockIdx.x * N * I + threadIdx.x * I;
  T values[I], result[I];
  {loads}
  {body}
  {barrier}
  {stores}
  {dummy}
}}
"""


def make_launcher(case, blocks):
    if case.operation not in _CORE_OPERATIONS:
        from _advanced import make_launcher as advanced_launcher

        return advanced_launcher(case, blocks)
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop
    from cuda.bindings import driver

    threads, items, width = case.threads, case.items, case.group_size
    is_block = case.group == "block"
    operation = case.operation
    value_type = {"int32": cutlass.Int32, "float32": cutlass.Float32}[
        case.dtype
    ]
    size = blocks * threads * items
    algorithm = case.options.get("algorithm", None)
    mode = case.options.get("mode", "")
    layout = case.options.get("output_layout", "striped")
    descending = case.options.get("descending", False)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        source_aux: cute.Pointer,
        out: cute.Pointer,
        out_aux: cute.Pointer,
    ):
        thread = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        offset = block * threads * items + thread * items
        inputs = cute.make_tensor(source, cute.make_layout(size))
        outputs = cute.make_tensor(out, cute.make_layout(size))
        aux_in = cute.make_tensor(source_aux, cute.make_layout(size))
        aux_out = cute.make_tensor(out_aux, cute.make_layout(size))
        if cutlass.const_expr(is_block):
            group = coop.this_block()
        elif cutlass.const_expr(width == 32):
            group = coop.this_warp()
        else:
            group = coop.this_warp().group_by(width)
        lane = thread % width
        group_index = block * (threads // width) + thread // width
        values = coop.ThreadData(items, dtype=value_type)
        if cutlass.const_expr(operation != "load"):
            for item in cutlass.range_constexpr(items):
                values[item] = inputs[offset + item]
        if cutlass.const_expr(operation == "load"):
            result = coop.load(
                group,
                source,
                items_per_thread=items,
                algorithm=algorithm,
                offset=block * threads * items,
            )
        elif cutlass.const_expr(operation == "store"):
            coop.store(
                group,
                out,
                values,
                algorithm=algorithm,
                offset=block * threads * items,
            )
        elif cutlass.const_expr(operation in ("sum", "reduce")):
            if cutlass.const_expr(operation == "reduce"):
                aggregate = coop.reduce(
                    group, values, binary_op="min", algorithm=algorithm
                )
            elif cutlass.const_expr(is_block):
                aggregate = coop.sum(group, values, algorithm=algorithm)
            else:
                aggregate = coop.sum(group, values[0])
            if lane == 0:
                outputs[group_index] = aggregate
        elif cutlass.const_expr(
            operation
            in (
                "inclusive_sum",
                "exclusive_sum",
                "inclusive_scan",
                "exclusive_scan",
            )
        ):
            if cutlass.const_expr(is_block):
                scan_input = values
            else:
                scan_input = values[0]
            if cutlass.const_expr(operation == "inclusive_sum"):
                scanned = coop.inclusive_sum(
                    group, scan_input, algorithm=algorithm
                )
            elif cutlass.const_expr(operation == "exclusive_sum"):
                scanned = coop.exclusive_sum(
                    group, scan_input, algorithm=algorithm
                )
            elif cutlass.const_expr(operation == "inclusive_scan"):
                scanned = coop.inclusive_scan(
                    group, scan_input, scan_op="max", algorithm=algorithm
                )
            else:
                scanned = coop.exclusive_scan(
                    group,
                    scan_input,
                    scan_op="max",
                    initial_value=cutlass.Int32(-2147483648),
                    algorithm=algorithm,
                )
            result = coop.ThreadData(items, dtype=value_type)
            if cutlass.const_expr(is_block):
                for item in cutlass.range_constexpr(items):
                    result[item] = scanned[item]
            else:
                result[0] = scanned
        elif cutlass.const_expr(operation == "reduce_batched"):
            reduced = coop.reduce_batched(group, values, output_layout=layout)
            for item in cutlass.range_constexpr((items + width - 1) // width):
                if cutlass.const_expr(layout == "striped"):
                    batch = lane + item * width
                else:
                    batch = lane * ((items + width - 1) // width) + item
                if batch < items:
                    outputs[group_index * items + batch] = reduced[item]
        elif cutlass.const_expr(operation == "exchange"):
            result = coop.exchange(group, values, mode=mode)
        elif cutlass.const_expr(operation == "shuffle"):
            if cutlass.const_expr(mode in ("up", "down")):
                result = coop.shuffle(group, values, mode=mode)
                if cutlass.const_expr(mode == "up"):
                    if lane == 0:
                        result[0] = value_type(0)
                else:
                    if lane == threads - 1:
                        result[items - 1] = value_type(0)
            else:
                shifted = coop.shuffle(group, values[0], mode=mode, distance=3)
                result = coop.ThreadData(1, dtype=value_type)
                result[0] = shifted
                if cutlass.const_expr(mode == "offset"):
                    if lane + 3 >= threads:
                        result[0] = value_type(0)
        elif cutlass.const_expr(operation == "merge_sort_keys"):
            result = coop.merge_sort_keys(group, values, descending=descending)
        else:
            associated = coop.ThreadData(items, dtype=cutlass.Int32)
            for item in cutlass.range_constexpr(items):
                associated[item] = aux_in[offset + item]
            result, paired = coop.merge_sort_pairs(
                group, values, associated, descending=descending
            )
            for item in cutlass.range_constexpr(items):
                aux_out[offset + item] = paired[item]
        if cutlass.const_expr(
            operation not in ("store", "sum", "reduce", "reduce_batched")
        ):
            for item in cutlass.range_constexpr(items):
                outputs[offset + item] = result[item]
        if cutlass.const_expr(operation != "merge_sort_pairs"):
            if block == 0 and thread == 0:
                aux_out[0] = cutlass.Int32(0)

    @cute.jit
    def launcher(
        source: cute.Pointer,
        source_aux: cute.Pointer,
        out: cute.Pointer,
        out_aux: cute.Pointer,
        stream: driver.CUstream,
    ):
        kernel(source, source_aux, out, out_aux).launch(
            grid=(blocks, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launcher


class PreparedKernel:
    def __init__(
        self, cubin, symbol, stream, pointers, grid, block, shared, metadata
    ):
        from cuda.core import LaunchConfig, ObjectCode, Stream

        self.module = ObjectCode.from_cubin(str(cubin))
        self.kernel = self.module.get_kernel(symbol)
        self.stream = Stream.from_handle(int(stream))
        self.config = LaunchConfig(grid=grid, block=block, shmem_size=shared)
        self.arguments = tuple(
            np.uint64(int(pointers[name])) for name in metadata["abi"]
        )
        self.metadata = metadata
        self._compiled = None
        self.compiled_launch = None

    def launch(self):
        from cuda.core import launch

        launch(self.stream, self.config, self.kernel, *self.arguments)

    def close(self):
        self.kernel = None
        self.module = None
        self._compiled = None
        self.compiled_launch = None


def prepare(
    case,
    inputs,
    outputs,
    stream,
    blocks,
    provider,
    artifacts,
    *,
    arch=None,
    cccl_root=None,
    toolkit=None,
):
    """Compile outside timing and use a common raw launch path."""
    from cuda.bindings import driver

    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    cccl_root = (
        Path(cccl_root) if cccl_root else Path(__file__).resolve().parents[3]
    )
    device = _checked(driver.cuCtxGetDevice())
    if arch is None:
        major = _checked(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
        )
        minor = _checked(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            )
        )
        arch = f"sm_{major}{minor}"
    pointers = {
        "input": inputs["input"],
        "input_aux": inputs["input_aux"],
        "output": outputs["output"],
        "output_aux": outputs["output_aux"],
    }
    grid, block, shared = (blocks, 1, 1), (case.threads, 1, 1), 0
    metadata = {
        "case": asdict(case),
        "provider": provider,
        "arch": arch,
        "workload": "one independent tile per group; one collective invocation",
        "cccl_root": str(cccl_root),
        "abi": ["input", "input_aux", "output", "output_aux"],
    }
    compiled = None
    if provider in ("cpp", "cub"):
        source = artifacts / "kernel.cu"
        source.write_text(native_source(case))
        cubin = artifacts / "kernel.cubin"
        nvcc = (
            str(Path(toolkit) / "bin/nvcc") if toolkit else shutil.which("nvcc")
        )
        if not nvcc:
            raise RuntimeError("nvcc is required to compile the CUB baseline")
        command = [
            nvcc,
            "-std=c++17",
            "-O3",
            f"-arch={arch}",
            "--cubin",
            str(source),
            "-o",
            str(cubin),
        ]
        for include in ("cub", "thrust", "libcudacxx/include"):
            command.extend(["-I", str(cccl_root / include)])
        completed = subprocess.run(command, text=True, capture_output=True)
        (artifacts / "compile.log").write_text(
            completed.stdout + completed.stderr
        )
        if completed.returncode:
            raise RuntimeError(f"nvcc failed: see {artifacts / 'compile.log'}")
        metadata.update(
            compile_command=command,
            nvcc_version=subprocess.check_output(
                [nvcc, "--version"], text=True
            ),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        symbol = "benchmark"
        metadata["parameter_layout"] = [
            {"index": index, "offset": 8 * index, "size": 8, "binding": name}
            for index, name in enumerate(metadata["abi"])
        ]
        metadata["parameter_layout_source"] = "native signature"
    elif provider == "cutlass":
        import cutlass
        from cutlass import cute
        from cutlass.base_dsl.compiler import (
            DumpDir,
            GPUArch,
            KeepCUBIN,
            KeepPTX,
        )
        from cutlass.cute.runtime import make_ptr
        from cutlass.cutlass_dsl import CuTeDSL

        from cuda.coop.cutlass._compiler._nvrtc import resolve_compile_context

        context = resolve_compile_context(("cub/version.cuh",))
        expected_roots = {
            (cccl_root / include).resolve()
            for include in ("cub", "thrust", "libcudacxx/include")
        }
        if not expected_roots.issubset(
            {Path(include).resolve() for include in context.include_dirs}
        ):
            raise RuntimeError("CUB and CuTe must use the same CCCL headers")
        metadata["provider_compile_context"] = asdict(context)

        value_type = {"int32": cutlass.Int32, "float32": cutlass.Float32}[
            case.dtype
        ]
        typed = [
            make_ptr(
                dtype, int(pointer), cute.AddressSpace.gmem, assumed_align=16
            )
            for pointer, dtype in zip(
                pointers.values(),
                (value_type, cutlass.Int32, value_type, cutlass.Int32),
            )
        ]
        dump = artifacts / "cute"
        dump.mkdir(exist_ok=True)
        launcher = make_launcher(case, blocks)
        metadata["compiler_route_environment"] = os.environ.get(
            "CUTE_DSL_USE_EXTENSION_COMPILER"
        )
        metadata["compiler_route"] = (
            "extension"
            if CuTeDSL._get_dsl().envar.use_extension_compiler
            else "legacy"
        )
        compiled = cute.compile[
            (GPUArch(arch), KeepCUBIN, KeepPTX, DumpDir(str(dump)))
        ](launcher, *typed, stream)
        # Lazy module loading must finish before capture. No output is consumed
        # here; the caller poisons and checks the raw cubin independently.
        compiled(*typed, stream)
        _checked(driver.cuStreamSynchronize(stream))
        graph = capture(stream, lambda: compiled(*typed, stream))
        try:
            _, count = _checked(driver.cuGraphGetNodes(graph, 0))
            if count != 1:
                raise RuntimeError(
                    f"Expected one CuTe kernel node, found {count}"
                )
            nodes, _ = _checked(driver.cuGraphGetNodes(graph, count))
            parameters = _checked(driver.cuGraphKernelNodeGetParams(nodes[0]))
            layout = _captured_parameters(parameters, pointers)
            metadata["abi"] = [item["binding"] for item in layout]
            metadata["parameter_layout"] = layout
            metadata["parameter_layout_source"] = "captured driver launch"
            symbol = _checked(driver.cuFuncGetName(parameters.func)).decode()
            grid = (
                parameters.gridDimX,
                parameters.gridDimY,
                parameters.gridDimZ,
            )
            block = (
                parameters.blockDimX,
                parameters.blockDimY,
                parameters.blockDimZ,
            )
            shared = parameters.sharedMemBytes
            if grid != (blocks, 1, 1) or block != (case.threads, 1, 1):
                raise RuntimeError(
                    "Compiled CuTe launch geometry differs from the case"
                )
        finally:
            _checked(driver.cuGraphDestroy(graph))
        candidates = list(dump.rglob("*.cubin"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one final cubin, found {candidates}")
        cubin = artifacts / "kernel.cubin"
        shutil.copyfile(candidates[0], cubin)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    metadata.update(
        cubin=str(cubin.resolve()),
        cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
        symbol=symbol,
        grid=list(grid),
        block=list(block),
        dynamic_shared_bytes=shared,
    )
    metadata["builder_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (artifacts / "kernel.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    result = PreparedKernel(
        cubin, symbol, stream, pointers, grid, block, shared, metadata
    )
    result._compiled = compiled
    if compiled is not None:
        result.compiled_launch = lambda: compiled(*typed, stream)
    return result
