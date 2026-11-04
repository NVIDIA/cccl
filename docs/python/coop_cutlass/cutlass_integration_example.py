# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Call a CUB block reduction from a CuTe kernel, in one self-contained file.

This script isolates the mechanisms ``cuda.coop.cutlass`` uses to integrate
with the CUTLASS CuTe DSL. It has none of the package's planning, thread
group, or payload abstractions. The kernel sums one ``int`` per thread with
``cub::BlockReduce`` and thread zero writes the total.

The numbered comments mark the integration points:

1. ``block_sum`` is an ordinary Python function that runs while CuTe traces
   the kernel body. Nothing is registered with the compiler ahead of time.
2. ``cute._get_launch_facts()`` supplies the exact block shape. It selects the
   ``cub::BlockReduce`` specialization, which must match the launch.
3. ``cute.ffi`` emits a call to an ``extern "C"`` symbol that has no
   definition yet.
4. A trace-finalize hook compiles that definition with NVRTC to LTO-IR and
   names the file in the ``gpu.module``'s ``link-libraries`` attribute.
5. CUB scratch is allocated by CuTe after NVRTC reports its exact ``sizeof``
   and ``alignof``. Placeholder operands traced in step 3 are replaced inside
   the finalized module.

Run it on Linux from a CCCL checkout with a compatible CUTLASS 4.9 CuTe DSL,
CUDA 13 driver and toolkit, ``cuda-bindings``, and NumPy. The example uses
the launch-facts and trace-finalization interfaces validated with the 4.9
candidate; see the accompanying documentation for setup details::

   python docs/python/coop_cutlass/cutlass_integration_example.py

Set ``CUDA_COOP_CUTLASS_EXAMPLE_PRINT_IR=1`` to print the finalized MLIR module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import tempfile
from pathlib import Path

import cutlass
import numpy as np
from cutlass import cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import CuTeDSL, T
from cutlass.memory import SmemAllocator

from cuda.bindings import driver, nvrtc

_BLOCK = (32, 4, 1)
_THREADS = _BLOCK[0] * _BLOCK[1] * _BLOCK[2]
_PROBE = "example_layout_probe"

# docs: start cute-cub-cpp
_PREAMBLE = f"""\
#include <cub/block/block_reduce.cuh>

// Layout probe (step 5). NVRTC evaluates the template arguments while it
// compiles the wrappers, and nvrtcGetLoweredName returns the mangled
// instantiation, which spells sizeof and alignof as decimal integers.
template <unsigned long long Size, unsigned long long Alignment>
__device__ unsigned char {_PROBE} = 0;
"""

_WRAPPER = """\
extern "C" __device__ int {symbol}(int value, unsigned int smem_addr,
                                   int smem_bytes) {{
  using reduce_t = {cpp_type};
  using storage_t = typename reduce_t::TempStorage;
  if (smem_bytes < (int) sizeof(storage_t) ||
      (smem_addr & (alignof(storage_t) - 1)) != 0) {{
    asm volatile("trap;");
  }}
  // CuTe passes a 32-bit shared-window address; CUB wants a generic pointer.
  unsigned long long generic_addr;
  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_addr)
      : "l"((unsigned long long) smem_addr));
  auto& storage = *reinterpret_cast<storage_t*>(generic_addr);
  int result = reduce_t(storage).Sum(value);
  __syncthreads();  // The scratch may be reused by a later call.
  return result;
}}
"""
# docs: end cute-cub-cpp


# docs: start cute-cub-requests
@dataclasses.dataclass(frozen=True)
class _Request:
    """One C++ definition to generate: a BlockReduce for one block shape."""

    block_dim: tuple[int, int, int]

    @property
    def cpp_type(self) -> str:
        x, y, z = self.block_dim
        return (
            f"::cub::BlockReduce<int, {x}, "
            f"::cub::BLOCK_REDUCE_WARP_REDUCTIONS, {y}, {z}>"
        )

    @property
    def symbol(self) -> str:
        x, y, z = self.block_dim
        return f"example_block_sum_i32_{x}x{y}x{z}"

    @property
    def probe(self) -> str:
        """NVRTC name expression whose mangled form encodes the layout."""

        storage = f"typename {self.cpp_type}::TempStorage"
        return f"&{_PROBE}<(sizeof({storage})), (alignof({storage}))>"


@dataclasses.dataclass(frozen=True)
class _ScratchUse:
    """Placeholder operands of one traced call, replaced at finalization."""

    request: _Request
    kernel_op: ir.Operation
    smem_addr: ir.Value
    smem_bytes: ir.Value


# One trace at a time. The real package keys this state by compile options
# and MLIR module so nested and concurrent traces stay separate.
_requests: dict[str, _Request] = {}
_scratch_uses: list[_ScratchUse] = []
# The LTO-IR file must outlive the device link, which happens after the hook.
_ARTIFACTS = tempfile.TemporaryDirectory(
    prefix="cuda-coop-cutlass-example-",
    dir=os.environ.get("CUDA_COOP_CUTLASS_EXAMPLE_ARTIFACT_DIR") or None,
)
# docs: end cute-cub-requests


def _check(result):
    """Raise on CUDA Driver or NVRTC errors and unwrap the returned values."""

    if int(result[0]):
        raise RuntimeError(f"CUDA call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


# docs: start cute-cub-entry
def block_sum(value: cutlass.Int32) -> cutlass.Int32:
    """Emit a call to a CUB block sum while CuTe traces the kernel.

    ``value`` wraps an MLIR value of the kernel body. The returned ``Int32``
    wraps the call result, which ``cub::BlockReduce::Sum`` defines only in
    thread zero.
    """

    # 2. Exact launch facts. ``.launch(grid=..., block=...)`` attaches them to
    #    the kernel operation before tracing its body, so the block shape is
    #    a Python tuple here. An upper bound such as maxntid would not do.
    facts = cute._get_launch_facts()
    if facts.exact_block_dim is None:
        raise RuntimeError("block_sum requires a static block shape")
    request = _Request(facts.exact_block_dim)
    _requests[request.symbol] = request

    # 4. Make sure the finalize hook runs for this trace. Registering the
    #    same callable twice is a no-op.
    CuTeDSL._get_dsl().register_trace_finalize_hook(_finalize_trace)

    # 5. Scratch placeholders. The C++ layout is unknown until NVRTC runs,
    #    so trace two i32 constants and remember them with their kernel.
    smem_addr = arith.constant(T.i32(), 0)
    smem_bytes = arith.constant(T.i32(), 0)
    _scratch_uses.append(
        _ScratchUse(request, _enclosing_kernel(), smem_addr, smem_bytes)
    )

    # 3. Declare a private prototype in the gpu.module and emit func.call.
    #    No definition exists yet; the device linker resolves it later.
    result = cute.ffi(
        name=request.symbol,
        params_types=[cutlass.Int32, cutlass.Uint32, cutlass.Int32],
        return_type=cutlass.Int32,
    )(value, cutlass.Uint32(smem_addr), cutlass.Int32(smem_bytes))
    return cutlass.Int32(result)


def _enclosing_kernel() -> ir.Operation:
    """Walk from the insertion point to the kernel that owns the scratch."""

    op = ir.InsertionPoint.current.block.owner
    while op is not None:
        operation = getattr(op, "operation", op)
        if operation.name == "cuda.kernel" or (
            operation.name == "lir.func"
            and "gpu.kernel" in operation.attributes
        ):
            return operation
        op = operation.parent
    raise RuntimeError("block_sum must be called inside a @cute.kernel body")


# docs: end cute-cub-entry


# docs: start cute-cub-finalize
def _finalize_trace(dsl, module: ir.Module, function_name: str) -> None:
    """Compile requests and complete the module before CuTe hashes it.

    CuTe calls every registered hook once per trace, after the Python body
    returns and before the module hash, the compile-cache lookup, and the
    device link. Everything emitted during tracing is visible here and can
    still be changed.
    """

    del function_name
    requests, uses = list(_requests.values()), list(_scratch_uses)
    _requests.clear()
    _scratch_uses.clear()
    if not requests:
        return  # A trace without block_sum calls needs nothing from us.

    source = _PREAMBLE + "".join(
        _WRAPPER.format(symbol=r.symbol, cpp_type=r.cpp_type) for r in requests
    )
    # The DSL already knows the target, for example "sm_120a".
    arch = "compute_" + dsl.envar.arch.removeprefix("sm_")
    ltoir, layouts = _compile_ltoir(source, arch, [r.probe for r in requests])
    digest = hashlib.sha256(ltoir).hexdigest()[:16]
    path = os.path.join(_ARTIFACTS.name, f"{digest}.ltoir")
    Path(path).write_bytes(ltoir)

    _allocate_scratch(uses, {r.symbol: layouts[r.probe] for r in requests})
    _attach_link_library(module, path)
    if os.environ.get("CUDA_COOP_CUTLASS_EXAMPLE_PRINT_IR"):
        print(module)


def _attach_link_library(module: ir.Module, path: str) -> None:
    """Name the LTO-IR file on every gpu.module.

    CuTe merges the attribute into its ``link-libraries`` compile option and
    passes the files to the device linker with the kernel's own LTO-IR.
    """

    for op in module.body.operations:
        if op.name != "gpu.module":
            continue
        paths = {path}
        if "link-libraries" in op.attributes:
            paths.update(attr.value for attr in op.attributes["link-libraries"])
        op.attributes["link-libraries"] = ir.ArrayAttr.get(
            [ir.StringAttr.get(item) for item in sorted(paths)]
        )


# docs: end cute-cub-finalize


# docs: start cute-cub-nvrtc
def _compile_ltoir(
    source: str, arch: str, probes: list[str]
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    """Compile the wrappers to LTO-IR and read back the probed layouts.

    Relocatable device code with ``-dlto`` yields an LTO-IR container, the
    input kind CuTe's device link already consumes for the kernel itself, so
    the linker can inline the wrapper into the kernel.
    """

    program = _check(
        nvrtc.nvrtcCreateProgram(source.encode(), b"wrappers.cu", 0, [], [])
    )
    try:
        for probe in probes:
            _check(nvrtc.nvrtcAddNameExpression(program, probe.encode()))
        options = [
            b"--std=c++17",
            b"--relocatable-device-code=true",
            b"-dlto",
            f"--gpu-architecture={arch}".encode(),
            *(f"--include-path={path}".encode() for path in _include_dirs()),
        ]
        if nvrtc.nvrtcCompileProgram(program, len(options), options)[0]:
            log = bytearray(_check(nvrtc.nvrtcGetProgramLogSize(program)))
            _check(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(f"NVRTC failed:\n{log.decode(errors='replace')}")
        layouts = {}
        for probe in probes:
            lowered = _check(nvrtc.nvrtcGetLoweredName(program, probe.encode()))
            # For example _Z20example_layout_probeILy32ELy4EE -> (32, 4).
            match = re.search(r"ILy(\d+)ELy(\d+)EE", str(lowered))
            layouts[probe] = (int(match.group(1)), int(match.group(2)))
        blob = bytearray(_check(nvrtc.nvrtcGetLTOIRSize(program)))
        _check(nvrtc.nvrtcGetLTOIR(program, blob))
        return bytes(blob), layouts
    finally:
        nvrtc.nvrtcDestroyProgram(program)


def _include_dirs() -> list[str]:
    """Select this checkout's CCCL headers, then the CUDA runtime headers."""

    root = Path(
        os.environ.get("CUDA_COOP_CUTLASS_CCCL_ROOT")
        or Path(__file__).resolve().parents[3]
    )
    if not (root / "cub" / "cub" / "version.cuh").is_file():
        raise RuntimeError(
            f"{root} is not a CCCL checkout; set CUDA_COOP_CUTLASS_CCCL_ROOT"
        )
    from cuda.pathfinder import find_nvidia_header_directory

    cuda_include = find_nvidia_header_directory("cudart")
    if cuda_include is None:
        raise RuntimeError("CUDA runtime headers not found; set CUDA_HOME")
    return [
        str(root / "cub"),
        str(root / "thrust"),
        str(root / "libcudacxx" / "include"),
        str(cuda_include),
    ]


# docs: end cute-cub-nvrtc


# docs: start cute-cub-scratch
def _allocate_scratch(
    uses: list[_ScratchUse], layouts: dict[str, tuple[int, int]]
) -> None:
    """Allocate CUB scratch at each kernel entry and patch the traced calls.

    ``SmemAllocator`` only emits ``memref.alloca`` operations. The CuTe
    compiler lays them out with the kernel's own shared memory and sizes the
    launch automatically. The traced placeholders become dead constants.
    """

    by_kernel: dict[ir.Operation, list[_ScratchUse]] = {}
    for use in uses:
        by_kernel.setdefault(use.kernel_op, []).append(use)
    for kernel_op, kernel_uses in by_kernel.items():
        entry_block = kernel_op.regions[0].blocks[0]
        with ir.InsertionPoint.at_block_begin(entry_block):
            allocator = SmemAllocator()
            for use in kernel_uses:
                size, alignment = layouts[use.request.symbol]
                pointer = allocator.allocate(size, alignment)
                address = llvm.ptrtoint(T.i32(), pointer.to_llvm_ptr())
                use.smem_addr.replace_all_uses_with(address)
                use.smem_bytes.replace_all_uses_with(
                    arith.constant(T.i32(), size)
                )


# docs: end cute-cub-scratch


# docs: start cute-cub-kernel
@cute.kernel
def sum_block(source: cute.Pointer, destination: cute.Pointer):
    tidx, tidy, _ = cute.arch.thread_idx()
    rank = tidx + tidy * _BLOCK[0]
    inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
    output = cute.make_tensor(destination, cute.make_layout(1))
    total = block_sum(inputs[rank])  # 1. Traced like any other CuTe call.
    if rank == 0:
        output[0] = total


@cute.jit
def launch(source: cute.Pointer, destination: cute.Pointer):
    sum_block(source, destination).launch(grid=1, block=_BLOCK)


# docs: end cute-cub-kernel


def main() -> None:
    """Compile once, launch once, and compare with NumPy."""

    cutlass.cuda.initialize_cuda_context()
    values = np.arange(_THREADS, dtype=np.int32) - 40
    result = np.zeros(1, dtype=np.int32)
    src = _check(driver.cuMemAlloc(values.nbytes))
    try:
        dst = _check(driver.cuMemAlloc(result.nbytes))
        try:
            _check(driver.cuMemcpyHtoD(src, values.ctypes.data, values.nbytes))
            source = make_ptr(
                cutlass.Int32,
                int(src),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            destination = make_ptr(
                cutlass.Int32,
                int(dst),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            # docs: start cute-cub-main
            # Tracing, the finalize hook, NVRTC, and the device link all happen
            # inside cute.compile. The returned callable launches the cubin.
            compiled = cute.compile(launch, source, destination)
            compiled(source, destination)
            # docs: end cute-cub-main
            _check(driver.cuCtxSynchronize())
            _check(driver.cuMemcpyDtoH(result.ctypes.data, dst, result.nbytes))
        finally:
            _check(driver.cuMemFree(dst))
    finally:
        _check(driver.cuMemFree(src))
    expected = int(values.sum())
    if int(result[0]) != expected:
        raise AssertionError(f"block sum {int(result[0])} != {expected}")
    print(f"block sum verified: {expected}")


if __name__ == "__main__":
    main()
