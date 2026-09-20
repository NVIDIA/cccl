# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the CUTLASS Developer Guide's Debugger Walkthrough."""

import argparse
import os
import tempfile

# Import the kernel DSL before cuda.coop so automatic registration sees it.
# isort: off
import numpy as np
import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop  # First breakpoint: observe backend registration.
from cuda.bindings import driver
# isort: on


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def main(*, algorithm="direct"):
    # docs: start cutlass-debug-kernel
    @cute.kernel
    def copy_tile(source: cute.Pointer, destination: cute.Pointer):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        scratch = coop.TempStorage()
        coop.load(block, source, items, algorithm=algorithm, temp_storage=scratch)
        coop.store(block, destination, items, algorithm=algorithm, temp_storage=scratch)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        copy_tile(source, destination).launch(grid=1, block=128)

    # docs: end cutlass-debug-kernel

    cutlass.cuda.initialize_cuda_context()
    source = np.arange(256, dtype=np.int32)
    destination = np.full_like(source, -1)
    source_device = _check(driver.cuMemAlloc(source.nbytes))
    try:
        destination_device = _check(driver.cuMemAlloc(destination.nbytes))
        try:
            _check(
                driver.cuMemcpyHtoD(source_device, source.ctypes.data, source.nbytes)
            )
            source_pointer = make_ptr(
                cutlass.Int32,
                int(source_device),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            destination_pointer = make_ptr(
                cutlass.Int32,
                int(destination_device),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )

            # docs: start cutlass-debug-launches
            compiled = cute.compile(launch, source_pointer, destination_pointer)
            for iteration in range(2):
                destination.fill(-1)
                _check(
                    driver.cuMemcpyHtoD(
                        destination_device, destination.ctypes.data, destination.nbytes
                    )
                )
                compiled(source_pointer, destination_pointer)
                _check(driver.cuCtxSynchronize())
                _check(
                    driver.cuMemcpyDtoH(
                        destination.ctypes.data, destination_device, destination.nbytes
                    )
                )
                np.testing.assert_array_equal(destination, source)
                print(f"Launch {iteration + 1} ({algorithm}): copy verified")
            # docs: end cutlass-debug-launches
        finally:
            _check(driver.cuMemFree(destination_device))
    finally:
        _check(driver.cuMemFree(source_device))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=("direct", "transpose"), default="direct"
    )
    args = parser.parse_args()
    # Each debug session reaches NVRTC; the second launch still reuses its kernel.
    with tempfile.TemporaryDirectory(prefix="cuda-coop-cutlass-debug-") as cache_dir:
        os.environ["CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR"] = cache_dir
        main(algorithm=args.algorithm)
