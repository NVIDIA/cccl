# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check independent histogram types, striped bin ownership, and fresh counts.

A host bincount checks each block's last input tile. Output slots at or
beyond the bin count must be zero. Both algorithms must preserve the
original samples, including Sort, which uses a private mutable copy.
"""

import re
import shutil
import subprocess

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import cutlass_dtype, device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


class _Readonly:
    """Copy sample items into a tuple that Histogram cannot write.

    Histogram must accept this read-only payload and infer the sample dtype
    from its items when no dtype is set. The harness checks preservation
    through the original ThreadData.
    """

    def __init__(self, source):
        self.items_per_thread, self.dtype, self.alignment = (
            source.items_per_thread,
            source.dtype,
            source.alignment,
        )
        self._items = tuple(source)

    def __len__(self):
        return self.items_per_thread

    def __getitem__(self, index):
        return self._items[index]


def _run(
    api=coop,
    *,
    dtype=np.int32,
    counter_dtype=np.int64,
    algorithm="atomic",
    threads=64,
    bins=65,
    bins_per_thread=2,
    items_per_thread=3,
    reuse=False,
    sharing="shared",
    manual_sync=False,
    alignment=64,
    capacity=None,
    payload="thread_data",
    inferred=False,
    cute_selector=False,
    default_counter=False,
    compile_options=(),
):
    """Compare striped counters against each block's final input tile.

    Input items and output bins have independent extents and types. Plain CuTe
    stores bin slot i from thread t at t + i * threads, so the host sees bins
    in order. It also checks zero padding and every original sample.

    Reuse cases process four different tiles with the same scratch. Output
    must count only the last tile, which exposes counters left from earlier
    calls. Manual barriers protect reuse when automatic barriers are off.
    """

    value_type, counter_type = (
        cutlass_dtype(dtype),
        cutlass_dtype(counter_dtype),
    )
    blocks, repeats = 2, 4 if reuse else 1
    tile, projection = threads * items_per_thread, threads * bins_per_thread
    size = tile * blocks * repeats
    selector = (
        None
        if default_counter
        else counter_type
        if cute_selector
        else counter_dtype
    )

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        output: cute.Pointer,
        preserved: cute.Pointer,
        iterations: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        thread = cute.arch.thread_idx()[0]
        block_index = cute.arch.block_idx()[0]
        group = api.this_block()
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(output, cute.make_layout(projection * blocks)),
            counter_type,
        )
        originals = cute.recast_tensor(
            cute.make_tensor(preserved, cute.make_layout(size)), value_type
        )
        samples = api.ThreadData(
            items_per_thread,
            dtype=None if inferred else value_type,
            alignment=alignment,
        )
        for item in cutlass.range_constexpr(items_per_thread):
            samples[item] = value_type(0)
        if cutlass.const_expr(reuse):
            scratch = api.TempStorage(
                size_in_bytes=capacity,
                sharing=sharing,
                alignment=alignment,
                auto_sync=not manual_sync,
            )
        else:
            scratch = None
        for iteration in range(iterations):
            start = (
                block_index * repeats + iteration
            ) * tile + thread * items_per_thread
            for item in cutlass.range_constexpr(items_per_thread):
                samples[item] = inputs[start + item]
            if cutlass.const_expr(payload == "readonly"):
                values = _Readonly(samples)
            elif cutlass.const_expr(payload == "tensor"):
                values = samples.to_register_tensor()
            elif cutlass.const_expr(payload == "vector"):
                values = samples.to_tensor_ssa()
            else:
                values = samples
            counts = api.histogram(
                group,
                values,
                bins=bins,
                bins_per_thread=bins_per_thread,
                counter_dtype=selector,
                algorithm=algorithm,
                temp_storage=scratch,
            )
            for item in cutlass.range_constexpr(bins_per_thread):
                outputs[block_index * projection + thread + item * threads] = (
                    counts[item]
                )
            for item in cutlass.range_constexpr(items_per_thread):
                originals[start + item] = samples[item]
            if cutlass.const_expr(reuse and manual_sync):
                scratch.sync()

    @cute.jit
    def launch(
        source: cute.Pointer,
        output: cute.Pointer,
        preserved: cute.Pointer,
        iterations: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, output, preserved, iterations, items_per_thread).launch(
            grid=blocks, block=threads
        )

    source = ((np.arange(size) * 19 + np.arange(size) // tile) % bins).astype(
        dtype
    )
    source[:tile] = bins - 1
    output = np.full(projection * blocks, 99, dtype=counter_dtype)
    preserved = np.zeros_like(source)
    with (
        device_array(source) as src,
        device_array(output) as out,
        device_array(preserved) as keep,
    ):
        args = src, out, keep, cutlass.Int32(repeats)
        compiled = (
            cute.compile[compile_options](launch, *args, items_per_thread)
            if compile_options
            else cute.compile(launch, *args, items_per_thread)
        )
        compiled(*args)
    np.testing.assert_array_equal(preserved, source)
    for block in range(blocks):
        start = (block * repeats + repeats - 1) * tile
        expected = np.zeros(projection, dtype=counter_dtype)
        expected[:bins] = np.bincount(
            source[start : start + tile].astype(np.int64), minlength=bins
        )
        np.testing.assert_array_equal(
            output[block * projection : (block + 1) * projection], expected
        )


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize(
    "dtype", (np.uint8, np.int32, np.uint32, np.int64, np.uint64)
)
@pytest.mark.parametrize(
    "counter_dtype", (np.int32, np.uint32, np.int64, np.uint64)
)
@pytest.mark.parametrize("algorithm", ("atomic", "sort"))
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_counts_types_and_preservation(
    api, dtype, counter_dtype, algorithm, items_per_thread
):
    _run(
        api,
        dtype=dtype,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
        items_per_thread=items_per_thread,
    )


@pytest.mark.parametrize("algorithm", ("atomic", "sort"))
@pytest.mark.parametrize("threads,bins", ((1, 1), (7, 13), (32, 1), (64, 128)))
def test_bin_capacity(algorithm, threads, bins):
    _run(
        algorithm=algorithm,
        threads=threads,
        bins=bins,
        bins_per_thread=(bins + threads - 1) // threads,
    )


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("manual_sync", (False, True))
@pytest.mark.parametrize("algorithm", ("atomic", "sort"))
def test_fresh_counters_on_scratch_reuse(sharing, manual_sync, algorithm):
    """Discard earlier counts while reusing scratch across four input tiles.

    Both algorithms and sharing policies use the same last-tile oracle. Manual
    barriers replace automatic reuse barriers when requested by the test.
    """

    _run(
        sharing=sharing,
        manual_sync=manual_sync,
        algorithm=algorithm,
        reuse=True,
        alignment=128,
    )


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
def test_readonly_inference(api):
    _run(api, payload="readonly", inferred=True)


@pytest.mark.parametrize("payload", ("tensor", "vector"))
def test_qualified_register_payloads(payload):
    _run(cutlass_coop, payload=payload, cute_selector=True)


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
def test_default_counter(api):
    _run(api, default_counter=True, counter_dtype=np.int32)


def test_alignment_minimum():
    """Allow native scratch alignment to exceed a one-byte request.

    A large capacity isolates alignment from size while repeated histograms
    still check fresh counters and input preservation.
    """

    _run(reuse=True, alignment=1, capacity=8192)


def test_undersized_storage():
    with pytest.raises(
        (ValueError, DSLRuntimeError), match="size|capacity|bytes|storage"
    ):
        _run(reuse=True, capacity=1)


@pytest.mark.parametrize("algorithm", ("atomic", "sort"))
def test_final_cubin(tmp_path, algorithm):
    """Check that each generated histogram wrapper is fully inlined.

    Each algorithm first passes the host result checks. The final SASS must
    then contain no wrapper symbol and no CALL instruction. Register and
    shared-memory usage are not checked here.
    """

    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.skip("cuobjdump is required for final linked code inspection")
    _run(
        algorithm=algorithm,
        compile_options=(KeepCUBIN(True), DumpDir(str(tmp_path))),
    )
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output(
            [tool, "--dump-sass", str(cubin)], text=True
        )
        assert "cuda_coop_cutlass_histogram_" not in sass
        assert re.search(r"\bCALL\b", sass) is None


@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_documented_histogram(items_per_thread):
    """Execute the documented striped store of independently typed counters.

    The example maps 65 bins onto 128 output slots. The host checks each count
    and requires zero in every padding slot, for both input payload extents.
    """

    # docs: start cutlass-histogram
    @cute.kernel
    def count_samples(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        samples = coop.ThreadData(items_per_thread)
        coop.load(block, source, samples)
        counts = coop.histogram(
            block, samples, bins=65, bins_per_thread=2, counter_dtype=np.int64
        )
        coop.store(block, destination, counts, algorithm="striped")

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        count_samples(source, destination, items_per_thread).launch(
            grid=1, block=64
        )

    # docs: end cutlass-histogram

    source = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 65
    output = np.full(128, -1, dtype=np.int64)
    with device_array(source) as src, device_array(output) as out:
        launch(src, out, items_per_thread)
    np.testing.assert_array_equal(
        output[:65], np.bincount(source, minlength=65)
    )
    np.testing.assert_array_equal(output[65:], 0)
