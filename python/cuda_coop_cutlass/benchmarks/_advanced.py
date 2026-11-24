# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Matched advanced primitives and independent host references.

The native Histogram, TopK and Run Length Decode paths retain the package's
C++ compatibility adapters. Their checked boundaries and output contracts
are part of the measured operation; NumPy provides independent references.
"""

# Long lines below are the emitted C++ translation unit.
# ruff: noqa: E501

import numpy as np


def cases():
    from _cases import BenchmarkCase

    result = []

    def add(name, operation, **options):
        suffix = (
            name[len(operation) :].strip(".")
            if name.startswith(operation)
            else ""
        )
        case_name = f"block128_{operation}_int32_i4"
        if suffix:
            case_name += "_" + suffix.replace(".", "_")
        result.append(
            BenchmarkCase(
                name=case_name,
                operation=operation,
                options={"advanced": True, **options},
            )
        )

    for algorithm in ("atomic", "sort"):
        add("histogram." + algorithm, "histogram", algorithm=algorithm, bins=64)
    for direction in ("min", "max"):
        for payload in ("keys", "pairs"):
            operation = f"topk_{direction}_{payload}"
            add(operation, operation, k=17)
    for descending in (False, True):
        order = "descending" if descending else "ascending"
        for payload in ("keys", "pairs"):
            operation = "radix_sort_" + payload
            add(operation + "." + order, operation, descending=descending)
        add(
            "radix_rank_keys." + order, "radix_rank_keys", descending=descending
        )
    add("run_length_decode.window", "run_length_decode", offset=18)
    add("run_length_decode.into", "run_length_decode_into")
    for direction in ("left", "right"):
        add(
            "adjacent_difference." + direction,
            "adjacent_difference",
            direction=direction,
        )
    for mode in ("heads", "tails", "heads_and_tails"):
        add("discontinuity." + mode, "discontinuity", mode=mode)
    return result


def make_arrays(case, blocks):
    tile = case.threads * case.items
    local = np.arange(tile, dtype=np.int32)
    values = np.tile((37 * local) % tile - tile // 2, (blocks, 1))
    values += np.arange(blocks, dtype=np.int32)[:, None]
    inputs = {
        "input": values.reshape(-1).copy(),
        "input_aux": np.zeros(1, np.int32),
    }
    expected = {
        "output": np.zeros(blocks * tile, np.int32),
        "output_aux": np.zeros(1, np.int32),
    }
    operation = case.operation
    if operation == "histogram":
        bins = case.options["bins"]
        values = np.mod(values, bins)
        inputs["input"] = values.reshape(-1).copy()
        output = np.zeros((blocks, case.threads), np.int32)
        for block in range(blocks):
            output[block, :bins] = np.bincount(values[block], minlength=bins)
        expected["output"] = output.reshape(-1)
    elif operation.startswith("topk_") or operation.startswith("radix_sort_"):
        descending = operation.startswith("topk_max") or case.options.get(
            "descending", False
        )
        order = np.argsort(
            -values if descending else values, axis=1, kind="stable"
        )
        if operation.startswith("topk_"):
            order = order[:, : case.options["k"]]
        expected["output"] = np.take_along_axis(values, order, axis=1).reshape(
            -1
        )
        if operation.endswith("pairs"):
            inputs["input_aux"] = np.tile(local, blocks)
            expected["output_aux"] = order.astype(np.int32).reshape(-1)
    elif operation == "radix_rank_keys":
        digits = np.bitwise_and(values, 15)
        if case.options["descending"]:
            digits = -digits
        order = np.argsort(digits, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.int32)
        np.put_along_axis(
            ranks, order, np.broadcast_to(local, order.shape), axis=1
        )
        expected["output"] = ranks.reshape(-1)
    elif operation.startswith("run_length_decode"):
        lengths = np.tile(np.array([1, 2, 3, 2], np.int32), blocks * tile // 4)
        inputs["input_aux"] = lengths
        decoded = np.stack(
            [
                np.repeat(values[block], lengths.reshape(blocks, tile)[block])
                for block in range(blocks)
            ]
        )
        if operation == "run_length_decode":
            offset = case.options["offset"]
            expected["output"] = decoded[:, offset : offset + tile].reshape(-1)
        else:
            expected["output"] = decoded.reshape(-1)
            expected["output_aux"] = np.full(blocks, 2 * tile, np.int32)
    elif operation in ("adjacent_difference", "discontinuity"):
        values = np.tile(local // 3, (blocks, 1)) % 11
        values += np.arange(blocks, dtype=np.int32)[:, None]
        inputs["input"] = values.reshape(-1).copy()
        if operation == "adjacent_difference":
            output = values.copy()
            if case.options["direction"] == "left":
                output[:, 1:] -= values[:, :-1]
            else:
                output[:, :-1] -= values[:, 1:]
            expected["output"] = output.reshape(-1)
        else:
            heads = np.ones_like(values)
            tails = np.ones_like(values)
            heads[:, 1:] = values[:, 1:] != values[:, :-1]
            tails[:, :-1] = values[:, :-1] != values[:, 1:]
            mode = case.options["mode"]
            expected["output"] = (tails if mode == "tails" else heads).reshape(
                -1
            )
            if mode == "heads_and_tails":
                expected["output_aux"] = tails.reshape(-1)
    else:
        raise ValueError(f"Unsupported advanced operation: {operation}")
    return inputs, expected


def assert_result(case, observed, expected):
    if case.operation.startswith("topk_"):
        k = case.options["k"]
        keys = observed["output"].reshape(-1, k)
        expected_keys = expected["output"].reshape(-1, k)
        order = np.argsort(keys, axis=1)
        expected_order = np.argsort(expected_keys, axis=1)
        np.testing.assert_array_equal(
            np.take_along_axis(keys, order, axis=1),
            np.take_along_axis(expected_keys, expected_order, axis=1),
        )
        if case.operation.endswith("pairs"):
            np.testing.assert_array_equal(
                np.take_along_axis(
                    observed["output_aux"].reshape(-1, k), order, axis=1
                ),
                np.take_along_axis(
                    expected["output_aux"].reshape(-1, k),
                    expected_order,
                    axis=1,
                ),
            )
        else:
            np.testing.assert_array_equal(
                observed["output_aux"], expected["output_aux"]
            )
    else:
        for key, values in expected.items():
            np.testing.assert_array_equal(observed[key], values)


def native_source(case):
    threads, items = case.threads, case.items
    tile = threads * items
    operation = case.operation
    prefix = ""
    headers = []
    load_aux = False
    output_items = items
    output_tile = tile
    store_aux = False
    custom_store = ""
    if operation == "histogram":
        from cuda.coop.cutlass._planning.block.histogram import (
            BLOCK_HISTOGRAM_TYPE,
        )

        prefix = BLOCK_HISTOGRAM_TYPE.code
        headers = ["cub/block/block_histogram.cuh", "cuda/std/type_traits"]
        algorithm = (
            "ATOMIC" if case.options["algorithm"] == "atomic" else "SORT"
        )
        implementation = f"cub::BlockHistogramCoop<int, {threads}, {items}, {case.options['bins']}, 1, int, cub::BLOCK_HISTO_{algorithm}>"
        action = "implementation(storage).Histogram(values, result);"
        output_items, output_tile = 1, threads
    elif operation.startswith("topk_"):
        from cuda.coop.cutlass._planning.block.topk import BLOCK_TOPK_TYPE

        prefix = BLOCK_TOPK_TYPE.code
        headers = ["cub/block/block_topk.cuh"]
        pairs = operation.endswith("pairs")
        value_type = "int" if pairs else "cub::NullType"
        implementation = (
            f"cub::BlockTopKCoop<int, {threads}, {items}, {value_type}>"
        )
        method = operation.removeprefix("topk_") + "_full"
        args = "values, aux" if pairs else "values"
        action = f"implementation(storage).{method}({args}, {case.options['k']}ll, {tile}ll);"
        action += f"\n  for (int i = 0; i < {items}; ++i) {{ result[i] = values[i]; }}"
        load_aux = store_aux = pairs
        output_tile = case.options["k"]
    elif operation.startswith("radix_sort_"):
        headers = ["cub/block/block_radix_sort.cuh"]
        pairs = operation.endswith("pairs")
        value_type = "int" if pairs else "cub::NullType"
        implementation = (
            f"cub::BlockRadixSort<int, {threads}, {items}, {value_type}>"
        )
        method = "SortDescending" if case.options["descending"] else "Sort"
        args = "values, aux" if pairs else "values"
        action = f"implementation(storage).{method}({args});"
        action += f"\n  for (int i = 0; i < {items}; ++i) {{ result[i] = values[i]; }}"
        load_aux = store_aux = pairs
    elif operation == "radix_rank_keys":
        headers = [
            "cub/block/block_radix_rank.cuh",
            "cub/block/radix_rank_sort_operations.cuh",
        ]
        descending = str(case.options["descending"]).lower()
        implementation = f"cub::BlockRadixRank<{threads}, 4, {descending}>"
        action = f"""unsigned int bits[{items}];
  for (int i = 0; i < {items}; ++i) {{ bits[i] = static_cast<unsigned int>(values[i]) ^ 0x80000000u; }}
  implementation(storage).RankKeys(bits, result, cub::BFEDigitExtractor<unsigned int>(0, 4));"""
    elif operation.startswith("run_length_decode"):
        from cuda.coop.cutlass._planning.block.run_length import (
            BLOCK_RUN_LENGTH_DECODE_DRIVER,
        )

        prefix = BLOCK_RUN_LENGTH_DECODE_DRIVER.code
        headers = [
            "cub/block/block_run_length_decode.cuh",
            "cub/block/block_scan.cuh",
            "cuda/std/limits",
            "cuda/std/type_traits",
        ]
        implementation = f"cub::BlockRunLengthDecodeCoop<int, int, unsigned int, long long, {threads}, {items}, {items}>"
        load_aux = True
        if operation == "run_length_decode":
            action = f"""unsigned int total[1], relative[{items}];
  implementation(storage).Window(values, aux, result, total, relative, {case.options["offset"]}ll);"""
        else:
            output_tile = 2 * tile
            action = f"""unsigned int total;
  implementation(storage).Into(values, aux, output + blockIdx.x * {output_tile}, {output_tile}ll, 0ll, total);"""
            custom_store = "if (threadIdx.x == 0) { output_aux[blockIdx.x] = static_cast<int>(total); }"
    elif operation == "adjacent_difference":
        headers = [
            "cub/block/block_adjacent_difference.cuh",
            "cuda/std/functional",
        ]
        implementation = f"cub::BlockAdjacentDifference<int, {threads}>"
        method = (
            "SubtractLeft"
            if case.options["direction"] == "left"
            else "SubtractRight"
        )
        action = f"implementation(storage).{method}(values, result, cuda::std::minus<int>{{}});"
    elif operation == "discontinuity":
        headers = ["cub/block/block_discontinuity.cuh", "cuda/std/functional"]
        implementation = f"cub::BlockDiscontinuity<int, {threads}>"
        mode = case.options["mode"]
        if mode == "heads_and_tails":
            action = "implementation(storage).FlagHeadsAndTails(result, aux, values, cuda::std::not_equal_to<int>{});"
            store_aux = True
        else:
            method = "FlagHeads" if mode == "heads" else "FlagTails"
            action = f"implementation(storage).{method}(result, values, cuda::std::not_equal_to<int>{{}});"
    else:
        raise ValueError(operation)
    headers += [
        "cub/block/block_load.cuh",
        "cub/block/block_store.cuh",
        "cuda/std/__memory/assume_aligned.h",
    ]
    aux_load = (
        f"cub::LoadDirectBlockedVectorized(threadIdx.x, const_cast<int*>(input_aux) + blockIdx.x * {tile}, aux);"
        if load_aux
        else ""
    )
    if custom_store:
        store = custom_store
    elif output_tile == threads * output_items:
        store = f"cub::StoreDirectBlockedVectorized(threadIdx.x, output + blockIdx.x * {output_tile}, result);"
        if store_aux:
            store += f"\n  cub::StoreDirectBlockedVectorized(threadIdx.x, output_aux + blockIdx.x * {output_tile}, aux);"
    else:
        store = f"""for (int i = 0; i < {output_items}; ++i) {{
    const int index = threadIdx.x * {output_items} + i;
    if (index < {output_tile}) {{
      output[blockIdx.x * {output_tile} + index] = result[i];
      {f"output_aux[blockIdx.x * {output_tile} + index] = aux[i];" if store_aux else ""}
    }}
  }}"""
    if not store_aux and not custom_store:
        store += "\n  if (blockIdx.x == 0 && threadIdx.x == 0) { output_aux[0] = 0; }"
    aux_declaration = f"int aux[{items}];" if load_aux or store_aux else ""
    result_declaration = (
        f"int result[{output_items}];" if not custom_store else ""
    )
    return (
        "// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.\n"
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
        + "\n".join(f"#include <{header}>" for header in headers)
        + f"""
{prefix}
extern "C" __global__ void benchmark(const int* input, const int* input_aux, int* output, int* output_aux)
{{
  input = cuda::std::assume_aligned<16>(input);
  input_aux = cuda::std::assume_aligned<16>(input_aux);
  output = cuda::std::assume_aligned<16>(output);
  output_aux = cuda::std::assume_aligned<16>(output_aux);
  using implementation = {implementation};
  __shared__ typename implementation::TempStorage storage;
  int values[{items}];
  {aux_declaration}
  {result_declaration}
  // The read-only CUB helper takes a non-const pointer.
  cub::LoadDirectBlockedVectorized(threadIdx.x, const_cast<int*>(input) + blockIdx.x * {tile}, values);
  {aux_load}
  {action}
  __syncthreads();
  {store}
}}
"""
    )


def make_launcher(case, blocks):
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop
    from cuda.bindings import driver

    operation = case.operation
    items, threads = case.items, case.threads
    tile, size = items * threads, blocks * items * threads
    pairs = operation.endswith("pairs")
    run_length = operation.startswith("run_length_decode")
    histogram = operation == "histogram"
    bulk = operation == "run_length_decode_into"
    k = case.options.get("k", tile)
    output_tile = k if operation.startswith("topk_") else tile
    if histogram:
        output_tile = threads
    if bulk:
        output_tile = 2 * tile
    output_items = 1 if histogram else items
    aux_output = pairs or (
        operation == "discontinuity"
        and case.options["mode"] == "heads_and_tails"
    )
    aux_size = blocks if bulk else blocks * output_tile if aux_output else 1
    source_aux_size = size if pairs or run_length else 1
    topk = getattr(coop, operation) if operation.startswith("topk_") else None
    descending = case.options.get("descending", False)
    algorithm = case.options.get("algorithm", "atomic")
    bins = case.options.get("bins", 64)
    direction = case.options.get("direction", "left")
    mode = case.options.get("mode", "heads")
    offset = case.options.get("offset", 0)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        source_aux: cute.Pointer,
        out: cute.Pointer,
        out_aux: cute.Pointer,
    ):
        thread = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        sources = cute.make_tensor(source, cute.make_layout(size))
        auxiliaries = cute.make_tensor(
            source_aux, cute.make_layout(source_aux_size)
        )
        outputs = cute.make_tensor(out, cute.make_layout(blocks * output_tile))
        auxiliary_outputs = cute.make_tensor(
            out_aux, cute.make_layout(aux_size)
        )
        group = coop.this_block()
        values = coop.ThreadData(items, dtype=cutlass.Int32)
        aux = coop.ThreadData(items, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(items):
            values[item] = sources[block * tile + thread * items + item]
            aux[item] = cutlass.Int32(0)
            if cutlass.const_expr(pairs or run_length):
                aux[item] = auxiliaries[block * tile + thread * items + item]
        result = coop.ThreadData(output_items, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(output_items):
            result[item] = cutlass.Int32(0)
        if cutlass.const_expr(histogram):
            result = coop.histogram(
                group,
                values,
                bins=bins,
                bins_per_thread=1,
                counter_dtype=cutlass.Int32,
                algorithm=algorithm,
            )
        elif cutlass.const_expr(operation.startswith("topk_")):
            if cutlass.const_expr(pairs):
                result, aux = topk(group, values, aux, k=k)
            else:
                result = topk(group, values, k=k)
        elif cutlass.const_expr(operation == "radix_sort_keys"):
            result = coop.radix_sort_keys(group, values, descending=descending)
        elif cutlass.const_expr(operation == "radix_sort_pairs"):
            result, aux = coop.radix_sort_pairs(
                group, values, aux, descending=descending
            )
        elif cutlass.const_expr(operation == "radix_rank_keys"):
            result = coop.radix_rank_keys(
                group, values, begin_bit=0, radix_bits=4, descending=descending
            )
        elif cutlass.const_expr(operation == "run_length_decode"):
            result = coop.run_length_decode(
                group,
                values,
                aux,
                decoded_items_per_thread=items,
                decoded_window_offset=offset,
            )
        elif cutlass.const_expr(bulk):
            destination = cute.make_tensor(
                outputs.iterator + block * output_tile,
                cute.make_layout(output_tile),
            )
            total = coop.run_length_decode_into(
                group, values, aux, destination, decoded_items_per_thread=items
            )
            if thread == 0:
                auxiliary_outputs[block] = cutlass.Int32(total)
        elif cutlass.const_expr(operation == "adjacent_difference"):
            result = coop.adjacent_difference(
                group, values, direction=direction
            )
        elif cutlass.const_expr(mode == "heads_and_tails"):
            result, aux = coop.discontinuity(group, values, mode=mode)
        else:
            result = coop.discontinuity(group, values, mode=mode)
        if cutlass.const_expr(not bulk):
            for item in cutlass.range_constexpr(output_items):
                index = thread * output_items + item
                if index < output_tile:
                    outputs[block * output_tile + index] = result[item]
            if cutlass.const_expr(aux_output):
                for item in cutlass.range_constexpr(output_items):
                    index = thread * output_items + item
                    if index < output_tile:
                        auxiliary_outputs[block * output_tile + index] = aux[
                            item
                        ]
            if cutlass.const_expr(not aux_output):
                if block == 0:
                    if thread == 0:
                        auxiliary_outputs[0] = cutlass.Int32(0)

    @cute.jit
    def launch(
        source: cute.Pointer,
        source_aux: cute.Pointer,
        out: cute.Pointer,
        out_aux: cute.Pointer,
        stream: driver.CUstream,
    ):
        kernel(source, source_aux, out, out_aux).launch(
            grid=blocks, block=threads, stream=stream
        )

    return launch
