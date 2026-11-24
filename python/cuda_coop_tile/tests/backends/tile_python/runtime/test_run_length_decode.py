# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.e2e import configure_e2e_environment, import_e2e_modules


@pytest.mark.parametrize(
    "value_dtype,length_dtype,run_count,decoded_items",
    [
        ("int32", "uint32", 128, 256),
        ("int32", "uint32", 32, 64),
        ("float32", "uint64", 64, 32),
        ("float32", "uint64", 256, 128),
        ("float64", "uint32", 128, 128),
        ("int8", "uint32", 128, 256),
        ("uint8", "uint32", 128, 256),
        ("int16", "uint64", 128, 256),
        ("uint16", "uint64", 128, 256),
        ("uint32", "uint32", 128, 256),
        ("int64", "uint64", 128, 256),
        ("uint64", "uint64", 128, 256),
        ("float16", "uint32", 128, 256),
        ("bfloat16", "uint64", 128, 256),
    ],
)
def test_run_length_decode_windows(
    monkeypatch, value_dtype, length_dtype, run_count, decoded_items
):
    configure_e2e_environment(monkeypatch)
    global ct, coop
    torch, ct, coop = import_e2e_modules()
    value_type = getattr(torch, value_dtype)
    length_type = getattr(torch, length_dtype)
    sentinel = (1 << (32 if length_dtype == "uint32" else 64)) - 1
    values = [i % 31 + 1 for i in range(run_count)]
    if value_dtype.startswith("int"):
        values = [value - 16 for value in values]
    if value_dtype in ("int64", "uint64"):
        high_bits = -(1 << 40) if value_dtype == "int64" else 1 << 40
        values = [value + high_bits for value in values]
    if value_dtype.startswith("float") or value_dtype == "bfloat16":
        values = [(value - 16) / 4 for value in values]
    # CUB permits a positive prefix followed by zero-length padding runs.
    lengths = [
        i % 5 + 1 if i < max(run_count - 32, run_count // 2) else 0
        for i in range(run_count)
    ]
    decoded = [
        value for value, count in zip(values, lengths) for _ in range(count)
    ]
    relative = [offset for count in lengths for offset in range(count)]
    inputs = torch.tensor(values, dtype=value_type, device="cuda")
    counts = torch.tensor(lengths, dtype=length_type, device="cuda")
    out = torch.empty(decoded_items, dtype=value_type, device="cuda")
    out_with_offsets = torch.empty_like(out)
    offsets = torch.empty(decoded_items, dtype=length_type, device="cuda")

    worker_warps = (
        min(run_count, decoded_items) // 32
        if min(run_count, decoded_items) < 128
        else None
    )

    @ct.kernel(num_worker_warps=worker_warps)
    def kernel(inp, counts, window, out, out_with_offsets, offsets):
        values = ct.load(inp, (0,), shape=(run_count,))
        lengths = ct.load(counts, (0,), shape=(run_count,))
        start = ct.load(window, (0,), shape=(1,))
        result = coop.run_length_decode(
            values,
            lengths,
            decoded_items=decoded_items,
            decoded_window_offset=start,
        )
        result_pair, relative = coop.run_length_decode_with_offsets(
            values,
            lengths,
            decoded_items=decoded_items,
            decoded_window_offset=start,
        )
        ct.store(out, (0,), result)
        ct.store(out_with_offsets, (0,), result_pair)
        ct.store(offsets, (0,), relative)
        ct.store(inp, (0,), values)
        ct.store(counts, (0,), lengths)

    large_offset = 1 << (31 if length_dtype == "uint32" else 40)
    for start in (0, 7, len(decoded) - 3, len(decoded), large_offset):
        window = torch.tensor([start], dtype=length_type, device="cuda")
        ct.launch(
            torch.cuda.current_stream(),
            (1,),
            kernel,
            (inputs, counts, window, out, out_with_offsets, offsets),
        )
        torch.cuda.synchronize()
        expected = (
            decoded[start : start + decoded_items] + [0] * decoded_items
        )[:decoded_items]
        expected_offsets = (
            relative[start : start + decoded_items] + [sentinel] * decoded_items
        )[:decoded_items]
        assert out.cpu().tolist() == expected
        assert out_with_offsets.cpu().tolist() == expected
        assert offsets.cpu().tolist() == expected_offsets

    assert inputs.cpu().tolist() == values
    assert counts.cpu().tolist() == lengths
    counts.zero_()
    window.zero_()
    ct.launch(
        torch.cuda.current_stream(),
        (1,),
        kernel,
        (inputs, counts, window, out, out_with_offsets, offsets),
    )
    torch.cuda.synchronize()
    assert out.cpu().tolist() == [0] * decoded_items
    assert out_with_offsets.cpu().tolist() == [0] * decoded_items
    assert offsets.cpu().tolist() == [sentinel] * decoded_items


@pytest.mark.parametrize("length_dtype", ["uint32", "uint64"])
def test_run_length_decode_runtime_scalar_offset(monkeypatch, length_dtype):
    configure_e2e_environment(monkeypatch)
    global ct, coop
    torch, ct, coop = import_e2e_modules()
    length_type = getattr(torch, length_dtype)
    sentinel = (1 << (32 if length_dtype == "uint32" else 64)) - 1
    values = list(range(128))
    lengths = [i % 3 + 1 if i < 96 else 0 for i in range(128)]
    decoded = [
        value for value, count in zip(values, lengths) for _ in range(count)
    ]
    relative = [offset for count in lengths for offset in range(count)]
    inputs = torch.tensor(values, dtype=torch.int32, device="cuda")
    counts = torch.tensor(lengths, dtype=length_type, device="cuda")
    output = torch.empty(64, dtype=torch.int32, device="cuda")
    paired = torch.empty_like(output)
    offsets = torch.empty(64, dtype=length_type, device="cuda")

    @ct.kernel(num_worker_warps=2)
    def kernel(inputs, counts, start, output, paired, offsets):
        values = ct.load(inputs, (0,), shape=(128,))
        lengths = ct.load(counts, (0,), shape=(128,))
        result = coop.run_length_decode(
            values, lengths, decoded_items=64, decoded_window_offset=start
        )
        result_pair, relative = coop.run_length_decode_with_offsets(
            values, lengths, decoded_items=64, decoded_window_offset=start
        )
        ct.store(output, (0,), result)
        ct.store(paired, (0,), result_pair)
        ct.store(offsets, (0,), relative)

    for start in (7, len(decoded) - 3, len(decoded) + 5):
        ct.launch(
            torch.cuda.current_stream(),
            (1,),
            kernel,
            (inputs, counts, start, output, paired, offsets),
        )
        torch.cuda.synchronize()
        expected = (decoded[start : start + 64] + [0] * 64)[:64]
        expected_offsets = (relative[start : start + 64] + [sentinel] * 64)[:64]
        assert output.cpu().tolist() == expected
        assert paired.cpu().tolist() == expected
        assert offsets.cpu().tolist() == expected_offsets


@pytest.mark.parametrize(
    "decoded_items,offset,length_dtype,message",
    [
        (0, 0, "uint32", "extents must be positive"),
        (16, 0, "uint32", "multiples of 32"),
        (96, 0, "uint32", "power of two"),
        (True, 0, "uint32", "decoded_items must be an int"),
        (256, -1, "uint32", "nonnegative"),
        (256, 1 << 32, "uint32", "representable"),
        (256, 0.5, "uint32", "must be an integer"),
        (256, 0, "int32", "length dtype"),
    ],
)
def test_run_length_decode_compile_diagnostics(
    monkeypatch, decoded_items, offset, length_dtype, message
):
    configure_e2e_environment(monkeypatch)
    global ct, coop
    torch, ct, coop = import_e2e_modules()
    from cuda.tile._exception import TileTypeError

    @ct.kernel
    def kernel(inp, counts, out):
        values = ct.load(inp, (0,), shape=(128,))
        lengths = ct.load(counts, (0,), shape=(128,))
        result = coop.run_length_decode(
            values,
            lengths,
            decoded_items=decoded_items,
            decoded_window_offset=offset,
        )
        ct.store(out, (0,), result)

    values = torch.ones(128, device="cuda", dtype=torch.int32)
    lengths = torch.ones(128, device="cuda", dtype=getattr(torch, length_dtype))
    out = torch.empty(256, device="cuda", dtype=torch.int32)
    with pytest.raises(TileTypeError, match=message):
        ct.launch(
            torch.cuda.current_stream(), (1,), kernel, (values, lengths, out)
        )
