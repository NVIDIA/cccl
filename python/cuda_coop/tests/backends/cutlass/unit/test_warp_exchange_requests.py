# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_warp_exchange_requests_follow_core_semantics():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._dsl.warp import _provider as provider

    exchange = provider._warp_exchange_request(
        mode="scatter_to_striped",
        value_type=Int32,
        rank_type=Int32,
        threads_in_warp=8,
        block_threads=64,
        items_per_thread=2,
    )
    assert exchange.kind == "warp_thread_data_exchange"
    assert exchange.mode == "scatter_to_striped"
    assert exchange.value_type is Int32
    assert exchange.rank_type is Int32
    assert exchange.logical_warp_threads == 8
    assert exchange.block_threads == 64
    assert exchange.items_per_thread == 2
    assert exchange.symbol_name.endswith("_x2_w8_b64")

    blocked = provider._warp_exchange_request(
        mode="striped_to_blocked",
        value_type=Int32,
        rank_type=None,
        threads_in_warp=16,
        block_threads=128,
        items_per_thread=4,
    )
    assert blocked.mode == "striped_to_blocked"
    assert blocked.rank_type is None
    assert blocked.logical_warp_threads == 16
    assert blocked.block_threads == 128
    assert blocked.items_per_thread == 4
    assert blocked.symbol_name.endswith("_x4_w16_b128")


@pytest.mark.parametrize("block_threads", [0, -1, 40])
def test_warp_exchange_requests_reject_invalid_block_threads(block_threads):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._dsl.warp import _provider as provider

    with pytest.raises(ValueError, match="block_threads|multiple"):
        provider._warp_exchange_request(
            mode="blocked_to_striped",
            value_type=Int32,
            rank_type=None,
            threads_in_warp=16,
            block_threads=block_threads,
            items_per_thread=2,
        )


def test_warp_exchange_source_uses_exact_logical_warp_count():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._dsl.warp import _provider as provider

    request = provider._warp_exchange_request(
        mode="scatter_to_striped",
        value_type=Int32,
        rank_type=Int32,
        threads_in_warp=8,
        block_threads=64,
        items_per_thread=2,
    )
    source = "\n".join(provider._render_cub_warp_exchange(request))

    assert "temp_storage[\n      8];" in source
    assert "1024" not in source


def test_warp_exchange_frontend_carries_exact_block_threads(monkeypatch):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")

    from cuda.coop.cutlass._dsl.warp import _exchange
    from cuda.coop.cutlass._dsl.warp import _provider as provider

    captured = {}

    def capture_provider_exchange(**kwargs):
        captured.update(kwargs)
        return "exchange-result"

    monkeypatch.setattr(
        _exchange, "_infer_block_dim", lambda *_args, **_kwargs: (8, 4, 2)
    )
    monkeypatch.setattr(provider, "provider_exchange", capture_provider_exchange)

    assert (
        _exchange._exchange_provider(
            value=object(),
            mode="scatter_to_striped",
            ranks=object(),
            threads_in_warp=8,
        )
        == "exchange-result"
    )
    assert captured["block_threads"] == 64
