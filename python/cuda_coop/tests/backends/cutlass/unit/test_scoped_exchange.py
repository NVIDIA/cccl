# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_warp_dispatch_maps_thread_data_for_non_native_provider():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        calls = []

        def increment(**payload):
            calls.append(payload)
            return payload["value"] + 10

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("exchange", increment)

        values = coop.ThreadData.from_values(1, 2, dtype=int)
        result = coop._warp.exchange_striped_to_blocked(
            values,
            threads_in_warp=16,
        )

        assert isinstance(result, coop.ThreadData)
        assert result.values("exchange") == (11, 12)
        assert [call["value"] for call in calls] == [1, 2]
        assert [call["mode"] for call in calls] == [
            "striped_to_blocked",
            "striped_to_blocked",
        ]
        assert [call["threads_in_warp"] for call in calls] == [16, 16]
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_exchange_uses_implementation_owned_storage_for_all_block_modes():
    script = textwrap.dedent(
        """
        import warnings

        import cuda.coop.cutlass as coop

        warnings.simplefilter("error")

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        capture._preserves_launch_metadata = True
        capture._uses_planned_temp_storage = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("exchange", capture)

        values = coop.ThreadData.from_values(1, 2, 3, dtype=int)
        ranks = coop.ThreadData.from_values(0, 1, 2, dtype=int)
        flags = coop.ThreadData.from_values(1, 1, 1, dtype=int)
        storage = coop._block.TempStorage(size_in_bytes=1)

        common = coop._block.exchange_striped_to_blocked(
            values,
            launch_metadata={"threads_per_block": 64},
            temp_storage=storage,
        )
        assert common["value"] is values
        assert common["mode"] == "striped_to_blocked"
        assert common["launch_metadata"] == {"threads_per_block": 64}

        warp_striped = coop._block.exchange_blocked_to_warp_striped(
            values,
            launch_metadata={"threads_per_block": 16},
            temp_storage=storage,
        )
        assert warp_striped["mode"] == "blocked_to_warp_striped"

        scatter = coop._block.exchange_scatter_to_blocked(
            values,
            ranks,
            launch_metadata={"threads_per_block": 32},
            temp_storage=storage,
        )
        assert scatter["ranks"] is ranks

        flagged = coop._block.exchange_scatter_to_striped_flagged(
            values,
            ranks,
            flags,
            launch_metadata={"threads_per_block": 32},
            temp_storage=storage,
        )
        assert flagged["valid_flags"] is flags
        assert storage.uses == ()
        assert storage.size_in_bytes == 1
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
