# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_topk_temp_storage_uses_public_primitive_names():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        class I64:
            width = 64

        seen = []

        def capture(**payload):
            seen.append(dict(payload))
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("topk_max_keys", capture)
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("topk_min_pairs", capture)
        keys = coop.ThreadData.from_values(1, 2, dtype=I32)
        values = coop.ThreadData.from_values(3, 4, dtype=I64)
        temp_storage = coop._block.TempStorage(size_in_bytes=16384)

        coop._block.topk_max_keys(
            keys,
            1,
            temp_storage=temp_storage,
            threads_per_block=64,
        )
        coop._block.topk_min_pairs(
            keys,
            values,
            1,
            temp_storage=temp_storage,
            dim=(8, 8, 1),
        )
        assert all("launch_metadata" not in payload for payload in seen)
        assert temp_storage.uses[0].primitive_name == "topk_max_keys"
        assert temp_storage.uses[0].required_size_in_bytes >= 64 * 2 * 4
        assert temp_storage.uses[0].required_alignment == 16
        assert temp_storage.uses[1].primitive_name == "topk_min_pairs"
        assert temp_storage.uses[1].required_size_in_bytes >= 64 * 2 * 8
        assert temp_storage.uses[1].required_alignment == 16
        assert temp_storage.slice_for_primitive("radix_sort_keys") is None
        assert temp_storage.slice_for_primitive("radix_sort_pairs") is None
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_topk_pair_temp_storage_accounts_for_key_and_value_payloads():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass._dsl._temp_storage import _topk_cub_temp_storage_requirement

        keys_size, keys_alignment = _topk_cub_temp_storage_requirement(
            block_threads=1024,
            items_per_thread=16,
            key_bytes=4,
        )
        pairs_size, pairs_alignment = _topk_cub_temp_storage_requirement(
            block_threads=1024,
            items_per_thread=16,
            key_bytes=4,
            value_bytes=4,
        )

        assert pairs_size > keys_size
        assert pairs_size >= 8 + 1024 * 16 * (4 + 4)
        assert keys_alignment == 16
        assert pairs_alignment == 16
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
