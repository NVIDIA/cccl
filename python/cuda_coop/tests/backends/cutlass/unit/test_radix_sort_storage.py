# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_sort_pair_temp_storage_accounts_for_key_and_value_widths():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        class I64:
            width = 64

        class U8:
            width = 8

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_sort_pairs", capture)
        keys = coop.ThreadData.from_values(1, 2, dtype=I32)
        values = coop.ThreadData.from_values(3, 4, dtype=I64)
        temp_storage = coop._block.TempStorage(size_in_bytes=1024)

        payload = coop._block.radix_sort_pairs(
            keys,
            values,
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 32},
        )

        assert payload["keys"] is keys
        assert payload["values"] is values
        assert temp_storage.uses[0].required_size_in_bytes == 32 * 2 * (4 + 8)
        assert temp_storage.required_alignment == 8

        values_u8 = coop.ThreadData.from_values(3, 4, dtype=U8)
        u8_temp_storage = coop._block.TempStorage(size_in_bytes=1024)
        coop._block.radix_sort_pairs(
            keys,
            values_u8,
            temp_storage=u8_temp_storage,
            launch_metadata={"threads_per_block": 32},
        )
        assert u8_temp_storage.uses[0].required_size_in_bytes == 32 * 2 * (4 + 4)
        assert u8_temp_storage.required_alignment == 4
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
