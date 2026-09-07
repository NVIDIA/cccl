# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_histogram_uses_implementation_owned_storage_for_both_algorithms():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop
        from cuda.coop.cutlass._dsl.block import _histogram

        class U8:
            width = 8

        class I64:
            width = 64

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        capture._uses_planned_temp_storage = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("histogram", capture)
        assert _histogram._histogram_provider._uses_planned_temp_storage

        samples = coop.ThreadData.from_values(1, 2, 3, 4, dtype=U8)
        launch_metadata = {"threads_per_block": 64}

        atomic_temp_storage = coop._block.TempStorage(size_in_bytes=1)
        coop._block.histogram(
            samples,
            bins=64,
            bins_per_thread=1,
            counter_dtype=I64,
            algorithm="atomic",
            temp_storage=atomic_temp_storage,
            launch_metadata=launch_metadata,
        )
        assert atomic_temp_storage.uses == ()

        sort_temp_storage = coop._block.TempStorage(size_in_bytes=1)
        coop._block.histogram(
            samples,
            bins=64,
            bins_per_thread=1,
            counter_dtype=I64,
            algorithm="sort",
            temp_storage=sort_temp_storage,
            launch_metadata=launch_metadata,
        )
        assert sort_temp_storage.uses == ()
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
