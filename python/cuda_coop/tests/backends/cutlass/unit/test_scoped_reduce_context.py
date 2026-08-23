# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_reduce_temp_storage_planning_uses_context_thread_data():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class U64:
            width = 64

        def capture(**payload):
            return payload

        thread_data = coop.ThreadData.from_values(1, 2, dtype=U64)
        temp_storage = coop._block.TempStorage(size_in_bytes=1024, sharing="shared")
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", capture)

        payload = coop._block.sum(
            1,
            thread_data=thread_data,
            temp_storage=temp_storage,
            launch_metadata={"block": (16, 3, 1)},
        )

        assert payload == {
            "value": 1,
            "args": (),
            "launch_metadata": {"block": (16, 3, 1)},
        }
        assert temp_storage.required_size_in_bytes == 64 * 2 * 8
        assert temp_storage.capacity_size_in_bytes == 1024
        assert temp_storage.required_alignment == 8
        assert len(temp_storage.uses) == 1
        use = temp_storage.uses[0]
        assert use.primitive_name == "sum"
        assert use.required_size_in_bytes == 64 * 2 * 8
        assert use.required_alignment == 8
        assert use.byte_offset_in_bytes == 0
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_reduce_provider_omits_explicit_none_context_fields():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", capture)

        payload = coop._block.sum(
            1,
            thread_data=None,
            temp_storage=None,
            launch_metadata={"threads_per_block": 32},
        )

        assert payload == {"value": 1, "args": ()}
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_launch_preserving_reduce_omits_explicit_none_context_fields():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        capture._preserves_launch_metadata = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", capture)

        payload = coop._block.sum(
            1,
            thread_data=None,
            temp_storage=None,
            launch_metadata={"threads_per_block": 32},
        )

        assert payload == {
            "value": 1,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
