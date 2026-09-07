# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_warp_dispatch_uses_common_single_phase_context():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        calls = []

        def capture(**payload):
            calls.append(payload)
            return payload["keys"]

        capture._supports_native_thread_data = True
        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("merge_sort_keys", capture)

        keys = coop.ThreadData.from_values(3, 1, dtype=I32)
        result = coop._warp.merge_sort_keys(
            keys,
            threads_in_warp=16,
            thread_data=None,
        )

        assert result is keys
        assert len(calls) == 1
        assert calls[0]["keys"] is keys
        assert calls[0]["threads_in_warp"] == 16
        assert "thread_data" not in calls[0]
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_warp_dispatch_rejects_temp_storage_context():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("merge_sort_keys", capture)

        keys = coop.ThreadData.from_values(3, 1, dtype=I32)
        try:
            coop._warp.merge_sort_keys(
                keys,
                temp_storage=coop._warp.TempStorage(size_in_bytes=128),
            )
        except NotImplementedError as exc:
            assert "TempStorage planning" in str(exc)
        else:
            raise AssertionError("warp TempStorage dispatch should be rejected")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
