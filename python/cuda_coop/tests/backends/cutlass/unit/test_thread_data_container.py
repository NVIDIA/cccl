# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_thread_data_value_container():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class U64:
            width = 64

        data = coop.ThreadData.from_values(10, 20, dtype=U64)

        assert len(data) == 2
        assert data.items_per_thread == 2
        assert data.dtype is U64
        assert data[0] == 10
        assert data[1] == 20
        assert data.values("test") == (10, 20)

        data[1] = 30
        assert tuple(data) == (10, 30)

        try:
            tuple(coop.ThreadData(1))
        except ValueError as exc:
            assert "ThreadData iteration requires" in str(exc)
            assert "thread_data_iter" not in str(exc)
        else:
            raise AssertionError("uninitialized ThreadData iteration should fail")

        try:
            coop.ThreadData(0)
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError("zero-item ThreadData should be rejected")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
