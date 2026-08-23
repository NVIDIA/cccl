# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_row_sum_requires_explicit_temp_storage():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("row_sum", capture)

        try:
            coop._block.row_sum(
                1,
                rows_per_block=1,
                warps_per_row=4,
                launch_metadata={"threads_per_block": 128},
            )
        except ValueError as exc:
            assert "requires TempStorage" in str(exc)
            assert "CUB row reduction uses shared memory" in str(exc)
        else:
            raise AssertionError("row_sum without TempStorage should fail")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_registered_row_sum_provider_rechecks_launch_geometry():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass._dsl.block._reduce import _row_sum_provider

        try:
            _row_sum_provider(
                value=object(),
                rows_per_block=1,
                warps_per_row=4,
                launch_metadata={"threads_per_block": 64},
            )
        except ValueError as exc:
            assert "launch block has 64 threads" in str(exc)
            assert "expected exactly 128" in str(exc)
        else:
            raise AssertionError("registered provider skipped its launch check")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_row_sum_temp_storage_is_row_shaped():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("row_sum", capture)

        temp_storage = coop._block.TempStorage(size_in_bytes=20)
        value = I32()
        payload = coop._block.row_sum(
            value,
            rows_per_block=1,
            warps_per_row=4,
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 128},
        )

        assert payload["value"] is value
        assert payload["args"] == ()
        assert payload["rows_per_block"] == 1
        assert payload["warps_per_row"] == 4
        assert temp_storage.required_size_in_bytes == 20
        assert temp_storage.required_alignment == 4
        assert len(temp_storage.uses) == 1
        assert temp_storage.uses[0].primitive_name == "row_sum"
        assert temp_storage.uses[0].required_size_in_bytes == 20

        mismatched_storage = coop._block.TempStorage(size_in_bytes=20)
        try:
            coop._block.row_sum(
                value,
                rows_per_block=1,
                warps_per_row=4,
                temp_storage=mismatched_storage,
                launch_metadata={"threads_per_block": 64},
            )
        except ValueError as exc:
            assert "launch block has 64 threads" in str(exc)
            assert "expected exactly 128" in str(exc)
        else:
            raise AssertionError("row_sum accepted a mismatched launch block")
        assert mismatched_storage.uses == ()

        undersized_storage = coop._block.TempStorage(size_in_bytes=16)
        try:
            coop._block.row_sum(
                value,
                rows_per_block=1,
                warps_per_row=4,
                temp_storage=undersized_storage,
                launch_metadata={"threads_per_block": 128},
            )
        except ValueError as exc:
            assert "TempStorage size_in_bytes is smaller" in str(exc)
            assert "16 < 20" in str(exc)
        else:
            raise AssertionError("row_sum accepted its legacy undersized scratch")
        assert undersized_storage.uses == ()

        missing_launch_storage = coop._block.TempStorage(size_in_bytes=20)
        try:
            coop._block.row_sum(
                value,
                rows_per_block=1,
                warps_per_row=4,
                temp_storage=missing_launch_storage,
            )
        except ValueError as exc:
            assert "requires launch metadata or kernel reqntid" in str(exc)
        else:
            raise AssertionError("row_sum dispatch skipped launch validation")
        assert missing_launch_storage.uses == ()
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_row_sum_temp_storage_uses_core_geometry_validation():
    script = textwrap.dedent(
        """
        from numbers import Integral

        from cuda.coop.cutlass._dsl._temp_storage import (
            infer_temp_storage_requirement,
        )

        class IntegralScalar:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return self.value

            def __lt__(self, other):
                return self.value < other

        Integral.register(IntegralScalar)

        class U8:
            width = 8

        class I32:
            width = 32

        class F64:
            width = 64

        class U16:
            width = 16

        assert infer_temp_storage_requirement(
            "row_sum",
            {
                "value": I32(),
                "rows_per_block": IntegralScalar(2),
                "warps_per_row": IntegralScalar(4),
                "launch_metadata": {"threads_per_block": 256},
            },
        ) == (40, 4)

        assert infer_temp_storage_requirement(
            "row_sum",
            {
                "value": U8(),
                "rows_per_block": 1,
                "warps_per_row": 4,
                "launch_metadata": {"threads_per_block": 128},
            },
        ) == (8, 4)

        assert infer_temp_storage_requirement(
            "row_sum",
            {
                "value": F64(),
                "rows_per_block": 1,
                "warps_per_row": 4,
                "launch_metadata": {"threads_per_block": 128},
            },
        ) == (40, 8)

        assert infer_temp_storage_requirement(
            "row_sum",
            {
                "value": F64(),
                "rows_per_block": 4,
                "warps_per_row": 1,
                "launch_metadata": {"threads_per_block": 128},
            },
        ) == (1, 8)

        try:
            infer_temp_storage_requirement(
                "row_sum",
                {
                    "value": U16(),
                    "rows_per_block": 1,
                    "warps_per_row": 4,
                    "launch_metadata": {"threads_per_block": 128},
                },
            )
        except ValueError as exc:
            assert "supports scalar widths of 1, 4, 8 bytes" in str(exc)
        else:
            raise AssertionError("row_sum accepted an unreviewed scalar layout")

        try:
            infer_temp_storage_requirement(
                "row_sum",
                {
                    "value": I32(),
                    "rows_per_block": 1,
                    "warps_per_row": 33,
                    "launch_metadata": {"threads_per_block": 1024},
                },
            )
        except ValueError as exc:
            assert "row_sum TempStorage sizing" in str(exc)
            assert "warps_per_row must be <= 32" in str(exc)
        else:
            raise AssertionError("TempStorage should share core row validation")

        try:
            infer_temp_storage_requirement(
                "row_sum",
                {
                    "value": I32(),
                    "rows_per_block": 2,
                    "warps_per_row": 17,
                    "launch_metadata": {"threads_per_block": 1024},
                },
            )
        except ValueError as exc:
            assert "must fit in one CUDA thread block" in str(exc)
        else:
            raise AssertionError("oversized row shape should fail during sizing")

        assert infer_temp_storage_requirement(
            "row_sum",
            {
                "value": I32(),
                "rows_per_block": 1,
                "warps_per_row": 4,
                "launch_metadata": {"threads_per_block": 64},
            },
        ) == (20, 4)

        assert infer_temp_storage_requirement(
            "row_sum",
            {
                "value": I32(),
                "rows_per_block": 1,
                "warps_per_row": 4,
            },
        ) == (20, 4)
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
