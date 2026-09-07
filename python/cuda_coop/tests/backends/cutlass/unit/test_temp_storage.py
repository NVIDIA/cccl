# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support._subprocess import run_python_with_source as _run_python_with_source


def test_temp_storage_validates_explicit_capacity():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        class U8:
            width = 8

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)
        temp_storage = coop._block.TempStorage(size_in_bytes=1)

        try:
            coop._block.TempStorage(size_in_bytes=64, alignment=3)
        except ValueError as exc:
            assert "power of 2" in str(exc)
        else:
            raise AssertionError("non-power-of-two alignment should be rejected")

        try:
            coop._block.sum(
                I32(),
                temp_storage=temp_storage,
                launch_metadata={"threads_per_block": 32},
            )
        except ValueError as exc:
            assert "smaller than required" in str(exc)
        else:
            raise AssertionError("undersized TempStorage should be rejected")

        u8_temp_storage = coop._block.TempStorage(size_in_bytes=128)
        coop._block.sum(
            U8(),
            temp_storage=u8_temp_storage,
            launch_metadata={"threads_per_block": 32},
        )
        assert u8_temp_storage.uses[-1].required_size_in_bytes == 128
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_launch_metadata_required_when_temp_storage_omitted():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", capture)

        try:
            coop._block.sum(1)
        except ValueError as exc:
            assert "requires launch_metadata" in str(exc)
            assert "TempStorage is omitted" in str(exc)
        else:
            raise AssertionError("omitted TempStorage should require launch metadata")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_launch_metadata_multiplies_dimensional_block_fields():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)
        temp_storage = coop._block.TempStorage(size_in_bytes=512)

        coop._block.sum(
            I32(),
            temp_storage=temp_storage,
            launch_metadata={"block_dim_x": 16, "block_dim_y": 2, "block_dim_z": 1},
        )

        assert temp_storage.required_size_in_bytes == 32 * 4
        assert temp_storage.capacity_size_in_bytes == 512
        assert temp_storage.uses[0].required_size_in_bytes == 32 * 4
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_partial_dimensional_launch_metadata_is_unknown():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)
        temp_storage = coop._block.TempStorage(size_in_bytes=512)

        try:
            coop._block.sum(
                I32(),
                temp_storage=temp_storage,
                launch_metadata={"block_dim_x": 16, "block_dim_y": 2},
            )
        except ValueError as exc:
            assert "requires launch_metadata" in str(exc)
        else:
            raise AssertionError("partial block_dim_* metadata should be unknown")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_shared_temp_storage_infers_size_without_an_environment_toggle():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        storage = coop._block.TempStorage(sharing="shared")
        assert storage.is_deferred
        assert storage.capacity_size_in_bytes is None
        assert storage.alignment is None
        assert storage.auto_sync is True
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_temp_storage_requires_launch_metadata_for_sizing():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)

        try:
            coop._block.sum(I32(), temp_storage=coop._block.TempStorage(size_in_bytes=128))
        except ValueError as exc:
            assert "requires launch_metadata" in str(exc)
        else:
            raise AssertionError("TempStorage sizing should require launch metadata")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_shared_temp_storage_uses_the_max_explicit_capacity():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("exclusive_sum", lambda **payload: payload)
        temp_storage = coop._block.TempStorage(size_in_bytes=128, sharing="shared")

        coop._block.sum(
            I32(),
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 32},
        )
        coop._block.exclusive_sum(
            I32(),
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 32},
        )

        assert temp_storage.required_size_in_bytes == 128
        assert [use.byte_offset_in_bytes for use in temp_storage.uses] == [0, 0]
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_shared_temp_storage_strengthens_in_place_at_offset_zero():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        class I64:
            width = 64

        class V32:
            dtype = I32

        class V64:
            dtype = I64

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("exclusive_sum", lambda **payload: payload)
        temp_storage = coop._block.TempStorage(size_in_bytes=1024, sharing="shared")
        launch_metadata = {"threads_per_block": 32}

        coop._block.sum(V32(), temp_storage=temp_storage, launch_metadata=launch_metadata)
        coop._block.exclusive_sum(
            V32(),
            temp_storage=temp_storage,
            launch_metadata=launch_metadata,
        )

        coop._block.sum(V64(), temp_storage=temp_storage, launch_metadata=launch_metadata)

        assert temp_storage.required_size_in_bytes == 32 * 8
        assert [use.byte_offset_in_bytes for use in temp_storage.uses] == [0, 0, 0]
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_temp_storage_rejects_thread_data_without_width():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", capture)
        temp_storage = coop._block.TempStorage(size_in_bytes=128)
        thread_data = coop.ThreadData(2)

        try:
            coop._block.sum(thread_data, temp_storage=temp_storage)
        except ValueError as exc:
            assert "requires a dtype with positive width" in str(exc)
        else:
            raise AssertionError("ThreadData without dtype/values should fail sizing")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_temp_storage_rejects_unknown_value_width():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class UnknownWidth:
            pass

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)

        try:
            coop._block.sum(
                UnknownWidth(),
                temp_storage=coop._block.TempStorage(size_in_bytes=128),
                launch_metadata={"threads_per_block": 32},
            )
        except ValueError as exc:
            assert "item width could not be inferred" in str(exc)
        else:
            raise AssertionError("unknown temp-storage item width should fail")

        try:
            coop._block.sum(
                1,
                temp_storage=coop._block.TempStorage(size_in_bytes=128),
                launch_metadata={"threads_per_block": 32},
            )
        except ValueError as exc:
            assert "item width could not be inferred" in str(exc)
        else:
            raise AssertionError("raw Python scalar width should fail")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_temp_storage_planning_rolls_back_on_provider_failure():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        def fail(**payload):
            raise TypeError("provider rejected payload")

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", fail)
        temp_storage = coop._block.TempStorage(size_in_bytes=128)

        try:
            coop._block.sum(
                I32(),
                temp_storage=temp_storage,
                launch_metadata={"threads_per_block": 32},
            )
        except TypeError as exc:
            assert "provider rejected payload" in str(exc)
        else:
            raise AssertionError("failing provider should raise")

        assert temp_storage.uses == ()
        assert temp_storage.required_size_in_bytes == 0
        assert temp_storage.capacity_size_in_bytes == 128
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_value_dtype_width_overrides_context_thread_data_width():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        class I64:
            width = 64

        class DslValue:
            dtype = I64

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)
        thread_data = coop.ThreadData.from_values(1, dtype=I32)
        temp_storage = coop._block.TempStorage(size_in_bytes=256)

        coop._block.sum(
            DslValue(),
            thread_data=thread_data,
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 32},
        )

        assert temp_storage.uses[0].required_size_in_bytes == 32 * 8
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_context_thread_data_width_can_be_inferred_from_values():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I64:
            width = 64

        class V64:
            dtype = I64

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)
        thread_data = coop.ThreadData.from_values(V64(), V64())
        temp_storage = coop._block.TempStorage(size_in_bytes=1024)

        coop._block.sum(
            1,
            thread_data=thread_data,
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 32},
        )

        assert temp_storage.uses[0].required_size_in_bytes == 32 * 2 * 8
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_temp_storage_rejects_unknown_primitive_sizing():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop
        from cuda.coop.cutlass._dsl.block._dispatch import dispatch_primitive, register_primitive_impl

        register_primitive_impl("future", impl=lambda **payload: payload)

        try:
            dispatch_primitive(
                "future",
                kwargs={"temp_storage": coop._block.TempStorage(size_in_bytes=128)},
            )
        except NotImplementedError as exc:
            assert "TempStorage sizing is not known" in str(exc)
        else:
            raise AssertionError("unknown TempStorage primitive should fail sizing")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_temp_storage_accepts_explicit_manual_sync():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        temp_storage = coop.TempStorage(size_in_bytes=128, auto_sync=False)

        assert not temp_storage.is_deferred
        assert not temp_storage.auto_sync
        assert temp_storage.scope == "cuda.coop.cutlass"
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_exclusive_temp_storage_assigns_disjoint_call_slices():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", lambda **payload: payload)
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("exclusive_sum", lambda **payload: payload)
        temp_storage = coop.TempStorage(size_in_bytes=512, sharing="exclusive")
        assert not temp_storage.auto_sync

        coop._block.sum(
            I32(),
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 32},
        )
        coop._block.exclusive_sum(
            I32(),
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 32},
        )

        assert len(temp_storage.uses) == 2
        first, second = temp_storage.uses
        expected_second_offset = (
            (first.required_size_in_bytes + second.required_alignment - 1)
            // second.required_alignment
            * second.required_alignment
        )
        assert first.byte_offset_in_bytes == 0
        assert second.byte_offset_in_bytes == expected_second_offset
        assert temp_storage.required_size_in_bytes == (
            second.byte_offset_in_bytes + second.required_size_in_bytes
        )

        try:
            coop.TempStorage(
                size_in_bytes=512,
                sharing="exclusive",
                auto_sync=True,
            )
        except ValueError as exc:
            assert "sharing='exclusive'" in str(exc)
        else:
            raise AssertionError("exclusive TempStorage accepted auto_sync=True")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
