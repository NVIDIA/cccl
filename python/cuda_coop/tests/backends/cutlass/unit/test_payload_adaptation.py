# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support.source import run_python_with_source


def test_cutlass_store_auto_adapts_direct_thread_payloads():
    script = textwrap.dedent(
        """
        import importlib
        import sys

        cutlass_module = sys.modules["cutlass"]
        cute_module = sys.modules["cutlass.cute"]

        class Arch:
            @staticmethod
            def thread_idx():
                return 0, 0, 0

            @staticmethod
            def block_dim():
                return 4, 1, 1

        cute_module.arch = Arch
        cutlass_module.cute = cute_module

        class FakeVector:
            shape = (2,)
            dtype = int

            def __init__(self, base):
                self.base = base

            def __getitem__(self, idx):
                return self.base + idx

        class UntypedVector:
            shape = (2,)

            def __init__(self, base):
                self.base = base

            def __getitem__(self, idx):
                return self.base + idx

        class FakeDslUint32:
            __module__ = "cutlass.base_dsl.typing"

            def __init__(self, value):
                self.value = value

        class MemoryProtocolTensor:
            __cuda_array_interface__ = {
                "shape": (2,),
                "typestr": "<i4",
                "data": (0, False),
                "version": 3,
            }
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        def exercise(module_name, expected_block_scope, expected_warp_scope, forbidden):
            coop = importlib.import_module(module_name)

            block_out = [None] * 8
            coop._block.store(block_out, FakeVector(10))
            assert block_out[:2] == [10, 11]
            assert block_out[2:] == [None] * 6

            warp_out = [None] * 8
            coop._warp.store(warp_out, FakeVector(20), threads_in_warp=4)
            assert warp_out[:2] == [20, 21]
            assert warp_out[2:] == [None] * 6

            typed_block_out = [None] * 8
            coop._block.store(typed_block_out, UntypedVector(40), dtype=FakeDslUint32)
            assert [
                value.value for value in typed_block_out[:2]
            ] == [40, 41]
            assert typed_block_out[2:] == [None] * 6

            typed_warp_out = [None] * 8
            coop._warp.store(
                typed_warp_out,
                UntypedVector(50),
                threads_in_warp=4,
                dtype=FakeDslUint32,
            )
            assert [value.value for value in typed_warp_out[:2]] == [50, 51]
            assert typed_warp_out[2:] == [None] * 6

            for store, kwargs in (
                (coop._block.store, {}),
                (coop._warp.store, {"threads_in_warp": 4}),
            ):
                try:
                    store([None] * 8, FakeVector(30), dtype=str, **kwargs)
                except TypeError as exc:
                    assert "dtype does not match value.dtype" in str(exc)
                else:
                    raise AssertionError("mismatched direct payload dtype was accepted")

            for store, expected_scope in (
                (coop._block.store, expected_block_scope),
                (coop._warp.store, expected_warp_scope),
            ):
                try:
                    store([None] * 8, MemoryProtocolTensor())
                except TypeError as exc:
                    message = str(exc)
                    assert (
                        f"{expected_scope}.store could not auto-convert "
                        "'value' payload to ThreadData"
                    ) in message
                    assert "per-thread register payload" in message
                    for scope in forbidden:
                        assert scope not in message
                else:
                    raise AssertionError("memory-backed store payload was accepted")

        exercise(
            "cuda.coop.cutlass",
            "cuda.coop.cutlass._block",
            "cuda.coop.cutlass._warp",
            ("cuda.coop.cutlass._dsl",),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_root_load_store_dispatches_existing_cutlass_arrays():
    script = textwrap.dedent(
        """
        import importlib
        import sys
        import types

        import cuda.coop.cutlass as root

        calls = []

        class FakeDtype:
            bytes = 4

        class OtherDtype:
            bytes = 4

        class FakeVector:
            shape = (2,)
            dtype = FakeDtype

            def __init__(self, prefix="v", dtype=FakeDtype):
                self.prefix = prefix
                self.dtype = dtype

            def __getitem__(self, idx):
                return f"{self.prefix}{idx}"

        class PublicArray:
            pass

        class FakeArray(PublicArray):
            dtype = FakeDtype

            def __init__(self, source="array"):
                self.source = source

            def load(self, base, vector_size=None, **kwargs):
                calls.append(("load", self.source, base, vector_size, kwargs))
                return {
                    "source": self.source,
                    "base": base,
                    "vector_size": vector_size,
                    "kwargs": kwargs,
                }

            def store(self, value, idx=0, vector_size=None, **kwargs):
                calls.append(("store", self.source, value, idx, vector_size, kwargs))

        FakeArray.__module__ = "cutlass.base_dsl.array"
        FakeArray.__qualname__ = "Array"

        class StoreOnlyArray(PublicArray):
            dtype = FakeDtype

            def __init__(self, source="store-only"):
                self.source = source

            def store(self, value, idx=0, vector_size=None, **kwargs):
                calls.append(
                    ("store-only", self.source, value, idx, vector_size, kwargs)
                )

        StoreOnlyArray.__module__ = "cutlass.base_dsl.array"
        StoreOnlyArray.__qualname__ = "Array"

        class LoadOnlyArray(PublicArray):
            dtype = FakeDtype

            def __init__(self, source="load-only"):
                self.source = source

            def load(self, base, vector_size=None, **kwargs):
                calls.append(("load-only", self.source, base, vector_size, kwargs))
                return {
                    "source": self.source,
                    "base": base,
                    "vector_size": vector_size,
                    "kwargs": kwargs,
                }

        LoadOnlyArray.__module__ = "cutlass.base_dsl.array"
        LoadOnlyArray.__qualname__ = "Array"

        thread_state = {"idx": (7, 0, 0), "block_dim": (32, 1, 1)}

        def thread_idx():
            calls.append(("thread_idx",))
            return thread_state["idx"]

        def block_dim():
            calls.append(("block_dim",))
            return thread_state["block_dim"]

        def make_array_view(source, **kwargs):
            raise AssertionError("existing cutlass.Array values should not be wrapped")

        sys.modules["cutlass"] = types.SimpleNamespace(
            Array=PublicArray,
            make_array_view=make_array_view,
            cute=types.SimpleNamespace(
                arch=types.SimpleNamespace(
                    thread_idx=thread_idx,
                    block_dim=block_dim,
                )
            ),
        )

        def assert_raises(expected, fn):
            try:
                fn()
            except (TypeError, ValueError) as exc:
                message = str(exc)
                assert expected in message, message
            else:
                raise AssertionError("expected TypeError")

        loaded = root._block.load(
            FakeArray("block-in"),
            items_per_thread=2,
            offset=5,
            dtype=FakeDtype,
            is_nontemporal=True,
        )
        assert loaded["source"] == "block-in"
        assert loaded["base"] == 19
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 4
        assert loaded["kwargs"]["is_nontemporal"] is True

        calls.clear()
        loaded = root._block.load(
            FakeArray("block-inferred"),
            items_per_thread=2,
            offset=5,
        )
        assert loaded["source"] == "block-inferred"
        assert loaded["base"] == 19
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 4

        calls.clear()
        block_load = root._block.make_load(
            FakeDtype,
            threads_per_block=32,
            items_per_thread=2,
            offset=5,
        )
        assert block_load.scope == "cuda.coop.cutlass._block"
        assert block_load.primitive is root._block.load
        loaded = block_load(FakeArray("factory-block-in"))
        assert loaded["source"] == "factory-block-in"
        assert loaded["base"] == 19
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 4

        calls.clear()
        inferred_block_load = root._block.make_load(
            threads_per_block=32,
            items_per_thread=2,
            offset=5,
        )
        assert inferred_block_load.scope == "cuda.coop.cutlass._block"
        assert inferred_block_load.primitive is root._block.load
        loaded = inferred_block_load(FakeArray("factory-block-inferred"))
        assert loaded["source"] == "factory-block-inferred"
        assert loaded["base"] == 19
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 4

        calls.clear()
        root._block.store(
            FakeArray("block-out"),
            root.ThreadData.from_values(1, 2, dtype=FakeDtype),
            offset=5,
            dtype=FakeDtype,
        )
        assert calls[-1] == (
            "store",
            "block-out",
            (1, 2),
            19,
            2,
            {
                "alignment": 4,
                "is_volatile": False,
                "is_nontemporal": False,
                "ordering": "not_atomic",
                "syncscope": None,
                "loc": None,
                "ip": None,
            },
        )

        calls.clear()
        block_store = root._block.make_store(
            FakeDtype,
            threads_per_block=32,
            items_per_thread=2,
            offset=5,
        )
        assert block_store.scope == "cuda.coop.cutlass._block"
        assert block_store.primitive is root._block.store
        block_store(FakeArray("factory-block-out"), FakeVector("b"))
        assert calls[-1][0:5] == (
            "store",
            "factory-block-out",
            ("b0", "b1"),
            19,
            2,
        )

        calls.clear()
        inferred_block_store = root._block.make_store(
            threads_per_block=32,
            items_per_thread=2,
            offset=5,
        )
        assert inferred_block_store.scope == "cuda.coop.cutlass._block"
        assert inferred_block_store.primitive is root._block.store
        inferred_block_store(FakeArray("factory-block-out-inferred"), FakeVector("i"))
        assert calls[-1][0:5] == (
            "store",
            "factory-block-out-inferred",
            ("i0", "i1"),
            19,
            2,
        )

        calls.clear()
        call_dtype_block_load = root._block.make_load(
            threads_per_block=32,
            items_per_thread=2,
            offset=5,
        )
        loaded = call_dtype_block_load(
            FakeArray("factory-block-call-dtype"),
            dtype=FakeDtype,
        )
        assert loaded["source"] == "factory-block-call-dtype"
        assert loaded["base"] == 19
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 4

        calls.clear()
        call_dtype_block_store = root._block.make_store(
            threads_per_block=32,
            items_per_thread=2,
            offset=5,
        )
        call_dtype_block_store(
            FakeArray("factory-block-out-call-dtype"),
            FakeVector("c"),
            dtype=FakeDtype,
        )
        assert calls[-1][0:5] == (
            "store",
            "factory-block-out-call-dtype",
            ("c0", "c1"),
            19,
            2,
        )

        calls.clear()
        root._block.store(
            StoreOnlyArray("block-store-only"),
            root.ThreadData.from_values(3, 4, dtype=FakeDtype),
            offset=5,
        )
        assert calls[-1][0:5] == (
            "store-only",
            "block-store-only",
            (3, 4),
            19,
            2,
        )

        calls.clear()
        block_store_only_factory = root._block.make_store(
            items_per_thread=2,
            offset=5,
        )
        block_store_only_factory(StoreOnlyArray("factory-store-only"), FakeVector("s"))
        assert calls[-1][0:5] == (
            "store-only",
            "factory-store-only",
            ("s0", "s1"),
            19,
            2,
        )

        assert_raises(
            "cuda.coop.cutlass._block.load cutlass.Array operand must support load",
            lambda: root._block.load(
                StoreOnlyArray("bad-load"),
                items_per_thread=2,
            ),
        )
        bad_load_factory = root._block.make_load(
            items_per_thread=2,
        )
        assert_raises(
            "cuda.coop.cutlass._block.load cutlass.Array operand must support load",
            lambda: bad_load_factory(StoreOnlyArray("bad-factory-load")),
        )
        assert_raises(
            "cuda.coop.cutlass._block.store cutlass.Array operand must support store",
            lambda: root._block.store(
                LoadOnlyArray("bad-store"),
                FakeVector("bad-store"),
            ),
        )
        bad_store_factory = root._block.make_store(
            items_per_thread=2,
        )
        assert_raises(
            "cuda.coop.cutlass._block.store cutlass.Array operand must support store",
            lambda: bad_store_factory(
                LoadOnlyArray("bad-factory-store"),
                FakeVector("bad-factory-store"),
            ),
        )

        calls.clear()
        thread_state["idx"] = (23, 0, 0)
        loaded = root._warp.load(
            FakeArray("warp-in"),
            items_per_thread=2,
            offset=5,
            threads_in_warp=16,
        )
        assert loaded["source"] == "warp-in"
        assert loaded["base"] == 51
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 4

        calls.clear()
        root._warp.store(
            FakeArray("warp-out"),
            FakeVector("w"),
            offset=5,
            dtype=FakeDtype,
            threads_in_warp=16,
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "warp-out", "w0", 51, None),
            ("store", "warp-out", "w1", 52, None),
        ]

        calls.clear()
        inferred_warp_load = root._warp.make_load(
            items_per_thread=2,
            offset=5,
            threads_in_warp=16,
        )
        assert inferred_warp_load.scope == "cuda.coop.cutlass._warp"
        assert inferred_warp_load.primitive is root._warp.load
        loaded = inferred_warp_load(FakeArray("factory-warp-inferred"))
        assert loaded["source"] == "factory-warp-inferred"
        assert loaded["base"] == 51
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 4

        calls.clear()
        inferred_warp_store = root._warp.make_store(
            items_per_thread=2,
            offset=5,
            threads_in_warp=16,
        )
        assert inferred_warp_store.scope == "cuda.coop.cutlass._warp"
        assert inferred_warp_store.primitive is root._warp.store
        inferred_warp_store(FakeArray("factory-warp-out-inferred"), FakeVector("z"))
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "factory-warp-out-inferred", "z0", 51, None),
            ("store", "factory-warp-out-inferred", "z1", 52, None),
        ]

        assert_raises(
            "cuda.coop.cutlass._block.load dtype= does not match cutlass.Array dtype",
            lambda: root._block.load(
                FakeArray("bad-dtype"),
                items_per_thread=2,
                dtype=OtherDtype,
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.store value dtype does not match "
                "cutlass.Array dtype"
            ),
            lambda: root._block.store(
                FakeArray("bad-store-dtype"),
                FakeVector("bad", dtype=OtherDtype),
            ),
        )

        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_root_auto_adapts_native_thread_data_payload_names():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class I32:
            width = 32

        class U32:
            width = 32
            signed = False

        class FakeVector:
            shape = (2,)
            dtype = I32

            def __init__(self, base):
                self.base = base

            def __getitem__(self, idx):
                return self.base + idx

        def capture_native(**payload):
            return payload

        capture_native._supports_native_thread_data = True

        def assert_thread_data(value, expected):
            assert isinstance(value, coop.ThreadData)
            assert value.dtype is I32
            assert value.values("payload") == expected

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("histogram", capture_native)
        histogram_payload = coop._block.histogram(
            FakeVector(10),
            bins=16,
            bins_per_thread=2,
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(histogram_payload["samples"], (10, 11))
        assert histogram_payload["bins"] == 16
        assert histogram_payload["bins_per_thread"] == 2

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("run_length_decode", capture_native)
        run_length_payload = coop._block.run_length_decode(
            FakeVector(20),
            FakeVector(30),
            decoded_items_per_thread=2,
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(run_length_payload["run_values"], (20, 21))
        assert_thread_data(run_length_payload["run_lengths"], (30, 31))

        relative_offsets = coop.ThreadData(2, dtype=U32)
        total_decoded_size = coop.ThreadData(1, dtype=U32)
        unsigned_lengths_payload = coop._block.run_length_decode(
            FakeVector(120),
            FakeVector(130),
            decoded_items_per_thread=2,
            relative_offsets=relative_offsets,
            total_decoded_size=total_decoded_size,
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(unsigned_lengths_payload["run_values"], (120, 121))
        assert isinstance(unsigned_lengths_payload["run_lengths"], coop.ThreadData)
        assert unsigned_lengths_payload["run_lengths"].dtype is U32
        assert unsigned_lengths_payload["run_lengths"].values("payload") == (130, 131)
        assert unsigned_lengths_payload["relative_offsets"] is relative_offsets
        assert unsigned_lengths_payload["total_decoded_size"] is total_decoded_size

        explicit_dtype_payload = coop._block.run_length_decode(
            FakeVector(180),
            FakeVector(190),
            decoded_items_per_thread=2,
            decoded_offset_dtype=U32,
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(explicit_dtype_payload["run_values"], (180, 181))
        assert isinstance(explicit_dtype_payload["run_lengths"], coop.ThreadData)
        assert explicit_dtype_payload["run_lengths"].dtype is U32
        assert explicit_dtype_payload["run_lengths"].values("payload") == (190, 191)

        run_length_parent = coop._block.run_length(
            FakeVector(140),
            FakeVector(150),
            runs_per_thread=2,
            decoded_items_per_thread=2,
            launch_metadata={"threads_per_block": 32},
        )
        run_length_parent_payload = run_length_parent.decode()
        assert_thread_data(run_length_parent_payload["run_values"], (140, 141))
        assert_thread_data(run_length_parent_payload["run_lengths"], (150, 151))

        parent_relative_offsets = coop.ThreadData(2, dtype=U32)
        run_length_parent = coop._block.run_length(
            FakeVector(160),
            FakeVector(170),
            runs_per_thread=2,
            decoded_items_per_thread=2,
            total_decoded_size=total_decoded_size,
            launch_metadata={"threads_per_block": 32},
        )
        unsigned_parent_payload = run_length_parent.decode(
            relative_offsets=parent_relative_offsets,
        )
        assert_thread_data(unsigned_parent_payload["run_values"], (160, 161))
        assert isinstance(unsigned_parent_payload["run_lengths"], coop.ThreadData)
        assert unsigned_parent_payload["run_lengths"].dtype is U32
        assert unsigned_parent_payload["run_lengths"].values("payload") == (170, 171)
        assert unsigned_parent_payload["relative_offsets"] is parent_relative_offsets
        assert unsigned_parent_payload["total_decoded_size"] is total_decoded_size

        explicit_dtype_parent = coop._block.run_length(
            FakeVector(200),
            FakeVector(210),
            runs_per_thread=2,
            decoded_items_per_thread=2,
            decoded_offset_dtype=U32,
            launch_metadata={"threads_per_block": 32},
        )
        explicit_dtype_parent_payload = explicit_dtype_parent.decode()
        assert_thread_data(explicit_dtype_parent_payload["run_values"], (200, 201))
        assert isinstance(
            explicit_dtype_parent_payload["run_lengths"],
            coop.ThreadData,
        )
        assert explicit_dtype_parent_payload["run_lengths"].dtype is U32
        assert explicit_dtype_parent_payload["run_lengths"].values("payload") == (
            210,
            211,
        )

        capture_native._uses_planned_temp_storage = True
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("exchange", capture_native)
        exchange_payload = coop._block.exchange_scatter_to_striped_flagged(
            FakeVector(40),
            FakeVector(50),
            FakeVector(60),
            launch_metadata={"threads_per_block": 32},
            temp_storage=coop._block.TempStorage(size_in_bytes=4096),
        )
        assert_thread_data(exchange_payload["value"], (40, 41))
        assert_thread_data(exchange_payload["ranks"], (50, 51))
        assert_thread_data(exchange_payload["valid_flags"], (60, 61))
        assert exchange_payload["mode"] == "scatter_to_striped_flagged"

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("exchange", capture_native)
        warp_exchange_payload = coop._warp.exchange_scatter_to_striped(
            FakeVector(70),
            FakeVector(80),
            threads_in_warp=16,
        )
        assert_thread_data(warp_exchange_payload["value"], (70, 71))
        assert_thread_data(warp_exchange_payload["ranks"], (80, 81))
        assert warp_exchange_payload["threads_in_warp"] == 16
        assert warp_exchange_payload["mode"] == "scatter_to_striped"

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_rank", capture_native)
        prefix = coop.ThreadData(1, dtype=I32)
        radix_rank_payload = coop._block.radix_rank(
            FakeVector(90),
            begin_bit=0,
            end_bit=4,
            exclusive_digit_prefix=prefix,
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(radix_rank_payload["keys"], (90, 91))
        assert radix_rank_payload["exclusive_digit_prefix"] is prefix

        prefix_vector = FakeVector(100)
        radix_rank_output_payload = coop._block.radix_rank(
            FakeVector(110),
            begin_bit=0,
            end_bit=4,
            exclusive_digit_prefix=prefix_vector,
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(radix_rank_output_payload["keys"], (110, 111))
        assert radix_rank_output_payload["exclusive_digit_prefix"] is prefix_vector

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("merge_sort_pairs", capture_native)
        merge_sort_payload = coop._block.merge_sort_pairs(
            FakeVector(120),
            FakeVector(130),
            valid_items=2,
            oob_default=999,
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(merge_sort_payload["keys"], (120, 121))
        assert_thread_data(merge_sort_payload["values"], (130, 131))
        assert merge_sort_payload["valid_items"] == 2
        assert merge_sort_payload["oob_default"] == 999
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_auto_adapt_payload_slots_are_intentional():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass._dsl import _single_phase

        assert _single_phase.THREAD_PAYLOAD_ARG_NAMES == (
            "value",
            "key",
            "keys",
            "values",
            "samples",
            "run_values",
            "run_lengths",
            "ranks",
            "valid_flags",
        )
        assert _single_phase.THREAD_DATA_MAPPING_ARG_NAMES == (
            "value",
            "key",
            "keys",
            "values",
        )
        assert set(_single_phase.THREAD_DATA_MAPPING_ARG_NAMES) < set(
            _single_phase.THREAD_PAYLOAD_ARG_NAMES
        )
        assert "exclusive_digit_prefix" not in _single_phase.THREAD_PAYLOAD_ARG_NAMES
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr
