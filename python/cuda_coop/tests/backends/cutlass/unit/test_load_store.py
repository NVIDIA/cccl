# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support.source import run_python_with_source


def test_cutlass_payload_selector_normalizes_prims():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop
        from cuda.coop.cutlass._payload import normalize_payload_selector

        def normalize(value):
            return normalize_payload_selector(
                value,
                scope="cuda.coop.cutlass._block",
                primitive_name="load",
                allowed=(coop.Payload.PRIMS,),
                choices_text="prims",
            )

        assert normalize(None) is None
        assert normalize("prims") is coop.Payload.PRIMS
        assert normalize(coop.Payload.PRIMS) is coop.Payload.PRIMS

        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_root_dispatch_defaults_to_tensor_route():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass import _root_load_store

        def tensor_load(source, *args, **kwargs):
            return source, args, kwargs

        assert _root_load_store.dispatch_load(
            "source",
            3,
            scope="cuda.coop.cutlass._block",
            tensor_load=tensor_load,
            marker=True,
        ) == ("source", (3,), {"marker": True})
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_scoped_offset_alone_stays_on_tensor_route():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass import _root_load_store

        calls = []

        def tensor_load(source, *args, **kwargs):
            calls.append(("load", source, args, kwargs))
            return "loaded"

        def tensor_store(destination, value, *args, **kwargs):
            calls.append(("store", destination, value, args, kwargs))

        assert _root_load_store._supplied_prims_load_store_controls(
            {"offset": 7}
        ) == ()
        assert _root_load_store._supplied_prims_load_store_controls(
            {"bounds_check": False, "offset": 7}
        ) == ("bounds_check",)
        assert _root_load_store._supplied_prims_load_store_controls(
            {"is_nontemporal": True}
        ) == ("is_nontemporal",)

        assert _root_load_store.dispatch_load(
            "source",
            scope="cuda.coop.cutlass._block",
            tensor_load=tensor_load,
            offset=7,
        ) == "loaded"
        _root_load_store.dispatch_store(
            "destination",
            "value",
            scope="cuda.coop.cutlass._warp",
            tensor_store=tensor_store,
            offset=9,
        )
        assert calls == [
            ("load", "source", (), {"offset": 7}),
            ("store", "destination", "value", (), {"offset": 9}),
        ]
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_dtype_route_avoids_prims():
    script = textwrap.dedent(
        """
        import sys

        from cuda.coop.cutlass import _root_load_store

        class CutlassDtype:
            bytes = 4

        CutlassDtype.__module__ = "cutlass.base_dsl.typing"

        class Value:
            dtype = CutlassDtype

        calls = []

        def tensor_load(source, *args, **kwargs):
            calls.append(("load", source, args, kwargs))
            return "tensor-load"

        def tensor_store(destination, value, *args, **kwargs):
            calls.append(("store", destination, value, args, kwargs))

        assert "cuda.coop.cutlass._prims_adapter" not in sys.modules
        assert _root_load_store._supplied_prims_load_store_controls(
            {"dtype": CutlassDtype}
        ) == ()
        assert _root_load_store.dispatch_load(
            "source",
            scope="cuda.coop.cutlass._block",
            tensor_load=tensor_load,
            dtype=CutlassDtype,
        ) == "tensor-load"
        value = Value()
        _root_load_store.dispatch_store(
            "destination",
            value,
            scope="cuda.coop.cutlass._block",
            tensor_store=tensor_store,
        )
        assert calls == [
            ("load", "source", (), {"dtype": CutlassDtype}),
            ("store", "destination", value, (), {}),
        ]
        assert "cuda.coop.cutlass._prims_adapter" not in sys.modules
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_root_accepts_public_cutlass_array_type():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop
        from cuda.coop.cutlass import _prims

        calls = []

        class FakeDtype:
            bytes = 4

        class FakeArray:
            dtype = FakeDtype

            def __init__(self, label="array"):
                self.label = label

            def load(self, base, vector_size=None, **kwargs):
                calls.append(("load", self.label, base, vector_size, kwargs))
                return {
                    "label": self.label,
                    "base": base,
                    "vector_size": vector_size,
                    "kwargs": kwargs,
                }

            def store(self, value, idx=0, vector_size=None, **kwargs):
                calls.append(("store", self.label, value, idx, vector_size, kwargs))

        FakeArray.__module__ = "cutlass.base_dsl.array"
        FakeArray.__qualname__ = "Array"

        def thread_idx():
            calls.append(("thread_idx",))
            return 3, 0, 0

        def block_dim():
            calls.append(("block_dim",))
            return 32, 1, 1

        arch = types.SimpleNamespace(thread_idx=thread_idx, block_dim=block_dim)
        cute = types.ModuleType("cutlass.cute")
        cute.arch = arch
        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass.Array = FakeArray
        cutlass.cute = cute
        sys.modules["cutlass"] = cutlass
        sys.modules["cutlass.cute"] = cute

        class Array:
            def load(self, *args, **kwargs):
                return args, kwargs

        Array.__module__ = "cutlass.base_dsl.array"

        assert not _prims.is_cutlass_array_operand(Array())

        loaded = coop._block.load(FakeArray("in"), items_per_thread=2)
        assert loaded["label"] == "in"
        assert loaded["base"] == 6
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 8

        coop._block.store(
            FakeArray("out"),
            coop.ThreadData.from_values(10, 11, dtype=FakeDtype),
        )
        assert calls[-1] == (
            "store",
            "out",
            (10, 11),
            6,
            2,
            {
                "alignment": 8,
                "is_volatile": False,
                "is_nontemporal": False,
                "ordering": "not_atomic",
                "syncscope": None,
                "loc": None,
                "ip": None,
            },
        )

        try:
            coop.ThreadData.from_payload(FakeArray("payload"))
        except TypeError as exc:
            assert "a scoped load for cutlass.Array values" in str(exc)
        else:
            raise AssertionError("memory-backed cutlass.Array was accepted")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_prims_movement_uses_cutlass_cute_arch():
    script = textwrap.dedent(
        """
        import sys
        import types

        calls = []
        arch = types.SimpleNamespace(
            thread_idx=lambda: calls.append("thread_idx") or (3, 2, 1),
            block_dim=lambda: calls.append("block_dim") or (8, 4, 2),
        )
        cutlass = sys.modules["cutlass"]
        cute = sys.modules["cutlass.cute"]
        cute.arch = arch
        cutlass.cute = cute
        from cuda.coop.cutlass import _prims_adapter

        linear_tid, block_threads = _prims_adapter._linear_thread_and_block_threads()
        assert linear_tid == 51
        assert block_threads == 64
        assert calls == ["thread_idx", "block_dim"]
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_prims_scoped_load_wraps_cutlass_array_view_lazily():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop

        assert "cuda.coop.cutlass._prims_adapter" not in sys.modules
        assert coop._block.load.__module__ == "cuda.coop.cutlass._block"
        assert coop._warp.load.__module__ == "cuda.coop.cutlass._warp"

        calls = []

        class FakeDtype:
            bytes = 4

        class DynamicOffset:
            def __add__(self, value):
                return 5 + value

            def __eq__(self, value):
                raise AssertionError("dynamic offset must not be compared in Python")

        class FakeArray:
            def __init__(self, source):
                self.source = source
                self.dtype = FakeDtype

            def load(self, base, vector_size=None, **kwargs):
                calls.append(("load", self.source, base, vector_size, kwargs))
                if vector_size is None:
                    return ("scalar", self.source, base)
                return {
                    "source": self.source,
                    "base": base,
                    "vector_size": vector_size,
                    "kwargs": kwargs,
                }

        FakeArray.__module__ = "cutlass.base_dsl.array"
        FakeArray.__qualname__ = "Array"

        thread_state = {"idx": (7, 0, 0), "block_dim": (32, 1, 1)}

        def thread_idx():
            calls.append(("thread_idx",))
            return thread_state["idx"]

        def block_dim():
            calls.append(("block_dim",))
            return thread_state["block_dim"]

        def make_array_view(source, *, dtype=None, bounds_check=False, loc=None, ip=None):
            calls.append(("make_array_view", source, dtype, bounds_check, loc, ip))
            return FakeArray(source)

        def if_generate(cond, true_fn, false_fn=None, return_types=None):
            calls.append(("if_generate", cond, return_types))
            if cond:
                return true_fn()
            if false_fn is None:
                return None
            return false_fn()

        sys.modules["cutlass"] = types.SimpleNamespace(
            Array=FakeArray,
            if_generate=if_generate,
            make_array_view=make_array_view,
            cute=types.SimpleNamespace(
                arch=types.SimpleNamespace(
                    thread_idx=thread_idx,
                    block_dim=block_dim,
                )
            ),
        )

        loaded = coop._block.load(
            "keys",
            items_per_thread=3,
            dtype=FakeDtype,
            offset=5,
            is_nontemporal=True,
        )

        assert loaded["source"] == "keys"
        assert loaded["base"] == 26
        assert loaded["vector_size"] == 3
        assert loaded["kwargs"]["alignment"] == 4
        assert loaded["kwargs"]["is_nontemporal"] is True
        assert calls == [
            ("make_array_view", "keys", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "keys",
                26,
                3,
                {
                    "alignment": 4,
                    "is_volatile": False,
                    "is_nontemporal": True,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        typed_array = FakeArray("typed-values")
        loaded = coop._block.load(
            typed_array,
            items_per_thread=2,
            offset=5,
        )

        assert loaded["source"] == "typed-values"
        assert loaded["base"] == 19
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 4
        assert calls == [
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "typed-values",
                19,
                2,
                {
                    "alignment": 4,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        base_limited_array = FakeArray("base-limited")
        base_limited_array.align = 4
        loaded = coop._block.load(
            base_limited_array,
            items_per_thread=2,
        )
        assert loaded["base"] == 14
        assert loaded["kwargs"]["alignment"] == 4

        calls.clear()
        loaded = coop._block.load(
            FakeArray("dynamic-offset"),
            items_per_thread=2,
            offset=DynamicOffset(),
        )
        assert loaded["base"] == 19
        assert loaded["kwargs"]["alignment"] == 4

        calls.clear()
        loaded = coop._block.load(
            FakeArray("none-offset"),
            items_per_thread=2,
            offset=None,
        )
        assert loaded["base"] == 14
        assert loaded["kwargs"]["alignment"] == 8
        assert [call[0:4] for call in calls if call[0] == "load"] == [
            ("load", "none-offset", 14, 2),
        ]

        calls.clear()
        array = FakeArray("values")
        loaded = coop._warp.load(
            array,
            2,
            algorithm="vectorize",
            alignment=16,
            threads_in_warp=32,
        )

        assert loaded["source"] == "values"
        assert loaded["base"] == 14
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 16
        assert calls == [
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "values",
                14,
                2,
                {
                    "alignment": 16,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        partial = coop._block.load(
            "partial",
            items_per_thread=3,
            dtype=FakeDtype,
            offset=5,
            num_valid_items=23,
            oob_default=-7,
            is_invariant=True,
        )

        assert isinstance(partial, coop.ThreadData)
        assert partial.dtype is FakeDtype
        assert tuple(partial[item] for item in range(3)) == (
            ("scalar", "partial", 26),
            ("scalar", "partial", 27),
            -7,
        )
        assert calls == [
            ("make_array_view", "partial", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            ("if_generate", True, [FakeDtype]),
            (
                "load",
                "partial",
                26,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": True,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            ("if_generate", True, [FakeDtype]),
            (
                "load",
                "partial",
                27,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": True,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            ("if_generate", False, [FakeDtype]),
        ]

        calls.clear()
        striped = coop._block.load(
            "striped",
            items_per_thread=3,
            dtype=FakeDtype,
            offset=5,
            algorithm="striped",
            threads_per_block=32,
            is_volatile=True,
        )

        assert isinstance(striped, coop.ThreadData)
        assert striped.dtype is FakeDtype
        assert tuple(striped[item] for item in range(3)) == (
            ("scalar", "striped", 12),
            ("scalar", "striped", 44),
            ("scalar", "striped", 76),
        )
        assert calls == [
            ("make_array_view", "striped", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "striped",
                12,
                None,
                {
                    "alignment": None,
                    "is_volatile": True,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "load",
                "striped",
                44,
                None,
                {
                    "alignment": None,
                    "is_volatile": True,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "load",
                "striped",
                76,
                None,
                {
                    "alignment": None,
                    "is_volatile": True,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        striped_partial = coop._block.load(
            FakeArray("striped-partial"),
            items_per_thread=3,
            dtype=FakeDtype,
            offset=5,
            algorithm="striped",
            threads_per_block=32,
            valid_items=40,
            oob_default=-9,
        )
        assert tuple(striped_partial) == (
            ("scalar", "striped-partial", 12),
            ("scalar", "striped-partial", 44),
            -9,
        )
        assert [call[1] for call in calls if call[0] == "if_generate"] == [
            True,
            True,
            False,
        ]

        calls.clear()
        striped_dim = coop._block.load(
            "striped-dim",
            payload=coop.Payload.PRIMS,
            items_per_thread=3,
            dtype=FakeDtype,
            offset=5,
            algorithm="striped",
            dim=(16, 1, 1),
        )

        assert isinstance(striped_dim, coop.ThreadData)
        assert striped_dim.dtype is FakeDtype
        assert tuple(striped_dim[item] for item in range(3)) == (
            ("scalar", "striped-dim", 12),
            ("scalar", "striped-dim", 28),
            ("scalar", "striped-dim", 44),
        )
        assert calls == [
            ("make_array_view", "striped-dim", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "striped-dim",
                12,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "load",
                "striped-dim",
                28,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "load",
                "striped-dim",
                44,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        thread_state["idx"] = (23, 0, 0)
        warp_striped = coop._warp.load(
            array,
            items_per_thread=2,
            algorithm="striped",
            threads_in_warp=16,
        )

        assert isinstance(warp_striped, coop.ThreadData)
        assert warp_striped.dtype is FakeDtype
        assert tuple(warp_striped[item] for item in range(2)) == (
            ("scalar", "values", 39),
            ("scalar", "values", 55),
        )
        assert calls == [
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "values",
                39,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "load",
                "values",
                55,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        warp_partial = coop._warp.load(
            array,
            items_per_thread=2,
            threads_in_warp=16,
            offset=5,
            valid_items=47,
            oob_default=-1,
        )
        assert tuple(warp_partial) == (("scalar", "values", 51), -1)
        assert [call[1] for call in calls if call[0] == "if_generate"] == [
            True,
            False,
        ]
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_prims_offset_alignment_preserves_proven_widths():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass import _prims_adapter

        class IndexValue:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        class Dtype:
            bytes = IndexValue(4)

        class ZeroDtype:
            bytes = 0

        infer = _prims_adapter._infer_offset_alignment
        assert infer(2, Dtype, 4) == 8
        assert infer(2, Dtype, 5) == 4
        assert infer(2, Dtype, -4) == 8
        assert infer(2, Dtype, -5) == 4
        assert infer(4, Dtype, 1) == 4
        assert infer(2, Dtype, object()) == 4
        assert infer(2, Dtype, 4, base_alignment=4) == 4
        assert infer(2, Dtype, 4, base_alignment=32) == 8
        assert infer(2, Dtype, 4, base_alignment=12) == 4
        assert infer(2, Dtype, 4, base_alignment=-8) == 8
        assert infer(2, Dtype, 4, base_alignment=0) == 8
        assert infer(2, Dtype, 4, base_alignment=object()) == 8
        assert infer(2, Dtype, 0, base_alignment=2) == 2
        assert infer(2, Dtype, object(), base_alignment=2) == 2
        assert infer(2, object(), 0) is None
        assert infer(2, object(), 0, base_alignment=4) == 4
        assert infer(2, object(), 0, base_alignment=1) == 1
        assert infer(2, object(), 5, base_alignment=4) == 1
        assert infer(2, object(), object()) == 1
        assert infer(2, ZeroDtype, 0) is None
        assert infer(2, ZeroDtype, 0, base_alignment=4) == 4
        assert infer(2, ZeroDtype, 5, base_alignment=4) == 1
        assert infer(2, ZeroDtype, object()) == 1
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_prims_scoped_store_wraps_cutlass_array_view_lazily():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop

        assert "cuda.coop.cutlass._prims_adapter" not in sys.modules
        assert coop._block.store.__module__ == "cuda.coop.cutlass._block"
        assert coop._warp.store.__module__ == "cuda.coop.cutlass._warp"
        assert hasattr(coop._block.store, "__wrapped__")
        assert hasattr(coop._warp.store, "__wrapped__")

        calls = []

        class FakeDtype:
            bytes = 4

        class DynamicOffset:
            def __add__(self, value):
                return 5 + value

            def __eq__(self, value):
                raise AssertionError("dynamic offset must not be compared in Python")

        class FakeVector:
            shape = (3,)
            dtype = FakeDtype

            def __init__(self, prefix="v", shape=(3,)):
                self.prefix = prefix
                self.shape = shape

            def __getitem__(self, idx):
                return f"{self.prefix}{idx}"

        class FakeArray:
            dtype = FakeDtype

            def __init__(self, source):
                self.source = source

            def load(self, *args, **kwargs):
                raise AssertionError("store test should not load")

            def store(self, value, idx=0, vector_size=None, **kwargs):
                calls.append(("store", self.source, value, idx, vector_size, kwargs))

        FakeArray.__module__ = "cutlass.base_dsl.array"
        FakeArray.__qualname__ = "Array"

        thread_state = {"idx": (7, 0, 0), "block_dim": (32, 1, 1)}

        def thread_idx():
            calls.append(("thread_idx",))
            return thread_state["idx"]

        def block_dim():
            calls.append(("block_dim",))
            return thread_state["block_dim"]

        def make_array_view(source, *, dtype=None, bounds_check=False, loc=None, ip=None):
            calls.append(("make_array_view", source, dtype, bounds_check, loc, ip))
            return FakeArray(source)

        def if_generate(cond, true_fn, false_fn=None, return_types=None):
            calls.append(("if_generate", cond, return_types))
            if cond:
                return true_fn()
            if false_fn is None:
                return None
            return false_fn()

        sys.modules["cutlass"] = types.SimpleNamespace(
            Array=FakeArray,
            if_generate=if_generate,
            make_array_view=make_array_view,
            cute=types.SimpleNamespace(
                arch=types.SimpleNamespace(
                    thread_idx=thread_idx,
                    block_dim=block_dim,
                )
            ),
        )

        coop._block.store(
            "out",
            FakeVector(),
            algorithm="vectorize",
            offset=5,
            is_nontemporal=True,
        )

        assert calls == [
            ("make_array_view", "out", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "store",
                "out",
                ("v0", "v1", "v2"),
                26,
                3,
                {
                    "alignment": 4,
                    "is_volatile": False,
                    "is_nontemporal": True,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        base_limited_array = FakeArray("base-limited")
        base_limited_array.align = 4
        coop._block.store(
            base_limited_array,
            FakeVector("a", shape=(2,)),
        )
        store_calls = [call for call in calls if call[0] == "store"]
        assert [call[0:5] for call in store_calls] == [
            ("store", "base-limited", ("a0", "a1"), 14, 2),
        ]
        assert store_calls[-1][5]["alignment"] == 4

        calls.clear()
        coop._block.store(
            FakeArray("dynamic-offset"),
            FakeVector("d", shape=(2,)),
            offset=DynamicOffset(),
        )
        store_calls = [call for call in calls if call[0] == "store"]
        assert [call[0:5] for call in store_calls] == [
            ("store", "dynamic-offset", ("d0", "d1"), 19, 2),
        ]
        assert store_calls[-1][5]["alignment"] == 4

        calls.clear()
        coop._block.store(
            FakeArray("none-offset"),
            FakeVector("n"),
            offset=None,
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "none-offset", ("n0", "n1", "n2"), 21, 3),
        ]
        assert [call[5]["alignment"] for call in calls if call[0] == "store"] == [
            4,
        ]

        calls.clear()
        coop._block.store(
            "partial",
            FakeVector("p"),
            offset=5,
            num_valid_items=23,
            is_volatile=True,
        )

        assert calls == [
            ("make_array_view", "partial", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            ("if_generate", True, None),
            (
                "store",
                "partial",
                "p0",
                26,
                None,
                {
                    "alignment": None,
                    "is_volatile": True,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            ("if_generate", True, None),
            (
                "store",
                "partial",
                "p1",
                27,
                None,
                {
                    "alignment": None,
                    "is_volatile": True,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            ("if_generate", False, None),
        ]

        calls.clear()
        coop._block.store(
            "striped",
            FakeVector("s"),
            payload=coop.Payload.PRIMS,
            offset=5,
            algorithm="striped",
            threads_per_block=32,
        )

        assert calls == [
            ("make_array_view", "striped", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "store",
                "striped",
                "s0",
                12,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "store",
                "striped",
                "s1",
                44,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "store",
                "striped",
                "s2",
                76,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        coop._block.store(
            FakeArray("striped-partial"),
            FakeVector("t"),
            offset=5,
            algorithm="striped",
            threads_per_block=32,
            valid_items=40,
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "striped-partial", "t0", 12, None),
            ("store", "striped-partial", "t1", 44, None),
        ]
        assert [call[1] for call in calls if call[0] == "if_generate"] == [
            True,
            True,
            False,
        ]

        calls.clear()
        thread_state["idx"] = (23, 0, 0)
        coop._warp.store(
            FakeArray("warp-out"),
            FakeVector("w", shape=(2,)),
            algorithm="vectorize",
            offset=5,
            threads_in_warp=16,
            ordering="monotonic",
        )

        assert calls == [
            ("thread_idx",),
            ("block_dim",),
            (
                "store",
                "warp-out",
                "w0",
                51,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "monotonic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "store",
                "warp-out",
                "w1",
                52,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "monotonic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        coop._warp.store(
            FakeArray("warp-striped"),
            FakeVector("q", shape=(2,)),
            algorithm="striped",
            threads_in_warp=16,
        )

        assert calls == [
            ("thread_idx",),
            ("block_dim",),
            (
                "store",
                "warp-striped",
                "q0",
                39,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "store",
                "warp-striped",
                "q1",
                55,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        coop._warp.store(
            FakeArray("warp-partial"),
            FakeVector("r", shape=(2,)),
            threads_in_warp=16,
            offset=5,
            valid_items=47,
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "warp-partial", "r0", 51, None),
        ]
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_prims_scoped_load_store_factories_bind_prims_helpers():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as root
        import cuda.coop.cutlass as coop

        calls = []

        class FakeDtype:
            bytes = 4

        class FakeVector:
            shape = (2,)
            dtype = FakeDtype

            def __init__(self, prefix="v"):
                self.prefix = prefix

            def __getitem__(self, idx):
                return f"{self.prefix}{idx}"

        class FakeArray:
            dtype = FakeDtype

            def __init__(self, source):
                self.source = source

            def load(self, base, vector_size=None, **kwargs):
                calls.append(("load", self.source, base, vector_size, kwargs))
                if vector_size is None:
                    return f"{self.source}:{base}"
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

        def thread_idx():
            calls.append(("thread_idx",))
            return 3, 0, 0

        def block_dim():
            calls.append(("block_dim",))
            return 32, 1, 1

        def make_array_view(source, *, dtype=None, bounds_check=False, loc=None, ip=None):
            calls.append(("make_array_view", source, dtype, bounds_check, loc, ip))
            return FakeArray(source)

        sys.modules["cutlass"] = types.SimpleNamespace(
            Array=FakeArray,
            make_array_view=make_array_view,
            cute=types.SimpleNamespace(
                arch=types.SimpleNamespace(
                    thread_idx=thread_idx,
                    block_dim=block_dim,
                )
            ),
        )

        for name in ("make_load", "make_store"):
            assert getattr(coop._block, name).__module__ == (
                "cuda.coop.cutlass._block"
            )
            assert getattr(coop._warp, name).__module__ == (
                "cuda.coop.cutlass._warp"
            )
            assert hasattr(getattr(coop._block, name), "__wrapped__")
            assert hasattr(getattr(coop._warp, name), "__wrapped__")
            assert name in coop._block.__all__
            assert name in coop._warp.__all__
            assert getattr(root._block, name) is getattr(coop._block, name)

        def assert_raises(expected, fn):
            try:
                fn()
            except Exception as exc:
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError("expected exception")

        for scope_name in ("block", "warp"):
            scope = getattr(coop, f"_{scope_name}")
            for factory_name in ("make_load", "make_store"):
                factory = getattr(scope, factory_name)
                assert_raises(
                    (
                        f"cuda.coop.cutlass._{scope_name}.{factory_name} "
                        "items_per_thread must be an int"
                    ),
                    lambda factory=factory: factory(
                        FakeDtype,
                        items_per_thread=object(),
                    ),
                )
                assert_raises(
                    (
                        f"cuda.coop.cutlass._{scope_name}.{factory_name} "
                        "items_per_thread must be an int"
                    ),
                    lambda factory=factory: factory(
                        FakeDtype,
                        items_per_thread=True,
                    ),
                )
                assert_raises(
                    (
                        f"cuda.coop.cutlass._{scope_name}.{factory_name} "
                        "items_per_thread must be positive"
                    ),
                    lambda factory=factory: factory(FakeDtype, items_per_thread=0),
                )

        for factory_name in ("make_load", "make_store"):
            factory = getattr(coop._warp, factory_name)
            assert_raises(
                (
                    f"cuda.coop.cutlass._warp.{factory_name} "
                    "threads_in_warp must be an int"
                ),
                lambda factory=factory: factory(
                    FakeDtype,
                    items_per_thread=2,
                    threads_in_warp=object(),
                ),
            )
            assert_raises(
                (
                    f"cuda.coop.cutlass._warp.{factory_name} "
                    "threads_in_warp must be positive"
                ),
                lambda factory=factory: factory(
                    FakeDtype,
                    items_per_thread=2,
                    threads_in_warp=0,
                ),
            )
            assert_raises(
                (
                    f"cuda.coop.cutlass._warp.{factory_name} "
                    "threads_in_warp must be <= 32"
                ),
                lambda factory=factory: factory(
                    FakeDtype,
                    items_per_thread=2,
                    threads_in_warp=64,
                ),
            )
            assert_raises(
                (
                    f"cuda.coop.cutlass._warp.{factory_name} "
                    "threads_in_warp must be a power of two"
                ),
                lambda factory=factory: factory(
                    FakeDtype,
                    items_per_thread=2,
                    threads_in_warp=12,
                ),
            )

        block_load = coop._block.make_load(
            FakeDtype,
            threads_per_block=32,
            items_per_thread=2,
            algorithm="vectorize",
            offset=4,
            payload=coop.Payload.PRIMS,
        )
        assert block_load.scope == "cuda.coop.cutlass._block"
        assert block_load.primitive is coop._block.load
        loaded = block_load("block-in")
        assert loaded["source"] == "block-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2

        block_store = coop._block.make_store(
            FakeDtype,
            threads_per_block=32,
            items_per_thread=2,
            algorithm="vectorize",
            offset=4,
            payload=coop.Payload.PRIMS,
        )
        assert block_store.scope == "cuda.coop.cutlass._block"
        assert block_store.primitive is coop._block.store
        block_store("block-out", FakeVector("b"))

        warp_store = coop._warp.make_store(
            FakeDtype,
            items_per_thread=2,
            algorithm="vectorize",
            threads_in_warp=16,
            offset=4,
        )
        assert warp_store.scope == "cuda.coop.cutlass._warp"
        assert warp_store.primitive is coop._warp.store
        warp_store(FakeArray("warp-out"), FakeVector("w"))

        assert calls == [
            ("make_array_view", "block-in", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "block-in",
                10,
                2,
                {
                    "alignment": 8,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            ("make_array_view", "block-out", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "store",
                "block-out",
                ("b0", "b1"),
                10,
                2,
                {
                    "alignment": 8,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            ("thread_idx",),
            ("block_dim",),
            (
                "store",
                "warp-out",
                "w0",
                10,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "store",
                "warp-out",
                "w1",
                11,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        inferred_block_load = coop._block.make_load(
            items_per_thread=2,
            algorithm="vectorize",
            offset=4,
            payload=coop.Payload.PRIMS,
        )
        assert inferred_block_load.scope == "cuda.coop.cutlass._block"
        assert inferred_block_load.primitive is coop._block.load
        loaded = inferred_block_load("inferred-block-in")
        assert loaded["source"] == "inferred-block-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 8

        inferred_block_store = coop._block.make_store(
            items_per_thread=2,
            algorithm="vectorize",
            offset=4,
            payload=coop.Payload.PRIMS,
        )
        assert inferred_block_store.scope == "cuda.coop.cutlass._block"
        assert inferred_block_store.primitive is coop._block.store
        inferred_block_store("inferred-block-out", FakeVector("i"))

        inferred_warp_load = coop._warp.make_load(
            items_per_thread=2,
            threads_in_warp=16,
            offset=4,
        )
        assert inferred_warp_load.scope == "cuda.coop.cutlass._warp"
        assert inferred_warp_load.primitive is coop._warp.load
        loaded = inferred_warp_load(FakeArray("inferred-warp-in"))
        assert loaded["source"] == "inferred-warp-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 8

        inferred_warp_store = coop._warp.make_store(
            items_per_thread=2,
            threads_in_warp=16,
            offset=4,
        )
        assert inferred_warp_store.scope == "cuda.coop.cutlass._warp"
        assert inferred_warp_store.primitive is coop._warp.store
        inferred_warp_store(FakeArray("inferred-warp-out"), FakeVector("j"))

        assert calls[0:4] == [
            ("make_array_view", "inferred-block-in", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "inferred-block-in",
                10,
                2,
                {
                    "alignment": 8,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]
        assert calls[4:8] == [
            ("make_array_view", "inferred-block-out", None, False, None, None),
            ("thread_idx",),
            ("block_dim",),
            (
                "store",
                "inferred-block-out",
                ("i0", "i1"),
                10,
                2,
                {
                    "alignment": 8,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]
        assert calls[8:11] == [
            ("thread_idx",),
            ("block_dim",),
            (
                "load",
                "inferred-warp-in",
                10,
                2,
                {
                    "alignment": 8,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "is_invariant": False,
                    "is_invariant_group": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]
        assert calls[11:] == [
            ("thread_idx",),
            ("block_dim",),
            (
                "store",
                "inferred-warp-out",
                "j0",
                10,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
            (
                "store",
                "inferred-warp-out",
                "j1",
                11,
                None,
                {
                    "alignment": None,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]

        calls.clear()
        call_dtype_block_load = coop._block.make_load(
            items_per_thread=2,
            algorithm="vectorize",
            offset=4,
            payload=coop.Payload.PRIMS,
        )
        loaded = call_dtype_block_load("call-dtype-block-in", dtype=FakeDtype)
        assert loaded["source"] == "call-dtype-block-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 8

        call_dtype_block_store = coop._block.make_store(
            items_per_thread=2,
            algorithm="vectorize",
            offset=4,
            payload=coop.Payload.PRIMS,
        )
        call_dtype_block_store(
            "call-dtype-block-out",
            FakeVector("k"),
            dtype=FakeDtype,
        )
        assert calls[-1][0:5] == (
            "store",
            "call-dtype-block-out",
            ("k0", "k1"),
            10,
            2,
        )

        calls.clear()
        call_dtype_warp_load = coop._warp.make_load(
            items_per_thread=2,
            threads_in_warp=16,
            offset=4,
        )
        loaded = call_dtype_warp_load(FakeArray("call-dtype-warp-in"), dtype=FakeDtype)
        assert loaded["source"] == "call-dtype-warp-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 8

        call_dtype_warp_store = coop._warp.make_store(
            items_per_thread=2,
            threads_in_warp=16,
            offset=4,
        )
        call_dtype_warp_store(
            FakeArray("call-dtype-warp-out"),
            FakeVector("l"),
            dtype=FakeDtype,
        )
        assert [call[0:5] for call in calls if call[0] == "store"][-2:] == [
            ("store", "call-dtype-warp-out", "l0", 10, None),
            ("store", "call-dtype-warp-out", "l1", 11, None),
        ]

        calls.clear()
        payload_block_load = coop._block.load(
            "payload-block-in",
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            dtype=FakeDtype,
            offset=4,
        )
        assert payload_block_load["source"] == "payload-block-in"
        assert payload_block_load["base"] == 10
        coop._block.store(
            "payload-block-out",
            FakeVector("p"),
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            dtype=FakeDtype,
            offset=4,
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "payload-block-out", ("p0", "p1"), 10, 2),
        ]

        calls.clear()
        payload_block_factory = coop._block.make_load(
            items_per_thread=2,
            algorithm="vectorize",
            offset=4,
            payload=coop.Payload.PRIMS,
        )
        assert dict(payload_block_factory.bound_kwargs)["payload"] is coop.Payload.PRIMS
        loaded = payload_block_factory(
            "payload-factory-block-in",
            payload=coop.Payload.PRIMS,
        )
        assert loaded["source"] == "payload-factory-block-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2

        payload_warp_factory = coop._warp.make_store(
            items_per_thread=2,
            threads_in_warp=16,
            offset=4,
            payload=coop.Payload.PRIMS,
        )
        assert dict(payload_warp_factory.bound_kwargs)["payload"] is coop.Payload.PRIMS
        payload_warp_factory(
            FakeArray("payload-factory-warp-out"),
            FakeVector("q"),
            payload=coop.Payload.PRIMS,
        )
        assert [call[0:5] for call in calls if call[0] == "store"][-2:] == [
            ("store", "payload-factory-warp-out", "q0", 10, None),
            ("store", "payload-factory-warp-out", "q1", 11, None),
        ]

        calls.clear()
        metadata_block_load = coop._block.make_load(
            items_per_thread=2,
            algorithm="striped",
            offset=4,
            payload=coop.Payload.PRIMS,
            launch_metadata={"threads_per_block": 8},
        )
        loaded = metadata_block_load("metadata-block-in")
        assert isinstance(loaded, coop.ThreadData)
        assert loaded.dtype is FakeDtype
        assert tuple(loaded[item] for item in range(2)) == (
            "metadata-block-in:7",
            "metadata-block-in:15",
        )
        assert [call[0:4] for call in calls if call[0] == "load"] == [
            ("load", "metadata-block-in", 7, None),
            ("load", "metadata-block-in", 15, None),
        ]

        calls.clear()
        metadata_block_store = coop._block.make_store(
            items_per_thread=2,
            algorithm="striped",
            offset=4,
            launch_config={"block": (8, 1, 1)},
        )
        metadata_block_store(FakeArray("metadata-block-out"), FakeVector("m"))
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "metadata-block-out", "m0", 7, None),
            ("store", "metadata-block-out", "m1", 15, None),
        ]

        metadata_warp_load = coop._warp.make_load(
            items_per_thread=2,
            threads_in_warp=16,
            payload=coop.Payload.PRIMS,
        )
        metadata_warp_store = coop._warp.make_store(
            items_per_thread=2,
            threads_in_warp=16,
            payload=coop.Payload.PRIMS,
        )
        for alias in ("launch_metadata", "launch_meta", "launch", "launch_config"):
            metadata_kwargs = {alias: {"threads_per_block": 32}}
            assert_raises(
                (
                    "cuda.coop.cutlass._warp.load does not accept launch metadata "
                    f"keyword argument(s): {alias}"
                ),
                lambda metadata_kwargs=metadata_kwargs: coop._warp.load(
                    FakeArray("metadata-warp-in"),
                    items_per_thread=2,
                    threads_in_warp=16,
                    **metadata_kwargs,
                ),
            )
            assert_raises(
                (
                    "cuda.coop.cutlass._warp.store does not accept launch metadata "
                    f"keyword argument(s): {alias}"
                ),
                lambda metadata_kwargs=metadata_kwargs: coop._warp.store(
                    FakeArray("metadata-warp-out"),
                    FakeVector("mw"),
                    items_per_thread=2,
                    threads_in_warp=16,
                    **metadata_kwargs,
                ),
            )
            assert_raises(
                (
                    "cuda.coop.cutlass._warp.make_load does not accept launch metadata "
                    f"keyword argument(s): {alias}"
                ),
                lambda metadata_kwargs=metadata_kwargs: coop._warp.make_load(
                    items_per_thread=2,
                    threads_in_warp=16,
                    payload=coop.Payload.PRIMS,
                    **metadata_kwargs,
                ),
            )
            assert_raises(
                (
                    "cuda.coop.cutlass._warp.make_store does not accept launch metadata "
                    f"keyword argument(s): {alias}"
                ),
                lambda metadata_kwargs=metadata_kwargs: coop._warp.make_store(
                    items_per_thread=2,
                    threads_in_warp=16,
                    payload=coop.Payload.PRIMS,
                    **metadata_kwargs,
                ),
            )
            assert_raises(
                (
                    "cuda.coop.cutlass._warp.load does not accept launch metadata "
                    f"keyword argument(s): {alias}"
                ),
                lambda metadata_kwargs=metadata_kwargs: metadata_warp_load(
                    FakeArray("deferred-metadata-warp-in"),
                    **metadata_kwargs,
                ),
            )
            assert_raises(
                (
                    "cuda.coop.cutlass._warp.store does not accept launch metadata "
                    f"keyword argument(s): {alias}"
                ),
                lambda metadata_kwargs=metadata_kwargs: metadata_warp_store(
                    FakeArray("deferred-metadata-warp-out"),
                    FakeVector("dmw"),
                    **metadata_kwargs,
                ),
            )

        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_root_load_store_payload_selector_routes_prims_sources():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop

        calls = []

        class FakeDtype:
            bytes = 4

        class PrimsDtype:
            bytes = 4

        PrimsDtype.__module__ = "cutlass.base_dsl.typing"

        class FakeVector:
            shape = (2,)
            dtype = FakeDtype

            def __init__(self, prefix="v"):
                self.prefix = prefix

            def __getitem__(self, idx):
                return f"{self.prefix}{idx}"

        class FakeArray:
            dtype = FakeDtype

            def __init__(self, source):
                self.source = source

            def load(self, base, vector_size=None, **kwargs):
                calls.append(("load", self.source, base, vector_size, kwargs))
                if vector_size is None:
                    return f"{self.source}:{base}"
                return {
                    "source": self.source,
                    "base": base,
                    "vector_size": vector_size,
                    "kwargs": kwargs,
                }

            def store(self, value, idx=0, vector_size=None, **kwargs):
                calls.append(("store", self.source, value, idx, vector_size, kwargs))

        def thread_idx():
            calls.append(("thread_idx",))
            return 3, 0, 0

        def block_dim():
            calls.append(("block_dim",))
            return 32, 1, 1

        def make_array_view(source, *, dtype=None, bounds_check=False, loc=None, ip=None):
            calls.append(("make_array_view", source, dtype, bounds_check, loc, ip))
            return FakeArray(source)

        sys.modules["cutlass"] = types.SimpleNamespace(
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
            except Exception as exc:
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError("expected exception")

        loaded = coop._block.load(
            "block-in",
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            dtype=FakeDtype,
            offset=4,
        )
        assert loaded["source"] == "block-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 8
        assert calls[0] == ("make_array_view", "block-in", None, False, None, None)

        calls.clear()
        loaded = coop._block.load(
            "dtype-block-in",
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            dtype=PrimsDtype,
        )
        assert loaded["source"] == "dtype-block-in"
        assert loaded["base"] == 6
        assert loaded["vector_size"] == 2
        assert loaded["kwargs"]["alignment"] == 8
        assert calls[0] == ("make_array_view", "dtype-block-in", None, False, None, None)

        calls.clear()
        loaded = coop._block.load(
            "implicit-block-in",
            items_per_thread=2,
            dtype=FakeDtype,
            offset=4,
            bounds_check=True,
        )
        assert loaded["source"] == "implicit-block-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        assert calls[0] == ("make_array_view", "implicit-block-in", None, True, None, None)

        calls.clear()
        coop._block.store(
            "block-out",
            FakeVector("b"),
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            dtype=FakeDtype,
            offset=4,
        )
        assert calls[0] == ("make_array_view", "block-out", None, False, None, None)
        assert calls[-1][0:5] == ("store", "block-out", ("b0", "b1"), 10, 2)

        calls.clear()
        coop._block.store(
            "dtype-block-out",
            coop.ThreadData.from_values("td0", "td1", dtype=PrimsDtype),
            payload=coop.Payload.PRIMS,
        )
        assert calls[0] == ("make_array_view", "dtype-block-out", None, False, None, None)
        assert calls[-1][0:5] == ("store", "dtype-block-out", ("td0", "td1"), 6, 2)

        calls.clear()
        coop._block.store(
            "implicit-block-out",
            FakeVector("bi"),
            items_per_thread=2,
            dtype=FakeDtype,
            offset=4,
            bounds_check=True,
        )
        assert calls[0] == (
            "make_array_view",
            "implicit-block-out",
            None,
            True,
            None,
            None,
        )
        assert calls[-1][0:5] == ("store", "implicit-block-out", ("bi0", "bi1"), 10, 2)

        calls.clear()
        block_load = coop._block.make_load(
            items_per_thread=2,
            payload=coop.Payload.PRIMS,
            dtype=FakeDtype,
            offset=4,
        )
        assert block_load.scope == "cuda.coop.cutlass._block"
        assert block_load.primitive is coop._block.load
        assert "payload" in block_load.overridable_kwargs
        assert "offset" in block_load.overridable_kwargs
        loaded = block_load("factory-block-in")
        assert loaded["source"] == "factory-block-in"
        assert loaded["base"] == 10
        loaded = block_load("factory-block-in-offset", offset=8)
        assert loaded["source"] == "factory-block-in-offset"
        assert loaded["base"] == 14
        loaded = block_load("factory-block-in-prims", payload=coop.Payload.PRIMS)
        assert loaded["source"] == "factory-block-in-prims"
        assert loaded["base"] == 10

        block_dtype_load = coop._block.make_load(
            PrimsDtype,
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
        )
        assert block_dtype_load.scope == "cuda.coop.cutlass._block"
        assert block_dtype_load.primitive is coop._block.load
        assert dict(block_dtype_load.bound_kwargs)["payload"] is coop.Payload.PRIMS
        assert "payload" in block_dtype_load.overridable_kwargs
        loaded = block_dtype_load("factory-block-in-dtype")
        assert loaded["source"] == "factory-block-in-dtype"
        assert loaded["base"] == 6

        block_late_dtype_load = coop._block.make_load(items_per_thread=2)
        loaded = block_late_dtype_load(
            "factory-block-in-late-dtype",
            dtype=PrimsDtype,
            payload=coop.Payload.PRIMS,
        )
        assert loaded["source"] == "factory-block-in-late-dtype"
        assert loaded["base"] == 6

        block_payload_load = coop._block.make_load(
            items_per_thread=2,
            payload=coop.Payload.PRIMS,
            dtype=FakeDtype,
        )
        assert "payload" in block_payload_load.overridable_kwargs
        assert "offset" in block_payload_load.overridable_kwargs
        assert "offset" not in dict(block_payload_load.bound_kwargs)
        assert "bounds_check" not in dict(block_payload_load.bound_kwargs)

        block_implicit_prims_load = coop._block.make_load(
            items_per_thread=2,
            dtype=FakeDtype,
            offset=4,
            bounds_check=True,
        )
        assert block_implicit_prims_load.scope == "cuda.coop.cutlass._block"
        implicit_bound = dict(block_implicit_prims_load.bound_kwargs)
        assert implicit_bound["payload"] is coop.Payload.PRIMS
        assert implicit_bound["offset"] == 4
        assert implicit_bound["bounds_check"] is True
        calls.clear()
        loaded = block_implicit_prims_load("implicit-factory-block-in")
        assert loaded["source"] == "implicit-factory-block-in"
        assert loaded["base"] == 10
        assert calls[0] == (
            "make_array_view",
            "implicit-factory-block-in",
            None,
            True,
            None,
            None,
        )

        block_load_string = coop._block.make_load(
            items_per_thread=2,
            payload="prims",
            dtype=FakeDtype,
            offset=4,
        )
        assert dict(block_load_string.bound_kwargs)["payload"] is coop.Payload.PRIMS
        assert "payload" in block_load_string.overridable_kwargs
        assert "offset" in block_load_string.overridable_kwargs
        loaded = block_load_string("factory-block-in-string")
        assert loaded["source"] == "factory-block-in-string"
        assert loaded["base"] == 10

        calls.clear()
        block_store = coop._block.make_store(
            items_per_thread=2,
            payload=coop.Payload.PRIMS,
            dtype=FakeDtype,
            offset=4,
        )
        assert block_store.scope == "cuda.coop.cutlass._block"
        assert block_store.primitive is coop._block.store
        assert "payload" in block_store.overridable_kwargs
        assert "offset" in block_store.overridable_kwargs
        block_store("factory-block-out", FakeVector("f"))
        block_store("factory-block-out-prims", FakeVector("g"), payload=coop.Payload.PRIMS)
        block_store("factory-block-out-offset", FakeVector("h"), offset=8)
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "factory-block-out", ("f0", "f1"), 10, 2),
            ("store", "factory-block-out-prims", ("g0", "g1"), 10, 2),
            ("store", "factory-block-out-offset", ("h0", "h1"), 14, 2),
        ]

        calls.clear()
        block_dtype_store = coop._block.make_store(
            PrimsDtype,
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
        )
        assert block_dtype_store.scope == "cuda.coop.cutlass._block"
        assert block_dtype_store.primitive is coop._block.store
        assert dict(block_dtype_store.bound_kwargs)["payload"] is coop.Payload.PRIMS
        block_dtype_store(
            "factory-block-out-dtype",
            coop.ThreadData.from_values("bd0", "bd1", dtype=PrimsDtype),
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "factory-block-out-dtype", ("bd0", "bd1"), 6, 2),
        ]

        calls.clear()
        loaded = coop._warp.load(
            "warp-in",
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            dtype=FakeDtype,
            threads_in_warp=16,
            offset=4,
        )
        assert loaded["source"] == "warp-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2

        calls.clear()
        loaded = coop._warp.load(
            "dtype-warp-in",
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            dtype=PrimsDtype,
            threads_in_warp=16,
        )
        assert loaded["source"] == "dtype-warp-in"
        assert loaded["base"] == 6
        assert loaded["vector_size"] == 2
        assert calls[0] == ("make_array_view", "dtype-warp-in", None, False, None, None)

        calls.clear()
        loaded = coop._warp.load(
            "implicit-warp-in",
            items_per_thread=2,
            dtype=FakeDtype,
            threads_in_warp=16,
            offset=4,
            bounds_check=True,
        )
        assert loaded["source"] == "implicit-warp-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        assert calls[0] == ("make_array_view", "implicit-warp-in", None, True, None, None)

        calls.clear()
        coop._warp.store(
            "warp-out",
            FakeVector("ws"),
            payload=coop.Payload.PRIMS,
            dtype=FakeDtype,
            threads_in_warp=16,
            offset=4,
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "warp-out", "ws0", 10, None),
            ("store", "warp-out", "ws1", 11, None),
        ]

        calls.clear()
        coop._warp.store(
            "dtype-warp-out",
            coop.ThreadData.from_values("wtd0", "wtd1", dtype=PrimsDtype),
            payload=coop.Payload.PRIMS,
            threads_in_warp=16,
        )
        assert calls[0] == ("make_array_view", "dtype-warp-out", None, False, None, None)
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "dtype-warp-out", "wtd0", 6, None),
            ("store", "dtype-warp-out", "wtd1", 7, None),
        ]

        calls.clear()
        coop._warp.store(
            "implicit-warp-out",
            FakeVector("wi"),
            dtype=FakeDtype,
            threads_in_warp=16,
            offset=4,
            bounds_check=True,
        )
        assert calls[0] == (
            "make_array_view",
            "implicit-warp-out",
            None,
            True,
            None,
            None,
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "implicit-warp-out", "wi0", 10, None),
            ("store", "implicit-warp-out", "wi1", 11, None),
        ]

        calls.clear()
        warp_load = coop._warp.make_load(
            items_per_thread=2,
            payload=coop.Payload.PRIMS,
            dtype=FakeDtype,
            threads_in_warp=16,
            offset=4,
        )
        assert warp_load.scope == "cuda.coop.cutlass._warp"
        assert warp_load.primitive is coop._warp.load
        assert "payload" in warp_load.overridable_kwargs
        assert "offset" in warp_load.overridable_kwargs
        loaded = warp_load("factory-warp-in")
        assert loaded["source"] == "factory-warp-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        loaded = warp_load("factory-warp-in-offset", offset=8)
        assert loaded["source"] == "factory-warp-in-offset"
        assert loaded["base"] == 14
        assert loaded["vector_size"] == 2
        loaded = warp_load("factory-warp-in-prims", payload=coop.Payload.PRIMS)
        assert loaded["source"] == "factory-warp-in-prims"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2

        warp_dtype_load = coop._warp.make_load(
            PrimsDtype,
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            threads_in_warp=16,
        )
        assert warp_dtype_load.scope == "cuda.coop.cutlass._warp"
        assert warp_dtype_load.primitive is coop._warp.load
        assert dict(warp_dtype_load.bound_kwargs)["payload"] is coop.Payload.PRIMS
        loaded = warp_dtype_load("factory-warp-in-dtype")
        assert loaded["source"] == "factory-warp-in-dtype"
        assert loaded["base"] == 6
        assert loaded["vector_size"] == 2

        warp_late_dtype_load = coop._warp.make_load(
            items_per_thread=2,
            threads_in_warp=16,
        )
        loaded = warp_late_dtype_load(
            "factory-warp-in-late-dtype",
            dtype=PrimsDtype,
            payload=coop.Payload.PRIMS,
        )
        assert loaded["source"] == "factory-warp-in-late-dtype"
        assert loaded["base"] == 6
        assert loaded["vector_size"] == 2

        warp_payload_load = coop._warp.make_load(
            items_per_thread=2,
            payload=coop.Payload.PRIMS,
            dtype=FakeDtype,
            threads_in_warp=16,
        )
        assert "payload" in warp_payload_load.overridable_kwargs
        assert "offset" in warp_payload_load.overridable_kwargs
        assert "offset" not in dict(warp_payload_load.bound_kwargs)
        assert "bounds_check" not in dict(warp_payload_load.bound_kwargs)

        warp_implicit_prims_load = coop._warp.make_load(
            items_per_thread=2,
            dtype=FakeDtype,
            threads_in_warp=16,
            offset=4,
            bounds_check=True,
        )
        assert warp_implicit_prims_load.scope == "cuda.coop.cutlass._warp"
        implicit_bound = dict(warp_implicit_prims_load.bound_kwargs)
        assert implicit_bound["payload"] is coop.Payload.PRIMS
        assert implicit_bound["offset"] == 4
        assert implicit_bound["bounds_check"] is True
        calls.clear()
        loaded = warp_implicit_prims_load("implicit-factory-warp-in")
        assert loaded["source"] == "implicit-factory-warp-in"
        assert loaded["base"] == 10
        assert loaded["vector_size"] == 2
        assert calls[0] == (
            "make_array_view",
            "implicit-factory-warp-in",
            None,
            True,
            None,
            None,
        )

        calls.clear()
        warp_store = coop._warp.make_store(
            items_per_thread=2,
            payload=coop.Payload.PRIMS,
            dtype=FakeDtype,
            threads_in_warp=16,
            offset=4,
        )
        assert warp_store.scope == "cuda.coop.cutlass._warp"
        assert warp_store.primitive is coop._warp.store
        assert "payload" in warp_store.overridable_kwargs
        assert "offset" in warp_store.overridable_kwargs
        warp_store("factory-warp-out", FakeVector("w"))
        warp_store("factory-warp-out-prims", FakeVector("wc"), payload=coop.Payload.PRIMS)
        warp_store("factory-warp-out-offset", FakeVector("wo"), offset=8)
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "factory-warp-out", "w0", 10, None),
            ("store", "factory-warp-out", "w1", 11, None),
            ("store", "factory-warp-out-prims", "wc0", 10, None),
            ("store", "factory-warp-out-prims", "wc1", 11, None),
            ("store", "factory-warp-out-offset", "wo0", 14, None),
            ("store", "factory-warp-out-offset", "wo1", 15, None),
        ]

        calls.clear()
        warp_dtype_store = coop._warp.make_store(
            PrimsDtype,
            payload=coop.Payload.PRIMS,
            items_per_thread=2,
            threads_in_warp=16,
        )
        assert warp_dtype_store.scope == "cuda.coop.cutlass._warp"
        assert warp_dtype_store.primitive is coop._warp.store
        assert dict(warp_dtype_store.bound_kwargs)["payload"] is coop.Payload.PRIMS
        warp_dtype_store(
            "factory-warp-out-dtype",
            coop.ThreadData.from_values("wd0", "wd1", dtype=PrimsDtype),
        )
        assert [call[0:5] for call in calls if call[0] == "store"] == [
            ("store", "factory-warp-out-dtype", "wd0", 6, None),
            ("store", "factory-warp-out-dtype", "wd1", 7, None),
        ]

        assert_raises(
            "cuda.coop.cutlass._block.load payload must be prims",
            lambda: coop._block.load(
                "bad-payload",
                payload="bogus",
                items_per_thread=2,
            ),
        )
        assert_raises(
            "cuda.coop.cutlass._block.make_load payload must be prims",
            lambda: coop._block.make_load(
                items_per_thread=2,
                payload="bogus",
            ),
        )
        assert_raises(
            "cuda.coop.cutlass._warp.make_store payload must be prims",
            lambda: coop._warp.make_store(
                items_per_thread=2,
                payload="bogus",
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_prims_scoped_load_reports_unsupported_prims_array_modes():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as root
        import cuda.coop.cutlass as coop

        class FakeDtype:
            bytes = 4

        class DynamicInt:
            pass

        class PublicArray:
            pass

        class FakeArray(PublicArray):
            dtype = FakeDtype

            def load(self, *args, **kwargs):
                return args, kwargs

        FakeArray.__module__ = "cutlass.base_dsl.array"
        FakeArray.__qualname__ = "Array"

        class StoreOnlyArray(PublicArray):
            dtype = FakeDtype

            def store(self, *args, **kwargs):
                return args, kwargs

        StoreOnlyArray.__module__ = "cutlass.base_dsl.array"
        StoreOnlyArray.__qualname__ = "Array"

        sys.modules["cutlass"] = types.SimpleNamespace(
            Array=PublicArray,
            cute=types.SimpleNamespace(
                arch=types.SimpleNamespace(
                    thread_idx=lambda: (0, 0, 0),
                    block_dim=lambda: (32, 1, 1),
                )
            ),
        )

        def assert_raises(expected, fn):
            try:
                fn()
            except Exception as exc:
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError("expected exception")

        assert_raises(
            "requires items_per_thread",
            lambda: coop._block.load(FakeArray()),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.load cutlass.Array operand must "
                "support load"
            ),
            lambda: coop._block.load(StoreOnlyArray(), items_per_thread=2),
        )
        assert_raises(
            "duplicate items_per_thread",
            lambda: coop._block.load(FakeArray(), 2, items_per_thread=2),
        )
        for algorithm in (
            "transpose",
            "warp_transpose",
            "warp-transpose-timesliced",
        ):
            expected_algorithm = algorithm.replace("-", "_")
            assert_raises(
                f"algorithm '{expected_algorithm}' is not implemented yet",
                lambda algorithm=algorithm: coop._block.load(
                    FakeArray(),
                    items_per_thread=2,
                    algorithm=algorithm,
                ),
            )
        assert_raises(
            "requires valid_items",
            lambda: coop._warp.load(
                FakeArray(),
                items_per_thread=2,
                oob_default=0,
            ),
        )
        assert_raises(
            "requires oob_default",
            lambda: coop._warp.load(
                FakeArray(),
                items_per_thread=2,
                valid_items=3,
            ),
        )
        assert_raises(
            "accepts only one of valid_items or num_valid_items",
            lambda: coop._warp.load(
                FakeArray(),
                items_per_thread=2,
                valid_items=3,
                num_valid_items=3,
                oob_default=0,
            ),
        )
        assert_raises(
            "explicit alignment is not implemented",
            lambda: coop._warp.load(
                FakeArray(),
                items_per_thread=2,
                valid_items=3,
                oob_default=0,
                alignment=8,
            ),
        )
        assert_raises(
            "cuda.coop.cutlass._warp.load dtype= does not match cutlass.Array dtype",
            lambda: coop._warp.load(FakeArray(), items_per_thread=2, dtype=object),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.load bounds_check= is only "
                "accepted when wrapping a source with cutlass.make_array_view"
            ),
            lambda: coop._block.load(
                FakeArray(),
                items_per_thread=2,
                bounds_check=True,
            ),
        )
        dynamic_threads = DynamicInt()
        assert_raises(
            (
                "cuda.coop.cutlass._block.load threads_per_block must be "
                "a compile-time positive int"
            ),
            lambda: coop._block.load(
                FakeArray(),
                items_per_thread=2,
                threads_per_block=dynamic_threads,
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.load launch metadata has invalid "
                "thread-count key(s): threads_per_block"
            ),
            lambda: coop._block.load(
                FakeArray(),
                items_per_thread=2,
                launch_metadata={"threads_per_block": dynamic_threads},
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.load got multiple launch metadata "
                "aliases: launch_metadata, launch"
            ),
            lambda: coop._block.load(
                FakeArray(),
                items_per_thread=2,
                launch_metadata={"threads_per_block": 32},
                launch={"threads_per_block": 32},
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.load dim must be a compile-time "
                "positive int"
            ),
            lambda: coop._block.load(
                FakeArray(),
                items_per_thread=2,
                dim=(dynamic_threads, 1, 1),
            ),
        )
        assert_raises(
            "cuda.coop.cutlass._block.load does not accept threads_in_warp",
            lambda: coop._block.load(
                FakeArray(),
                items_per_thread=2,
                threads_in_warp=16,
            ),
        )
        assert_raises(
            "cuda.coop.cutlass._warp.load threads_in_warp must be an int",
            lambda: coop._warp.load(
                FakeArray(),
                items_per_thread=2,
                threads_in_warp=dynamic_threads,
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._warp.load does not accept "
                "threads_per_block or dim"
            ),
            lambda: coop._warp.load(
                FakeArray(),
                items_per_thread=2,
                dim=(32, 1, 1),
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.load threads_per_block must be "
                "a compile-time positive int"
            ),
            lambda: root._block.load(
                FakeArray(),
                payload=root.Payload.PRIMS,
                items_per_thread=2,
                threads_per_block=dynamic_threads,
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.load launch metadata has invalid "
                "thread-count key(s): block"
            ),
            lambda: root._block.load(
                FakeArray(),
                    payload=root.Payload.PRIMS,
                items_per_thread=2,
                launch_config={"block": (dynamic_threads, 1, 1)},
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_prims_scoped_store_reports_unsupported_prims_array_modes():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as root
        import cuda.coop.cutlass as coop

        calls = []

        class FakeDtype:
            bytes = 4

        class OtherDtype:
            bytes = 4

        class DynamicInt:
            pass

        class PublicArray:
            pass

        class FakeArray(PublicArray):
            dtype = FakeDtype

            def load(self, *args, **kwargs):
                return args, kwargs

            def store(self, *args, **kwargs):
                return args, kwargs

        FakeArray.__module__ = "cutlass.base_dsl.array"
        FakeArray.__qualname__ = "Array"

        class StoreOnlyArray(PublicArray):
            dtype = FakeDtype

            def store(self, *args, **kwargs):
                calls.append(("store-only", args, kwargs))

        StoreOnlyArray.__module__ = "cutlass.base_dsl.array"
        StoreOnlyArray.__qualname__ = "Array"

        class LoadOnlyArray(PublicArray):
            dtype = FakeDtype

            def load(self, *args, **kwargs):
                return args, kwargs

        LoadOnlyArray.__module__ = "cutlass.base_dsl.array"
        LoadOnlyArray.__qualname__ = "Array"

        sys.modules["cutlass"] = types.SimpleNamespace(
            Array=PublicArray,
            cute=types.SimpleNamespace(
                arch=types.SimpleNamespace(
                    thread_idx=lambda: (0, 0, 0),
                    block_dim=lambda: (32, 1, 1),
                )
            ),
        )

        class FakeVector:
            shape = (2,)
            dtype = FakeDtype

            def __getitem__(self, idx):
                return idx

        class OtherVector:
            shape = (2,)
            dtype = OtherDtype

            def __getitem__(self, idx):
                return idx

        def assert_raises(expected, fn):
            try:
                fn()
            except Exception as exc:
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError("expected exception")

        assert_raises(
            "value must be ThreadData",
            lambda: coop._block.store(FakeArray(), object()),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.store cutlass.Array operand must "
                "support store"
            ),
            lambda: coop._block.store(LoadOnlyArray(), FakeVector()),
        )
        for algorithm in (
            "transpose",
            "warp_transpose",
            "warp-transpose-timesliced",
        ):
            expected_algorithm = algorithm.replace("-", "_")
            assert_raises(
                f"algorithm '{expected_algorithm}' is not implemented yet",
                lambda algorithm=algorithm: coop._block.store(
                    FakeArray(),
                    FakeVector(),
                    algorithm=algorithm,
                ),
            )
        assert_raises(
            "accepts only one of valid_items or num_valid_items",
            lambda: coop._warp.store(
                FakeArray(),
                FakeVector(),
                valid_items=1,
                num_valid_items=1,
            ),
        )
        assert_raises(
            "explicit alignment is not implemented",
            lambda: coop._warp.store(
                FakeArray(),
                FakeVector(),
                valid_items=1,
                alignment=8,
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.store bounds_check= is only "
                "accepted when wrapping a source with cutlass.make_array_view"
            ),
            lambda: coop._block.store(
                FakeArray(),
                FakeVector(),
                bounds_check=True,
            ),
        )
        dynamic_threads = DynamicInt()
        assert_raises(
            (
                "cuda.coop.cutlass._block.store threads_per_block must be "
                "a compile-time positive int"
            ),
            lambda: coop._block.store(
                FakeArray(),
                FakeVector(),
                threads_per_block=dynamic_threads,
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.store launch metadata has invalid "
                "thread-count key(s): threads_per_block"
            ),
            lambda: coop._block.store(
                FakeArray(),
                FakeVector(),
                launch_metadata={"threads_per_block": dynamic_threads},
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.store dim must be a compile-time "
                "positive int"
            ),
            lambda: coop._block.store(
                FakeArray(),
                FakeVector(),
                dim=(dynamic_threads, 1, 1),
            ),
        )
        assert_raises(
            "cuda.coop.cutlass._block.store does not accept threads_in_warp",
            lambda: coop._block.store(
                FakeArray(),
                FakeVector(),
                threads_in_warp=16,
            ),
        )
        assert_raises(
            "cuda.coop.cutlass._warp.store threads_in_warp must be an int",
            lambda: coop._warp.store(
                FakeArray(),
                FakeVector(),
                threads_in_warp=dynamic_threads,
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._warp.store does not accept "
                "threads_per_block or dim"
            ),
            lambda: coop._warp.store(
                FakeArray(),
                FakeVector(),
                threads_per_block=32,
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._warp.store does not accept "
                "threads_per_block or dim"
            ),
            lambda: root._warp.store(
                FakeArray(),
                FakeVector(),
                payload=root.Payload.PRIMS,
                dim=(32, 1, 1),
            ),
        )
        assert_raises(
            (
                "cuda.coop.cutlass._block.store value dtype does not match "
                "cutlass.Array dtype"
            ),
            lambda: coop._block.store(FakeArray(), OtherVector()),
        )
        coop._block.store(StoreOnlyArray(), FakeVector(), offset=4)
        assert calls == [
            (
                "store-only",
                (
                    (0, 1),
                    4,
                    2,
                ),
                {
                    "alignment": 8,
                    "is_volatile": False,
                    "is_nontemporal": False,
                    "ordering": "not_atomic",
                    "syncscope": None,
                    "loc": None,
                    "ip": None,
                },
            ),
        ]
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr
