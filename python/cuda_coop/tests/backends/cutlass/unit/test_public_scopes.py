# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support.source import run_python_with_source


def test_cutlass_scoped_secondary_adapters_are_stable():
    script = textwrap.dedent(
        """
        import importlib
        import inspect

        import cuda.coop.cutlass as root

        for scope_name, names in {
            "block": ("make_sum", "make_radix_sort_keys", "make_exchange"),
            "warp": ("make_sum", "make_merge_sort_keys", "make_exchange"),
        }.items():
            root_scope = getattr(root, f"_{scope_name}")
            private_scope = importlib.import_module(
                f"cuda.coop.cutlass._dsl.{scope_name}"
            )
            for name in names:
                first = getattr(root_scope, name)
                second = getattr(root_scope, name)
                assert first is second
                assert first.__module__ == root_scope.__name__
                assert first.__wrapped__ is getattr(private_scope, name)

        histogram_params = inspect.signature(root._block.make_histogram).parameters
        for name in ("bins", "bins_per_thread", "algorithm"):
            assert histogram_params[name].kind is inspect.Parameter.KEYWORD_ONLY

        for scope_name in ("block", "warp"):
            for factory_name in ("make_load", "make_store"):
                factory = getattr(getattr(root, f"_{scope_name}"), factory_name)
                try:
                    factory(object, items_per_thread=object())
                except TypeError as exc:
                    assert str(exc) == (
                        f"cuda.coop.cutlass._{scope_name}.{factory_name} "
                        "items_per_thread must be an int"
                    )
                else:
                    raise AssertionError("expected static items_per_thread error")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_do_not_forward_non_exported_make_helpers():
    script = textwrap.dedent(
        """
        import importlib

        import cuda.coop.cutlass as coop

        cute_block = importlib.import_module("cuda.coop.cutlass._dsl.block")

        def make_private_probe():
            raise AssertionError("should not be forwarded")

        cute_block.make_private_probe = make_private_probe
        assert "make_private_probe" not in cute_block.__all__
        assert "make_private_probe" not in coop._block.__dict__
        assert "make_private_probe" not in dir(coop._block)

        try:
            getattr(coop._block, "make_private_probe")
        except AttributeError as exc:
            assert coop._block.__name__ in str(exc)
        else:
            raise AssertionError("non-exported make_* helper should not forward")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_do_not_forward_backend_private_modules():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        private_names = (
            "_api",
            "_dispatch",
            "_factory",
            "_provider",
            "_sort",
        )
        helper_names = (
            "get_public_attr",
            "install_public_exports",
            "public_dir",
        )

        for scope in (coop._block, coop._warp):
            for name in private_names:
                assert name not in dir(scope), (scope.__name__, name)
                try:
                    getattr(scope, name)
                except AttributeError as exc:
                    assert scope.__name__ in str(exc)
                else:
                    raise AssertionError(
                        f"{scope.__name__} unexpectedly forwarded {name}"
                    )

            for name in helper_names:
                assert name not in dir(scope), (scope.__name__, name)
                assert not hasattr(scope, name), (scope.__name__, name)

            assert scope.make_sum.__module__ == scope.__name__
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_only_forward_exported_names():
    script = textwrap.dedent(
        """
        import importlib

        import cuda.coop.cutlass as root

        private_block = importlib.import_module("cuda.coop.cutlass._dsl.block")
        private_warp = importlib.import_module("cuda.coop.cutlass._dsl.warp")

        def accidental_secondary():
            return "secondary"

        for backend_scope, root_scope in (
            (private_block, root._block),
            (private_warp, root._warp),
        ):
            backend_scope.accidental_helper = object()
            backend_scope.make_accidental_secondary = accidental_secondary

            for name in ("accidental_helper", "make_accidental_secondary"):
                assert name not in dir(root_scope)
                try:
                    getattr(root_scope, name)
                except AttributeError as exc:
                    assert root_scope.__name__ in str(exc)
                else:
                    raise AssertionError(f"unexpectedly forwarded {name}")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scope_forwards_row_sum_scalar_temp_storage():
    script = textwrap.dedent(
        """
        import importlib

        class I32:
            width = 32

        def capture(**payload):
            return payload

        capture._supports_native_thread_data = True

        module_name = "cuda.coop.cutlass"
        coop = importlib.import_module(module_name)
        value = I32()
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("row_sum", capture)
        temp_storage = coop._block.TempStorage(size_in_bytes=20, sharing="shared")

        payload = coop._block.row_sum(
            value,
            rows_per_block=1,
            warps_per_row=4,
            temp_storage=temp_storage,
            launch_metadata={"threads_per_block": 128},
        )

        assert coop._block.row_sum.__module__ == f"{module_name}._block"
        assert temp_storage.scope == "cuda.coop.cutlass"
        assert payload["value"] is value
        assert payload["args"] == ()
        assert payload["rows_per_block"] == 1
        assert payload["warps_per_row"] == 4
        assert len(temp_storage.uses) == 1
        assert temp_storage.uses[0].primitive_name == "row_sum"
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scope_rejects_row_sum_thread_payload():
    script = textwrap.dedent(
        """
        import importlib

        class I32:
            width = 32

        class FakeVector:
            shape = (2,)
            dtype = I32

            def __getitem__(self, idx):
                return idx

        def reject_thread_data(**payload):
            value = payload["value"]
            assert value.items_per_thread == 2
            assert value.dtype is I32
            raise TypeError(
                "cuda.coop.cutlass._dsl.block.row_sum currently expects a scalar value"
            )

        reject_thread_data._supports_native_thread_data = True

        def assert_rejected(module_name, expected_scope, forbidden_scopes):
            coop = importlib.import_module(module_name)
            getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("row_sum", reject_thread_data)
            temp_storage = coop._block.TempStorage(size_in_bytes=20, sharing="shared")

            try:
                coop._block.row_sum(
                    FakeVector(),
                    rows_per_block=1,
                    warps_per_row=4,
                    temp_storage=temp_storage,
                    launch_metadata={"threads_per_block": 128},
                )
            except TypeError as exc:
                message = str(exc)
                assert f"{expected_scope}.row_sum currently expects a scalar value" in message
                for forbidden_scope in forbidden_scopes:
                    assert forbidden_scope not in message
            else:
                raise AssertionError("row_sum accepted a ThreadData payload")

        assert_rejected(
            "cuda.coop.cutlass",
            "cuda.coop.cutlass._block",
            ("cuda.coop.cutlass._dsl.block",),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_scoped_factories_rewrite_returned_factory_scope():
    script = textwrap.dedent(
        """
        import importlib

        class FakeVector:
            shape = (2,)
            dtype = int

            def __init__(self, base):
                self.base = base

            def __getitem__(self, idx):
                return self.base + idx

        class MemoryArray:
            space = object()
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        def capture(**payload):
            return payload

        def assert_thread_data(value, expected):
            assert isinstance(value, coop.ThreadData)
            payloads = value.values("factory")
            assert [payload["keys"] for payload in payloads] == expected

        def assert_memory_payload_rejected(factory, expected_scope, forbidden):
            try:
                factory(MemoryArray())
            except TypeError as exc:
                message = str(exc)
                assert (
                    f"{expected_scope}.radix_sort_keys could not auto-convert "
                    "'keys' payload to ThreadData"
                ) in message
                assert "per-thread register payload" in message
                for scope in forbidden:
                    assert scope not in message
            else:
                raise AssertionError("memory-backed factory payload was accepted")

        def factory_args(scope_name, factory_name):
            if scope_name == "block" and factory_name == "make_histogram":
                return (int, int)
            if scope_name == "block" and factory_name == "make_run_length":
                return (int,)
            return (int,)

        def assert_all_factories_rewritten(coop, expected_scopes, forbidden):
            for scope_name, expected_scope in expected_scopes.items():
                scope = getattr(coop, f"_{scope_name}")
                for name in sorted(
                    candidate for candidate in dir(scope)
                    if candidate.startswith("make_")
                ):
                    factory = getattr(scope, name)(
                        *factory_args(scope_name, name)
                    )
                    assert factory.scope == expected_scope, name
                    assert factory.primitive.__module__ == expected_scope, name
                    for forbidden_scope in forbidden:
                        assert forbidden_scope not in factory.scope, name
                        assert (
                            forbidden_scope not in factory.primitive.__module__
                        ), name

        def exercise(coop_module, expected_block_scope, expected_warp_scope, forbidden):
            global coop
            coop = (
                importlib.import_module(coop_module)
                if isinstance(coop_module, str)
                else coop_module
            )
            assert_all_factories_rewritten(
                coop,
                {
                    "block": expected_block_scope,
                    "warp": expected_warp_scope,
                },
                forbidden,
            )

            getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_sort_keys", capture)
            block_factory = coop._block.make_radix_sort_keys(
                int,
                threads_per_block=32,
            )
            assert block_factory.scope == expected_block_scope
            assert block_factory.primitive.__module__ == expected_block_scope
            block_result = block_factory(FakeVector(10), begin_bit=1, end_bit=7)
            assert_thread_data(block_result, [10, 11])
            for payload in block_result.values("block"):
                assert payload["args"] == ()
                assert payload["begin_bit"] == 1
                assert payload["end_bit"] == 7
                assert payload["descending"] is False
            assert_memory_payload_rejected(
                block_factory,
                expected_block_scope,
                forbidden,
            )

            getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("merge_sort_keys", capture)
            warp_factory = coop._warp.make_merge_sort_keys(
                int,
                threads_in_warp=32,
            )
            assert warp_factory.scope == expected_warp_scope
            assert warp_factory.primitive.__module__ == expected_warp_scope
            warp_result = warp_factory(FakeVector(20))
            assert_thread_data(warp_result, [20, 21])
            for payload in warp_result.values("warp"):
                assert payload["args"] == ()
                assert payload["threads_in_warp"] == 32

        root = importlib.import_module("cuda.coop.cutlass")
        exercise(
            root,
            "cuda.coop.cutlass._block",
            "cuda.coop.cutlass._warp",
            ("cuda.coop.cutlass._dsl",),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_scoped_factory_rewrite_does_not_mutate_backend_factory():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass._scoped_api import _rewrite_return_scope
        from cuda.coop.cutlass._dsl._factory import _PrimitiveFactory

        def primitive(*args, **kwargs):
            return args, kwargs

        backend_factory = _PrimitiveFactory(
            scope="cuda.coop.cutlass._dsl.block",
            factory_name="make_probe",
            primitive=primitive,
            bound_kwargs=(("value", 1),),
        )

        first_factory = _rewrite_return_scope(
            backend_factory,
            source="cuda.coop.cutlass._dsl.block",
            target="cuda.coop.cutlass._block",
            module_name="cuda.coop.cutlass._block",
        )
        second_factory = _rewrite_return_scope(
            backend_factory,
            source="cuda.coop.cutlass._dsl.block",
            target="cuda.coop.cutlass._block",
            module_name="cuda.coop.cutlass._block",
        )

        assert first_factory is not backend_factory
        assert second_factory is not backend_factory
        assert first_factory is not second_factory
        assert backend_factory.scope == "cuda.coop.cutlass._dsl.block"
        assert backend_factory.primitive is primitive
        assert backend_factory.bound_kwargs == (("value", 1),)

        assert first_factory.scope == "cuda.coop.cutlass._block"
        assert first_factory.primitive.__module__ == "cuda.coop.cutlass._block"
        assert first_factory.primitive.__wrapped__ is primitive
        assert first_factory.bound_kwargs == backend_factory.bound_kwargs

        assert second_factory.scope == "cuda.coop.cutlass._block"
        assert second_factory.primitive.__module__ == "cuda.coop.cutlass._block"
        assert second_factory.primitive.__wrapped__ is primitive
        assert second_factory.bound_kwargs == backend_factory.bound_kwargs
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_root_factory_payload_selector_is_deferred_call_default():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop
        from cuda.coop.cutlass import _root_load_store
        from cuda.coop.cutlass._dsl._factory import _PrimitiveFactory

        calls = []

        def primitive(*args, **kwargs):
            calls.append((args, kwargs))
            return args, kwargs

        factory_calls = []

        def make_factory(*args, **kwargs):
            factory_calls.append((args, kwargs))
            assert args == ("factory-arg",)
            assert kwargs == {"valid_items": 4}
            return _PrimitiveFactory(
                scope="cuda.coop.cutlass._block",
                factory_name="make_load",
                primitive=lambda *args, **kwargs: None,
                bound_kwargs=tuple(kwargs.items()),
                overridable_kwargs=("valid_items",),
            )

        make_factory.__module__ = "cuda.coop.cutlass._block"
        make_factory.__name__ = "make_load"

        factory = _root_load_store.root_factory(
            make_factory,
            primitive,
            "factory-arg",
            payload=coop.Payload.PRIMS,
            valid_items=4,
        )

        assert factory.primitive is primitive
        assert factory.bound_kwargs == (
            ("payload", coop.Payload.PRIMS),
            ("valid_items", 4),
        )
        assert factory.overridable_kwargs == ("valid_items", "payload")
        assert factory_calls == [(("factory-arg",), {"valid_items": 4})]

        args, kwargs = factory("source")
        assert args == ("source",)
        assert kwargs == {"payload": coop.Payload.PRIMS, "valid_items": 4}

        args, kwargs = factory("source", payload=coop.Payload.PRIMS, valid_items=8)
        assert args == ("source",)
        assert kwargs == {"payload": coop.Payload.PRIMS, "valid_items": 8}
        assert calls == [
            (("source",), {"payload": coop.Payload.PRIMS, "valid_items": 4}),
            (("source",), {"payload": coop.Payload.PRIMS, "valid_items": 8}),
        ]

        string_factory = _root_load_store.root_factory(
            make_factory,
            primitive,
            "factory-arg",
            payload="prims",
            valid_items=4,
        )
        assert string_factory.bound_kwargs == (
            ("payload", coop.Payload.PRIMS),
            ("valid_items", 4),
        )

        try:
            _root_load_store.root_factory(
                make_factory,
                primitive,
                "factory-arg",
                payload="bogus",
                valid_items=4,
            )
        except ValueError as exc:
            assert (
                    "cuda.coop.cutlass._block.make_load payload must be prims"
            ) in str(exc)
        else:
            raise AssertionError("expected invalid factory payload selector error")

        assert factory_calls == [
            (("factory-arg",), {"valid_items": 4}),
            (("factory-arg",), {"valid_items": 4}),
        ]
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_tensor_load_store_is_default():
    script = textwrap.dedent(
        """
        import sys

        cutlass_module = sys.modules["cutlass"]
        cute_module = sys.modules["cutlass.cute"]

        class Arch:
            @staticmethod
            def thread_idx():
                return 0, 0, 0

            @staticmethod
            def block_dim():
                return 8, 1, 1

        cute_module.arch = Arch
        cutlass_module.cute = cute_module

        import cuda.coop.cutlass as coop

        source = list(range(16))
        loaded = coop._block.load(source, items_per_thread=2)
        assert loaded.values("block-load") == (0, 1)

        output = [None] * 16
        coop._block.store(output, coop.ThreadData.from_values(10, 11))
        assert output[:2] == [10, 11]

        block_load = coop._block.make_load(items_per_thread=2)
        assert all(name != "payload" for name, _ in block_load.bound_kwargs)
        assert block_load(source).values("factory") == (0, 1)

        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_scoped_factories_forward_runtime_value_operands():
    script = textwrap.dedent(
        """
        import importlib

        class DynamicInt:
            def __init__(self, name):
                self.name = name

        class FakeVector:
            shape = (2,)
            dtype = int

            def __init__(self, base):
                self.base = base

            def __getitem__(self, idx):
                return self.base + idx

        def capture(**payload):
            return payload

        pair_payloads = []

        def capture_pairs(**payload):
            pair_payloads.append(payload)
            return payload["keys"], payload["values"]

        begin_bit = DynamicInt("begin")
        end_bit = DynamicInt("end")
        runtime_k = DynamicInt("k")
        runtime_valid = DynamicInt("valid")
        factory_begin_bit = DynamicInt("factory-begin")
        factory_end_bit = DynamicInt("factory-end")
        factory_radix_bits = DynamicInt("factory-radix-bits")
        factory_valid = DynamicInt("factory-valid")

        def assert_common_payload(
            payload,
            expected_key,
            descending,
            expected_begin_bit=begin_bit,
            expected_end_bit=end_bit,
        ):
            assert payload["keys"] == expected_key
            assert payload["args"] == ()
            assert payload["begin_bit"] is expected_begin_bit
            assert payload["end_bit"] is expected_end_bit
            assert payload["descending"] is descending
            assert payload["launch_metadata"] == {"threads_per_block": 32}

        def exercise(module_name, expected_scope):
            coop = importlib.import_module(module_name)
            block = coop._block

            getattr(block, "_backend", block)._api.register_provider_impl("radix_sort_keys", capture)
            radix_sort = block.make_radix_sort_keys(
                int,
                threads_per_block=32,
                begin_bit=factory_begin_bit,
                end_bit=factory_end_bit,
                descending=False,
            )
            assert radix_sort.scope == expected_scope
            assert radix_sort.primitive.__module__ == expected_scope
            factory_sorted_keys = radix_sort(FakeVector(8))
            assert isinstance(factory_sorted_keys, coop.ThreadData)
            assert factory_sorted_keys.dtype is int
            for payload, expected_key in zip(
                factory_sorted_keys.values("factory-radix-default"),
                (8, 9),
            ):
                assert_common_payload(
                    payload,
                    expected_key,
                    False,
                    factory_begin_bit,
                    factory_end_bit,
                )
            sorted_keys = radix_sort(
                FakeVector(10),
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=True,
            )
            assert isinstance(sorted_keys, coop.ThreadData)
            assert sorted_keys.dtype is int
            for payload, expected_key in zip(
                sorted_keys.values("factory-radix"),
                (10, 11),
            ):
                assert_common_payload(payload, expected_key, True)

            getattr(block, "_backend", block)._api.register_provider_impl("topk_max_keys", capture)
            topk_keys = block.make_topk_max_keys(
                int,
                threads_per_block=32,
                num_valid=factory_valid,
                begin_bit=factory_begin_bit,
                end_bit=factory_end_bit,
            )
            factory_top_keys = topk_keys(FakeVector(18), runtime_k)
            assert isinstance(factory_top_keys, coop.ThreadData)
            assert factory_top_keys.dtype is int
            for payload, expected_key in zip(
                factory_top_keys.values("factory-topk-default"),
                (18, 19),
            ):
                assert_common_payload(
                    payload,
                    expected_key,
                    True,
                    factory_begin_bit,
                    factory_end_bit,
                )
                assert payload["k"] is runtime_k
                assert payload["num_valid"] is factory_valid
            top_keys = topk_keys(
                FakeVector(20),
                runtime_k,
                num_valid=runtime_valid,
                begin_bit=begin_bit,
                end_bit=end_bit,
            )
            assert isinstance(top_keys, coop.ThreadData)
            assert top_keys.dtype is int
            for payload, expected_key in zip(
                top_keys.values("factory-topk"),
                (20, 21),
            ):
                assert_common_payload(payload, expected_key, True)
                assert payload["k"] is runtime_k
                assert payload["num_valid"] is runtime_valid

            getattr(block, "_backend", block)._api.register_provider_impl("sum", capture)
            sum_factory = block.make_sum(
                int,
                threads_per_block=32,
                num_valid=factory_valid,
            )
            assert sum_factory.scope == expected_scope
            assert sum_factory.primitive.__module__ == expected_scope
            factory_sum_payload = sum_factory(100)
            assert factory_sum_payload["value"] == 100
            assert factory_sum_payload["args"] == ()
            assert factory_sum_payload["num_valid"] is factory_valid
            assert factory_sum_payload["launch_metadata"] == {"threads_per_block": 32}
            sum_payload = sum_factory(101, valid_items=runtime_valid)
            assert sum_payload["value"] == 101
            assert sum_payload["args"] == ()
            assert sum_payload["valid_items"] is runtime_valid
            assert "num_valid" not in sum_payload
            assert sum_payload["launch_metadata"] == {"threads_per_block": 32}

            getattr(block, "_backend", block)._api.register_provider_impl("reduce", capture)
            reduce_factory = block.make_reduce(
                int,
                threads_per_block=32,
                binary_op="max",
                num_valid=factory_valid,
            )
            assert reduce_factory.scope == expected_scope
            assert reduce_factory.primitive.__module__ == expected_scope
            factory_reduce_payload = reduce_factory(102)
            assert factory_reduce_payload["value"] == 102
            assert factory_reduce_payload["args"] == ()
            assert factory_reduce_payload["binary_op"] == "max"
            assert factory_reduce_payload["num_valid"] is factory_valid
            assert factory_reduce_payload["launch_metadata"] == {
                "threads_per_block": 32
            }
            reduce_payload = reduce_factory(103, num_valid=runtime_valid)
            assert reduce_payload["value"] == 103
            assert reduce_payload["args"] == ()
            assert reduce_payload["binary_op"] == "max"
            assert reduce_payload["num_valid"] is runtime_valid
            assert reduce_payload["launch_metadata"] == {"threads_per_block": 32}

            getattr(block, "_backend", block)._api.register_provider_impl("exclusive_sum", capture)
            runtime_block_aggregate = object()
            exclusive_sum_factory = block.make_exclusive_sum(
                int,
                threads_per_block=32,
            )
            exclusive_sum_payload = exclusive_sum_factory(
                104,
                block_aggregate=runtime_block_aggregate,
            )
            assert exclusive_sum_payload == {
                "value": 104,
                "args": (),
                "block_aggregate": runtime_block_aggregate,
                "launch_metadata": {"threads_per_block": 32},
            }

            getattr(block, "_backend", block)._api.register_provider_impl("inclusive_scan", capture)
            runtime_scan_aggregate = object()
            inclusive_scan_factory = block.make_inclusive_scan(
                int,
                scan_op="max",
                threads_per_block=32,
            )
            inclusive_scan_payload = inclusive_scan_factory(
                105,
                block_aggregate=runtime_scan_aggregate,
            )
            assert inclusive_scan_payload == {
                "value": 105,
                "args": (),
                "scan_op": "max",
                "initial_value": None,
                "block_aggregate": runtime_scan_aggregate,
                "launch_metadata": {"threads_per_block": 32},
            }

            getattr(block, "_backend", block)._api.register_provider_impl("topk_min_pairs", capture_pairs)
            topk_pairs = block.make_topk_min_pairs(
                int,
                int,
                threads_per_block=32,
                num_valid=factory_valid,
                begin_bit=factory_begin_bit,
                end_bit=factory_end_bit,
            )
            factory_top_pair_keys, factory_top_pair_values = topk_pairs(
                FakeVector(28),
                FakeVector(48),
                runtime_k,
            )
            assert isinstance(factory_top_pair_keys, coop.ThreadData)
            assert isinstance(factory_top_pair_values, coop.ThreadData)
            assert factory_top_pair_keys.values("factory-topk-pair-defaults") == (28, 29)
            assert factory_top_pair_values.values("factory-topk-value-defaults") == (
                48,
                49,
            )
            assert len(pair_payloads) == 2
            for payload, expected_key, expected_value in zip(
                pair_payloads,
                (28, 29),
                (48, 49),
            ):
                assert_common_payload(
                    payload,
                    expected_key,
                    False,
                    factory_begin_bit,
                    factory_end_bit,
                )
                assert payload["values"] == expected_value
                assert payload["k"] is runtime_k
                assert payload["num_valid"] is factory_valid
            pair_payloads.clear()
            top_pair_keys, top_pair_values = topk_pairs(
                FakeVector(30),
                FakeVector(50),
                runtime_k,
                num_valid=runtime_valid,
                begin_bit=begin_bit,
                end_bit=end_bit,
            )
            assert isinstance(top_pair_keys, coop.ThreadData)
            assert isinstance(top_pair_values, coop.ThreadData)
            assert top_pair_keys.values("factory-topk-pairs") == (30, 31)
            assert top_pair_values.values("factory-topk-values") == (50, 51)
            assert len(pair_payloads) == 2
            for payload, expected_key, expected_value in zip(
                pair_payloads,
                (30, 31),
                (50, 51),
            ):
                assert_common_payload(payload, expected_key, False)
                assert payload["values"] == expected_value
                assert payload["k"] is runtime_k
                assert payload["num_valid"] is runtime_valid
            pair_payloads.clear()

            getattr(block, "_backend", block)._api.register_provider_impl("radix_rank", capture)
            radix_rank = block.make_radix_rank(
                int,
                threads_per_block=32,
                begin_bit=factory_begin_bit,
                radix_bits=factory_radix_bits,
                descending=True,
            )
            factory_ranks = radix_rank(FakeVector(68))
            assert isinstance(factory_ranks, coop.ThreadData)
            assert factory_ranks.dtype is int
            for payload, expected_key in zip(
                factory_ranks.values("factory-rank-default"),
                (68, 69),
            ):
                assert payload["keys"] == expected_key
                assert payload["args"] == ()
                assert payload["begin_bit"] is factory_begin_bit
                assert payload["end_bit"] is None
                assert payload["radix_bits"] is factory_radix_bits
                assert payload["descending"] is True
                assert payload["launch_metadata"] == {"threads_per_block": 32}
            radix_bits_ranks = radix_rank(
                FakeVector(70),
                begin_bit=begin_bit,
                radix_bits=factory_radix_bits,
                descending=False,
            )
            assert isinstance(radix_bits_ranks, coop.ThreadData)
            assert radix_bits_ranks.dtype is int
            for payload, expected_key in zip(
                radix_bits_ranks.values("factory-rank-bits"),
                (70, 71),
            ):
                assert payload["keys"] == expected_key
                assert payload["args"] == ()
                assert payload["begin_bit"] is begin_bit
                assert payload["end_bit"] is None
                assert payload["radix_bits"] is factory_radix_bits
                assert payload["descending"] is False
                assert payload["launch_metadata"] == {"threads_per_block": 32}
            end_bit_ranks = radix_rank(
                FakeVector(72),
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=False,
            )
            assert isinstance(end_bit_ranks, coop.ThreadData)
            assert end_bit_ranks.dtype is int
            for payload, expected_key in zip(
                end_bit_ranks.values("factory-rank-end-bit"),
                (72, 73),
            ):
                assert payload["keys"] == expected_key
                assert payload["args"] == ()
                assert payload["begin_bit"] is begin_bit
                assert payload["end_bit"] is end_bit
                assert payload["radix_bits"] is None
                assert payload["descending"] is False
                assert payload["launch_metadata"] == {"threads_per_block": 32}

            getattr(block, "_backend", block)._api.register_provider_impl("merge_sort_keys", capture)
            merge_sort_keys = block.make_merge_sort_keys(
                int,
                threads_per_block=32,
                descending=True,
                valid_items=factory_valid,
                oob_default=-5,
            )
            factory_merge_keys = merge_sort_keys(FakeVector(74))
            assert isinstance(factory_merge_keys, coop.ThreadData)
            assert factory_merge_keys.dtype is int
            for payload, expected_key in zip(
                factory_merge_keys.values("factory-merge-key-defaults"),
                (74, 75),
            ):
                assert payload["keys"] == expected_key
                assert payload["args"] == ()
                assert payload["descending"] is True
                assert payload["valid_items"] is factory_valid
                assert payload["oob_default"] == -5
                assert payload["launch_metadata"] == {"threads_per_block": 32}
            override_merge_keys = merge_sort_keys(
                FakeVector(78),
                valid_items=runtime_valid,
                oob_default=-7,
            )
            assert isinstance(override_merge_keys, coop.ThreadData)
            assert override_merge_keys.dtype is int
            for payload, expected_key in zip(
                override_merge_keys.values("factory-merge-key-overrides"),
                (78, 79),
            ):
                assert payload["keys"] == expected_key
                assert payload["args"] == ()
                assert payload["descending"] is True
                assert payload["valid_items"] is runtime_valid
                assert payload["oob_default"] == -7
                assert payload["launch_metadata"] == {"threads_per_block": 32}

            getattr(block, "_backend", block)._api.register_provider_impl("merge_sort_pairs", capture_pairs)
            merge_sort_pairs = block.make_merge_sort_pairs(
                int,
                int,
                threads_per_block=32,
                descending=True,
                valid_items=factory_valid,
                oob_default=-3,
            )
            factory_merge_pair_keys, factory_merge_pair_values = merge_sort_pairs(
                FakeVector(76),
                FakeVector(96),
            )
            assert isinstance(factory_merge_pair_keys, coop.ThreadData)
            assert isinstance(factory_merge_pair_values, coop.ThreadData)
            assert factory_merge_pair_keys.values("factory-merge-pair-defaults") == (
                76,
                77,
            )
            assert factory_merge_pair_values.values("factory-merge-value-defaults") == (
                96,
                97,
            )
            assert len(pair_payloads) == 2
            for payload, expected_key, expected_value in zip(
                pair_payloads,
                (76, 77),
                (96, 97),
            ):
                assert payload["keys"] == expected_key
                assert payload["values"] == expected_value
                assert payload["args"] == ()
                assert payload["descending"] is True
                assert payload["valid_items"] is factory_valid
                assert payload["oob_default"] == -3
                assert payload["launch_metadata"] == {"threads_per_block": 32}
            pair_payloads.clear()
            merge_pair_keys, merge_pair_values = merge_sort_pairs(
                FakeVector(80),
                FakeVector(100),
                valid_items=runtime_valid,
                oob_default=-1,
            )
            assert isinstance(merge_pair_keys, coop.ThreadData)
            assert isinstance(merge_pair_values, coop.ThreadData)
            assert merge_pair_keys.values("factory-merge-pairs") == (80, 81)
            assert merge_pair_values.values("factory-merge-values") == (100, 101)
            assert len(pair_payloads) == 2
            for payload, expected_key, expected_value in zip(
                pair_payloads,
                (80, 81),
                (100, 101),
            ):
                assert payload["keys"] == expected_key
                assert payload["values"] == expected_value
                assert payload["args"] == ()
                assert payload["descending"] is True
                assert payload["valid_items"] is runtime_valid
                assert payload["oob_default"] == -1
                assert payload["launch_metadata"] == {"threads_per_block": 32}
            pair_payloads.clear()

        exercise("cuda.coop.cutlass", "cuda.coop.cutlass._block")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_warp_factories_forward_context_kwargs():
    script = textwrap.dedent(
        """
        import importlib

        FACTORIES = {
            "make_sum": "sum",
            "make_max": "max",
            "make_min": "min",
            "make_exclusive_sum": "exclusive_sum",
            "make_inclusive_sum": "inclusive_sum",
            "make_exclusive_scan": "exclusive_scan",
            "make_inclusive_scan": "inclusive_scan",
        }

        def assert_message(fn, expected, forbidden=()):
            try:
                fn()
            except Exception as exc:
                message = str(exc)
                assert expected in message, message
                for token in forbidden:
                    assert token not in message, message
            else:
                raise AssertionError("expected diagnostic was not raised")

        class DynamicInt:
            def __init__(self, name):
                self.name = name

        def capture(**payload):
            return payload

        def capture_native(**payload):
            return payload

        capture_native._supports_native_thread_data = True

        def exercise(module_name, expected_scope, forbidden):
            coop = importlib.import_module(module_name)
            warp = coop._warp
            runtime_valid = DynamicInt("valid_items")
            factory_valid = DynamicInt("factory-valid-items")

            assert warp.__name__ == expected_scope

            getattr(warp, "_backend", warp)._api.register_provider_impl("sum", capture)
            sum_factory = warp.make_sum(
                int,
                threads_in_warp=16,
                valid_items=factory_valid,
            )
            assert sum_factory.scope == expected_scope
            assert sum_factory.primitive.__module__ == expected_scope
            sum_payload = sum_factory(5)
            assert sum_payload == {
                "value": 5,
                "args": (),
                "threads_in_warp": 16,
                "valid_items": factory_valid,
            }
            sum_override_payload = sum_factory(6, valid_items=runtime_valid)
            assert sum_override_payload == {
                "value": 6,
                "args": (),
                "threads_in_warp": 16,
                "valid_items": runtime_valid,
            }

            getattr(warp, "_backend", warp)._api.register_provider_impl("max", capture)
            max_factory = warp.make_max(
                int,
                threads_in_warp=8,
                valid_items=factory_valid,
            )
            assert max_factory.scope == expected_scope
            assert max_factory.primitive.__module__ == expected_scope
            max_payload = max_factory(7)
            assert max_payload == {
                "value": 7,
                "args": (),
                "threads_in_warp": 8,
                "valid_items": factory_valid,
            }
            max_override_payload = max_factory(8, valid_items=runtime_valid)
            assert max_override_payload == {
                "value": 8,
                "args": (),
                "threads_in_warp": 8,
                "valid_items": runtime_valid,
            }

            getattr(warp, "_backend", warp)._api.register_provider_impl("sum", capture_native)
            native_sum_factory = warp.make_sum(
                int,
                threads_in_warp=16,
                valid_items=factory_valid,
            )
            thread_items = coop.ThreadData.from_values(10, 11, dtype=int)
            native_sum_payload = native_sum_factory(thread_items)
            assert native_sum_payload["value"] is thread_items
            assert native_sum_payload["args"] == ()
            assert native_sum_payload["threads_in_warp"] == 16
            assert native_sum_payload["valid_items"] is factory_valid
            native_sum_override_payload = native_sum_factory(
                thread_items,
                valid_items=runtime_valid,
            )
            assert native_sum_override_payload["value"] is thread_items
            assert native_sum_override_payload["args"] == ()
            assert native_sum_override_payload["threads_in_warp"] == 16
            assert native_sum_override_payload["valid_items"] is runtime_valid

            getattr(warp, "_backend", warp)._api.register_provider_impl("reduce", capture)
            reduce_factory = warp.make_reduce(
                int,
                binary_op="bit_xor",
                threads_in_warp=32,
                valid_items=factory_valid,
            )
            reduce_payload = reduce_factory(9)
            assert reduce_payload == {
                "value": 9,
                "args": (),
                "binary_op": "bit_xor",
                "threads_in_warp": 32,
                "valid_items": factory_valid,
            }
            reduce_override_payload = reduce_factory(10, valid_items=runtime_valid)
            assert reduce_override_payload == {
                "value": 10,
                "args": (),
                "binary_op": "bit_xor",
                "threads_in_warp": 32,
                "valid_items": runtime_valid,
            }

            getattr(warp, "_backend", warp)._api.register_provider_impl("min", capture)
            min_factory = warp.make_min(
                int,
                threads_in_warp=4,
                valid_items=factory_valid,
            )
            min_payload = min_factory(11)
            assert min_payload == {
                "value": 11,
                "args": (),
                "threads_in_warp": 4,
                "valid_items": factory_valid,
            }
            min_override_payload = min_factory(12, valid_items=runtime_valid)
            assert min_override_payload == {
                "value": 12,
                "args": (),
                "threads_in_warp": 4,
                "valid_items": runtime_valid,
            }

            getattr(warp, "_backend", warp)._api.register_provider_impl("exclusive_sum", capture)
            runtime_aggregate = object()
            exclusive_sum_factory = warp.make_exclusive_sum(
                int,
                threads_in_warp=16,
                valid_items=factory_valid,
            )
            exclusive_sum_payload = exclusive_sum_factory(13)
            assert exclusive_sum_payload == {
                "value": 13,
                "args": (),
                "threads_in_warp": 16,
                "valid_items": factory_valid,
            }
            exclusive_sum_override_payload = exclusive_sum_factory(
                14,
                valid_items=runtime_valid,
            )
            assert exclusive_sum_override_payload == {
                "value": 14,
                "args": (),
                "threads_in_warp": 16,
                "valid_items": runtime_valid,
            }
            exclusive_sum_aggregate_payload = exclusive_sum_factory(
                14,
                valid_items=runtime_valid,
                warp_aggregate=runtime_aggregate,
            )
            assert exclusive_sum_aggregate_payload == {
                "value": 14,
                "args": (),
                "threads_in_warp": 16,
                "valid_items": runtime_valid,
                "warp_aggregate": runtime_aggregate,
            }

            getattr(warp, "_backend", warp)._api.register_provider_impl("inclusive_scan", capture)
            runtime_scan_aggregate = object()
            inclusive_scan_factory = warp.make_inclusive_scan(
                int,
                scan_op="max",
                threads_in_warp=16,
                valid_items=factory_valid,
            )
            inclusive_scan_payload = inclusive_scan_factory(15)
            assert inclusive_scan_payload == {
                "value": 15,
                "args": (),
                "scan_op": "max",
                "initial_value": None,
                "threads_in_warp": 16,
                "valid_items": factory_valid,
            }
            inclusive_scan_override_payload = inclusive_scan_factory(
                16,
                valid_items=runtime_valid,
            )
            assert inclusive_scan_override_payload == {
                "value": 16,
                "args": (),
                "scan_op": "max",
                "initial_value": None,
                "threads_in_warp": 16,
                "valid_items": runtime_valid,
            }
            inclusive_scan_aggregate_payload = inclusive_scan_factory(
                16,
                valid_items=runtime_valid,
                warp_aggregate=runtime_scan_aggregate,
            )
            assert inclusive_scan_aggregate_payload == {
                "value": 16,
                "args": (),
                "scan_op": "max",
                "initial_value": None,
                "threads_in_warp": 16,
                "valid_items": runtime_valid,
                "warp_aggregate": runtime_scan_aggregate,
            }

            for factory_name, primitive_name in FACTORIES.items():
                temp_storage = warp.TempStorage(size_in_bytes=1)
                factory = getattr(warp, factory_name)(
                    int,
                    threads_in_warp=32,
                    temp_storage=temp_storage,
                )
                assert factory.scope == expected_scope
                assert factory.primitive.__module__ == expected_scope
                assert_message(
                    lambda factory=factory: factory(1),
                    (
                        f"{expected_scope}.{primitive_name} does not support "
                        "explicit TempStorage planning yet"
                    ),
                    forbidden=forbidden,
                )

                assert_message(
                    lambda factory_name=factory_name: getattr(warp, factory_name)(
                        int, methods=object()
                    ),
                    (
                        f"{expected_scope}.{factory_name} does not support "
                        "compatibility factory argument 'methods'"
                    ),
                    forbidden=forbidden,
                )

        exercise(
            "cuda.coop.cutlass",
            "cuda.coop.cutlass._warp",
            ("cuda.coop.cutlass._dsl.warp",),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_diagnostics_use_private_scope():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def assert_message(fn, expected):
            try:
                fn()
            except Exception as exc:
                message = str(exc)
                assert expected in message, message
                assert "cuda.coop.cutlass._dsl" not in message
            else:
                raise AssertionError("expected diagnostic was not raised")

        assert_message(
            lambda: coop._block.store([], object()),
            "cuda.coop.cutlass._block.store value must be ThreadData",
        )
        assert_message(
            lambda: coop._block.load([], items_per_thread=2, oob_default=0),
            "cuda.coop.cutlass._block.load oob_default requires valid_items",
        )
        for scope in (coop._block, coop._warp):
            assert_message(
                lambda scope=scope: getattr(scope, "definitely_missing"),
                f"module {scope.__name__!r} has no attribute "
                + repr("definitely_missing"),
            )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_diagnostics_rewrite_backend_root_thread_data_mentions():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def fail_with_backend_scope(**_payload):
            raise TypeError(
                "pass a cuda.coop.cutlass._dsl.ThreadData value through "
                "cuda.coop.cutlass._dsl.block.radix_sort_keys"
            )

        def assert_message(fn, expected, forbidden=()):
            try:
                fn()
            except TypeError as exc:
                message = str(exc)
                assert expected in message, message
                for token in forbidden:
                    assert token not in message, message
            else:
                raise AssertionError("expected diagnostic was not raised")

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_sort_keys", fail_with_backend_scope)
        assert_message(
            lambda: coop._block.radix_sort_keys(1),
            "pass a cuda.coop.cutlass.ThreadData value through "
            "cuda.coop.cutlass._block.radix_sort_keys",
            forbidden=("cuda.coop.cutlass._dsl",),
        )

        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr
