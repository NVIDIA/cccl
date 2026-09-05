# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register-payload adapter coverage for CUTLASS group and scoped APIs."""

from __future__ import annotations

import textwrap

from ..support.source import run_python_with_source as _run_python_with_source

_FAKE_CUTLASS_REDUCE_PREAMBLE = textwrap.dedent(
    """
    import sys
    import types

    class AddressSpace:
        rmem = object()
        gmem = object()

    class DialectAddressSpace:
        rmem = object()
        gmem = object()

    for module_name in (
        "cutlass",
        "cutlass._mlir",
        "cutlass._mlir.dialects",
    ):
        module = types.ModuleType(module_name)
        module.__path__ = []
        sys.modules[module_name] = module
    sys.modules["cutlass"].AddressSpace = AddressSpace

    cute_module = types.ModuleType("cutlass._mlir.dialects.cute")
    cute_module.AddressSpace = DialectAddressSpace
    sys.modules[cute_module.__name__] = cute_module

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass._dsl import _launch

    _launch.current_kernel_launch_facts = lambda: LaunchFacts(exact_block_dim=64)

    provider_calls = []
    provider = types.ModuleType("cuda.coop.cutlass._dsl._cudax_reduce_provider")

    def provider_reduce(**kwargs):
        provider_calls.append(kwargs)
        return kwargs["value"]

    provider.provider_reduce = provider_reduce
    sys.modules[provider.__name__] = provider
    """
)


_FAKE_CUTLASS_EXPORT_PREAMBLE = textwrap.dedent(
    """
    import math
    import sys
    import types

    class Numeric:
        def __init__(self, value):
            self.value = value.value if isinstance(value, Numeric) else value

        def __eq__(self, other):
            return type(self) is type(other) and self.value == other.value

    class Int32(Numeric):
        pass

    class Float32(Numeric):
        pass

    Numeric.__module__ = "cutlass.base_dsl.typing"
    Int32.__module__ = "cutlass.base_dsl.typing"
    Float32.__module__ = "cutlass.base_dsl.typing"

    class AddressSpace:
        rmem = object()

    class VectorValue:
        def __init__(self, values, dtype):
            self.values = values
            self.dtype = dtype

    class Vector:
        calls = []

        @staticmethod
        def from_elements(values, dtype):
            values = tuple(
                value if isinstance(value, dtype) else dtype(value)
                for value in values
            )
            Vector.calls.append((values, dtype))
            return VectorValue(values, dtype)

    class TensorSSA:
        def __init__(self, vector, shape, dtype):
            self._values = vector.values
            self.shape = shape
            self.dtype = dtype

        def numel(self):
            return len(self._values)

        def __getitem__(self, idx):
            return self._values[idx]

    class RegisterTensor:
        memspace = AddressSpace.rmem

        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.element_type = dtype
            self._values = [None] * math.prod(
                dimension
                for mode in shape
                for dimension in (mode if isinstance(mode, tuple) else (mode,))
            )

        def store(self, value):
            self._values[:] = value._values

        def __getitem__(self, idx):
            return self._values[idx]

        def __setitem__(self, idx, value):
            self._values[idx] = value

    cutlass_module = sys.modules["cutlass"]
    cutlass_module.AddressSpace = AddressSpace
    cutlass_module.Vector = Vector

    base_module = sys.modules["cutlass.base_dsl"]
    typing_module = types.ModuleType("cutlass.base_dsl.typing")
    typing_module.Numeric = Numeric
    sys.modules[typing_module.__name__] = typing_module
    base_module.typing = typing_module

    cute_module = sys.modules["cutlass.cute"]
    cute_module.TensorSSA = TensorSSA
    cute_module.make_rmem_tensor_like = (
        lambda value: RegisterTensor(value.shape, value.dtype)
    )
    sys.modules[cute_module.__name__] = cute_module
    cutlass_module.cute = cute_module

    import cuda.coop.cutlass as coop
    """
)


def test_thread_data_exports_tensorssa_and_fresh_register_tensor():
    script = _FAKE_CUTLASS_EXPORT_PREAMBLE + textwrap.dedent(
        """
        values = tuple(Int32(value) for value in (10, 20, 30, 40))
        thread_data = coop.ThreadData.from_values(*values, dtype=Int32)

        flat_ssa = thread_data.to_tensor_ssa()
        assert flat_ssa.shape == (4,)
        assert flat_ssa.dtype is Int32
        assert tuple(flat_ssa[idx] for idx in range(4)) == values

        nested_ssa = thread_data.to_tensor_ssa(shape=(2, 2))
        assert nested_ssa.shape == (2, 2)
        assert tuple(nested_ssa[idx] for idx in range(4)) == values
        ssa_round_trip = coop.ThreadData.from_payload(nested_ssa)
        assert ssa_round_trip.dtype is Int32
        assert ssa_round_trip.values("round-trip") == values

        cast_ssa = thread_data.to_tensor_ssa(dtype=Float32)
        assert cast_ssa.dtype is Float32
        assert tuple(value.value for value in cast_ssa._values) == (10, 20, 30, 40)
        assert all(isinstance(value, Float32) for value in cast_ssa._values)

        class ProducerShape:
            shape = ((2, 1), 2)
            dtype = Float32

        like_ssa = thread_data.to_tensor_ssa(like=ProducerShape())
        assert like_ssa.shape == ((2, 1), 2)
        assert like_ssa.dtype is Float32
        assert tuple(value.value for value in like_ssa._values) == (10, 20, 30, 40)

        explicit_ssa = thread_data.to_tensor_ssa(
            dtype=Int32,
            shape=(4,),
            like=ProducerShape(),
        )
        assert explicit_ssa.shape == (4,)
        assert explicit_ssa.dtype is Int32

        register_tensor = thread_data.to_register_tensor(shape=((2, 1), 2))
        assert register_tensor.shape == ((2, 1), 2)
        assert register_tensor.memspace is AddressSpace.rmem
        assert tuple(register_tensor[idx] for idx in range(4)) == values
        rmem_round_trip = coop.ThreadData.from_payload(register_tensor)
        assert rmem_round_trip.values("round-trip") == values

        register_tensor[0] = Int32(99)
        assert thread_data[0] == Int32(10)
        assert nested_ssa[0] == Int32(10)
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_export_validation_and_domain_safety():
    script = _FAKE_CUTLASS_EXPORT_PREAMBLE + textwrap.dedent(
        """
        from cuda.coop._core import ResultVisibility
        from cuda.coop.cutlass._value_metadata import (
            DefinedThreadConstraint,
            DefinedThreadDomain,
            DefinedThreadDomainKind,
            ValueGroupMetadata,
            attach_thread_data_metadata,
        )

        def assert_raises(exc_type, message, fn):
            try:
                fn()
            except exc_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"{exc_type.__name__} was not raised")

        initialized = coop.ThreadData.from_values(Int32(1), Int32(2), dtype=Int32)
        for shape in ((), (2, object()), (3,)):
            assert_raises(
                ValueError,
                "shape",
                lambda shape=shape: initialized.to_tensor_ssa(shape=shape),
            )
        assert Vector.calls == []

        assert_raises(
            TypeError,
            "requires dtype",
            lambda: coop.ThreadData.from_values(1, 2).to_tensor_ssa(),
        )
        assert_raises(
            TypeError,
            "CUTLASS Numeric",
            lambda: initialized.to_tensor_ssa(dtype=object),
        )
        assert_raises(
            ValueError,
            "initialized before use",
            lambda: coop.ThreadData(2, dtype=Int32).to_tensor_ssa(),
        )

        class MissingLikeMetadata:
            pass

        class NoneLikeMetadata:
            shape = None
            dtype = None

        class InaccessibleLikeMetadata:
            @property
            def shape(self):
                raise RuntimeError("shape unavailable")

            @property
            def dtype(self):
                raise RuntimeError("dtype unavailable")

        assert_raises(
            TypeError,
            "accessible dtype metadata",
            lambda: initialized.to_tensor_ssa(like=MissingLikeMetadata()),
        )
        assert_raises(
            TypeError,
            "non-None dtype metadata",
            lambda: initialized.to_tensor_ssa(like=NoneLikeMetadata()),
        )
        assert_raises(
            TypeError,
            "accessible dtype metadata",
            lambda: initialized.to_tensor_ssa(like=InaccessibleLikeMetadata()),
        )
        assert_raises(
            TypeError,
            "accessible shape metadata",
            lambda: initialized.to_tensor_ssa(
                dtype=Int32,
                like=MissingLikeMetadata(),
            ),
        )
        assert initialized.to_tensor_ssa(
            dtype=Int32,
            shape=(2,),
            like=InaccessibleLikeMetadata(),
        ).shape == (2,)

        constrained = coop.ThreadData.from_values(Int32(1), Int32(2), dtype=Int32)
        attach_thread_data_metadata(
            constrained,
            ValueGroupMetadata(
                DefinedThreadDomain(
                    frozenset(
                        {
                            DefinedThreadConstraint(
                                DefinedThreadDomainKind.ROOTS,
                                ("block", 32),
                            )
                        }
                    )
                ),
                ResultVisibility.GROUP_ROOT,
            ),
        )
        for export in (constrained.to_tensor_ssa, constrained.to_register_tensor):
            assert_raises(
                ValueError,
                "must remain ThreadData",
                export,
            )

        all_callers = coop.ThreadData.from_values(Int32(3), Int32(4), dtype=Int32)
        attach_thread_data_metadata(
            all_callers,
            ValueGroupMetadata(
                DefinedThreadDomain.all_callers(),
                ResultVisibility.ALL_MEMBERS,
            ),
        )
        assert all_callers.to_tensor_ssa().shape == (2,)
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_group_reduce_auto_converts_register_payloads_and_preserves_scalars():
    script = _FAKE_CUTLASS_REDUCE_PREAMBLE + textwrap.dedent(
        """
        class RegisterTensor:
            memspace = AddressSpace.rmem
            shape = (2,)
            element_type = "i32"

            def __getitem__(self, idx):
                return f"r{idx}"

        class TensorSSA:
            shape = ((2, 2),)
            dtype = "i64"

            def numel(self):
                return 4

            def __getitem__(self, idx):
                return f"ssa{idx}"

        class NestedRegisterTensor:
            memspace = AddressSpace.rmem
            shape = ((2, 2),)
            element_type = "i16"

            def __getitem__(self, idx):
                return f"nr{idx}"

        block = coop.this_block()
        rmem_result = coop.reduce(
            block,
            RegisterTensor(),
        )
        assert isinstance(rmem_result, coop.ThreadData)
        assert rmem_result.items_per_thread == 2
        assert rmem_result.dtype == "i32"
        assert rmem_result.values("reduce") == ("r0", "r1")
        assert provider_calls[-1]["value"] is rmem_result

        tensor_ssa_result = coop.reduce(
            block,
            TensorSSA(),
        )
        assert isinstance(tensor_ssa_result, coop.ThreadData)
        assert tensor_ssa_result.items_per_thread == 4
        assert tensor_ssa_result.dtype == "i64"
        assert tensor_ssa_result.values("reduce") == (
            "ssa0",
            "ssa1",
            "ssa2",
            "ssa3",
        )
        assert provider_calls[-1]["value"] is tensor_ssa_result

        nested_rmem_result = coop.reduce(
            block,
            NestedRegisterTensor(),
        )
        assert isinstance(nested_rmem_result, coop.ThreadData)
        assert nested_rmem_result.items_per_thread == 4
        assert nested_rmem_result.dtype == "i16"
        assert nested_rmem_result.values("reduce") == (
            "nr0",
            "nr1",
            "nr2",
            "nr3",
        )
        assert provider_calls[-1]["value"] is nested_rmem_result

        scalar = object()
        scalar_result = coop.reduce(block, scalar)
        assert scalar_result is scalar
        assert provider_calls[-1]["value"] is scalar
        assert len(provider_calls) == 4
        assert all(
            call["launch"].exact_block_dim == (64, 1, 1)
            for call in provider_calls
        )
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_group_reduce_rejects_memory_and_non_static_register_payloads():
    script = _FAKE_CUTLASS_REDUCE_PREAMBLE + textwrap.dedent(
        """
        class GlobalTensor:
            memspace = AddressSpace.gmem
            shape = (2,)

            def __getitem__(self, idx):
                raise AssertionError("memory-backed payload must not be indexed")

        class MissingShapeRegisterTensor:
            memspace = AddressSpace.rmem

            def __getitem__(self, idx):
                raise AssertionError("missing-shape payload must not be indexed")

        class DynamicShapeRegisterTensor:
            memspace = AddressSpace.rmem
            shape = (object(),)

            def __getitem__(self, idx):
                raise AssertionError("dynamic-shape payload must not be indexed")

        class EmptyTensorSSA:
            dtype = "i32"

            def numel(self):
                return 0

            def __getitem__(self, idx):
                raise AssertionError("empty TensorSSA must not be indexed")

        class DynamicTensorSSA:
            dtype = "i32"

            def numel(self):
                return object()

            def __getitem__(self, idx):
                raise AssertionError("dynamic TensorSSA must not be indexed")

        def assert_raises(exc_type, message, fn):
            try:
                fn()
            except exc_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"{exc_type.__name__} was not raised")

        block = coop.this_block()
        assert_raises(
            TypeError,
            "requires a per-thread register payload",
            lambda: coop.reduce(
                block,
                GlobalTensor(),
            ),
        )
        for payload in (
            MissingShapeRegisterTensor(),
            DynamicShapeRegisterTensor(),
            EmptyTensorSSA(),
            DynamicTensorSSA(),
        ):
            assert_raises(
                TypeError,
                "could not infer items_per_thread",
                lambda payload=payload: coop.reduce(
                    block,
                    payload,
                ),
            )

        assert provider_calls == []
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


_FAKE_GROUP_PROVIDER_PREAMBLE = textwrap.dedent(
    """
    import sys
    import types

    class AddressSpace:
        rmem = object()
        gmem = object()

    class DialectAddressSpace:
        rmem = object()
        gmem = object()

    for module_name in (
        "cutlass",
        "cutlass._mlir",
        "cutlass._mlir.dialects",
    ):
        module = types.ModuleType(module_name)
        module.__path__ = []
        sys.modules[module_name] = module
    sys.modules["cutlass"].AddressSpace = AddressSpace

    cute_module = types.ModuleType("cutlass._mlir.dialects.cute")
    cute_module.AddressSpace = DialectAddressSpace
    sys.modules[cute_module.__name__] = cute_module

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFactOrigin, LaunchFacts
    from cuda.coop.cutlass._dsl import _launch

    _launch.current_kernel_launch_facts = lambda: LaunchFacts(
        exact_block_dim=32,
        provenance=(
            LaunchFactOrigin(
                fact="exact_block_dim",
                source="test_compiler",
                verified=True,
            ),
        ),
    )

    provider_calls = {}

    def install_provider(module_suffix, *function_names):
        module_name = f"cuda.coop.cutlass._dsl.{module_suffix}"
        module = types.ModuleType(module_name)
        for function_name in function_names:
            def capture(*, _name=function_name, **kwargs):
                provider_calls[_name] = kwargs
                return kwargs
            setattr(module, function_name, capture)
        sys.modules[module_name] = module

    install_provider("_cudax_reduce_provider", "provider_reduce")
    install_provider("_cub_scan_provider", "provider_scan")
    install_provider("_cub_exchange_provider", "provider_exchange")
    install_provider("_cub_load_store_provider", "provider_store")
    install_provider(
        "_cub_adjacent_difference_provider",
        "provider_adjacent_difference",
    )
    install_provider("_cub_discontinuity_provider", "provider_discontinuity")
    install_provider("_cub_shuffle_provider", "provider_shuffle")
    install_provider("_cub_histogram_provider", "provider_histogram")
    install_provider(
        "_cub_run_length_decode_provider",
        "provider_run_length_decode",
    )
    install_provider(
        "_cub_radix_provider",
        "provider_radix_sort_keys",
        "provider_radix_sort_pairs",
        "provider_radix_rank",
    )
    install_provider("_cub_merge_sort_provider", "provider_merge_sort")

    class RegisterTensor:
        memspace = AddressSpace.rmem
        shape = (2,)
        element_type = "i32"

        def __init__(self, base):
            self.base = base

        def __getitem__(self, idx):
            return self.base + idx

    class TensorSSA:
        shape = (2,)
        dtype = "i32"

        def __init__(self, base):
            self.base = base

        def numel(self):
            return 2

        def __getitem__(self, idx):
            return self.base + idx

    def assert_thread_data(value, base):
        assert isinstance(value, coop.ThreadData)
        assert value.items_per_thread == 2
        assert value.dtype == "i32"
        assert value.values("payload") == (base, base + 1)
    """
)


def test_group_first_primitives_auto_convert_rmem_and_tensorssa_inputs():
    script = _FAKE_GROUP_PROVIDER_PREAMBLE + textwrap.dedent(
        """
        block = coop.this_block()
        unary_calls = (
            (
                "provider_reduce",
                "value",
                lambda value: coop.reduce(block, value),
            ),
            (
                "provider_scan",
                "value",
                lambda value: coop.scan(block, value),
            ),
            (
                "provider_exchange",
                "value",
                lambda value: coop.exchange(block, value),
            ),
            (
                "provider_adjacent_difference",
                "value",
                lambda value: coop.adjacent_difference(
                    block,
                    value,
                ),
            ),
            (
                "provider_discontinuity",
                "value",
                lambda value: coop.discontinuity(
                    block,
                    value,
                ),
            ),
            (
                "provider_shuffle",
                "value",
                lambda value: coop.shuffle(block, value),
            ),
            (
                "provider_histogram",
                "samples",
                lambda value: coop.histogram(
                    block,
                    value,
                    bins=32,
                ),
            ),
            (
                "provider_radix_sort_keys",
                "keys",
                lambda value: coop.radix_sort_keys(
                    block,
                    value,
                ),
            ),
            (
                "provider_radix_rank",
                "keys",
                lambda value: coop.radix_rank(
                    block,
                    value,
                ),
            ),
            (
                "provider_merge_sort",
                "keys",
                lambda value: coop.merge_sort_keys(
                    block,
                    value,
                ),
            ),
        )

        for case_index, (provider_name, arg_name, invoke) in enumerate(unary_calls):
            for kind_index, payload_type in enumerate((RegisterTensor, TensorSSA)):
                base = 1000 + 100 * case_index + 10 * kind_index
                result = invoke(payload_type(base))
                assert result is provider_calls[provider_name]
                assert_thread_data(result[arg_name], base)

        for kind_index, payload_type in enumerate((RegisterTensor, TensorSSA)):
            value_base = 3000 + 100 * kind_index
            length_base = 3100 + 100 * kind_index
            decoded = coop.run_length_decode(
                block,
                payload_type(value_base),
                payload_type(length_base),
                decoded_items_per_thread=2,
            )
            assert_thread_data(decoded["run_values"], value_base)
            assert_thread_data(decoded["run_lengths"], length_base)

            key_base = 4000 + 100 * kind_index
            value_base = 4100 + 100 * kind_index
            radix = coop.radix_sort_pairs(
                block,
                payload_type(key_base),
                payload_type(value_base),
            )
            assert_thread_data(radix["keys"], key_base)
            assert_thread_data(radix["values"], value_base)

            merge = coop.merge_sort_pairs(
                block,
                payload_type(key_base),
                payload_type(value_base),
            )
            assert_thread_data(merge["keys"], key_base)
            assert_thread_data(merge["values"], value_base)

            destination = object()
            assert (
                coop.store(
                    block,
                    destination,
                    payload_type(value_base),
                )
                is None
            )
            assert provider_calls["provider_store"]["destination"] is destination
            assert_thread_data(
                provider_calls["provider_store"]["value"],
                value_base,
            )

        scalar = object()
        scalar_result = coop.radix_sort_keys(
            block,
            scalar,
        )
        assert scalar_result["keys"] is scalar
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_common_root_rejects_cutlass_register_payload_adaptation():
    script = _FAKE_GROUP_PROVIDER_PREAMBLE + textwrap.dedent(
        """
        from cuda import coop as common_coop
        from cuda.coop._core import root_api

        def assert_rejected(
            operation,
            invoke,
            expected,
            qualification="use cuda.coop.cutlass",
        ):
            before = provider_calls.copy()
            try:
                invoke()
            except TypeError as exc:
                message = str(exc)
                assert f"cuda.coop.{operation}" in message
                assert expected in message
                assert qualification in message
            else:
                raise AssertionError(
                    f"common-root {operation} accepted a CUTLASS register payload"
                )
            assert provider_calls == before

        with root_api._compiler_scope("cuda.coop.cutlass"):
            block = common_coop.this_block()
            destination = object()
            for payload_type in (RegisterTensor, TensorSSA):
                payload = payload_type(100)
                for operation, invoke in (
                    (
                        "exchange",
                        lambda: common_coop.exchange(block, payload),
                    ),
                    (
                        "adjacent_difference",
                        lambda: common_coop.adjacent_difference(block, payload),
                    ),
                    (
                        "discontinuity",
                        lambda: common_coop.discontinuity(block, payload),
                    ),
                    (
                        "shuffle",
                        lambda: common_coop.shuffle(block, payload),
                    ),
                ):
                    assert_rejected(
                        operation,
                        invoke,
                        "requires a fixed-size ThreadData",
                    )

                for operation, invoke in (
                    ("reduce", lambda: common_coop.reduce(block, payload)),
                    ("sum", lambda: common_coop.sum(block, payload)),
                    ("scan", lambda: common_coop.scan(block, payload)),
                    (
                        "exclusive_sum",
                        lambda: common_coop.exclusive_sum(block, payload),
                    ),
                    (
                        "inclusive_sum",
                        lambda: common_coop.inclusive_sum(block, payload),
                    ),
                    (
                        "exclusive_scan",
                        lambda: common_coop.exclusive_scan(block, payload),
                    ),
                    (
                        "inclusive_scan",
                        lambda: common_coop.inclusive_scan(block, payload),
                    ),
                    (
                        "store",
                        lambda: common_coop.store(block, destination, payload),
                    ),
                ):
                    assert_rejected(
                        operation,
                        invoke,
                        "accepts only a scalar or fixed-size ThreadData",
                    )

                for operation, invoke in (
                    (
                        "merge_sort_keys",
                        lambda: common_coop.merge_sort_keys(block, payload),
                    ),
                    (
                        "radix_sort_keys",
                        lambda: common_coop.radix_sort_keys(block, payload),
                    ),
                    (
                        "radix_rank",
                        lambda: common_coop.radix_rank(block, payload),
                    ),
                ):
                    assert_rejected(
                        operation,
                        invoke,
                        "requires a fixed-size ThreadData",
                        "use a backend-qualified import",
                    )

                for operation, invoke in (
                    (
                        "merge_sort_keys",
                        lambda: coop.merge_sort_keys(block, payload),
                    ),
                    (
                        "radix_sort_keys",
                        lambda: coop.radix_sort_keys(block, payload),
                    ),
                    (
                        "radix_rank",
                        lambda: coop.radix_rank(block, payload),
                    ),
                ):
                    with root_api._common_root_operation_scope(operation):
                        assert_rejected(
                            operation,
                            invoke,
                            "requires a fixed-size ThreadData",
                        )

        assert root_api._common_root_operation_name() is None
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_scoped_block_and_warp_dispatch_share_register_payload_adapter():
    script = _FAKE_GROUP_PROVIDER_PREAMBLE + textwrap.dedent(
        """
        class EmptyTensorSSA:
            dtype = "i32"

            def numel(self):
                return 0

            def __getitem__(self, idx):
                raise AssertionError("run_length validation must not index payloads")

        class DynamicTensorSSA:
            dtype = "i32"

            def numel(self):
                return object()

            def __getitem__(self, idx):
                raise AssertionError("run_length validation must not index payloads")

        def capture_native(**kwargs):
            return kwargs

        capture_native._supports_native_thread_data = True

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("reduce", capture_native)
        block_result = coop._block.reduce(
            RegisterTensor(10),
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(block_result["value"], 10)

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("merge_sort_pairs", capture_native)
        warp_result = coop._warp.merge_sort_pairs(
            TensorSSA(20),
            RegisterTensor(30),
            threads_in_warp=16,
        )
        assert_thread_data(warp_result["keys"], 20)
        assert_thread_data(warp_result["values"], 30)

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("topk_max_pairs", capture_native)
        topk_result = coop._block.topk_max_pairs(
            RegisterTensor(40),
            TensorSSA(50),
            1,
            launch_metadata={"threads_per_block": 32},
        )
        assert_thread_data(topk_result["keys"], 40)
        assert_thread_data(topk_result["values"], 50)

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("run_length_decode", capture_native)
        run_length = coop._block.run_length(
            RegisterTensor(60),
            TensorSSA(70),
            runs_per_thread=2,
            decoded_items_per_thread=2,
            launch_metadata={"threads_per_block": 32},
        )
        decoded = run_length.decode()
        assert_thread_data(decoded["run_values"], 60)
        assert_thread_data(decoded["run_lengths"], 70)

        calls_before_invalid = dict(provider_calls)
        for invalid_lengths in (EmptyTensorSSA(), DynamicTensorSSA()):
            try:
                coop._block.run_length(
                    RegisterTensor(80),
                    invalid_lengths,
                    runs_per_thread=2,
                    decoded_items_per_thread=2,
                    launch_metadata={"threads_per_block": 32},
                )
            except TypeError as exc:
                assert "could not infer items_per_thread" in str(exc), str(exc)
            else:
                raise AssertionError("invalid run_lengths payload was accepted")
        assert provider_calls == calls_before_invalid
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_input_adapter_does_not_claim_load_or_mutable_output_slots():
    script = _FAKE_GROUP_PROVIDER_PREAMBLE + textwrap.dedent(
        """
        block = coop.this_block()
        try:
            coop.load(block, object(), RegisterTensor(10))
        except TypeError as exc:
            assert "load output must be ThreadData" in str(exc)
        else:
            raise AssertionError("group-first load accepted an rmem destination")

        prefix = RegisterTensor(20)
        shuffled = coop.shuffle(
            block,
            RegisterTensor(30),
            mode="down",
            block_prefix=prefix,
        )
        assert shuffled["block_prefix"] is prefix

        digit_prefix = TensorSSA(40)
        ranked = coop.radix_rank(
            block,
            RegisterTensor(50),
            exclusive_digit_prefix=digit_prefix,
        )
        assert ranked["exclusive_digit_prefix"] is digit_prefix

        relative_offsets = TensorSSA(60)
        total_decoded_size = RegisterTensor(70)
        decoded = coop.run_length_decode(
            block,
            RegisterTensor(80),
            TensorSSA(90),
            decoded_items_per_thread=2,
            relative_offsets=relative_offsets,
            total_decoded_size=total_decoded_size,
        )
        assert decoded["relative_offsets"] is relative_offsets
        assert decoded["total_decoded_size"] is total_decoded_size
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
