# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support.source import run_python_with_source


def test_cutlass_private_scopes_dispatch_dynamic_radix_sort_and_topk_values():
    script = textwrap.dedent(
        """
        import importlib

        class DynamicInt:
            def __init__(self, name):
                self.name = name

        class FakeVector:
            shape = (2,)
            dtype = int

            def __init__(self, base=10):
                self.base = base

            def __getitem__(self, idx):
                return idx + self.base

        def capture(**payload):
            return payload

        def capture_pairs(**payload):
            return payload["keys"], payload["values"]

        begin_bit = DynamicInt("begin")
        end_bit = DynamicInt("end")
        runtime_k = DynamicInt("k")
        runtime_valid = DynamicInt("valid")
        runtime_threads = DynamicInt("threads")

        class MemoryArray:
            space = object()
            shape = (2,)

            def __getitem__(self, idx):
                return idx

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

        class RaisingMemoryProtocolTensor:
            shape = (2,)

            @property
            def __cuda_array_interface__(self):
                raise RuntimeError("not called")

            def __getitem__(self, idx):
                return idx

        class RaisingMemorySpacePayload:
            shape = (2,)

            @property
            def memspace(self):
                raise RuntimeError("memspace probe failed")

            def __getitem__(self, idx):
                return idx

        def exercise(module_name, expected_scope, forbidden_scopes):
            coop = importlib.import_module(module_name)
            block = coop._block
            keys = FakeVector()

            assert block.__name__ == expected_scope

            getattr(block, "_backend", block)._api.register_provider_impl("radix_sort_keys", capture)
            radix_result = block.radix_sort_keys(
                keys,
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=True,
            )
            assert isinstance(radix_result, coop.ThreadData)
            assert radix_result.dtype is int
            radix_payloads = radix_result.values("radix")
            assert [payload["keys"] for payload in radix_payloads] == [10, 11]
            for payload in radix_payloads:
                assert payload["args"] == ()
                assert payload["begin_bit"] is begin_bit
                assert payload["end_bit"] is end_bit
                assert payload["descending"] is True

            metadata_result = block.radix_sort_keys(
                keys,
                launch_metadata={"tag": "keep"},
            )
            assert isinstance(metadata_result, coop.ThreadData)
            for payload in metadata_result.values("metadata-radix"):
                assert payload["launch_metadata"] == {"tag": "keep"}

            getattr(block, "_backend", block)._api.register_provider_impl("topk_max_keys", capture)
            topk_result = block.topk_max_keys(
                keys,
                runtime_k,
                num_valid=runtime_valid,
                begin_bit=begin_bit,
                end_bit=end_bit,
            )
            assert isinstance(topk_result, coop.ThreadData)
            assert topk_result.dtype is int
            topk_payloads = topk_result.values("topk")
            assert [payload["keys"] for payload in topk_payloads] == [10, 11]
            for payload in topk_payloads:
                assert payload["k"] is runtime_k
                assert payload["args"] == ()
                assert payload["num_valid"] is runtime_valid
                assert payload["begin_bit"] is begin_bit
                assert payload["end_bit"] is end_bit
                assert payload["descending"] is True

            getattr(block, "_backend", block)._api.register_provider_impl("radix_sort_pairs", capture_pairs)
            sorted_pair_keys, sorted_pair_values = block.radix_sort_pairs(
                FakeVector(20),
                FakeVector(40),
                begin_bit=begin_bit,
                end_bit=end_bit,
            )
            assert isinstance(sorted_pair_keys, coop.ThreadData)
            assert isinstance(sorted_pair_values, coop.ThreadData)
            assert sorted_pair_keys.values("pair-keys") == (20, 21)
            assert sorted_pair_values.values("pair-values") == (40, 41)
            assert sorted_pair_keys.dtype is int
            assert sorted_pair_values.dtype is int

            for memory_payload in (
                MemoryArray(),
                MemoryProtocolTensor(),
                RaisingMemorySpacePayload(),
                RaisingMemoryProtocolTensor(),
            ):
                try:
                    block.radix_sort_keys(memory_payload)
                except TypeError as exc:
                    message = str(exc)
                    assert (
                        f"{expected_scope}.radix_sort_keys could not auto-convert "
                        "'keys' payload to ThreadData"
                    ) in message
                    assert "per-thread register payload" in message
                    for scope in forbidden_scopes:
                        assert scope not in message
                else:
                    raise AssertionError("memory-backed array was not rejected")

            for kwargs, expected in (
                (
                    {"threads_per_block": runtime_threads},
                    "must be a compile-time positive int",
                ),
                (
                    {"dim": (runtime_threads, 1, 1)},
                    "must be a compile-time positive int",
                ),
                (
                    {"launch_metadata": {"threads_per_block": runtime_threads}},
                    "launch metadata has invalid thread-count key(s): "
                    "threads_per_block",
                ),
                (
                    {"launch_config": {"block": (runtime_threads, 1, 1)}},
                    "launch metadata has invalid thread-count key(s): block",
                ),
                (
                    {"launch": {"threads_per_block": 32, "block": (32, 1, 1)}},
                    "launch metadata has multiple thread-count keys: "
                    "threads_per_block, block",
                ),
            ):
                try:
                    block.radix_sort_keys(keys, **kwargs)
                except TypeError as exc:
                    message = str(exc)
                    assert (
                        f"{expected_scope}.radix_sort_keys "
                        in message
                    ), message
                    assert expected in message
                    for scope in forbidden_scopes:
                        assert scope not in message
                else:
                    raise AssertionError("dynamic block-shape alias was not rejected")

        exercise(
            "cuda.coop.cutlass",
            "cuda.coop.cutlass._block",
            ("cuda.coop.cutlass._dsl.block",),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_ignore_none_payload_on_register_primitives():
    script = textwrap.dedent(
        """
        import importlib

        class FakeVector:
            shape = (2,)
            dtype = int

            def __getitem__(self, idx):
                return idx

        def capture_key(**payload):
            assert "payload" not in payload
            return payload["keys"]

        def capture_value(**payload):
            assert "payload" not in payload
            return payload["value"]

        def exercise(coop, expected_scope):
            expected_warp_scope = expected_scope.replace("._block", "._warp")

            getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_sort_keys", capture_key)
            direct_keys = coop._block.radix_sort_keys(FakeVector(), payload=None)
            assert isinstance(direct_keys, coop.ThreadData)
            assert direct_keys.values("direct-keys") == (0, 1)

            radix_sort = coop._block.make_radix_sort_keys(
                int,
                threads_per_block=32,
                payload=None,
            )
            assert radix_sort.scope == expected_scope
            assert radix_sort.primitive.__module__ == expected_scope
            assert "payload" not in dict(radix_sort.bound_kwargs)
            factory_keys = radix_sort(FakeVector(), payload=None)
            assert isinstance(factory_keys, coop.ThreadData)
            assert factory_keys.values("factory-keys") == (0, 1)

            getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("sum", capture_value)
            direct_sum = coop._warp.sum(FakeVector(), payload=None)
            assert isinstance(direct_sum, coop.ThreadData)
            assert direct_sum.values("direct-sum") == (0, 1)

            warp_sum = coop._warp.make_sum(
                int,
                threads_in_warp=16,
                payload=None,
            )
            assert warp_sum.scope == expected_warp_scope
            assert warp_sum.primitive.__module__ == expected_warp_scope
            assert "payload" not in dict(warp_sum.bound_kwargs)
            factory_sum = warp_sum(FakeVector(), payload=None)
            assert isinstance(factory_sum, coop.ThreadData)
            assert factory_sum.values("factory-sum") == (0, 1)

        root = importlib.import_module("cuda.coop.cutlass")

        exercise(root, "cuda.coop.cutlass._block")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_reject_payload_selectors_on_register_primitives():
    script = textwrap.dedent(
        """
        import importlib

        class FakeVector:
            shape = (2,)
            dtype = int

            def __getitem__(self, idx):
                return idx

        def assert_rejects_payload(fn, expected_scope, primitive_name, forbidden):
            try:
                fn()
            except TypeError as exc:
                message = str(exc)
                assert (
                    f"{expected_scope}.{primitive_name} does not accept payload="
                    in message
                ), message
                assert (
                    "payload selectors are only supported by load/store and "
                    "make_load/make_store"
                ) in message
                assert "per-thread register payloads" in message
                for scope in forbidden:
                    assert scope not in message
            else:
                raise AssertionError(
                    f"{expected_scope}.{primitive_name} accepted payload="
                )

        def exercise(coop, expected_scope, forbidden):
            selector = coop
            expected_warp_scope = expected_scope.replace("._block", "._warp")
            forbidden_warp = tuple(
                scope.replace(".block", ".warp") for scope in forbidden
            )
            assert_rejects_payload(
                lambda: coop._block.radix_sort_keys(
                    FakeVector(),
                    payload=selector,
                ),
                expected_scope,
                "radix_sort_keys",
                forbidden,
            )
            assert_rejects_payload(
                lambda: coop._warp.sum(
                    FakeVector(),
                    payload=selector,
                ),
                expected_warp_scope,
                "sum",
                forbidden_warp,
            )
            assert_rejects_payload(
                lambda: coop._block.topk_max_keys(
                    FakeVector(),
                    1,
                    payload=selector,
                ),
                expected_scope,
                "topk_max_keys",
                forbidden,
            )
            assert_rejects_payload(
                lambda: coop._block.make_radix_sort_keys(
                    int,
                    threads_per_block=32,
                    payload=selector,
                ),
                expected_scope,
                "make_radix_sort_keys",
                forbidden,
            )
            assert_rejects_payload(
                lambda: coop._warp.make_sum(
                    int,
                    threads_in_warp=16,
                    payload=selector,
                ),
                expected_warp_scope,
                "make_sum",
                forbidden_warp,
            )

            radix_sort = coop._block.make_radix_sort_keys(
                int,
                threads_per_block=32,
            )
            assert_rejects_payload(
                lambda: radix_sort(
                    FakeVector(),
                    payload=selector,
                ),
                expected_scope,
                "radix_sort_keys",
                forbidden,
            )
            warp_sum = coop._warp.make_sum(
                int,
                threads_in_warp=16,
            )
            assert_rejects_payload(
                lambda: warp_sum(
                    FakeVector(),
                    payload=selector,
                ),
                expected_warp_scope,
                "sum",
                forbidden_warp,
            )

        root = importlib.import_module("cuda.coop.cutlass")

        exercise(
            root,
            "cuda.coop.cutlass._block",
            ("cuda.coop.cutlass._dsl.block",),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_dispatch_dynamic_radix_rank_values():
    script = textwrap.dedent(
        """
        import importlib

        class DynamicInt:
            def __init__(self, name):
                self.name = name

        class FakeVector:
            shape = (2,)
            dtype = int

            def __init__(self, base=10):
                self.base = base

            def __getitem__(self, idx):
                return self.base + idx

        def capture(**payload):
            return payload

        begin_bit = DynamicInt("begin")
        end_bit = DynamicInt("end")
        radix_bits = DynamicInt("radix_bits")

        def exercise(module_name, expected_scope):
            coop = importlib.import_module(module_name)
            block = coop._block
            prefix = coop.ThreadData.from_values(100, 101, dtype=int)

            assert block.__name__ == expected_scope

            getattr(block, "_backend", block)._api.register_provider_impl("radix_rank", capture)
            end_bit_rank = block.radix_rank(
                FakeVector(20),
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=True,
                exclusive_digit_prefix=prefix,
            )
            assert isinstance(end_bit_rank, coop.ThreadData)
            assert end_bit_rank.dtype is int
            for payload in end_bit_rank.values("rank"):
                assert payload["keys"] in (20, 21)
                assert payload["args"] == ()
                assert payload["begin_bit"] is begin_bit
                assert payload["end_bit"] is end_bit
                assert payload["radix_bits"] is None
                assert payload["descending"] is True
                assert payload["exclusive_digit_prefix"] is prefix

            radix_bits_rank = block.radix_rank(
                FakeVector(30),
                begin_bit=begin_bit,
                radix_bits=radix_bits,
            )
            assert isinstance(radix_bits_rank, coop.ThreadData)
            assert radix_bits_rank.dtype is int
            for payload in radix_bits_rank.values("rank-bits"):
                assert payload["keys"] in (30, 31)
                assert payload["args"] == ()
                assert payload["begin_bit"] is begin_bit
                assert payload["end_bit"] is None
                assert payload["radix_bits"] is radix_bits
                assert payload["descending"] is False
                assert "exclusive_digit_prefix" not in payload

        exercise("cuda.coop.cutlass", "cuda.coop.cutlass._block")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_dispatch_dynamic_merge_sort_valid_items():
    script = textwrap.dedent(
        """
        import importlib

        class DynamicInt:
            def __init__(self, name):
                self.name = name

        class FakeVector:
            shape = (2,)
            dtype = int

            def __init__(self, base=10):
                self.base = base

            def __getitem__(self, idx):
                return self.base + idx

        def capture(**payload):
            return payload

        pair_payloads = []

        def capture_pairs(**payload):
            pair_payloads.append(payload)
            return payload["keys"], payload["values"]

        runtime_valid = DynamicInt("valid_items")
        sentinel = DynamicInt("oob_default")

        def exercise(module_name, expected_scope):
            coop = importlib.import_module(module_name)
            block = coop._block

            assert block.__name__ == expected_scope

            getattr(block, "_backend", block)._api.register_provider_impl("merge_sort_keys", capture)
            sorted_keys = block.merge_sort_keys(
                FakeVector(20),
                descending=True,
                valid_items=runtime_valid,
                oob_default=sentinel,
            )
            assert isinstance(sorted_keys, coop.ThreadData)
            for payload in sorted_keys.values("merge-keys"):
                assert payload["keys"] in (20, 21)
                assert payload["args"] == ()
                assert payload["descending"] is True
                assert payload["valid_items"] is runtime_valid
                assert payload["oob_default"] is sentinel

            getattr(block, "_backend", block)._api.register_provider_impl("merge_sort_pairs", capture_pairs)
            sorted_pair_keys, sorted_pair_values = block.merge_sort_pairs(
                FakeVector(30),
                FakeVector(50),
                valid_items=runtime_valid,
                oob_default=sentinel,
            )
            assert isinstance(sorted_pair_keys, coop.ThreadData)
            assert isinstance(sorted_pair_values, coop.ThreadData)
            assert sorted_pair_keys.values("merge-pair-keys") == (30, 31)
            assert sorted_pair_values.values("merge-pair-values") == (50, 51)
            assert len(pair_payloads) == 2
            for payload in pair_payloads:
                assert payload["args"] == ()
                assert payload["descending"] is False
                assert payload["valid_items"] is runtime_valid
                assert payload["oob_default"] is sentinel
            pair_payloads.clear()

        exercise("cuda.coop.cutlass", "cuda.coop.cutlass._block")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_dispatch_dynamic_run_length_decode_values():
    script = textwrap.dedent(
        """
        import importlib

        class DynamicInt:
            width = 32
            signed = False

            def __init__(self, name):
                self.name = name
                self.dtype = U32

            def ir_value(self):
                return self.name

        class U32:
            width = 32
            signed = False

        class FakeVector:
            shape = (2,)
            dtype = int

            def __init__(self, base):
                self.base = base

            def __getitem__(self, idx):
                return self.base + idx

        def capture(**payload):
            return payload

        runtime_offset = DynamicInt("decoded_window_offset")

        def exercise(module_name, expected_scope):
            coop = importlib.import_module(module_name)
            block = coop._block
            run_values = coop.ThreadData.from_values(10, 11, dtype=int)
            run_lengths = coop.ThreadData.from_values(1, 2, dtype=int)
            factory_total = object()
            runtime_total = object()

            assert block.__name__ == expected_scope

            getattr(block, "_backend", block)._api.register_provider_impl("run_length_decode", capture)
            direct_payload = block.run_length_decode(
                run_values,
                run_lengths,
                decoded_items_per_thread=2,
                decoded_window_offset=runtime_offset,
                total_decoded_size=runtime_total,
                launch_metadata={"threads_per_block": 32},
            )
            assert direct_payload["run_values"] is run_values
            assert direct_payload["run_lengths"] is run_lengths
            assert direct_payload["decoded_items_per_thread"] == 2
            assert direct_payload["decoded_window_offset"] is runtime_offset
            assert direct_payload["total_decoded_size"] is runtime_total
            assert direct_payload["launch_metadata"] == {"threads_per_block": 32}

            run_length = block.make_run_length(
                int,
                threads_per_block=32,
                decoded_items_per_thread=2,
                total_decoded_size=factory_total,
            )
            parent = run_length(run_values, run_lengths)
            factory_payload = parent.decode(decoded_window_offset=runtime_offset)
            assert factory_payload["run_values"] is run_values
            assert factory_payload["run_lengths"] is run_lengths
            assert factory_payload["decoded_window_offset"] is runtime_offset
            assert factory_payload["total_decoded_size"] is factory_total
            assert factory_payload["launch_metadata"] == {"threads_per_block": 32}

            override_parent = run_length(
                run_values,
                run_lengths,
                total_decoded_size=runtime_total,
            )
            override_payload = override_parent.decode(runtime_offset)
            assert override_payload["decoded_window_offset"] is runtime_offset
            assert override_payload["total_decoded_size"] is runtime_total

            typed_run_length = block.make_run_length(
                int,
                threads_per_block=32,
                runs_per_thread=2,
                decoded_items_per_thread=2,
                decoded_offset_dtype=U32,
            )
            typed_parent = typed_run_length(FakeVector(20), FakeVector(30))
            typed_payload = typed_parent.decode(decoded_window_offset=runtime_offset)
            assert isinstance(typed_payload["run_values"], coop.ThreadData)
            assert typed_payload["run_values"].values("run-values") == (20, 21)
            assert isinstance(typed_payload["run_lengths"], coop.ThreadData)
            assert typed_payload["run_lengths"].dtype is U32
            assert typed_payload["run_lengths"].values("run-lengths") == (30, 31)
            assert typed_payload["decoded_window_offset"] is runtime_offset
            assert typed_payload["launch_metadata"] == {"threads_per_block": 32}

        exercise("cuda.coop.cutlass", "cuda.coop.cutlass._block")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_private_scopes_require_static_histogram_bins():
    script = textwrap.dedent(
        """
        import importlib

        class DynamicInt:
            def __init__(self, name):
                self.name = name

        def capture(**payload):
            return payload

        runtime_bins = DynamicInt("bins")
        factory_bins = DynamicInt("factory-bins")

        def exercise(module_name, expected_scope):
            coop = importlib.import_module(module_name)
            block = coop._block
            samples = coop.ThreadData.from_values(1, 2, dtype=int)

            assert block.__name__ == expected_scope

            getattr(block, "_backend", block)._api.register_provider_impl("histogram", capture)
            try:
                block.histogram(
                    samples,
                    bins=runtime_bins,
                    bins_per_thread=2,
                    counter_dtype=int,
                    algorithm="sort",
                    launch_metadata={"threads_per_block": 32},
                )
            except NotImplementedError as exc:
                assert "bins must be trace-time static" in str(exc)
            else:
                raise AssertionError("dynamic direct histogram bins should fail")

            histogram = block.make_histogram(
                int,
                int,
                threads_per_block=32,
                bins=factory_bins,
                bins_per_thread=2,
                algorithm="sort",
            )
            try:
                histogram(samples)
            except NotImplementedError as exc:
                assert "bins must be trace-time static" in str(exc)
            else:
                raise AssertionError("dynamic factory histogram bins should fail")

            static_histogram = block.make_histogram(
                int,
                int,
                threads_per_block=32,
                bins=16,
                bins_per_thread=2,
                algorithm="sort",
            )
            default_payload = static_histogram(samples)
            assert default_payload["samples"] is samples
            assert default_payload["bins"] == 16
            assert default_payload["bins_per_thread"] == 2
            assert default_payload["counter_dtype"] is int
            assert default_payload["algorithm"] == "sort"
            assert default_payload["launch_metadata"] == {"threads_per_block": 32}

            try:
                static_histogram(samples, bins=runtime_bins)
            except NotImplementedError as exc:
                assert "bins must be trace-time static" in str(exc)
            else:
                raise AssertionError("dynamic histogram bin override should fail")

        exercise("cuda.coop.cutlass", "cuda.coop.cutlass._block")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr
