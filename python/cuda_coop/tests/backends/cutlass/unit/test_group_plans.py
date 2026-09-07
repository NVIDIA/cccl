# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support.source import run_python_with_source


def test_cutlass_group_sum_and_scan_aliases_route_canonical_semantics():
    script = textwrap.dedent(
        """
        import operator
        import sys
        import types

        import cuda.coop.cutlass as coop
        from cuda.coop._core import LaunchFacts
        from cuda.coop.cutlass._dsl import _launch

        _launch.current_kernel_launch_facts = lambda: LaunchFacts(
            exact_block_dim=32
        )

        calls = []
        reduce_provider = types.ModuleType(
            "cuda.coop.cutlass._dsl._cudax_reduce_provider"
        )
        scan_provider = types.ModuleType(
            "cuda.coop.cutlass._dsl._cub_scan_provider"
        )

        def provider_reduce(**kwargs):
            calls.append(("reduce", kwargs))
            return ("reduce-result", len(calls))

        def provider_scan(**kwargs):
            calls.append(("scan", kwargs))
            return ("scan-result", len(calls))

        reduce_provider.provider_reduce = provider_reduce
        scan_provider.provider_scan = provider_scan
        sys.modules[reduce_provider.__name__] = reduce_provider
        sys.modules[scan_provider.__name__] = scan_provider

        block = coop.this_block()
        value = coop.ThreadData.from_values(1, 2, dtype=int)
        storage = object()

        assert coop.sum(
            block,
            value,
            broadcast=False,
            valid_items=17,
            algorithm="raking",
        ) == ("reduce-result", 1)
        reduce_call = calls[-1][1]
        assert reduce_call["op"] == "sum"
        assert reduce_call["broadcast"] is False
        assert reduce_call["valid_items"] == 17
        assert reduce_call["algorithm"].name == "RAKING"

        cases = (
            ("scan", {"mode": "exclusive"}, "exclusive", "sum", None),
            ("exclusive_sum", {}, "exclusive", "sum", None),
            ("inclusive_sum", {}, "inclusive", "sum", None),
            (
                "exclusive_scan",
                {"scan_op": operator.mul, "initial_value": 1},
                "exclusive",
                "multiplies",
                1,
            ),
            (
                "inclusive_scan",
                {"scan_op": operator.add},
                "inclusive",
                "sum",
                None,
            ),
        )
        for name, kwargs, mode, op, initial_value in cases:
            result = getattr(coop, name)(
                block,
                value,
                algorithm="raking",
                temp_storage=storage,
                **kwargs,
            )
            assert result[0] == "scan-result"
            scan_call = calls[-1][1]
            assert scan_call["mode"] == mode
            assert scan_call["op"] == op
            assert scan_call["initial_value"] == initial_value
            assert scan_call["algorithm"].name == "RAKING"
            assert scan_call["temp_storage"] is storage
            assert scan_call["source"] == "cutlass_root"
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_group_adjacent_difference_root_and_scoped_share_plan():
    script = textwrap.dedent(
        """
        import operator
        import sys
        import types

        import cuda.coop.cutlass as coop
        from cuda.coop._core import LaunchFacts
        from cuda.coop.cutlass._dsl import _launch
        from cuda.coop.cutlass._group_adjacent_difference import (
            _make_group_adjacent_difference_plan,
        )

        launch_dim = [(8, 4, 2)]
        _launch.current_kernel_launch_facts = lambda: LaunchFacts(
            exact_block_dim=launch_dim[0]
        )

        calls = []
        provider = types.ModuleType(
            "cuda.coop.cutlass._dsl._cub_adjacent_difference_provider"
        )

        def provider_adjacent_difference(**kwargs):
            calls.append(kwargs)
            return ("adjacent-result", len(calls))

        provider.provider_adjacent_difference = provider_adjacent_difference
        sys.modules[provider.__name__] = provider

        value = coop.ThreadData.from_values(4, 7, dtype=int)
        block = coop.this_block()
        assert coop.adjacent_difference(
            block,
            value,
            direction="left",
            difference_op=operator.sub,
            valid_items=31,
            tile_predecessor_item=3,
        ) == ("adjacent-result", 1)
        assert calls[0]["group"].block_dim == (8, 4, 2)
        assert calls[0]["launch"].exact_block_dim == (8, 4, 2)
        assert calls[0]["direction"].value == "left"
        assert calls[0]["valid_items"] == 31
        assert calls[0]["tile_predecessor_item"] == 3
        assert calls[0]["source"] == "cutlass_root"

        launch_dim[0] = 64
        assert coop._block.adjacent_difference(
            value,
            block_adjacent_difference_type=(
                coop._block.BlockAdjacentDifferenceType.SubtractRight
            ),
            launch_metadata={"threads_per_block": 64},
        ) == ("adjacent-result", 2)
        assert calls[1]["group"].block_dim == (64, 1, 1)
        assert calls[1]["direction"].value == "right"
        assert calls[1]["source"] == "scoped_block"

        root_plan = _make_group_adjacent_difference_plan(
            group=block,
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            dtype=int,
            items_per_thread=2,
            direction="left",
            valid_items=31,
            tile_predecessor_item=3,
            source="root_test",
        )
        scoped_plan = _make_group_adjacent_difference_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            dtype=int,
            items_per_thread=2,
            direction="left",
            valid_items=17,
            tile_predecessor_item=999,
            source="scoped_test",
        )
        assert root_plan.implementation.method_name == "SubtractLeftPartialTile"
        assert root_plan.implementation.template_arguments["BLOCK_DIM_Y"] == 4
        assert root_plan.implementation.template_arguments["BLOCK_DIM_Z"] == 2
        assert root_plan.artifact_key == scoped_plan.artifact_key

        def assert_raises(error_type, message, callback):
            try:
                callback()
            except error_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"expected {error_type.__name__}: {message}")

        assert_raises(
            NotImplementedError,
            "only this_block",
            lambda: coop.adjacent_difference(
                coop.this_warp(),
                value,
            ),
        )
        assert_raises(
            NotImplementedError,
            "built-in subtraction",
            lambda: coop.adjacent_difference(
                block,
                value,
                difference_op=lambda left, right: left + right,
            ),
        )
        assert_raises(
            ValueError,
            "tile_successor_item is not valid for SubtractRightPartialTile",
            lambda: _make_group_adjacent_difference_plan(
                group=block,
                launch=LaunchFacts(exact_block_dim=64),
                dtype=int,
                items_per_thread=2,
                direction="right",
                valid_items=17,
                tile_successor_item=9,
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_discontinuity_shuffle_root_scoped_routes_and_failures():
    script = textwrap.dedent(
        """
        import operator
        import sys
        import types

        import cuda.coop.cutlass as coop
        from cuda.coop._core import LaunchFacts
        from cuda.coop.cutlass._dsl import _launch
        from cuda.coop.cutlass._group_discontinuity import (
            _make_group_discontinuity_plan,
        )
        from cuda.coop.cutlass._group_shuffle import _make_group_shuffle_plan

        launch_dim = [(8, 4, 2)]
        _launch.current_kernel_launch_facts = lambda: LaunchFacts(
            exact_block_dim=launch_dim[0]
        )

        calls = []
        discontinuity_provider = types.ModuleType(
            "cuda.coop.cutlass._dsl._cub_discontinuity_provider"
        )
        shuffle_provider = types.ModuleType(
            "cuda.coop.cutlass._dsl._cub_shuffle_provider"
        )

        def provider_discontinuity(**kwargs):
            calls.append(("discontinuity", kwargs))
            return ("discontinuity-result", len(calls))

        def provider_shuffle(**kwargs):
            calls.append(("shuffle", kwargs))
            return ("shuffle-result", len(calls))

        discontinuity_provider.provider_discontinuity = provider_discontinuity
        shuffle_provider.provider_shuffle = provider_shuffle
        sys.modules[discontinuity_provider.__name__] = discontinuity_provider
        sys.modules[shuffle_provider.__name__] = shuffle_provider

        block = coop.this_block()
        items = coop.ThreadData.from_values(1, 1, 2, dtype=int)
        assert coop.discontinuity(
            block,
            items,
            mode="heads_and_tails",
            flag_op=operator.ne,
            tile_predecessor_item=0,
            tile_successor_item=3,
        ) == ("discontinuity-result", 1)
        assert calls[-1][1]["group"].block_dim == (8, 4, 2)
        assert calls[-1][1]["mode"].value == "heads_and_tails"
        assert calls[-1][1]["source"] == "cutlass_root"

        launch_dim[0] = 64
        assert coop._block.discontinuity_flag_heads(
            items,
            tile_predecessor_item=0,
            launch_metadata={"threads_per_block": 64},
        ) == ("discontinuity-result", 2)
        assert calls[-1][1]["mode"].value == "heads"
        assert calls[-1][1]["source"] == "scoped_block"

        assert coop.shuffle(
            block,
            7,
            mode="rotate",
            distance=131,
        ) == ("shuffle-result", 3)
        assert calls[-1][1]["mode"].value == "rotate"
        assert calls[-1][1]["distance"] == 131

        class DynamicDistance:
            pass

        dynamic_distance = DynamicDistance()
        assert coop.shuffle(
            block,
            7,
            mode="offset",
            distance=dynamic_distance,
        ) == ("shuffle-result", 4)
        assert calls[-1][1]["distance"] is dynamic_distance

        block_suffix = coop.ThreadData(1, dtype=int)
        assert coop._block.shuffle_up(
            items,
            block_suffix=block_suffix,
            launch_metadata={"threads_per_block": 64},
        ) == ("shuffle-result", 5)
        assert calls[-1][1]["mode"].value == "up"
        assert calls[-1][1]["source"] == "scoped_block"

        root_disc = _make_group_discontinuity_plan(
            group=block,
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            dtype=int,
            flag_dtype=int,
            items_per_thread=3,
            mode="heads_and_tails",
            tile_predecessor_item=1,
            tile_successor_item=9,
            source="root_test",
        )
        scoped_disc = _make_group_discontinuity_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            dtype=int,
            flag_dtype=int,
            items_per_thread=3,
            mode="heads_and_tails",
            tile_predecessor_item=111,
            tile_successor_item=999,
            source="scoped_test",
        )
        assert root_disc.artifact_key == scoped_disc.artifact_key
        assert [result.name for result in root_disc.result.values] == [
            "head_flags", "tail_flags"
        ]

        root_shuffle = _make_group_shuffle_plan(
            group=block,
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            dtype=int,
            items_per_thread=3,
            mode="up",
            block_suffix=True,
            source="root_test",
        )
        scoped_shuffle = _make_group_shuffle_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            dtype=int,
            items_per_thread=3,
            mode="up",
            block_suffix=True,
            source="scoped_test",
        )
        assert root_shuffle.artifact_key == scoped_shuffle.artifact_key
        assert [result.name for result in root_shuffle.result.values] == [
            "value", "block_suffix"
        ]

        def assert_raises(error_type, message, callback):
            try:
                callback()
            except error_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"expected {error_type.__name__}: {message}")

        assert_raises(
            NotImplementedError,
            "built-in inequality",
            lambda: coop.discontinuity(
                block,
                items,
                flag_op=lambda left, right: False,
            ),
        )
        assert_raises(
            TypeError,
            "unexpected keyword argument 'valid_items'",
            lambda: coop.discontinuity(
                block,
                items,
                valid_items=7,
            ),
        )
        assert_raises(
            NotImplementedError,
            "scalar values support only",
            lambda: coop.shuffle(
                block, 7, mode="down"
            ),
        )
        assert_raises(
            NotImplementedError,
            "ThreadData supports only",
            lambda: coop.shuffle(
                block, items, mode="rotate"
            ),
        )
        assert_raises(
            NotImplementedError,
            "only distance=1",
            lambda: coop.shuffle(
                block, items, mode="up", distance=2,
            ),
        )
        assert_raises(
            TypeError,
            "dynamic distance is unsupported",
            lambda: coop.shuffle(
                block, items, mode="up", distance=object(),
            ),
        )
        assert_raises(
            ValueError,
            "Rotate distance must be non-negative",
            lambda: coop.shuffle(
                block, 7, mode="rotate", distance=-1,
            ),
        )
        assert_raises(
            NotImplementedError,
            "does not return block prefix or suffix",
            lambda: coop.shuffle(
                block, 7, mode="offset", block_prefix=coop.ThreadData(1),
            ),
        )
        assert_raises(
            ValueError,
            "block_prefix is valid only for ThreadData Down",
            lambda: coop.shuffle(
                block, items, mode="up", block_prefix=coop.ThreadData(1),
            ),
        )
        assert_raises(
            ValueError,
            "only one of block_prefix or block_suffix",
            lambda: coop.shuffle(
                block, items, mode="down",
                block_prefix=coop.ThreadData(1),
                block_suffix=coop.ThreadData(1),
            ),
        )
        assert_raises(
            NotImplementedError,
            "only this_block",
            lambda: coop.shuffle(
                coop.this_warp(), 7, mode="offset",
            ),
        )
        assert_raises(
            ValueError,
            "tile_successor_item is not valid for HEADS",
            lambda: _make_group_discontinuity_plan(
                group=block,
                launch=LaunchFacts(exact_block_dim=64),
                dtype=int,
                flag_dtype=int,
                items_per_thread=3,
                mode="heads",
                tile_successor_item=9,
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_group_radix_root_and_scoped_share_plans_and_static_rank_bits():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop
        from cuda.coop._core import GroupOperandKind, LaunchFacts
        from cuda.coop.cutlass._dsl import _launch
        from cuda.coop.cutlass._group_radix import (
            _make_group_radix_rank_plan,
            _make_group_radix_sort_plan,
        )

        _launch.current_kernel_launch_facts = lambda: LaunchFacts(
            exact_block_dim=(8, 4, 2)
        )

        calls = []
        provider = types.ModuleType("cuda.coop.cutlass._dsl._cub_radix_provider")

        def capture(name):
            def provider_call(**kwargs):
                calls.append((name, kwargs))
                return (name, len(calls))
            return provider_call

        provider.provider_radix_sort_keys = capture("keys")
        provider.provider_radix_sort_pairs = capture("pairs")
        provider.provider_radix_rank = capture("rank")
        sys.modules[provider.__name__] = provider

        block = coop.this_block()
        keys = coop.ThreadData.from_values(3, 1, dtype=int)
        values = coop.ThreadData.from_values(30, 10, dtype=int)
        assert coop.radix_sort_keys(
            block,
            keys,
            begin_bit=1,
            end_bit=9,
        ) == ("keys", 1)
        assert calls[-1][1]["group"].block_dim == (8, 4, 2)
        assert calls[-1][1]["source"] == "cutlass_root"

        assert coop._block.radix_sort_pairs(
            keys,
            values,
            descending=True,
            launch_metadata={"block": (8, 4, 2)},
        ) == ("pairs", 2)
        assert calls[-1][1]["source"] == "cutlass_scoped_block"
        assert calls[-1][1]["descending"] is True

        prefix = coop.ThreadData(4, dtype=int)
        assert coop.radix_rank(
            block,
            keys,
            begin_bit=0,
            radix_bits=8,
            exclusive_digit_prefix=prefix,
        ) == ("rank", 3)
        assert calls[-1][1]["begin_bit"] == 0
        assert calls[-1][1]["end_bit"] == 8
        assert calls[-1][1]["exclusive_digit_prefix"] is prefix

        root_sort = _make_group_radix_sort_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            key_dtype=int,
            value_dtype=None,
            items_per_thread=2,
            operand_kind=GroupOperandKind.ARRAY,
            descending=False,
            key_bit_width=32,
            source="root_test",
        )
        scoped_sort = _make_group_radix_sort_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            key_dtype=int,
            value_dtype=None,
            items_per_thread=2,
            operand_kind=GroupOperandKind.ARRAY,
            descending=False,
            key_bit_width=32,
            source="scoped_test",
        )
        assert root_sort.artifact_key == scoped_sort.artifact_key

        root_rank = _make_group_radix_rank_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=64),
            cub_key_dtype="unsigned int",
            input_dtype=int,
            items_per_thread=2,
            operand_kind=GroupOperandKind.ARRAY,
            begin_bit=0,
            end_bit=8,
            key_bit_width=32,
            descending=False,
            exclusive_digit_prefix_items_per_thread=4,
            source="root_test",
        )
        scoped_rank = _make_group_radix_rank_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=64),
            cub_key_dtype="unsigned int",
            input_dtype=int,
            items_per_thread=2,
            operand_kind=GroupOperandKind.ARRAY,
            begin_bit=0,
            end_bit=8,
            key_bit_width=32,
            descending=False,
            exclusive_digit_prefix_items_per_thread=4,
            source="scoped_test",
        )
        assert root_rank.artifact_key == scoped_rank.artifact_key
        assert (
            root_rank.implementation.template_arguments["SMEM_CONFIG"]
            == "cudaSharedMemBankSizeEightByte"
        )

        narrow_rank = _make_group_radix_rank_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=64),
            cub_key_dtype="unsigned int",
            input_dtype=int,
            items_per_thread=2,
            operand_kind=GroupOperandKind.ARRAY,
            begin_bit=0,
            end_bit=4,
            key_bit_width=32,
            descending=False,
            exclusive_digit_prefix_items_per_thread=1,
            source="root_test",
        )
        assert (
            narrow_rank.implementation.template_arguments["SMEM_CONFIG"]
            == "cudaSharedMemBankSizeFourByte"
        )

        def assert_raises(error_type, message, callback):
            try:
                callback()
            except error_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"expected {error_type.__name__}: {message}")

        assert_raises(
            TypeError,
            "trace-time static integer",
            lambda: coop.radix_rank(
                block,
                keys,
                begin_bit=object(),
                end_bit=8,
            ),
        )
        for unsupported_width in (9, 10, 30):
            assert_raises(
                ValueError,
                "<= 8",
                lambda unsupported_width=unsupported_width: coop.radix_rank(
                    block,
                    keys,
                    begin_bit=0,
                    end_bit=unsupported_width,
                ),
            )
        assert_raises(
            NotImplementedError,
            "only this_block",
            lambda: coop.radix_sort_keys(
                coop.this_warp(),
                keys,
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_group_merge_sort_root_and_scoped_share_typed_plans():
    script = textwrap.dedent(
        """
        import operator
        import sys
        import types

        import cuda.coop.cutlass as coop
        from cuda.coop._core import LaunchFacts
        from cuda.coop.cutlass._dsl import _launch
        from cuda.coop.cutlass._group_merge_sort import _make_group_merge_sort_plan

        launch_dim = [(8, 4, 2)]
        _launch.current_kernel_launch_facts = lambda: LaunchFacts(
            exact_block_dim=launch_dim[0]
        )

        calls = []
        provider = types.ModuleType("cuda.coop.cutlass._dsl._cub_merge_sort_provider")

        def provider_merge_sort(**kwargs):
            calls.append(kwargs)
            if kwargs["values"] is None:
                return ("sorted-keys", len(calls))
            return ("sorted-keys", "sorted-values", len(calls))

        provider.provider_merge_sort = provider_merge_sort
        sys.modules[provider.__name__] = provider

        keys = coop.ThreadData.from_values(4, 1, dtype=int)
        values = coop.ThreadData.from_values(40.0, 10.0, dtype=float)
        block = coop.this_block()
        assert coop.merge_sort_pairs(
            block,
            keys,
            values,
            compare_op=operator.gt,
            valid_items=31,
            oob_default=-999,
        ) == ("sorted-keys", "sorted-values", 1)
        assert calls[0]["group"].block_dim == (8, 4, 2)
        assert calls[0]["descending"] is True
        assert calls[0]["source"] == "cutlass_root"

        launch_dim[0] = 64
        assert coop._block.merge_sort_keys(
            keys,
            descending=False,
            launch_metadata={"threads_per_block": 64},
        ) == ("sorted-keys", 2)
        assert calls[1]["group"].kind == "block"
        assert calls[1]["source"] == "scoped_block"

        warp = coop.this_warp()
        assert coop.merge_sort_keys(
            warp,
            keys,
            descending=True,
        ) == ("sorted-keys", 3)
        assert calls[2]["group"].kind == "warp"
        assert calls[2]["source"] == "cutlass_root"

        assert coop._warp.merge_sort_pairs(
            keys,
            values,
            threads_in_warp=16,
            launch_metadata={"threads_per_block": 64},
        ) == ("sorted-keys", "sorted-values", 4)
        assert calls[3]["group"].kind == "threads_within_warp"
        assert calls[3]["group"].static_size == 16
        assert calls[3]["source"] == "scoped_warp"

        assert coop.merge_sort_keys(
            warp.group_by(16),
            keys,
        ) == ("sorted-keys", 5)
        assert calls[4]["group"].kind == "threads_within_warp"
        assert calls[4]["group"].static_size == 16
        assert calls[4]["source"] == "cutlass_root"

        root_plan = _make_group_merge_sort_plan(
            group=block,
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            key_dtype=int,
            value_dtype=float,
            items_per_thread=2,
            descending=True,
            valid_items=31,
            oob_default=-999,
            source="root_test",
        )
        scoped_plan = _make_group_merge_sort_plan(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
            key_dtype=int,
            value_dtype=float,
            items_per_thread=2,
            descending=True,
            valid_items=17,
            oob_default=999,
            source="scoped_test",
        )
        assert root_plan.artifact_key == scoped_plan.artifact_key
        assert [result.name for result in root_plan.result.values] == [
            "keys", "values"
        ]

        def assert_raises(error_type, message, callback):
            try:
                callback()
            except error_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"expected {error_type.__name__}: {message}")

        assert_raises(
            TypeError,
            "takes 2 positional arguments but 3 were given",
            lambda: coop.merge_sort_keys(
                block,
                keys,
                "unexpected",
            ),
        )
        assert_raises(
            NotImplementedError,
            "built-in ascending/less",
            lambda: coop.merge_sort_keys(
                block,
                keys,
                compare_op=lambda left, right: left != right,
            ),
        )
        assert_raises(
            NotImplementedError,
            "power-of-two block thread count",
            lambda: _make_group_merge_sort_plan(
                group=block,
                launch=LaunchFacts(exact_block_dim=48),
                key_dtype=int,
                value_dtype=None,
                items_per_thread=2,
                descending=False,
            ).require_supported(),
        )
        warp_partial_plan = _make_group_merge_sort_plan(
            group=warp,
            launch=LaunchFacts(exact_block_dim=64),
            key_dtype=int,
            value_dtype=None,
            items_per_thread=2,
            descending=False,
            valid_items=17,
            oob_default=-999,
        )
        warp_partial_plan.require_supported()
        assert warp_partial_plan.target.value == "cub_warp"
        assert [
            parameter.name
            for parameter in warp_partial_plan.implementation.parameters[0]
        ] == [
            "temp_storage",
            "keys",
            "compare_op",
            "valid_items",
            "oob_default",
        ]
        precondition = warp_partial_plan.participation.argument_preconditions[0]
        assert (precondition.minimum, precondition.maximum) == (0, 64)
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_group_exchange_root_surface_and_provider_boundary():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop
        import cuda.coop.cutlass as cute
        from cuda.coop._core import LaunchFacts
        from cuda.coop.cutlass._dsl import _launch
        from cuda.coop.cutlass._group_exchange import (
            _exchange,
            _make_group_exchange_plan,
        )

        launch_dim = [(8, 4, 1)]
        _launch.current_kernel_launch_facts = lambda: LaunchFacts(
            exact_block_dim=launch_dim[0]
        )

        calls = []
        provider = types.ModuleType(
            "cuda.coop.cutlass._dsl._cub_exchange_provider"
        )

        def provider_exchange(
            *,
            group,
            launch,
            value,
            mode,
            output=None,
            ranks=None,
            valid_flags=None,
            warp_time_slicing=False,
            source,
        ):
            calls.append(
                {
                    "group": group,
                    "launch": launch,
                    "value": value,
                    "mode": mode,
                    "output": output,
                    "ranks": ranks,
                    "valid_flags": valid_flags,
                    "warp_time_slicing": warp_time_slicing,
                    "source": source,
                }
            )
            return ("exchange-result", len(calls))

        provider.provider_exchange = provider_exchange
        sys.modules[provider.__name__] = provider
        cute._cub_exchange_provider = provider

        value = coop.ThreadData.from_values(10, 11, dtype=int)
        block = coop.this_block()
        assert coop.exchange(
            block,
            value,
            mode="blocked_to_striped",
        ) == ("exchange-result", 1)
        assert calls[0]["group"] is block
        assert calls[0]["launch"].exact_block_dim == (8, 4, 1)
        assert calls[0]["value"] is value
        assert calls[0]["mode"] == "blocked_to_striped"
        assert calls[0]["output"] is None
        assert calls[0]["source"] == "cutlass_root"

        output = coop.ThreadData(2, dtype=int)
        warp = coop.this_warp()
        launch_dim[0] = 64
        assert _exchange(
            warp,
            value,
            mode="striped_to_blocked",
            output=output,
            source="scoped_warp",
            launch_metadata={"threads_per_block": 64},
        ) == ("exchange-result", 2)
        assert calls[1]["group"] is warp
        assert calls[1]["launch"].exact_block_dim == (64, 1, 1)
        assert calls[1]["mode"] == "striped_to_blocked"
        assert calls[1]["output"] is output
        assert calls[1]["source"] == "scoped_warp"

        launch_dim[0] = (8, 4, 1)
        assert coop._block.exchange_blocked_to_striped(
            value,
            output=output,
            launch_metadata={"block": (8, 4, 1)},
        ) == ("exchange-result", 3)
        assert calls[2]["group"].kind == "block"
        assert calls[2]["launch"].exact_block_dim == (8, 4, 1)
        assert calls[2]["source"] == "scoped_block"

        launch_dim[0] = 64
        assert coop._warp.exchange_striped_to_blocked(
            value,
            output=output,
            launch_metadata={"threads_per_block": 64},
        ) == ("exchange-result", 4)
        assert calls[3]["group"].kind == "warp"
        assert calls[3]["launch"].exact_block_dim == (64, 1, 1)
        assert calls[3]["source"] == "scoped_warp"

        assert coop._block.exchange_blocked_to_warp_striped(
            value,
            launch_metadata={"threads_per_block": 64},
        ) == ("exchange-result", 5)
        assert calls[4]["group"].kind == "block"
        assert calls[4]["mode"] == "blocked_to_warp_striped"
        assert calls[4]["source"] == "scoped_block_compatibility"

        plan = _make_group_exchange_plan(
            group=block,
            launch=calls[0]["launch"],
            dtype=int,
            items_per_thread=2,
            mode="blocked_to_striped",
            source="root_test",
        )
        assert plan.call.source == "root_test"
        assert plan.implementation.struct_name == "BlockExchange"
        assert plan.implementation.method_name == "BlockedToStriped"
        assert plan.implementation.template_arguments == {
            "T": int,
            "BLOCK_DIM_X": 8,
            "ITEMS_PER_THREAD": 2,
            "WARP_TIME_SLICING": 0,
            "BLOCK_DIM_Y": 4,
            "BLOCK_DIM_Z": 1,
        }
        assert plan.temp_storage.ownership.value == "implementation"
        assert plan.temp_storage.cpp_type is None
        assert plan.temp_storage.instances is None
        scoped_plan = _make_group_exchange_plan(
            group=coop.this_block(),
            launch=calls[2]["launch"],
            dtype=int,
            items_per_thread=2,
            mode="blocked_to_striped",
            source="scoped_block",
        )
        assert scoped_plan.call.source == "scoped_block"
        assert scoped_plan.semantic_key == plan.semantic_key
        assert scoped_plan.artifact_key == plan.artifact_key

        assert coop.exchange is cute.exchange

        def assert_raises(error_type, message, callback):
            try:
                callback()
            except error_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"expected {error_type.__name__}: {message}")

        launch_dim[0] = 32

        assert_raises(
            TypeError,
            "unexpected keyword argument 'output'",
            lambda: coop.exchange(block, value, output=output),
        )
        assert_raises(
            TypeError,
            "unexpected keyword argument 'source'",
            lambda: coop.exchange(block, value, source="caller"),
        )
        assert_raises(
            TypeError,
            "value must be ThreadData",
            lambda: coop.exchange(
                block,
                7,
            ),
        )
        assert_raises(
            TypeError,
            "takes 2 positional arguments but 3 were given",
            lambda: coop.exchange(
                block,
                value,
                value,
            ),
        )
        too_many = coop.ThreadData.from_values(0, 1, 2, 3, 4, 5, dtype=int)
        assert_raises(
            NotImplementedError,
            "at most 5 items per thread",
            lambda: coop.exchange(
                block,
                too_many,
            ),
        )
        launch_dim[0] = 64
        assert_raises(
            NotImplementedError,
            "at most 5 items per thread",
            lambda: coop.exchange(
                warp,
                too_many,
            ),
        )
        launch_dim[0] = 32
        ranks = coop.ThreadData.from_values(0, 1, dtype=int)
        assert coop.exchange(
            block,
            value,
            mode="scatter_to_blocked",
            ranks=ranks,
            warp_time_slicing=True,
        ) == ("exchange-result", 6)
        assert calls[5]["mode"] == "scatter_to_blocked"
        assert calls[5]["ranks"] is ranks
        assert calls[5]["warp_time_slicing"] is True
        assert_raises(
            ValueError,
            "mode must be one of",
            lambda: coop.exchange(
                block,
                value,
                mode="not_an_exchange_mode",
            ),
        )
        assert_raises(
            NotImplementedError,
            "block, physical-warp, and logical-warp groups",
            lambda: coop.exchange(
                coop.this_cluster(),
                value,
            ),
        )
        launch_dim[0] = 48
        assert_raises(
            NotImplementedError,
            "every physical warp in the enclosing CTA to be complete",
            lambda: coop.exchange(
                coop.this_warp(),
                value,
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr
