# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ..support.source import run_python_with_source as _run_python_with_source

CUDA_COOP_CUTLASS_SCOPE = "cuda.coop.cutlass"
CUDA_COOP_CUTLASS_BLOCK_SCOPE = f"{CUDA_COOP_CUTLASS_SCOPE}._block"
CUDA_COOP_CUTLASS_WARP_SCOPE = f"{CUDA_COOP_CUTLASS_SCOPE}._warp"


def test_unregistered_primitive_raises_explicit_error():
    script = textwrap.dedent(
        f"""
        import cuda.coop.cutlass as coop
        from cuda.coop.cutlass._dsl.block import _dispatch

        CUDA_COOP_CUTLASS_BLOCK_SCOPE = {CUDA_COOP_CUTLASS_BLOCK_SCOPE!r}
        _dispatch._IMPLS.pop("exclusive_sum", None)

        try:
            coop._block.exclusive_sum(1)
        except NotImplementedError as exc:
            assert f"{{CUDA_COOP_CUTLASS_BLOCK_SCOPE}}.exclusive_sum" in str(exc)
            assert "no provider implementation registered yet" in str(exc)
        else:
            raise AssertionError("missing primitive should require a provider")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_unregistered_warp_primitive_raises_explicit_error():
    script = textwrap.dedent(
        f"""
        import cuda.coop.cutlass as coop
        from cuda.coop.cutlass._dsl.warp import _dispatch

        CUDA_COOP_CUTLASS_WARP_SCOPE = {CUDA_COOP_CUTLASS_WARP_SCOPE!r}
        _dispatch._IMPLS.pop("sum", None)

        try:
            coop._warp.sum(1)
        except NotImplementedError as exc:
            assert f"{{CUDA_COOP_CUTLASS_WARP_SCOPE}}.sum" in str(exc)
            assert "no provider implementation registered yet" in str(exc)
        else:
            raise AssertionError("missing primitive should require a warp provider")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_registered_provider_receives_primitive_payload():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        class DynamicInt:
            def __init__(self, label):
                self.label = label

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_sort_pairs", capture)
        payload = coop._block.radix_sort_pairs(
            "keys",
            "values",
            "extra",
            begin_bit=3,
            end_bit=9,
            descending=True,
            launch_metadata={"threads_per_block": 32},
            tag="scratch",
        )

        assert payload == {
            "keys": "keys",
            "values": "values",
            "args": ("extra",),
            "begin_bit": 3,
            "end_bit": 9,
            "descending": True,
            "launch_metadata": {"threads_per_block": 32},
            "tag": "scratch",
        }

        runtime_begin_bit = DynamicInt("begin_bit")
        runtime_end_bit = DynamicInt("end_bit")
        runtime_k = DynamicInt("k")
        runtime_valid = DynamicInt("valid_items")

        runtime_sort_payload = coop._block.radix_sort_pairs(
            "keys",
            "values",
            begin_bit=runtime_begin_bit,
            end_bit=runtime_end_bit,
        )
        assert runtime_sort_payload["begin_bit"] is runtime_begin_bit
        assert runtime_sort_payload["end_bit"] is runtime_end_bit

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("topk_max_keys", capture)
        runtime_topk_payload = coop._block.topk_max_keys(
            "keys",
            runtime_k,
            num_valid=runtime_valid,
            begin_bit=runtime_begin_bit,
            end_bit=runtime_end_bit,
        )
        assert runtime_topk_payload["k"] is runtime_k
        assert runtime_topk_payload["num_valid"] is runtime_valid
        assert runtime_topk_payload["begin_bit"] is runtime_begin_bit
        assert runtime_topk_payload["end_bit"] is runtime_end_bit
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_direct_block_calls_accept_launch_aliases():
    script = textwrap.dedent(
        """
        from numbers import Integral

        import cuda.coop.cutlass as coop

        class IntegralScalar:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return self.value

            def __lt__(self, other):
                return self.value < other

        Integral.register(IntegralScalar)

        def capture(**payload):
            return payload

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", capture)

        thread_count_payload = coop._block.sum(
            5,
            threads_per_block=32,
            tag="direct",
        )
        assert thread_count_payload == {
            "value": 5,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
            "tag": "direct",
        }

        tuple_dim_payload = coop._block.sum(6, threads_per_block=(4, 8, 1))
        assert tuple_dim_payload == {
            "value": 6,
            "args": (),
            "launch_metadata": {"block": (4, 8, 1)},
        }

        dim_payload = coop._block.sum(
            7,
            dim=32,
            launch_config={"tag": "keep"},
        )
        assert dim_payload == {
            "value": 7,
            "args": (),
            "launch_config": {"tag": "keep", "threads_per_block": 32},
        }

        compatible_metadata_payload = coop._block.sum(
            8,
            dim=(4, 8, 1),
            launch_metadata={"threads_per_block": 32, "tag": "keep"},
        )
        assert compatible_metadata_payload == {
            "value": 8,
            "args": (),
            "launch_metadata": {"threads_per_block": 32, "tag": "keep"},
        }

        try:
            coop._block.sum(9, threads_per_block=32, dim=64)
        except TypeError as exc:
            assert "conflicting threads_per_block and dim" in str(exc)
        else:
            raise AssertionError("conflicting direct block-size aliases should fail")

        try:
            coop._block.sum(
                10,
                dim=32,
                launch_metadata={"threads_per_block": 64},
            )
        except TypeError as exc:
            assert "conflicting launch metadata and threads_per_block" in str(exc)
        else:
            raise AssertionError("conflicting launch metadata should fail")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_registered_warp_provider_receives_primitive_payload():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        def native_capture(**payload):
            return payload

        native_capture._supports_native_thread_data = True

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("sum", capture)
        sum_payload = coop._warp.sum(
            5,
            threads_in_warp=16,
            launch_metadata={"threads_per_block": 64},
        )
        assert sum_payload == {
            "value": 5,
            "args": (),
            "threads_in_warp": 16,
            "launch_metadata": {"threads_per_block": 64},
        }

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("reduce", capture)
        reduce_payload = coop._warp.reduce(7, binary_op="max", marker="custom")
        assert reduce_payload == {
            "value": 7,
            "args": (),
            "binary_op": "max",
            "threads_in_warp": 32,
            "marker": "custom",
        }

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("exclusive_scan", capture)
        exclusive_payload = coop._warp.exclusive_scan(
            9,
            scan_op="max",
            initial_value=0,
            threads_in_warp=8,
        )
        assert exclusive_payload == {
            "value": 9,
            "args": (),
            "scan_op": "max",
            "initial_value": 0,
            "threads_in_warp": 8,
        }

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("inclusive_scan", capture)
        inclusive_payload = coop._warp.scan(
            11,
            mode="inclusive",
            scan_op="sum",
            threads_in_warp=4,
        )
        assert inclusive_payload == {
            "value": 11,
            "args": (),
            "scan_op": "sum",
            "initial_value": None,
            "threads_in_warp": 4,
        }

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("min", capture)
        min_payload = coop._warp.min(13, threads_in_warp=32)
        assert min_payload == {
            "value": 13,
            "args": (),
            "threads_in_warp": 32,
        }

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("max", capture)
        max_payload = coop._warp.max(17, threads_in_warp=32)
        assert max_payload == {
            "value": 17,
            "args": (),
            "threads_in_warp": 32,
        }

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("exchange", native_capture)
        values = coop.ThreadData.from_values(1, 2, dtype=int)
        ranks = coop.ThreadData.from_values(1, 0, dtype=int)
        striped_payload = coop._warp.exchange_striped_to_blocked(
            values,
            threads_in_warp=32,
        )
        assert striped_payload == {
            "value": values,
            "args": (),
            "output": None,
            "mode": "striped_to_blocked",
            "threads_in_warp": 32,
        }

        output = coop.ThreadData(2, dtype=int)
        default_output_payload = coop._warp.exchange(
            values,
            output,
            threads_in_warp=32,
        )
        assert default_output_payload == {
            "value": values,
            "args": (),
            "output": output,
            "mode": "striped_to_blocked",
            "threads_in_warp": 32,
        }

        blocked_payload = coop._warp.exchange_blocked_to_striped(
            values,
            output=output,
            threads_in_warp=32,
        )
        assert blocked_payload == {
            "value": values,
            "args": (),
            "output": output,
            "mode": "blocked_to_striped",
            "threads_in_warp": 32,
        }

        exchange_payload = coop._warp.exchange_scatter_to_striped(
            values,
            ranks,
            threads_in_warp=32,
        )
        assert exchange_payload == {
            "value": values,
            "args": (),
            "output": None,
            "mode": "scatter_to_striped",
            "threads_in_warp": 32,
            "ranks": ranks,
        }
        selector_output = coop.ThreadData(2, dtype=int)
        selector_payload = coop._warp.exchange(
            values,
            selector_output,
            ranks,
            warp_exchange_type=coop._warp.WarpExchangeType.ScatterToStriped,
            threads_in_warp=32,
        )
        assert selector_payload == {
            "value": values,
            "args": (),
            "output": selector_output,
            "mode": "scatter_to_striped",
            "threads_in_warp": 32,
            "ranks": ranks,
        }

        natural_scatter_selector_payload = coop._warp.exchange(
            values,
            ranks,
            warp_exchange_type=coop._warp.WarpExchangeType.ScatterToStriped,
            threads_in_warp=32,
        )
        assert natural_scatter_selector_payload == {
            "value": values,
            "args": (),
            "output": None,
            "mode": "scatter_to_striped",
            "threads_in_warp": 32,
            "ranks": ranks,
        }

        keyword_ranks_selector_payload = coop._warp.exchange(
            values,
            selector_output,
            ranks=ranks,
            warp_exchange_type=coop._warp.WarpExchangeType.ScatterToStriped,
            threads_in_warp=32,
        )
        assert keyword_ranks_selector_payload == {
            "value": values,
            "args": (),
            "output": selector_output,
            "mode": "scatter_to_striped",
            "threads_in_warp": 32,
            "ranks": ranks,
        }

        compatible_selector_payload = coop._warp.exchange(
            values,
            output=output,
            warp_exchange_type=3,
            ranks=ranks,
            threads_in_warp=16,
        )
        assert compatible_selector_payload == {
            "value": values,
            "args": (),
            "output": output,
            "mode": "scatter_to_striped",
            "threads_in_warp": 16,
            "ranks": ranks,
        }

        positional_ranks_payload = coop._warp.exchange(
            values,
            ranks,
            mode="scatter_to_striped",
            threads_in_warp=8,
        )
        assert positional_ranks_payload == {
            "value": values,
            "args": (),
            "output": None,
            "mode": "scatter_to_striped",
            "threads_in_warp": 8,
            "ranks": ranks,
        }

        try:
            coop._warp.exchange(
                values,
                selector_output,
                ranks,
                output=output,
                warp_exchange_type=coop._warp.WarpExchangeType.ScatterToStriped,
            )
        except TypeError as exc:
            assert "duplicate output" in str(exc)
        else:
            raise AssertionError("duplicate warp exchange output unexpectedly worked")

        try:
            coop._warp.exchange(
                values,
                mode="blocked_to_striped",
                warp_exchange_type=coop._warp.WarpExchangeType.ScatterToStriped,
            )
        except TypeError as exc:
            assert "conflicting mode and warp_exchange_type" in str(exc)
        else:
            raise AssertionError("conflicting warp exchange selector unexpectedly worked")

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("merge_sort_keys", native_capture)
        sorted_key_payload = coop._warp.merge_sort_keys(
            values,
            compare_op=">",
            threads_in_warp=16,
        )
        assert sorted_key_payload == {
            "keys": values,
            "args": (),
            "compare_op": ">",
            "descending": None,
            "threads_in_warp": 16,
            "valid_items": None,
            "oob_default": None,
        }

        getattr(coop._warp, "_backend", coop._warp)._api.register_provider_impl("merge_sort_pairs", native_capture)
        sorted_pair_payload = coop._warp.merge_sort_pairs(
            values,
            ranks,
            descending=False,
            threads_in_warp=8,
        )
        assert sorted_pair_payload == {
            "keys": values,
            "values": ranks,
            "args": (),
            "compare_op": None,
            "descending": False,
            "threads_in_warp": 8,
            "valid_items": None,
            "oob_default": None,
        }

        try:
            coop._warp.merge_sort_keys(values, valid_items=7)
        except ValueError as exc:
            assert "valid_items and oob_default" in str(exc)
        else:
            raise AssertionError("partial warp merge sort should require oob_default")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_warp_merge_sort_compare_op_validation():
    script = textwrap.dedent(
        """
        import operator

        from cuda.coop.cutlass._dsl.warp._sort import _normalize_compare_op

        assert _normalize_compare_op(None, None) is False
        assert _normalize_compare_op("<", None) is False
        assert _normalize_compare_op("greater", None) is True
        assert _normalize_compare_op(operator.lt, None) is False
        assert _normalize_compare_op(operator.gt, None) is True
        assert _normalize_compare_op(None, True) is True

        try:
            _normalize_compare_op("<", True)
        except ValueError as exc:
            assert "conflicting" in str(exc)
        else:
            raise AssertionError("conflicting compare direction should fail")

        def custom_compare(lhs, rhs):
            return lhs <= rhs

        try:
            _normalize_compare_op(custom_compare, None)
        except NotImplementedError as exc:
            assert "Arbitrary Python callables" in str(exc)
        else:
            raise AssertionError("unknown compare callable should fail")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_scoped_parity_entrypoints_route_to_provider_payloads():
    script = textwrap.dedent(
        """
        from numbers import Integral

        import cuda.coop.cutlass as coop

        class IntegralScalar:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return self.value

            def __lt__(self, other):
                return self.value < other

        Integral.register(IntegralScalar)

        captures = []

        def capture(**payload):
            captures.append(payload)
            return payload

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("inclusive_sum", capture)
        inclusive_payload = coop._block.inclusive_sum(
            5,
            launch_metadata={"threads_per_block": 32},
        )

        assert inclusive_payload == {
            "value": 5,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("exclusive_scan", capture)
        exclusive_scan_payload = coop._block.exclusive_scan(
            6,
            scan_op="max",
            initial_value=0,
            launch_metadata={"threads_per_block": 32},
        )
        assert exclusive_scan_payload == {
            "value": 6,
            "args": (),
            "scan_op": "max",
            "initial_value": 0,
            "launch_metadata": {"threads_per_block": 32},
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("inclusive_scan", capture)
        inclusive_scan_payload = coop._block.scan(
            8,
            mode="inclusive",
            scan_op="sum",
            launch_metadata={"threads_per_block": 32},
        )
        assert inclusive_scan_payload == {
            "value": 8,
            "args": (),
            "scan_op": "sum",
            "initial_value": None,
            "launch_metadata": {"threads_per_block": 32},
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("reduce", capture)
        reduce_payload = coop._block.reduce(7, binary_op="max", marker="custom")
        assert reduce_payload == {
            "value": 7,
            "args": (),
            "binary_op": "max",
            "marker": "custom",
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("adjacent_difference_subtract_left", capture)
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("adjacent_difference_subtract_right", capture)
        left_diff_payload = coop._block.adjacent_difference(
            10,
            block_adjacent_difference_type=coop._block.BlockAdjacentDifferenceType.SubtractLeft,
            launch_metadata={"threads_per_block": 32},
        )
        assert left_diff_payload == {
            "value": 10,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }
        right_diff_payload = coop._block.adjacent_difference_subtract_right(
            11,
            launch_metadata={"threads_per_block": 32},
        )
        assert right_diff_payload == {
            "value": 11,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }
        right_diff_selector_payload = coop._block.adjacent_difference(
            12,
            block_adjacent_difference_type=coop._block.BlockAdjacentDifferenceType.SubtractRight,
            difference_op="subtract",
            launch_metadata={"threads_per_block": 32},
        )
        assert right_diff_selector_payload == {
            "value": 12,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }
        right_diff_string_payload = coop._block.adjacent_difference(
            13,
            "right",
            launch_metadata={"threads_per_block": 32},
        )
        assert right_diff_string_payload == {
            "value": 13,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }
        try:
            coop._block.adjacent_difference(14, difference_op=lambda lhs, rhs: lhs + rhs)
        except NotImplementedError as exc:
            assert "Arbitrary Python difference_op callables" in str(exc)
        else:
            raise AssertionError("custom adjacent_difference callable unexpectedly worked")

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("discontinuity_flag_heads", capture)
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("discontinuity_flag_tails", capture)
        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("discontinuity_flag_heads_and_tails", capture)
        head_payload = coop._block.discontinuity(
            15,
            launch_metadata={"threads_per_block": 32},
        )
        assert head_payload == {
            "value": 15,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }
        tail_payload = coop._block.discontinuity(
            16,
            block_discontinuity_type=coop._block.BlockDiscontinuityType.TAILS,
            launch_metadata={"threads_per_block": 32},
        )
        assert tail_payload == {
            "value": 16,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }
        heads_tails_payload = coop._block.discontinuity(
            17,
            block_discontinuity_type="heads-and-tails",
            flag_op="!=",
            launch_metadata={"threads_per_block": 32},
        )
        assert heads_tails_payload == {
            "value": 17,
            "args": (),
            "launch_metadata": {"threads_per_block": 32},
        }
        try:
            coop._block.discontinuity(18, flag_op=lambda lhs, rhs: lhs == rhs)
        except NotImplementedError as exc:
            assert "Arbitrary Python flag_op callables" in str(exc)
        else:
            raise AssertionError("custom discontinuity callable unexpectedly worked")

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("shuffle", capture)
        shuffle_payload = coop._block.shuffle_rotate(
            12,
            distance=3,
            launch_metadata={"threads_per_block": 32},
        )
        assert shuffle_payload == {
            "value": 12,
            "args": (),
            "mode": "rotate",
            "distance": 3,
            "launch_metadata": {"threads_per_block": 32},
        }

        class ShuffleMode:
            name = "Up"

        shuffle_mode_payload = coop._block.shuffle(
            13,
            mode=ShuffleMode,
            launch_metadata={"threads_per_block": 32},
        )
        assert shuffle_mode_payload == {
            "value": 13,
            "args": (),
            "mode": "up",
            "distance": 1,
            "launch_metadata": {"threads_per_block": 32},
        }

        selector_shuffle_payload = coop._block.shuffle(
            13,
            block_shuffle_type=coop._block.BlockShuffleType.Rotate,
            distance=4,
            launch_metadata={"threads_per_block": 32},
        )
        assert selector_shuffle_payload == {
            "value": 13,
            "args": (),
            "mode": "rotate",
            "distance": 4,
            "launch_metadata": {"threads_per_block": 32},
        }

        compatible_selector_shuffle_payload = coop._block.shuffle(
            13,
            block_shuffle_type=4,
            distance=2,
            launch_metadata={"threads_per_block": 32},
        )
        assert compatible_selector_shuffle_payload == {
            "value": 13,
            "args": (),
            "mode": "down",
            "distance": 2,
            "launch_metadata": {"threads_per_block": 32},
        }

        try:
            coop._block.shuffle(
                13,
                mode="up",
                block_shuffle_type=coop._block.BlockShuffleType.Down,
            )
        except TypeError as exc:
            assert "conflicting mode and block_shuffle_type" in str(exc)
        else:
            raise AssertionError("conflicting block shuffle selector unexpectedly worked")

        shuffle_offset_payload = coop._block.shuffle_offset(
            14,
            distance=-2,
            launch_metadata={"threads_per_block": 32},
        )
        assert shuffle_offset_payload == {
            "value": 14,
            "args": (),
            "mode": "offset",
            "distance": -2,
            "launch_metadata": {"threads_per_block": 32},
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("exchange", capture)
        exchange_payload = coop._block.exchange_blocked_to_striped(
            "items",
            launch_metadata={"threads_per_block": 32},
        )
        assert exchange_payload == {
            "value": "items",
            "args": (),
            "output": None,
            "mode": "blocked_to_striped",
            "launch_metadata": {"threads_per_block": 32},
        }
        selector_output_payload = coop._block.exchange(
            "items",
            "out",
            block_exchange_type=coop._block.BlockExchangeType.BlockedToStriped,
            launch_metadata={"threads_per_block": 32},
        )
        assert selector_output_payload == {
            "value": "items",
            "args": (),
            "output": "out",
            "mode": "blocked_to_striped",
            "launch_metadata": {"threads_per_block": 32},
        }

        selector_scatter_payload = coop._block.exchange(
            "items",
            "ranks",
            block_exchange_type=coop._block.BlockExchangeType.ScatterToStriped,
            launch_metadata={"threads_per_block": 32},
        )
        assert selector_scatter_payload == {
            "value": "items",
            "args": (),
            "output": None,
            "mode": "scatter_to_striped",
            "ranks": "ranks",
            "launch_metadata": {"threads_per_block": 32},
        }

        selector_flagged_keywords_payload = coop._block.exchange(
            "items",
            "ranks",
            valid_flags="flags",
            block_exchange_type=coop._block.BlockExchangeType.ScatterToStripedFlagged,
            launch_metadata={"threads_per_block": 32},
        )
        assert selector_flagged_keywords_payload == {
            "value": "items",
            "args": (),
            "output": None,
            "mode": "scatter_to_striped_flagged",
            "ranks": "ranks",
            "valid_flags": "flags",
            "launch_metadata": {"threads_per_block": 32},
        }

        selector_flagged_payload = coop._block.exchange(
            "items",
            "out",
            "ranks",
            "flags",
            block_exchange_type=coop._block.BlockExchangeType.ScatterToStripedFlagged,
            launch_metadata={"threads_per_block": 32},
        )
        assert selector_flagged_payload == {
            "value": "items",
            "args": (),
            "output": "out",
            "mode": "scatter_to_striped_flagged",
            "ranks": "ranks",
            "valid_flags": "flags",
            "launch_metadata": {"threads_per_block": 32},
        }

        compatible_selector_payload = coop._block.exchange(
            "items",
            output="out",
            ranks="ranks",
            block_exchange_type=6,
            launch_metadata={"threads_per_block": 32},
        )
        assert compatible_selector_payload == {
            "value": "items",
            "args": (),
            "output": "out",
            "mode": "scatter_to_striped",
            "ranks": "ranks",
            "launch_metadata": {"threads_per_block": 32},
        }

        try:
            coop._block.exchange(
                "items",
                mode="blocked_to_striped",
                block_exchange_type=coop._block.BlockExchangeType.ScatterToStriped,
            )
        except TypeError as exc:
            assert "conflicting mode and block_exchange_type" in str(exc)
        else:
            raise AssertionError("conflicting block exchange selector unexpectedly worked")

        scatter_payload = coop._block.exchange_scatter_to_striped_flagged(
            "items",
            "ranks",
            "flags",
            launch_metadata={"threads_per_block": 32},
        )
        assert scatter_payload == {
            "value": "items",
            "args": (),
            "output": None,
            "mode": "scatter_to_striped_flagged",
            "ranks": "ranks",
            "valid_flags": "flags",
            "launch_metadata": {"threads_per_block": 32},
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("histogram", capture)
        class HistogramAlgorithm:
            name = "Block-Histo-Sort"

        histogram_payload = coop._block.histogram(
            "samples",
            bins=16,
            bins_per_thread=2,
            counter_dtype="counter",
            algorithm=HistogramAlgorithm,
            launch_metadata={"threads_per_block": 32},
        )
        assert histogram_payload == {
            "samples": "samples",
            "args": (),
            "bins": 16,
            "bins_per_thread": 2,
            "algorithm": "sort",
            "counter_dtype": "counter",
            "launch_metadata": {"threads_per_block": 32},
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("run_length_decode", capture)
        run_length_payload = coop._block.run_length_decode(
            "values",
            "lengths",
            decoded_items_per_thread=IntegralScalar(4),
            decoded_window_offset=2,
            relative_offsets="offsets",
            total_decoded_size="total",
            launch_metadata={"threads_per_block": 32},
        )
        assert run_length_payload == {
            "run_values": "values",
            "run_lengths": "lengths",
            "args": (),
            "decoded_items_per_thread": 4,
            "decoded_window_offset": 2,
            "relative_offsets": "offsets",
            "total_decoded_size": "total",
            "launch_metadata": {"threads_per_block": 32},
        }
        run_length = coop._block.run_length(
            "values",
            "lengths",
            decoded_items_per_thread=4,
            total_decoded_size="total",
            launch_metadata={"threads_per_block": 32},
        )
        assert isinstance(run_length, coop._block.BlockRunLengthDecode)
        run_length_wrapper_payload = run_length.decode(
            2,
            relative_offsets="offsets",
        )
        assert run_length_wrapper_payload == run_length_payload

        decoded_payloads = []

        def capture_decoded(**payload):
            decoded_payloads.append(payload)
            return coop.ThreadData.from_values(
                "d0",
                "d1",
                "d2",
                "d3",
                dtype="decoded",
            )

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("run_length_decode", capture_decoded)
        decoded_output = coop.ThreadData(4)
        relative_offsets_output = coop.ThreadData(4)
        decoded = run_length.decode(decoded_output, 2, relative_offsets_output)
        assert decoded is decoded_output
        assert decoded.dtype == "decoded"
        assert [decoded[idx] for idx in range(4)] == ["d0", "d1", "d2", "d3"]
        assert decoded_payloads[-1]["decoded_window_offset"] == 2
        assert decoded_payloads[-1]["relative_offsets"] is relative_offsets_output

        try:
            run_length.decode(
                decoded_output,
                2,
                relative_offsets_output,
                relative_offsets="duplicate",
            )
        except TypeError as exc:
            assert "duplicate relative_offsets" in str(exc)
        else:
            raise AssertionError("duplicate relative_offsets unexpectedly accepted")

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_rank", capture)
        rank_payload = coop._block.radix_rank(
            "keys",
            begin_bit=3,
            radix_bits=5,
            descending=True,
        )
        assert rank_payload == {
            "keys": "keys",
            "args": (),
            "begin_bit": 3,
            "end_bit": None,
            "radix_bits": 5,
            "descending": True,
        }
        prefix = coop.ThreadData(2)
        rank_prefix_payload = coop._block.radix_rank(
            "keys",
            begin_bit=0,
            end_bit=4,
            exclusive_digit_prefix=prefix,
        )
        assert rank_prefix_payload == {
            "keys": "keys",
            "args": (),
            "begin_bit": 0,
            "end_bit": 4,
            "radix_bits": None,
            "descending": False,
            "exclusive_digit_prefix": prefix,
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_sort_keys", capture)
        keys_payload = coop._block.radix_sort_keys_descending(
            "keys",
            begin_bit=2,
            end_bit=9,
        )
        assert keys_payload == {
            "keys": "keys",
            "args": (),
            "begin_bit": 2,
            "end_bit": 9,
            "descending": True,
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("radix_sort_pairs", capture)
        pairs_payload = coop._block.radix_sort_pairs_descending("keys", "values")
        assert pairs_payload == {
            "keys": "keys",
            "values": "values",
            "args": (),
            "begin_bit": 0,
            "end_bit": None,
            "descending": True,
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("topk_max_keys", capture)
        topk_keys_payload = coop._block.topk_max_keys(
            "keys",
            7,
            begin_bit=2,
            end_bit=9,
        )
        assert topk_keys_payload == {
            "keys": "keys",
            "k": 7,
            "args": (),
            "num_valid": None,
            "begin_bit": 2,
            "end_bit": 9,
            "descending": True,
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("topk_min_pairs", capture)
        topk_pairs_payload = coop._block.topk_min_pairs(
            "keys",
            "values",
            5,
            num_valid=11,
        )
        assert topk_pairs_payload == {
            "keys": "keys",
            "values": "values",
            "k": 5,
            "args": (),
            "num_valid": 11,
            "begin_bit": 0,
            "end_bit": None,
            "descending": False,
        }

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("merge_sort_keys", capture)
        merge_payload = coop._block.merge_sort_keys(
            "keys",
            descending=True,
            valid_items=13,
            oob_default=-1,
        )
        assert merge_payload == {
            "keys": "keys",
            "args": (),
            "descending": True,
            "valid_items": 13,
            "oob_default": -1,
        }
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_load_store_bind_conditional_loop_values():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop

        deferred = []

        def if_generate(condition, then_fn, else_fn=None, **kwargs):
            deferred.append((condition, then_fn, else_fn))
            return f"slot-{len(deferred)}"

        cutlass_mod = types.ModuleType("cutlass")
        cutlass_mod.__path__ = []
        cutlass_mod.if_generate = if_generate
        cute_mod = types.ModuleType("cutlass.cute")
        cute_mod.arch = types.SimpleNamespace(
            thread_idx=lambda: (0, 0, 0),
            block_dim=lambda: (1, 1, 1),
        )
        sys.modules["cutlass"] = cutlass_mod
        sys.modules["cutlass.cute"] = cute_mod

        try:
            coop._block.load(["a", "b"], coop.ThreadData(2, dtype=str), 4, 7, dtype=str)
        except TypeError as exc:
            assert "ambiguous positional args" in str(exc)
        else:
            raise AssertionError("ambiguous load positional args should fail")

        for algorithm in (
            "transpose",
            "warp_transpose",
            "warp_transpose_timesliced",
        ):
            try:
                coop._block.load(
                    ["a", "b"],
                    items_per_thread=1,
                    algorithm=algorithm,
                    dtype=str,
                )
            except NotImplementedError as exc:
                assert algorithm in str(exc)
            else:
                raise AssertionError("unsupported load algorithm should fail")

        loaded = coop._block.load(
            ["a", "b", "c"],
            items_per_thread=3,
            valid_items=2,
            oob_default="z",
            dtype=str,
        )
        assert tuple(loaded) == ("slot-1", "slot-2", "slot-3")
        assert [
            then_fn() if condition else else_fn()
            for condition, then_fn, else_fn in deferred
        ] == ["a", "b", "z"]

        direct_dim = coop._block.load(["q"], items_per_thread=1, dtype=str, dim=(1, 1, 1))
        assert tuple(direct_dim) == ("q",)

        try:
            coop._block.load(["q"], items_per_thread=1, dtype=str, threads_per_block=1, dim=2)
        except TypeError as exc:
            assert "conflicting threads_per_block and dim" in str(exc)
        else:
            raise AssertionError("conflicting load launch aliases should fail")

        try:
            coop._block.load(
                ["q"],
                items_per_thread=1,
                dtype=str,
                threads_per_block=1,
                launch_metadata={"threads_per_block": 2},
            )
        except TypeError as exc:
            assert "conflicting launch metadata and threads_per_block" in str(exc)
        else:
            raise AssertionError("conflicting load launch metadata should fail")

        deferred.clear()
        destination = {}
        coop._block.store(
            destination,
            coop.ThreadData.from_values("a", "b", "c", dtype=str),
            valid_items=2,
        )
        assert destination == {}
        for condition, then_fn, _ in deferred:
            if condition:
                then_fn()
        assert destination == {0: "a", 1: "b"}

        direct_dim_destination = {}
        coop._block.store(
            direct_dim_destination,
            coop.ThreadData.from_values("x", dtype=str),
            dim=1,
        )
        assert direct_dim_destination == {0: "x"}
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_warp_load_store_bind_logical_warp_tile_values():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop

        deferred = []

        def if_generate(condition, then_fn, else_fn=None, **kwargs):
            deferred.append((condition, then_fn, else_fn))
            return f"slot-{len(deferred)}"

        cutlass_mod = types.ModuleType("cutlass")
        cutlass_mod.__path__ = []
        cutlass_mod.if_generate = if_generate
        cute_mod = types.ModuleType("cutlass.cute")
        cute_mod.arch = types.SimpleNamespace(
            thread_idx=lambda: (34, 0, 0),
            block_dim=lambda: (64, 1, 1),
        )
        sys.modules["cutlass"] = cutlass_mod
        sys.modules["cutlass.cute"] = cute_mod

        source = list(range(128))
        direct = coop._warp.load(
            source,
            items_per_thread=2,
            threads_in_warp=32,
            dtype=int,
        )
        assert isinstance(direct, coop.ThreadData)
        assert tuple(direct) == (68, 69)

        output_without_dtype = coop.ThreadData(1)
        coop._warp.load(
            source,
            output_without_dtype,
            items_per_thread=1,
            dtype=int,
        )
        assert output_without_dtype.dtype is int

        try:
            coop._warp.load(
                source,
                coop.ThreadData(1, dtype=str),
                items_per_thread=1,
                dtype=int,
            )
        except TypeError as exc:
            assert "dtype does not match output.dtype" in str(exc)
        else:
            raise AssertionError("mismatched load output dtype should fail")

        failed_output = coop.ThreadData(2)
        try:
            coop._warp.load(
                source,
                failed_output,
                items_per_thread=1,
                dtype=int,
            )
        except ValueError as exc:
            assert "ThreadData.items_per_thread" in str(exc)
        else:
            raise AssertionError("mismatched load item count should fail")
        assert failed_output.dtype is None

        try:
            coop._warp.load(
                source,
                items_per_thread=1,
                oob_default=-1,
                dtype=int,
            )
        except TypeError as exc:
            assert "oob_default requires valid_items" in str(exc)
        else:
            raise AssertionError("oob_default without valid_items should fail")

        striped = coop._warp.load(
            source,
            items_per_thread=2,
            threads_in_warp=32,
            algorithm="striped",
            dtype=int,
        )
        assert tuple(striped) == (66, 98)

        for threads_in_warp in (0, 3, 64):
            try:
                coop._warp.load(
                    source,
                    items_per_thread=1,
                    threads_in_warp=threads_in_warp,
                    dtype=int,
                )
            except ValueError as exc:
                assert "threads_in_warp" in str(exc)
            else:
                raise AssertionError("invalid logical warp size should fail")

        try:
            coop._warp.load(
                source,
                items_per_thread=1,
                algorithm="transpose",
                dtype=int,
            )
        except NotImplementedError as exc:
            assert "transpose" in str(exc)
        else:
            raise AssertionError("unsupported warp load algorithm should fail")

        try:
            coop._warp.load(
                source,
                items_per_thread=1,
                algorithm="unknown",
                dtype=int,
            )
        except ValueError as exc:
            assert "direct" in str(exc)
            assert "striped" in str(exc)
            assert "vectorize" in str(exc)
            assert "transpose" not in str(exc)
        else:
            raise AssertionError("unknown warp load algorithm should fail")

        destination = {}
        coop._warp.store(
            destination,
            coop.ThreadData.from_values("a", "b", dtype=str),
            threads_in_warp=32,
        )
        assert destination == {68: "a", 69: "b"}

        untyped = coop.ThreadData.from_values("c")
        coop._warp.store(destination, untyped, threads_in_warp=32, dtype=str)
        assert untyped.dtype is None

        try:
            coop._warp.store(
                destination,
                coop.ThreadData.from_values("d", dtype=str),
                threads_in_warp=32,
                dtype=int,
            )
        except TypeError as exc:
            assert "dtype does not match value.dtype" in str(exc)
        else:
            raise AssertionError("mismatched store dtype should fail")

        failed_value = coop.ThreadData(2)
        try:
            coop._warp.store(
                destination,
                failed_value,
                items_per_thread=1,
                threads_in_warp=32,
                dtype=str,
            )
        except ValueError as exc:
            assert "ThreadData.items_per_thread" in str(exc)
        else:
            raise AssertionError("mismatched store item count should fail")
        assert failed_value.dtype is None

        destination.clear()
        coop._warp.store(
            destination,
            coop.ThreadData.from_values("a", "b", dtype=str),
            threads_in_warp=32,
            algorithm="striped",
        )
        assert destination == {66: "a", 98: "b"}

        deferred.clear()
        partial = coop._warp.load(
            source,
            items_per_thread=2,
            threads_in_warp=32,
            valid_items=69,
            oob_default=-1,
            dtype=int,
        )
        assert tuple(partial) == ("slot-1", "slot-2")
        assert [
            then_fn() if condition else else_fn()
            for condition, then_fn, else_fn in deferred
        ] == [68, -1]

        deferred.clear()
        destination.clear()
        coop._warp.store(
            destination,
            coop.ThreadData.from_values("a", "b", dtype=str),
            threads_in_warp=32,
            valid_items=69,
        )
        assert destination == {}
        for condition, then_fn, _ in deferred:
            if condition:
                then_fn()
        assert destination == {68: "a"}
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_builtin_reduce_rejects_unknown_ops_before_provider_import():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        try:
            coop._block.reduce(
                1,
                binary_op="not_an_op",
                launch_metadata={"threads_per_block": 32},
            )
        except NotImplementedError as exc:
            assert "supports sum" in str(exc)
            assert "max" in str(exc)
        else:
            raise AssertionError("builtin reduce should reject unknown reductions")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_builtin_scan_reduce_known_callable_aliases_before_provider_import():
    script = textwrap.dedent(
        """
        import operator

        from cuda.coop.cutlass._dsl.block._reduce import _normalize_reduce_op
        from cuda.coop.cutlass._dsl.block._scan import _normalize_scan_op

        class FakeNumpyUfunc:
            __module__ = "numpy"
            __name__ = "multiply"

            def __call__(self, lhs, rhs):
                return lhs * rhs

        assert _normalize_reduce_op(operator.add) == "sum"
        assert _normalize_reduce_op(operator.mul) == "multiplies"
        assert _normalize_reduce_op(operator.and_) == "bit_and"
        assert _normalize_reduce_op(operator.or_) == "bit_or"
        assert _normalize_reduce_op(operator.xor) == "bit_xor"
        assert _normalize_reduce_op(FakeNumpyUfunc()) == "multiplies"
        assert _normalize_scan_op(operator.xor) == "bit_xor"
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_builtin_scan_rejects_unknown_ops_and_inclusive_initial_value():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        try:
            coop._block.exclusive_scan(
                1,
                scan_op="not_an_op",
                launch_metadata={"threads_per_block": 32},
            )
        except NotImplementedError as exc:
            assert "supports sum" in str(exc)
        else:
            raise AssertionError("builtin scan should reject unknown scans")

        try:
            coop._block.inclusive_scan(
                1,
                initial_value=0,
                launch_metadata={"threads_per_block": 32},
            )
        except ValueError as exc:
            assert "initial_value is not supported" in str(exc)
        else:
            raise AssertionError("inclusive_scan should reject initial_value")

        try:
            coop._block.scan(1, mode="sideways")
        except ValueError as exc:
            assert "mode must be" in str(exc)
        else:
            raise AssertionError("invalid scan mode should fail")

        try:
            coop._block.radix_rank(
                1,
                begin_bit=2,
                end_bit=8,
                radix_bits=5,
                launch_metadata={"threads_per_block": 32},
            )
        except ValueError as exc:
            assert "radix_bits must match" in str(exc)
        else:
            raise AssertionError("mismatched radix_rank bit range should fail")

        try:
            coop._block.shuffle(
                1,
                mode="sideways",
                launch_metadata={"threads_per_block": 32},
            )
        except ValueError as exc:
            assert "mode must be" in str(exc)
        else:
            raise AssertionError("invalid shuffle mode should fail")

        try:
            coop._block.merge_sort_keys(1, valid_items=7)
        except ValueError as exc:
            assert "valid_items and oob_default" in str(exc)
        else:
            raise AssertionError("partial merge_sort_keys should require oob_default")
            """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_registered_provider_receives_opaque_context_keywords():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def capture(**payload):
            return payload

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", capture)
        payload = coop._block.sum(
            1,
            thread_data="external-thread-data",
            temp_storage="external-temp-storage",
        )

        assert payload == {
            "value": 1,
            "args": (),
            "thread_data": "external-thread-data",
            "temp_storage": "external-temp-storage",
        }
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_registered_provider_maps_thread_data_without_launch_metadata():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        def increment(**payload):
            return payload["value"] + 1

        getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("sum", increment)
        result = coop._block.sum(coop.ThreadData.from_values(1, 2))

        assert isinstance(result, coop.ThreadData)
        assert tuple(result) == (2, 3)
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_validate_no_extra_args_ignores_launch_metadata_and_keeps_hint():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass._dsl._scope import (
            validate_no_extra_block_args as validate_no_extra_args,
        )

        validate_no_extra_args(
            "sum",
            args=(),
            kwargs={"launch_metadata": {"threads_per_block": 32}},
            expected="currently expects one positional value",
        )

        try:
            validate_no_extra_args(
                "sum",
                args=(),
                kwargs={"unexpected": 1},
                expected="currently expects one positional value",
            )
        except TypeError as exc:
            assert "currently expects one positional value" in str(exc)
            assert "unexpected" in str(exc)
        else:
            raise AssertionError("unexpected kwargs should fail")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_reserved_payload_keys_cannot_be_overwritten():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        try:
            coop._block.exclusive_sum(1, value=2)
        except TypeError as exc:
            assert "reserved keyword argument" in str(exc)
            assert "value" in str(exc)
        else:
            raise AssertionError("value keyword should not override positional value")

        try:
            coop._block.radix_sort_keys("keys", args=())
        except TypeError as exc:
            assert "reserved keyword argument" in str(exc)
            assert "args" in str(exc)
        else:
            raise AssertionError("args keyword should not override captured args")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_register_provider_rejects_invalid_inputs():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        try:
            getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("", lambda **payload: payload)
        except ValueError as exc:
            assert "primitive_name" in str(exc)
        else:
            raise AssertionError("empty primitive name should be rejected")

        try:
            getattr(coop._block, "_backend", coop._block)._api.register_provider_impl("exclusive_sum", object())
        except TypeError as exc:
            assert "impl must be callable" in str(exc)
        else:
            raise AssertionError("non-callable provider should be rejected")
        """
    )

    result = _run_python_with_source(script)

    assert result.returncode == 0, result.stderr
