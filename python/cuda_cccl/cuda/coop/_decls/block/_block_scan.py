# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from typing import Any, Callable, Literal, Optional, Union

from numba.core import errors, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import Signature
from numba.extending import models, register_model, typeof_impl

import cuda.coop as coop

from ..._scan_op import ScanOp
from ..._typing import ScanOpType
from ...block._block_scan import _make_scan_rewrite as _make_block_scan_rewrite
from .. import (
    CoopAbstractTemplate,
    CoopDeclMixin,
    CoopInstanceTemplate,
    CoopInstanceTypeMixin,
    TempStorageType,
    ThreadDataType,
    process_algorithm,
    process_items_per_thread,
    register,
    validate_items_per_thread,
    validate_src_dst,
    validate_temp_storage,
)


# =============================================================================
# Scan
# =============================================================================
class CoopBlockScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.scan
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        initial_value: Optional[Any] = None,
        mode: Literal["exclusive", "inclusive"] = "exclusive",
        scan_op: ScanOpType = "+",
        block_prefix_callback_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        """
        This method defines the signature of the cooperative block scan
        function. It validates the parameters and returns a signature object.
        """
        return inspect.signature(CoopBlockScanDecl.signature).bind(
            src,
            dst,
            items_per_thread=items_per_thread,
            mode=mode,
            scan_op=scan_op,
            initial_value=initial_value,
            block_prefix_callback_op=block_prefix_callback_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def signature_instance(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        initial_value: Optional[Any] = None,
        *,
        items_per_thread: int = None,
        mode: Optional[Literal["exclusive", "inclusive"]] = None,
        scan_op: ScanOpType = None,
        block_prefix_callback_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return inspect.signature(CoopBlockScanDecl.signature_instance).bind(
            src,
            dst,
            initial_value=initial_value,
            items_per_thread=items_per_thread,
            mode=mode,
            scan_op=scan_op,
            block_prefix_callback_op=block_prefix_callback_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    @staticmethod
    def get_instance_type():
        return block_scan_instance_type

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        src = bound.arguments["src"]
        dst = bound.arguments.get("dst")

        src_is_array = isinstance(src, (types.Array, ThreadDataType))
        dst_is_array = isinstance(dst, (types.Array, ThreadDataType))
        src_is_scalar = isinstance(src, types.Number)
        dst_is_scalar = isinstance(dst, types.Number)

        scalar_return = dst is None and src_is_scalar

        if scalar_return:
            arglist = [src]
            pysig_params = [
                inspect.Parameter("src", inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            items_per_thread = bound.arguments.get("items_per_thread")
            items_per_thread_is_none_type = isinstance(items_per_thread, types.NoneType)
            if items_per_thread_is_none_type:
                arglist.append(items_per_thread)
                pysig_params.append(
                    inspect.Parameter(
                        "items_per_thread",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
                items_per_thread = None
            if not items_per_thread_is_none_type and items_per_thread is not None:
                maybe_literal = validate_items_per_thread(self, items_per_thread)
                if maybe_literal is not None:
                    items_per_thread = maybe_literal
                if isinstance(items_per_thread, types.IntegerLiteral):
                    if items_per_thread.literal_value != 1:
                        raise errors.TypingError(
                            f"{self.primitive_name} requires items_per_thread == 1 "
                            "for scalar inputs"
                        )
                elif isinstance(items_per_thread, types.Integer):
                    raise errors.TypingError(
                        f"{self.primitive_name} requires items_per_thread to be a "
                        "compile-time literal for scalar inputs"
                    )
                arglist.append(items_per_thread)
                pysig_params.append(
                    inspect.Parameter(
                        "items_per_thread",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
        else:
            if src_is_scalar or dst_is_scalar:
                raise errors.TypingError(
                    f"{self.primitive_name} requires scalar inputs to omit dst"
                )
            if not (src_is_array and dst_is_array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires src and dst to be both arrays "
                    "or src-only scalar input"
                )
            validate_src_dst(self, src, dst)
            arglist = [src, dst]
            pysig_params = [
                inspect.Parameter("src", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("dst", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ]

        if not scalar_return:
            items_per_thread = bound.arguments.get("items_per_thread")
            items_per_thread_is_none_type = isinstance(items_per_thread, types.NoneType)
            if items_per_thread_is_none_type:
                arglist.append(items_per_thread)
                pysig_params.append(
                    inspect.Parameter(
                        "items_per_thread",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
                items_per_thread = None
            if not items_per_thread_is_none_type and not (
                two_phase and items_per_thread is None
            ):
                process_items_per_thread(
                    self,
                    bound,
                    arglist,
                    two_phase,
                    target_array=(src, dst),
                )
                pysig_params.append(
                    inspect.Parameter(
                        "items_per_thread",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        initial_value = bound.arguments.get("initial_value")
        initial_value_is_none_type = isinstance(initial_value, types.NoneType)
        if initial_value_is_none_type:
            arglist.append(initial_value)
            pysig_params.append(
                inspect.Parameter(
                    "initial_value",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            initial_value = None
        if not initial_value_is_none_type:
            if isinstance(initial_value, types.IntegerLiteral):
                # If initial_value is an IntegerLiteral, we can use it directly.
                initial_value = initial_value.literal_value
            if initial_value is not None:
                arglist.append(initial_value)
                pysig_params.append(
                    inspect.Parameter(
                        "initial_value",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        mode = bound.arguments.get("mode")
        mode_is_none_type = isinstance(mode, types.NoneType)
        if mode_is_none_type:
            arglist.append(mode)
            pysig_params.append(
                inspect.Parameter(
                    "mode", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                )
            )
            mode = None
        if not mode_is_none_type:
            if mode is None:
                if not two_phase:
                    mode = "exclusive"
            elif isinstance(mode, types.StringLiteral):
                mode = mode.literal_value
            if mode is not None:
                if mode not in ("inclusive", "exclusive"):
                    raise errors.TypingError(
                        f"Invalid mode '{mode}' for {self.primitive_name}; expected "
                        "'inclusive' or 'exclusive'"
                    )
                arglist.append(mode)
                pysig_params.append(
                    inspect.Parameter(
                        "mode", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                    )
                )

        scan_op = bound.arguments.get("scan_op")
        scan_op_is_none_type = isinstance(scan_op, types.NoneType)
        if scan_op_is_none_type:
            arglist.append(scan_op)
            pysig_params.append(
                inspect.Parameter(
                    "scan_op", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                )
            )
            scan_op = None
        if not scan_op_is_none_type:
            if scan_op is None:
                if not two_phase:
                    scan_op = "+"
            if scan_op is not None:
                if isinstance(scan_op, types.StringLiteral):
                    scan_op = scan_op.literal_value
                try:
                    scan_op = ScanOp(scan_op)
                except ValueError as e:
                    raise errors.TypingError(
                        f"Invalid scan_op '{scan_op}' for {self.primitive_name}: {e}"
                    )
                arglist.append(scan_op)
                pysig_params.append(
                    inspect.Parameter(
                        "scan_op",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        block_prefix_callback_op = bound.arguments.get("block_prefix_callback_op")
        block_prefix_is_none_type = isinstance(block_prefix_callback_op, types.NoneType)
        if block_prefix_is_none_type:
            arglist.append(block_prefix_callback_op)
            pysig_params.append(
                inspect.Parameter(
                    "block_prefix_callback_op",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            block_prefix_callback_op = None
        if not block_prefix_is_none_type and block_prefix_callback_op is not None:
            # We can't do much validation here.
            arglist.append(block_prefix_callback_op)
            pysig_params.append(
                inspect.Parameter(
                    "block_prefix_callback_op",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )

        block_aggregate = bound.arguments.get("block_aggregate")
        block_aggregate_is_none_type = isinstance(block_aggregate, types.NoneType)
        if block_aggregate_is_none_type:
            arglist.append(block_aggregate)
            pysig_params.append(
                inspect.Parameter(
                    "block_aggregate",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            block_aggregate = None
        if not block_aggregate_is_none_type and block_aggregate is not None:
            if block_prefix_callback_op is not None:
                raise errors.TypingError(
                    f"{self.primitive_name} does not support block_aggregate when "
                    "block_prefix_callback_op is provided"
                )
            if not isinstance(block_aggregate, types.Array):
                raise errors.TypingError(
                    f"{self.primitive_name} requires block_aggregate to be a device "
                    "array"
                )
            if scalar_return:
                expected_dtype = src
            else:
                expected_dtype = src.dtype if isinstance(src, types.Array) else src
                if ThreadDataType is not None and isinstance(src, ThreadDataType):
                    expected_dtype = src.dtype
            if block_aggregate.dtype != expected_dtype:
                raise errors.TypingError(
                    f"{self.primitive_name} requires block_aggregate to have the same "
                    "dtype as the input"
                )
            arglist.append(block_aggregate)
            pysig_params.append(
                inspect.Parameter(
                    "block_aggregate",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )

        algorithm = bound.arguments.get("algorithm")
        algorithm_is_none_type = isinstance(algorithm, types.NoneType)
        if algorithm_is_none_type:
            arglist.append(algorithm)
            pysig_params.append(
                inspect.Parameter(
                    "algorithm",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            algorithm = None
        if not algorithm_is_none_type:
            if algorithm is None and two_phase:
                # Use the algorithm baked into the two-phase instance.
                pass
            else:
                process_algorithm(self, bound, arglist)
                pysig_params.append(
                    inspect.Parameter(
                        "algorithm",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        temp_storage = bound.arguments.get("temp_storage")
        temp_storage_is_none_type = isinstance(temp_storage, types.NoneType)
        if temp_storage_is_none_type:
            arglist.append(temp_storage)
            pysig_params.append(
                inspect.Parameter(
                    "temp_storage",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                )
            )
            temp_storage = None
        if not temp_storage_is_none_type:
            validate_temp_storage(self, temp_storage)
            if temp_storage is not None:
                arglist.append(temp_storage)
                pysig_params.append(
                    inspect.Parameter(
                        "temp_storage",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )

        return_type = src if scalar_return else types.void
        pysig = inspect.Signature(pysig_params)
        sig = Signature(return_type, tuple(arglist), recvr=None, pysig=pysig)

        return sig


@register
class CoopBlockExclusiveSumDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.exclusive_sum
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    forced_mode = "exclusive"
    forced_scan_op = "+"
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        prefix_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return CoopBlockScanDecl.signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=None,
            mode="exclusive",
            scan_op="+",
            block_prefix_callback_op=prefix_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        return CoopBlockScanDecl._validate_args_and_create_signature(
            self, bound, two_phase=two_phase
        )


@register
class CoopBlockInclusiveSumDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.inclusive_sum
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    forced_mode = "inclusive"
    forced_scan_op = "+"
    is_constructor = False
    minimum_num_args = 1

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        prefix_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return CoopBlockScanDecl.signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=None,
            mode="inclusive",
            scan_op="+",
            block_prefix_callback_op=prefix_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        return CoopBlockScanDecl._validate_args_and_create_signature(
            self, bound, two_phase=two_phase
        )


@register
class CoopBlockExclusiveScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.exclusive_scan
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    forced_mode = "exclusive"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        scan_op: ScanOpType = None,
        initial_value: Optional[Any] = None,
        prefix_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return CoopBlockScanDecl.signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            mode="exclusive",
            scan_op=scan_op,
            block_prefix_callback_op=prefix_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        return CoopBlockScanDecl._validate_args_and_create_signature(
            self, bound, two_phase=two_phase
        )


@register
class CoopBlockInclusiveScanDecl(CoopAbstractTemplate, CoopDeclMixin):
    key = coop.block.inclusive_scan
    impl_key = _make_block_scan_rewrite
    primitive_name = "coop.block.scan"
    algorithm_enum = coop.BlockScanAlgorithm
    default_algorithm = coop.BlockScanAlgorithm.RAKING
    forced_mode = "inclusive"
    is_constructor = False
    minimum_num_args = 2

    @staticmethod
    def signature(
        src: Union[types.Array, types.Number],
        dst: Union[types.Array, types.Number] = None,
        items_per_thread: int = None,
        scan_op: ScanOpType = None,
        initial_value: Optional[Any] = None,
        prefix_op: Optional[Callable] = None,
        block_aggregate: types.Array = None,
        algorithm: coop.BlockScanAlgorithm = None,
        temp_storage: Union[types.Array, TempStorageType] = None,
    ):
        return CoopBlockScanDecl.signature(
            src,
            dst,
            items_per_thread=items_per_thread,
            initial_value=initial_value,
            mode="inclusive",
            scan_op=scan_op,
            block_prefix_callback_op=prefix_op,
            block_aggregate=block_aggregate,
            algorithm=algorithm,
            temp_storage=temp_storage,
        )

    def _validate_args_and_create_signature(self, bound, two_phase=False):
        return CoopBlockScanDecl._validate_args_and_create_signature(
            self, bound, two_phase=two_phase
        )


class CoopBlockScanInstanceType(types.Type, CoopInstanceTypeMixin):
    decl_class = CoopBlockScanDecl

    def __init__(self):
        self.decl = self.decl_class()
        name = self.decl_class.primitive_name
        types.Type.__init__(self, name=name)
        CoopInstanceTypeMixin.__init__(self)

    def _validate_args_and_create_signature(self, *args, **kwargs):
        bound = self._bind_instance_signature(*args, **kwargs)
        return self.decl._validate_args_and_create_signature(bound, two_phase=True)


block_scan_instance_type = CoopBlockScanInstanceType()


@typeof_impl.register(coop.block.scan)
def typeof_block_scan_instance(*args, **kwargs):
    return block_scan_instance_type


@typeof_impl.register(coop.block.exclusive_sum)
def typeof_block_exclusive_sum_instance(*args, **kwargs):
    return block_scan_instance_type


@typeof_impl.register(coop.block.inclusive_sum)
def typeof_block_inclusive_sum_instance(*args, **kwargs):
    return block_scan_instance_type


@typeof_impl.register(coop.block.exclusive_scan)
def typeof_block_exclusive_scan_instance(*args, **kwargs):
    return block_scan_instance_type


@typeof_impl.register(coop.block.inclusive_scan)
def typeof_block_inclusive_scan_instance(*args, **kwargs):
    return block_scan_instance_type


@register
class CoopBlockScanInstanceDecl(CoopInstanceTemplate):
    key = block_scan_instance_type
    instance_type = block_scan_instance_type
    primitive_name = "coop.block.scan"


@register_model(CoopBlockScanInstanceType)
class CoopBlockScanInstanceModel(models.OpaqueModel):
    pass


@lower_constant(CoopBlockScanInstanceType)
def lower_constant_block_scan_instance_type(context, builder, typ, value):
    return context.get_dummy_value()
