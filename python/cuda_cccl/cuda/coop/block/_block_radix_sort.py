# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Any, Tuple, Union

import numba

from .._common import (
    CUB_BLOCK_SCAN_ALGOS,
    CudaSharedMemConfig,
    dim3,
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    Constant,
    Decomposer,
    Dependency,
    DependentArray,
    DependentPythonOperator,
    Invocable,
    TemplateParameter,
    TempStoragePointer,
    Value,
    numba_type_to_cpp,
    numba_type_to_wrapper,
)

if TYPE_CHECKING:
    import numpy as np

    from ._rewrite import CoopNode


TEMPLATE_PARAMETERS = [
    TemplateParameter("KeyT"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("ITEMS_PER_THREAD"),
    TemplateParameter("ValueT"),
    TemplateParameter("RADIX_BITS"),
    TemplateParameter("MEMOIZE_OUTER_SCAN"),
    TemplateParameter("INNER_SCAN_ALGORITHM"),
    TemplateParameter("SMEM_CONFIG"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
]


# N.B. In order to support multi-dimensional block dimensions, we have to
#      defaults for all the template parameters preceding the final Y and
#      Z dimensions.  This will be improved in the future, allowing users
#      to provide overrides for the default values.

TEMPLATE_PARAMETER_DEFAULTS = {
    "ValueT": "::cub::NullType",  # Indicates keys-only sort
    "RADIX_BITS": 4,
    "MEMOIZE_OUTER_SCAN": "true",
    "INNER_SCAN_ALGORITHM": CUB_BLOCK_SCAN_ALGOS["warp_scans"],
    "SMEM_CONFIG": str(CudaSharedMemConfig.BankSizeFourByte),
}


def _make_invocable_from_specialization(specialization) -> Invocable:
    lto_irs = specialization.get_lto_ir()
    if lto_irs and hasattr(lto_irs[0], "data"):
        return Invocable(
            ltoir_files=lto_irs,
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )

    # Backward-compatible fallback for raw-bytes LTO IR payloads.
    return Invocable(
        temp_files=[make_binary_tempfile(ltoir, ".ltoir") for ltoir in lto_irs],
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )


def _build_method_parameters(
    *,
    has_values: bool,
    has_bits: bool,
    has_decomposer: bool,
    explicit_temp_storage: bool,
    decomposer_ret_dtype: numba.types.Type = None,
):
    method = []
    if explicit_temp_storage:
        method.append(
            TempStoragePointer(
                numba.types.uint8,
                is_array_pointer=True,
                name="temp_storage",
            )
        )
    method.append(
        DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD"), name="keys")
    )
    if has_values:
        method.append(
            DependentArray(
                Dependency("ValueT"),
                Dependency("ITEMS_PER_THREAD"),
                name="values",
            )
        )
    if has_decomposer:
        method.append(
            DependentPythonOperator(
                Constant(decomposer_ret_dtype),
                [Dependency("KeyT")],
                Dependency("Decomposer"),
                name="decomposer",
            )
        )
    if has_bits:
        method.append(Value(numba.int32, name="begin_bit"))
        method.append(Value(numba.int32, name="end_bit"))
    return [method]


def _get_template_parameter_specializations(
    dtype: numba.types.Type,
    dim: dim3,
    items_per_thread: int,
    value_dtype: numba.types.Type = None,
) -> dict:
    """
    Returns a dictionary of template parameter specializations for the block
    radix sort algorithm.

    Args:
        dtype: Supplies the Numba data type.

        dim: Supplies the block dimensions.

        items_per_thread: Supplies the number of items each thread owns.

    Returns:
        A dictionary of template parameter specializations.
    """
    specialization = {
        "KeyT": dtype,
        "BLOCK_DIM_X": dim[0],
        "ITEMS_PER_THREAD": items_per_thread,
        "BLOCK_DIM_Y": dim[1],
        "BLOCK_DIM_Z": dim[2],
    }

    specialization.update(TEMPLATE_PARAMETER_DEFAULTS)
    if value_dtype is not None:
        specialization["ValueT"] = value_dtype

    return specialization


class _radix_sort_base(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        descending: bool,
        value_dtype: Union[str, type, "np.dtype", "numba.types.Type"] = None,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ) -> None:
        """
        Initialize a base block radix-sort primitive.

        :param dtype: Key dtype used for sorting.
        :type dtype: str | type | np.dtype | numba.types.Type

        :param threads_per_block: CUDA block dimensions as an int or
            ``(x, y, z)`` tuple.
        :type threads_per_block: int | Tuple[int, int] | Tuple[int, int, int] | dim3

        :param items_per_thread: Number of keys owned by each thread.
        :type items_per_thread: int

        :param descending: Sort order selector.
        :type descending: bool

        :param value_dtype: Optional value dtype for pair sorting.
        :type value_dtype: str | type | np.dtype | numba.types.Type, optional

        :param begin_bit: Optional inclusive start bit for partial-bit sorting.
        :type begin_bit: int, optional

        :param end_bit: Optional exclusive end bit for partial-bit sorting.
        :type end_bit: int, optional

        :param decomposer: Optional key decomposer. Not currently supported.
        :type decomposer: Any, optional

        :param blocked_to_striped: Whether to emit blocked-to-striped variants.
        :type blocked_to_striped: bool, optional

        :param unique_id: Optional unique suffix used for generated symbols.
        :type unique_id: int, optional

        :param temp_storage: Optional explicit temporary storage argument.
        :type temp_storage: Any, optional

        :param node: Internal rewrite node used by single-phase rewriting.
        :type node: CoopNode, optional

        :raises ValueError: If ``items_per_thread < 1``.
        :raises ValueError: If exactly one of ``begin_bit`` or ``end_bit`` is
            provided.
        :raises ValueError: If ``decomposer`` is provided (unsupported path).
        """
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")
        if (begin_bit is None) != (end_bit is None):
            raise ValueError("begin_bit and end_bit must be provided together")

        self.node = node
        self.temp_storage = temp_storage
        self.dim = normalize_dim_param(threads_per_block)
        self.dtype = normalize_dtype_param(dtype)
        self.value_dtype = (
            normalize_dtype_param(value_dtype) if value_dtype is not None else None
        )
        self.items_per_thread = items_per_thread
        self.descending = descending
        self.begin_bit = begin_bit
        self.end_bit = end_bit
        self.decomposer = decomposer
        self.blocked_to_striped = blocked_to_striped
        self.unique_id = unique_id

        if decomposer is not None:
            raise ValueError(
                "BlockRadixSort decomposer is not supported yet in cuda.coop. "
                "CUB requires tuple-of-references for custom key types and does "
                "not accept custom decomposers for built-in key types."
            )

        method_name = "SortDescending" if descending else "Sort"
        if blocked_to_striped:
            method_name = (
                "SortDescendingBlockedToStriped"
                if descending
                else "SortBlockedToStriped"
            )

        decomposer_op = None
        decomposer_ret_dtype = None
        if decomposer is not None:
            if isinstance(decomposer, Decomposer):
                decomposer_op = decomposer.op
                decomposer_ret_dtype = decomposer.ret_dtype
            else:
                decomposer_op = decomposer
                decomposer_ret_dtype = getattr(
                    decomposer, "ret_dtype", getattr(decomposer, "return_dtype", None)
                )
            if decomposer_ret_dtype is None:
                raise ValueError(
                    "decomposer requires a return dtype; use coop.Decomposer(op, ret_dtype)"
                )

        parameters = _build_method_parameters(
            has_values=self.value_dtype is not None,
            has_bits=begin_bit is not None,
            has_decomposer=decomposer is not None,
            explicit_temp_storage=temp_storage is not None,
            decomposer_ret_dtype=decomposer_ret_dtype,
        )

        type_definitions = None
        methods = getattr(self.dtype, "methods", None)
        if methods is not None and not methods:
            methods = None
        if methods is not None or numba_type_to_cpp(self.dtype) == "storage_t":
            type_definitions = [numba_type_to_wrapper(self.dtype, methods=methods)]

        self.algorithm = Algorithm(
            "BlockRadixSort",
            method_name,
            "block_radix_sort",
            ["cub/block/block_radix_sort.cuh"]
            + (["cuda/std/tuple"] if decomposer is not None else []),
            TEMPLATE_PARAMETERS,
            parameters,
            self,
            type_definitions=type_definitions,
            unique_id=unique_id,
        )
        specialization_kwds = _get_template_parameter_specializations(
            self.dtype, self.dim, self.items_per_thread, self.value_dtype
        )
        if decomposer_op is not None:
            specialization_kwds["Decomposer"] = decomposer_op
        self.specialization = self.algorithm.specialize(specialization_kwds)

    @classmethod
    def create(
        cls,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        descending: bool,
        value_dtype: Union[str, type, "np.dtype", "numba.types.Type"] = None,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
    ) -> Invocable:
        algo = cls(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            descending=descending,
            value_dtype=value_dtype,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            temp_storage=None,
        )
        specialization = algo.specialization
        return _make_invocable_from_specialization(specialization)


class radix_sort_keys(_radix_sort_base):
    """Performs an ascending block-wide radix sort over a :ref:`blocked arrangement <coop-flexible-data-arrangement>` of keys.

    Example:
        The code snippet below illustrates a sort of 512 integer keys that
        are partitioned in a :ref:`blocked arrangement <coop-flexible-data-arrangement>` across 128 threads
        where each thread owns 4 consecutive keys. We start by importing necessary modules:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``radix_sort_keys`` API:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort
            :end-before: example-end radix-sort

        Suppose the set of input ``thread_keys`` across the block of threads is
        ``{ [511, 510, 509, 508], [507, 506, 505, 504], ..., [3, 2, 1, 0] }``.
        The corresponding output ``thread_keys`` in those threads will be
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.

    Args:
        dtype: Data type of the keys to be sorted

        threads_per_block: The number of threads in a block, either an integer
            or a tuple of 2 or 3 integers

        items_per_thread: The number of items each thread owns

        decomposer: Not yet supported in cuda.coop. CUB requires a tuple-of-
            references decomposer for custom key types and does not accept
            custom decomposers for built-in key types. Passing one raises
            ``ValueError``.

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """

    def __init__(
        self,
        dtype,
        threads_per_block,
        items_per_thread,
        value_dtype: Union[str, type, "np.dtype", "numba.types.Type"] = None,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ):
        """
        Initialize ascending key-only block radix sort.

        :param dtype: Key dtype used for sorting.
        :param threads_per_block: CUDA block dimensions.
        :param items_per_thread: Number of keys owned by each thread.
        :param value_dtype: Optional value dtype for pair sorting.
        :param begin_bit: Optional inclusive start bit for partial sorting.
        :param end_bit: Optional exclusive end bit for partial sorting.
        :param decomposer: Optional key decomposer (unsupported).
        :param blocked_to_striped: Emit blocked-to-striped variant when true.
        :param unique_id: Optional unique suffix for generated symbols.
        :param temp_storage: Optional explicit temporary storage argument.
        :param node: Internal rewrite node used by single-phase rewriting.
        """
        super().__init__(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            descending=False,
            value_dtype=value_dtype,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            unique_id=unique_id,
            temp_storage=temp_storage,
            node=node,
        )

    @classmethod
    def create(
        cls,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        value_dtype: Union[str, type, "np.dtype", "numba.types.Type"] = None,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
    ) -> Invocable:
        algo = cls(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            value_dtype=value_dtype,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            temp_storage=None,
        )
        specialization = algo.specialization
        return _make_invocable_from_specialization(specialization)


class radix_sort_keys_descending(_radix_sort_base):
    """Performs an descending block-wide radix sort over a :ref:`blocked arrangement <coop-flexible-data-arrangement>` of keys.

    Example:
        The code snippet below illustrates a sort of 512 integer keys that
        are partitioned in a :ref:`blocked arrangement <coop-flexible-data-arrangement>` across 128 threads
        where each thread owns 4 consecutive keys. We start by importing necessary modules:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``radix_sort_keys`` API:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort-descending
            :end-before: example-end radix-sort-descending

        Suppose the set of input ``thread_keys`` across the block of threads is
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.
        The corresponding output ``thread_keys`` in those threads will be
        ``{ [511, 510, 509, 508], [507, 506, 505, 504], ..., [3, 2, 1, 0] }``.

    Args:
        dtype: Data type of the keys to be sorted

        threads_per_block: The number of threads in a block, either an integer
            or a tuple of 2 or 3 integers

        items_per_thread: The number of items each thread owns

        decomposer: Not yet supported in cuda.coop. CUB requires a tuple-of-
            references decomposer for custom key types and does not accept
            custom decomposers for built-in key types. Passing one raises
            ``ValueError``.

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """

    def __init__(
        self,
        dtype,
        threads_per_block,
        items_per_thread,
        value_dtype: Union[str, type, "np.dtype", "numba.types.Type"] = None,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ):
        """
        Initialize descending key-only block radix sort.

        :param dtype: Key dtype used for sorting.
        :param threads_per_block: CUDA block dimensions.
        :param items_per_thread: Number of keys owned by each thread.
        :param value_dtype: Optional value dtype for pair sorting.
        :param begin_bit: Optional inclusive start bit for partial sorting.
        :param end_bit: Optional exclusive end bit for partial sorting.
        :param decomposer: Optional key decomposer (unsupported).
        :param blocked_to_striped: Emit blocked-to-striped variant when true.
        :param unique_id: Optional unique suffix for generated symbols.
        :param temp_storage: Optional explicit temporary storage argument.
        :param node: Internal rewrite node used by single-phase rewriting.
        """
        super().__init__(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            descending=True,
            value_dtype=value_dtype,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            unique_id=unique_id,
            temp_storage=temp_storage,
            node=node,
        )

    @classmethod
    def create(
        cls,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        value_dtype: Union[str, type, "np.dtype", "numba.types.Type"] = None,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
    ) -> Invocable:
        algo = cls(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            value_dtype=value_dtype,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            temp_storage=None,
        )
        specialization = algo.specialization
        return _make_invocable_from_specialization(specialization)


class radix_sort_pairs(radix_sort_keys):
    def __init__(
        self,
        keys,
        values,
        threads_per_block,
        items_per_thread,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ):
        """
        Initialize ascending key-value pair block radix sort.

        :param keys: Key dtype.
        :param values: Value dtype.
        :param threads_per_block: CUDA block dimensions.
        :param items_per_thread: Number of items owned by each thread.
        :param begin_bit: Optional inclusive start bit for partial sorting.
        :param end_bit: Optional exclusive end bit for partial sorting.
        :param decomposer: Optional key decomposer (unsupported).
        :param blocked_to_striped: Emit blocked-to-striped variant when true.
        :param unique_id: Optional unique suffix for generated symbols.
        :param temp_storage: Optional explicit temporary storage argument.
        :param node: Internal rewrite node used by single-phase rewriting.
        """
        super().__init__(
            dtype=keys,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            value_dtype=values,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            unique_id=unique_id,
            temp_storage=temp_storage,
            node=node,
        )

    @classmethod
    def create(
        cls,
        keys: Union[str, type, "np.dtype", "numba.types.Type"],
        values: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
    ) -> Invocable:
        algo = cls(
            keys=keys,
            values=values,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            temp_storage=None,
        )
        specialization = algo.specialization
        return _make_invocable_from_specialization(specialization)


class radix_sort_pairs_descending(radix_sort_keys_descending):
    def __init__(
        self,
        keys,
        values,
        threads_per_block,
        items_per_thread,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ):
        """
        Initialize descending key-value pair block radix sort.

        :param keys: Key dtype.
        :param values: Value dtype.
        :param threads_per_block: CUDA block dimensions.
        :param items_per_thread: Number of items owned by each thread.
        :param begin_bit: Optional inclusive start bit for partial sorting.
        :param end_bit: Optional exclusive end bit for partial sorting.
        :param decomposer: Optional key decomposer (unsupported).
        :param blocked_to_striped: Emit blocked-to-striped variant when true.
        :param unique_id: Optional unique suffix for generated symbols.
        :param temp_storage: Optional explicit temporary storage argument.
        :param node: Internal rewrite node used by single-phase rewriting.
        """
        super().__init__(
            dtype=keys,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            value_dtype=values,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            unique_id=unique_id,
            temp_storage=temp_storage,
            node=node,
        )

    @classmethod
    def create(
        cls,
        keys: Union[str, type, "np.dtype", "numba.types.Type"],
        values: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        begin_bit: int = None,
        end_bit: int = None,
        decomposer: Any = None,
        blocked_to_striped: bool = False,
    ) -> Invocable:
        algo = cls(
            keys=keys,
            values=values,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            decomposer=decomposer,
            blocked_to_striped=blocked_to_striped,
            temp_storage=None,
        )
        specialization = algo.specialization
        return _make_invocable_from_specialization(specialization)


def _normalize_threads_per_block(kwargs, threads_per_block):
    kw = dict(kwargs)
    if threads_per_block is None:
        threads_per_block = kw.pop("dim", None)
    return kw, threads_per_block


def _build_radix_sort_keys_spec(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    kw, threads_per_block = _normalize_threads_per_block(kwargs, threads_per_block)
    spec = {
        "dtype": dtype,
        "threads_per_block": threads_per_block,
        "items_per_thread": items_per_thread,
    }
    spec.update(kw)
    return spec


def _build_radix_sort_pairs_spec(
    keys,
    values,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    kw, threads_per_block = _normalize_threads_per_block(kwargs, threads_per_block)
    spec = {
        "keys": keys,
        "values": values,
        "threads_per_block": threads_per_block,
        "items_per_thread": items_per_thread,
    }
    spec.update(kw)
    return spec


def _make_radix_sort_keys_two_phase(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_radix_sort_keys_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return radix_sort_keys.create(**spec)


def _make_radix_sort_keys_rewrite(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_radix_sort_keys_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return radix_sort_keys(**spec)


def _make_radix_sort_keys_descending_two_phase(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_radix_sort_keys_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return radix_sort_keys_descending.create(**spec)


def _make_radix_sort_keys_descending_rewrite(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_radix_sort_keys_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return radix_sort_keys_descending(**spec)


def _make_radix_sort_pairs_two_phase(
    keys,
    values,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_radix_sort_pairs_spec(
        keys=keys,
        values=values,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return radix_sort_pairs.create(**spec)


def _make_radix_sort_pairs_rewrite(
    keys,
    values,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_radix_sort_pairs_spec(
        keys=keys,
        values=values,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return radix_sort_pairs(**spec)


def _make_radix_sort_pairs_descending_two_phase(
    keys,
    values,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_radix_sort_pairs_spec(
        keys=keys,
        values=values,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return radix_sort_pairs_descending.create(**spec)


def _make_radix_sort_pairs_descending_rewrite(
    keys,
    values,
    threads_per_block=None,
    items_per_thread=1,
    **kwargs,
):
    spec = _build_radix_sort_pairs_spec(
        keys=keys,
        values=values,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        **kwargs,
    )
    return radix_sort_pairs_descending(**spec)
