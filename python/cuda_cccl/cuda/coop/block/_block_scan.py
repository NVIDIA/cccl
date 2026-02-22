# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
cuda.coop.block_scan
===========================

This module provides a set of :ref:`collective <collective-primitives>`
for computing parallel prefix scans of items partitioned across CUDA
thread blocks.  It is based on the :class:`cub.BlockScan` C++ class in
the CUB library.

Supported C++ APIs
++++++++++++++++++

The following :class:`cub.BlockScan` C++ APIs are supported:


    ExclusiveSum(T input, T &output)
    ExclusiveSum(T input, T &output, BlockPrefixCallbackOp &prefix_op)
    ExclusiveSum(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output)
    ExclusiveSum(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, BlockPrefixCallbackOp &prefix_op)

    InclusiveSum(T input, T &output)
    InclusiveSum(T input, T &output, BlockPrefixCallbackOp &prefix_op)
    InclusiveSum(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output)
    InclusiveSum(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, BlockPrefixCallbackOp &prefix_op)

    ExclusiveScan(T input, T &output, ScanOp scan_op)
    ExclusiveScan(T input, T &output, ScanOp scan_op, BlockPrefixCallbackOp &prefix_op)
    ExclusiveScan(T input, T &output, T initial_value, ScanOp scan_op)
    ExclusiveScan(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, ScanOp scan_op)
    ExclusiveScan(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, ScanOp scan_op, BlockPrefixCallbackOp &prefix_op)
    ExclusiveScanT(&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, T initial_value, ScanOp scan_op)

    InclusiveScan(T input, T &output, ScanOp scan_op)
    InclusiveScan(T input, T &output, ScanOp scan_op, BlockPrefixCallbackOp &prefix_op)
    InclusiveScan(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, ScanOp scan_op)
    InclusiveScan(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, ScanOp scan_op, BlockPrefixCallbackOp &prefix_op)
    InclusiveScanT(&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, T initial_value, ScanOp scan_op)

Unsupported C++ APIs
++++++++++++++++++++

This module does not support any of the :class:`cub.BlockScan` C++ APIs
that take a block aggregate reference as an argument.  That being said, the
`BlockPrefixCallbackOp` callable is supported, and thus, block aggregates can
be obtained using those measures.

The reason the `T &block_aggregate` pattern is not supported as it will usually
result in two output parameters, which we don't support in our underlying type
machinery (i.e. _types.py).

The unsupported APIs are as follows:

    ExclusiveSum(T input, T &output, T &block_aggregate)
    ExclusiveSum(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, T &block_aggregate)

    InclusiveSum(T input, T &output, T &block_aggregate)
    InclusiveSum(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, T &block_aggregate)

    ExclusiveScan(T input, T &output, ScanOp scan_op, T &block_aggregate)
    ExclusiveScan(T input, T &output, T initial_value, ScanOp scan_op, T &block_aggregate)
    ExclusiveScan(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, ScanOp scan_op, T &block_aggregate)
    ExclusiveScan(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, T initial_value, ScanOp scan_op, T &block_aggregate)

    InclusiveScan(T input, T &output, ScanOp scan_op, T &block_aggregate)
    InclusiveScan(T&)[ITEMS_PER_THREAD] input, T(&)[ITEMS_PER_THREAD] output, T initial_value, ScanOp scan_op, T &block_aggregate)
"""

from typing import Any, Callable, Literal

import numba

from .._common import (
    CUB_BLOCK_SCAN_ALGOS,
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._scan_op import (
    ScanOp,
)
from .._types import (
    Algorithm,
    Dependency,
    DependentArray,
    DependentCxxOperator,
    DependentPythonOperator,
    DependentReference,
    Invocable,
    Pointer,
    TemplateParameter,
    numba_type_to_wrapper,
)
from .._typing import (
    DimType,
    DtypeType,
    ScanOpType,
)


def make_scan(
    dtype: DtypeType,
    threads_per_block: DimType,
    items_per_thread: int,
    initial_value: Any = None,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: ScanOpType = "+",
    block_prefix_callback_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Creates a block-wide prefix scan primitive based on CUB's BlockScan
    APIs.

    The returned primitive is callable from a Numba CUDA kernel and
    supports sum and generic scan operators in inclusive and exclusive
    modes.

    Example:
        The snippet below creates a scan primitive and invokes the
        returned ``block_scan`` primitive inside a kernel.

        .. code-block:: python

           block_scan = coop.block.make_scan(
               dtype=numba.int32,
               threads_per_block=128,
               items_per_thread=4,
               mode="exclusive",
               scan_op="+",
           )

           @cuda.jit(link=block_scan.files)
           def kernel(input_arr, output_arr):
               tid = cuda.threadIdx.x
               thread_data = cuda.local.array(4, dtype=numba.int32)
               for i in range(4):
                   thread_data[i] = input_arr[tid * 4 + i]
               block_scan(thread_data, thread_data)
               for i in range(4):
                   output_arr[tid * 4 + i] = thread_data[i]

    :param dtype: Data type of the input and output values.
    :type dtype: DtypeType

    :param threads_per_block: Number of threads in the block. Can be an
        integer for 1D blocks or a tuple of two or three integers for 2D
        and 3D blocks.
    :type threads_per_block: DimType

    :param items_per_thread: Number of items owned by each thread.
        Must be greater than or equal to 1.
    :type items_per_thread: int, optional

    :param initial_value: Optional initial value for scan variants that
        support it.
    :type initial_value: Any, optional

    :param mode: Scan mode. Must be ``"exclusive"`` or ``"inclusive"``.
    :type mode: Literal["exclusive", "inclusive"], optional

    :param scan_op: Scan operator. The default is ``"+"``.
    :type scan_op: ScanOpType, optional

    :param block_prefix_callback_op: Optional block prefix callback
        operator invoked by the first warp.
    :type block_prefix_callback_op: Callable, optional

    :param algorithm: Scan algorithm. Must be ``"raking"``,
        ``"raking_memoize"``, or ``"warp_scans"``.
    :type algorithm:
        Literal["raking", "raking_memoize", "warp_scans"], optional

    :param methods: Optional method dictionary used for user-defined
        types.
    :type methods: dict, optional

    :raises ValueError: If ``algorithm`` is unsupported.
    :raises ValueError: If ``items_per_thread < 1``.
    :raises ValueError: If ``mode`` is not ``"exclusive"`` or
        ``"inclusive"``.
    :raises ValueError: If ``scan_op`` is unsupported.
    :raises ValueError: If ``initial_value`` is provided for sum scans.
    :raises ValueError: If ``initial_value`` is used with inclusive scans
        and ``items_per_thread == 1``.
    :raises ValueError: If ``initial_value`` is used with exclusive scans
        and ``items_per_thread == 1`` while
        ``block_prefix_callback_op`` is provided.
    :raises ValueError: If an initial value is required but cannot be
        inferred from ``dtype``.

    :returns: Callable primitive object that can be linked to and
        invoked from a CUDA kernel.
    :rtype: Callable
    """
    if algorithm not in CUB_BLOCK_SCAN_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

    if mode == "exclusive":
        cpp_func_prefix = "Exclusive"
    else:
        if mode != "inclusive":
            raise ValueError(f"Unsupported mode: {mode}")
        cpp_func_prefix = "Inclusive"

    # This will raise an error if scan_op is invalid.
    scan_op = ScanOp(scan_op)
    if scan_op.is_sum:
        # Make sure we specialize the correct CUB API for exclusive sum.
        cpp_function_name = f"{cpp_func_prefix}Sum"
    else:
        cpp_function_name = f"{cpp_func_prefix}Scan"

    # An initial value is not supported for inclusive and exclusive sums.
    if initial_value is not None and scan_op.is_sum:
        raise ValueError(
            "initial_value is not supported for inclusive and exclusive sums"
        )

    # An initial value is not supported for inclusive scans with a single
    # item per thread.
    invalid_initial_value = (
        initial_value is not None and items_per_thread == 1 and mode == "inclusive"
    )
    if invalid_initial_value:
        raise ValueError(
            "initial_value is not supported for inclusive scans with "
            "items_per_thread == 1"
        )

    # An initial value is not supported for exclusive scans with a
    # single item per thread and a block prefix callback operator.
    invalid_initial_value = (
        items_per_thread == 1
        and initial_value is not None
        and mode == "exclusive"
        and block_prefix_callback_op is not None
    )
    if invalid_initial_value:
        raise ValueError(
            "initial_value is not supported for exclusive scans with "
            "items_per_thread == 1 and a block prefix callback operator"
        )

    # An initial value is required for both inclusive and exclusive scans
    # when items_per_thread > 1 and a block prefix callback operator is
    # not supplied by the caller.
    initial_value_required = items_per_thread > 1 and block_prefix_callback_op is None
    if initial_value_required and initial_value is None:
        # We require an initial value, but one was not supplied.
        # Attempt to create a default value for the given dtype.
        # If we can't, raise an error.
        try:
            initial_value = dtype.cast_python_value(0)
        except (TypeError, NotImplementedError) as e:
            # We can't create a default value for the given dtype.
            # Raise an error.
            msg = (
                "initial_value is required for both inclusive and "
                "exclusive scans when items_per_thread > 1 and no "
                "block prefix callback operator has been supplied; "
                "attempted to create a default value for the given "
                f"dtype, but failed: {e}"
            )
            raise ValueError(msg) from e

    specialization_kwds = {
        "T": dtype,
        "BLOCK_DIM_X": dim[0],
        "ALGORITHM": CUB_BLOCK_SCAN_ALGOS[algorithm],
        "BLOCK_DIM_Y": dim[1],
        "BLOCK_DIM_Z": dim[2],
    }

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    if items_per_thread == 1:
        fake_return = True
    else:
        specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread
        fake_return = False

    # A "known" scan op is the standard set of associative operators,
    # e.g. ::cuda::std::plus<>, etc.  A "callable" scan op is a Python
    # callable that has been furnished by the caller.  Thus, we need to
    # generate different parameters for each case.
    if scan_op.is_known:

        def make_dependent_scan_op():
            return DependentCxxOperator(
                dep=Dependency("T"),
                cpp=scan_op.op_cpp,
            )

    elif scan_op.is_callable:

        def make_dependent_scan_op():
            return DependentPythonOperator(
                ret_dtype=Dependency("T"),
                arg_dtypes=[Dependency("T"), Dependency("T")],
                op=Dependency("ScanOp"),
            )

    if block_prefix_callback_op is not None:

        def make_dependent_block_prefix_callback_op():
            return DependentPythonOperator(
                ret_dtype=Dependency("T"),
                arg_dtypes=[Dependency("T")],
                op=Dependency("BlockPrefixCallbackOp"),
            )

    if scan_op.is_sum:
        if items_per_thread == 1:
            if block_prefix_callback_op is None:
                parameters = [
                    # Signature:
                    # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                    #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                    #     temp_storage
                    # ).<Inclusive|Exclusive>Sum(
                    #     T, # input
                    #     T& # output
                    # )
                    [
                        # temp_storage
                        Pointer(numba.uint8),
                        # T input
                        DependentReference(Dependency("T")),
                        # T& output
                        DependentReference(Dependency("T"), is_output=True),
                    ],
                ]
            else:
                parameters = [
                    # Signature:
                    # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                    #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                    #     temp_storage
                    # ).<Inclusive|Exclusive>Sum(
                    #     T,                     # input
                    #     T&,                    # output
                    #     BlockPrefixCallbackOp& # block_prefix_callback_op
                    # )
                    [
                        # temp_storage
                        Pointer(numba.uint8),
                        # T input
                        DependentReference(Dependency("T")),
                        # T& output
                        DependentReference(Dependency("T"), is_output=True),
                        # BlockPrefixCallbackOp& block_prefix_callback_op
                        make_dependent_block_prefix_callback_op(),
                    ],
                ]

        else:
            assert items_per_thread > 1, items_per_thread
            if block_prefix_callback_op is not None:
                parameters = [
                    # Signature:
                    # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                    #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                    #     temp_storage
                    # ).<Inclusive|Exclusive>Sum(
                    #     T (&)[ITEMS_PER_THREAD], # input
                    #     T (&)[ITEMS_PER_THREAD], # output
                    #     BlockPrefixCallbackOp&   # block_prefix_callback_op
                    # )
                    [
                        # temp_storage
                        Pointer(numba.uint8),
                        # T (&)[ITEMS_PER_THREAD] input
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # T (&)[ITEMS_PER_THREAD] output
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # BlockPrefixCallbackOp& block_prefix_callback_op
                        make_dependent_block_prefix_callback_op(),
                    ],
                ]
            else:
                parameters = [
                    # Signature:
                    # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                    #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                    #     temp_storage
                    # ).<Inclusive|Exclusive>Sum(
                    #     T (&)[ITEMS_PER_THREAD], # input
                    #     T (&)[ITEMS_PER_THREAD]  # output
                    # )
                    [
                        # temp_storage
                        Pointer(numba.uint8),
                        # T (&)[ITEMS_PER_THREAD] input
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # T (&)[ITEMS_PER_THREAD] output
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    ],
                ]

    elif scan_op.is_known or scan_op.is_callable:
        if items_per_thread == 1:
            if mode == "exclusive":
                if block_prefix_callback_op is not None:
                    assert initial_value is None
                    parameters = [
                        # Signature:
                        # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                        #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                        #     temp_storage
                        # )<ScanOp>ExclusiveScan(
                        #     T,                     # input
                        #     T&,                    # output
                        #     ScanOp,                # scan_op
                        #     BlockPrefixCallbackOp& # block_prefix_callback_op
                        # )
                        [
                            # temp_storage
                            Pointer(numba.uint8),
                            # T input
                            DependentReference(Dependency("T")),
                            # T& output
                            DependentReference(Dependency("T"), is_output=True),
                            # ScanOp scan_op
                            make_dependent_scan_op(),
                            # BlockPrefixCallbackOp& block_prefix_callback_op
                            make_dependent_block_prefix_callback_op(),
                        ],
                    ]
                else:
                    parameters = [
                        # Signature:
                        # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                        #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                        #     temp_storage
                        # )<ScanOp>ExclusiveScan(
                        #     T,     # input
                        #     T&,    # output
                        #     ScanOp # scan_op
                        # )
                        [
                            # temp_storage
                            Pointer(numba.uint8),
                            # T input
                            DependentReference(Dependency("T")),
                            # T& output
                            DependentReference(Dependency("T"), is_output=True),
                            # ScanOp scan_op
                            make_dependent_scan_op(),
                        ],
                        # Signature:
                        # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                        #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                        #     temp_storage
                        # )<ScanOp>ExclusiveScan(
                        #     T,     # input
                        #     T&,    # output
                        #     T,     # initial_value
                        #     ScanOp # scan_op
                        # )
                        [
                            # temp_storage
                            Pointer(numba.uint8),
                            # T input
                            DependentReference(Dependency("T")),
                            # T& output
                            DependentReference(Dependency("T"), is_output=True),
                            # T initial_value
                            DependentReference(Dependency("T")),
                            # ScanOp scan_op
                            make_dependent_scan_op(),
                        ],
                    ]

            else:
                assert mode == "inclusive" or initial_value is None
                if block_prefix_callback_op is not None:
                    parameters = [
                        # Signature:
                        # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                        #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                        #     temp_storage
                        # )<ScanOp><Inclusive|Exclusive>Scan(
                        #     T,                     # input
                        #     T&,                    # output
                        #     ScanOp,                # scan_op
                        #     BlockPrefixCallbackOp& # block_prefix_callback_op
                        # )
                        [
                            # temp_storage
                            Pointer(numba.uint8),
                            # T input
                            DependentReference(Dependency("T")),
                            # T& output
                            DependentReference(Dependency("T"), is_output=True),
                            # ScanOp scan_op
                            make_dependent_scan_op(),
                            # BlockPrefixCallbackOp& block_prefix_callback_op
                            make_dependent_block_prefix_callback_op(),
                        ],
                    ]
                else:
                    parameters = [
                        # Signature:
                        # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                        #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                        #     temp_storage
                        # )<ScanOp><Inclusive|Exclusive>Scan(
                        #     T,     # input
                        #     T&,    # output
                        #     ScanOp # scan_op
                        # )
                        [
                            # temp_storage
                            Pointer(numba.uint8),
                            # T input
                            DependentReference(Dependency("T")),
                            # T& output
                            DependentReference(Dependency("T"), is_output=True),
                            # ScanOp scan_op
                            make_dependent_scan_op(),
                        ],
                    ]

        else:
            assert items_per_thread > 1, items_per_thread
            if block_prefix_callback_op is not None:
                assert initial_value is None
                parameters = [
                    # Signature:
                    # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                    #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                    #     temp_storage
                    # )<ITEMS_PER_THREAD, ScanOp><Inclusive|Exclusive>Scan(
                    #     T (&)[ITEMS_PER_THREAD], # input
                    #     T (&)[ITEMS_PER_THREAD], # output
                    #     ScanOp,                  # scan_op
                    #     BlockPrefixCallbackOp&   # block_prefix_callback_op
                    # )
                    [
                        # temp_storage
                        Pointer(numba.uint8),
                        # T (&)[ITEMS_PER_THREAD] input
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # T (&)[ITEMS_PER_THREAD] output
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # ScanOp scan_op
                        make_dependent_scan_op(),
                        # BlockPrefixCallbackOp& block_prefix_callback_op
                        make_dependent_block_prefix_callback_op(),
                    ],
                ]
            else:
                parameters = [
                    # Signature:
                    # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                    #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                    #     temp_storage
                    # )<ITEMS_PER_THREAD, ScanOp><Inclusive|Exclusive>Scan(
                    #     T (&)[ITEMS_PER_THREAD], # input
                    #     T (&)[ITEMS_PER_THREAD], # output
                    #     ScanOp                   # scan_op
                    # )
                    [
                        # temp_storage
                        Pointer(numba.uint8),
                        # T (&)[ITEMS_PER_THREAD] input
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # T (&)[ITEMS_PER_THREAD] output
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # ScanOp scan_op
                        make_dependent_scan_op(),
                    ],
                    # Signature:
                    # void BlockScan<T, BLOCK_DIM_X, ALGORITHM,
                    #                   BLOCK_DIM_Y, BLOCK_DIM_Z>(
                    #     temp_storage
                    # )<ITEMS_PER_THREAD, ScanOp><Inclusive|Exclusive>Scan(
                    #     T (&)[ITEMS_PER_THREAD], # input
                    #     T (&)[ITEMS_PER_THREAD], # output
                    #     T,                       # initial_value
                    #     ScanOp                   # scan_op
                    # )
                    [
                        # temp_storage
                        Pointer(numba.uint8),
                        # T (&)[ITEMS_PER_THREAD] input
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # T (&)[ITEMS_PER_THREAD] output
                        DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                        # T initial_value
                        DependentReference(Dependency("T")),
                        # ScanOp scan_op
                        make_dependent_scan_op(),
                    ],
                ]

    else:
        # We shouldn't ever hit this; there are only three types of scan ops
        # supported: sum, known, and callable.
        raise RuntimeError("Unreachable code")

    # Invariant check: if we get here, parameters shouldn't be empty.
    assert parameters, "parameters should not be empty"

    # If we have a non-None `methods`, we're dealing with user-defined types.
    if methods is not None:
        type_definitions = [
            numba_type_to_wrapper(dtype, methods=methods),
        ]
    else:
        type_definitions = None

    template = Algorithm(
        "BlockScan",
        cpp_function_name,
        "block_scan",
        ["cub/block/block_scan.cuh"],
        template_parameters,
        parameters,
        fake_return=fake_return,
        type_definitions=type_definitions,
    )

    if scan_op.is_callable:
        specialization_kwds["ScanOp"] = scan_op.op

    if block_prefix_callback_op is not None:
        specialization_kwds["BlockPrefixCallbackOp"] = block_prefix_callback_op

    specialization = template.specialize(specialization_kwds)
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )


def make_exclusive_sum(
    dtype: DtypeType,
    threads_per_block: DimType,
    items_per_thread: int,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Creates an exclusive block-wide prefix sum primitive using addition
    (+) as the scan operator.

    Example:
        The code snippet below illustrates an exclusive prefix sum of
        512 integer items in a
        :ref:`blocked arrangement <flexible-data-arrangement>` across
        128 threads where each thread owns 4 consecutive items.

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        The following snippet shows how to invoke the returned
        ``block_exclusive_sum`` primitive:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum
            :end-before: example-end exclusive-sum

        Suppose the set of input ``thread_data`` across the block of
        threads is
        ``{ [1, 1, 1, 1], [1, 1, 1, 1], ..., [1, 1, 1, 1] }``.

        The corresponding output ``thread_data`` in those threads will be
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.

    :param dtype: Data type of the input and output values.
    :type dtype: DtypeType
    :param threads_per_block: Number of threads in the block.
    :type threads_per_block: DimType
    :param items_per_thread: Number of items owned by each thread.
    :type items_per_thread: int, optional
    :param prefix_op: Optional block prefix callback operator.
    :type prefix_op: Callable, optional
    :param algorithm: Scan algorithm.
    :type algorithm:
        Literal["raking", "raking_memoize", "warp_scans"], optional
    :param methods: Optional method dictionary for user-defined types.
    :type methods: dict, optional
    :returns: Callable primitive object for exclusive prefix sum.
    :rtype: Callable
    """
    return make_scan(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        mode="exclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )


def make_inclusive_sum(
    dtype: DtypeType,
    threads_per_block: DimType,
    items_per_thread: int,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Creates an inclusive block-wide prefix sum primitive using addition
    (+) as the scan operator.

    Example:
        The snippet below shows how to create and invoke the returned
        ``block_inclusive_sum`` primitive.

        .. code-block:: python

           block_inclusive_sum = coop.block.make_inclusive_sum(
               dtype=numba.int32,
               threads_per_block=128,
               items_per_thread=4,
           )

           @cuda.jit(link=block_inclusive_sum.files)
           def kernel(thread_data):
               block_inclusive_sum(thread_data, thread_data)

    :param dtype: Data type of the input and output values.
    :type dtype: DtypeType
    :param threads_per_block: Number of threads in the block.
    :type threads_per_block: DimType
    :param items_per_thread: Number of items owned by each thread.
    :type items_per_thread: int, optional
    :param prefix_op: Optional block prefix callback operator.
    :type prefix_op: Callable, optional
    :param algorithm: Scan algorithm.
    :type algorithm:
        Literal["raking", "raking_memoize", "warp_scans"], optional
    :param methods: Optional method dictionary for user-defined types.
    :type methods: dict, optional
    :returns: Callable primitive object for inclusive prefix sum.
    :rtype: Callable
    """
    return make_scan(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        mode="inclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )


def make_exclusive_scan(
    dtype: DtypeType,
    threads_per_block: DimType,
    scan_op: ScanOpType,
    items_per_thread: int,
    initial_value: Any = None,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Creates an exclusive block-wide prefix scan primitive with the
    specified scan operator.

    Example:
        The snippet below shows how to create and invoke the returned
        ``block_exclusive_scan`` primitive.

        .. code-block:: python

           block_exclusive_scan = coop.block.make_exclusive_scan(
               dtype=numba.int32,
               threads_per_block=128,
               scan_op="max",
               items_per_thread=4,
           )

           @cuda.jit(link=block_exclusive_scan.files)
           def kernel(thread_data):
               block_exclusive_scan(thread_data, thread_data)

    :param dtype: Data type of the input and output values.
    :type dtype: DtypeType
    :param threads_per_block: Number of threads in the block.
    :type threads_per_block: DimType
    :param scan_op: Scan operator.
    :type scan_op: ScanOpType
    :param items_per_thread: Number of items owned by each thread.
    :type items_per_thread: int, optional
    :param initial_value: Optional initial value when supported.
    :type initial_value: Any, optional
    :param prefix_op: Optional block prefix callback operator.
    :type prefix_op: Callable, optional
    :param algorithm: Scan algorithm.
    :type algorithm:
        Literal["raking", "raking_memoize", "warp_scans"], optional
    :param methods: Optional method dictionary for user-defined types.
    :type methods: dict, optional
    :returns: Callable primitive object for exclusive prefix scan.
    :rtype: Callable
    """
    return make_scan(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        initial_value=initial_value,
        mode="exclusive",
        scan_op=scan_op,
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )


def make_inclusive_scan(
    dtype: DtypeType,
    threads_per_block: DimType,
    scan_op: ScanOpType,
    items_per_thread: int,
    initial_value: Any = None,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Creates an inclusive block-wide prefix scan primitive with the
    specified scan operator.

    Example:
        The snippet below shows how to create and invoke the returned
        ``block_inclusive_scan`` primitive.

        .. code-block:: python

           block_inclusive_scan = coop.block.make_inclusive_scan(
               dtype=numba.int32,
               threads_per_block=128,
               scan_op="min",
               items_per_thread=4,
           )

           @cuda.jit(link=block_inclusive_scan.files)
           def kernel(thread_data):
               block_inclusive_scan(thread_data, thread_data)

    :param dtype: Data type of the input and output values.
    :type dtype: DtypeType
    :param threads_per_block: Number of threads in the block.
    :type threads_per_block: DimType
    :param scan_op: Scan operator.
    :type scan_op: ScanOpType
    :param items_per_thread: Number of items owned by each thread.
    :type items_per_thread: int, optional
    :param initial_value: Optional initial value when supported.
    :type initial_value: Any, optional
    :param prefix_op: Optional block prefix callback operator.
    :type prefix_op: Callable, optional
    :param algorithm: Scan algorithm.
    :type algorithm:
        Literal["raking", "raking_memoize", "warp_scans"], optional
    :param methods: Optional method dictionary for user-defined types.
    :type methods: dict, optional
    :returns: Callable primitive object for inclusive prefix scan.
    :rtype: Callable
    """
    return make_scan(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        initial_value=initial_value,
        mode="inclusive",
        scan_op=scan_op,
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )
