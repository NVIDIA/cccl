# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Any, Callable, Literal, Union

import numba

from cuda.cooperative.experimental._common import (
    CUB_BLOCK_SCAN_ALGOS,
    ScanOp,
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from cuda.cooperative.experimental._types import (
    Algorithm,
    Dependency,
    DependentArray,
    DependentFunction,
    DependentOperator,
    DependentReference,
    Invocable,
    Pointer,
    TemplateParameter,
    numba_type_to_wrapper,
)
from cuda.cooperative.experimental._typing import (
    DimType,
    ScanOpType,
)

if TYPE_CHECKING:
    import numpy as np


def _scan(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: DimType,
    items_per_thread: int = 1,
    initial_value: Any = None,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: ScanOpType = ScanOp("+"),
    block_prefix_callback_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
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

    # An initial value is not supported for inclusive scans with a single
    # item per thread.
    initial_value_not_supported = items_per_thread == 1 and mode == "inclusive"
    if initial_value is not None and initial_value_not_supported:
        raise ValueError(
            "initial_value is not supported for inclusive scans with "
            "items_per_thread == 1"
        )

    # An initial value is not supported for exclusive scans with a
    # single item per thread and a block prefix callback operator.
    initial_value_not_supported = (
        items_per_thread == 1
        and initial_value is not None
        and mode == "exclusive"
        and block_prefix_callback_op is not None
    )
    if initial_value_not_supported:
        raise ValueError(
            "initial_value is not supported for exclusive scans with "
            "items_per_thread == 1 and a block prefix callback operator"
        )

    # An initial value is required for both inclusive and exclusive scans
    # when items_per_thread > 1 and a block prefix callback operator is
    # not supplied by the caller.
    initial_value_required = items_per_thread > 1 and block_prefix_callback_op is None
    if initial_value_required and initial_value is None:
        raise ValueError(
            "initial_value is required for both inclusive and exclusive scans "
            "when items_per_thread > 1 and no block prefix callback operator "
            "has been supplied"
        )

    # This will raise an error if scan_op is invalid.
    scan_op = ScanOp(scan_op)
    if scan_op.is_sum:
        # Make sure we specialize the correct CUB API for exclusive sum.
        cpp_function_name = f"{cpp_func_prefix}Sum"
    else:
        cpp_function_name = f"{cpp_func_prefix}Scan"

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
            return DependentFunction(Dependency("ScanOp"), op=scan_op.op_cpp)
    elif scan_op.is_callable:

        def make_dependent_scan_op():
            return DependentOperator(
                Dependency("T"),
                [Dependency("T"), Dependency("T")],
                Dependency("ScanOp"),
            )

    if block_prefix_callback_op is not None:

        def make_dependent_block_prefix_callback_op():
            return DependentOperator(
                Dependency("T"),
                [Dependency("T")],
                Dependency("BlockPrefixCallbackOp"),
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
            if block_prefix_callback_op is None:
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
            else:
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

    elif scan_op.is_known or scan_op.is_callable:
        if items_per_thread == 1:
            if mode == "exclusive":
                if initial_value is not None:
                    # We've already verified earlier that block_prefix_callback_op
                    # isn't supplied by the caller if items_per_thread == 1.
                    assert block_prefix_callback_op is None
                    parameters = [
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

                elif block_prefix_callback_op is not None:
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
                    # We shouldn't ever hit this; if we do, our prior invariant
                    # checks have failed.
                    raise RuntimeError("Unreachable code")

            else:
                assert mode == "inclusive" or initial_value is None
                if block_prefix_callback_op is None:
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
            assert items_per_thread > 1, items_per_thread
            if initial_value is not None:
                # We've already verified earlier that block_prefix_callback_op
                # isn't supplied by the caller if items_per_thread > 1 and an
                # initial value is supplied.
                assert block_prefix_callback_op is None
                parameters = [
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
                        # ScanOp scan_op
                        make_dependent_scan_op(),
                    ],
                ]
            elif block_prefix_callback_op is not None:
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
                ]

    else:
        raise RuntimeError("Unreachable code")

    template = Algorithm(
        "BlockScan",
        cpp_function_name,
        "block_scan",
        ["cub/block/block_scan.cuh"],
        template_parameters,
        parameters,
        fake_return=fake_return,
        type_definitions=[numba_type_to_wrapper(dtype, methods=methods)],
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
        temp_storage_bytes=specialization.get_temp_storage_bytes(),
        algorithm=specialization,
    )


def exclusive_sum(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: DimType,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Computes an exclusive block-wide prefix scan using addition (+) as the
    scan operator.  The value of 0 is applied as the initial value, and is
    assigned to the first output element in the first thread.

    Example:
        The code snippet below illustrates an exclusive prefix sum of 512
        integer items that are partitioned in a
        :ref:`blocked arrangement <flexible-data-arrangement>` across 128
        threads where each thread owns 4 consecutive items.

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the
        ``exclusive_sum`` API:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum
            :end-before: example-end exclusive-sum

        Suppose the set of input ``thread_data`` across the block of threads is
        ``{ [1, 1, 1, 1], [1, 1, 1, 1], ..., [1, 1, 1, 1] }``.

        The corresponding output ``thread_data`` in those threads will be
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.

    Args:
        dtype: Supplies the data type of the input and output arrays.

        threads_per_block: Supplies the number of threads in the block, either
            as an integer for a 1D block or a tuple of two or three integers
            for a 2D or 3D block, respectively.

        items_per_thread: Supplies the number of items partitioned onto each
            thread.

        prefix_op: Optionally supplies a callable that will be invoked by the
            first warp of threads in a block with the block aggregate value;
            only the return value of the first lane in the warp is applied as
            the prefix value.

        algorithm: Optionally supplies the algorithm to use for the block-wide
            scan.  Must be one of the following: ``"raking"``,
            ``"raking_memoize"``, or ``"warp_scans"``.  The default is
            ``"raking"``.

        methods: Optionally supplies a dictionary of methods to use for
            user-defined types.  The default is ``None``.

    Returns:
        A callable that can be linked to a CUDA kernel and invoked to perform
        the block-wide exclusive prefix scan.

    """
    return _scan(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        mode="exclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )


def inclusive_sum(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: DimType,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Computes an inclusive block-wide prefix scan using addition (+) as the
    scan operator.

    Args:
        dtype: Supplies the data type of the input and output arrays.

        threads_per_block: Supplies the number of threads in the block, either
            as an integer for a 1D block or a tuple of two or three integers
            for a 2D or 3D block, respectively.

        items_per_thread: Supplies the number of items partitioned onto each
            thread.

        prefix_op: Optionally supplies a callable that will be invoked by the
            first warp of threads in a block with the block aggregate value;
            only the return value of the first lane in the warp is applied as
            the prefix value.

        algorithm: Optionally supplies the algorithm to use for the block-wide
            scan.  Must be one of the following: ``"raking"``,
            ``"raking_memoize"``, or ``"warp_scans"``.  The default is
            ``"raking"``.

        methods: Optionally supplies a dictionary of methods to use for
            user-defined types.  The default is ``None``.


    Returns:
        A callable that can be linked to a CUDA kernel and invoked to perform
        the block-wide inclusive prefix scan.
    """
    return _scan(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        mode="inclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )


def exclusive_scan(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: DimType,
    scan_op: ScanOpType,
    initial_value: Any = None,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Computes an exclusive block-wide prefix scan using the specified scan
    operator.

    Args:
        dtype: Supplies the data type of the input and output arrays.

        threads_per_block: Supplies the number of threads in the block, either
            as an integer for a 1D block or a tuple of two or three integers
            for a 2D or 3D block, respectively.

        scan_op: Supplies the scan operator to use for the block-wide scan.

        initial_value: Optionally supplies the initial value to use for the
            block-wide scan.  If a non-None value is supplied, `prefix_op`
            must be None.

        items_per_thread: Optionally supplies the number of items partitioned
            onto each thread.  Defaults to 1.

        prefix_op: Optionally supplies a callable that will be invoked by the
            first warp of threads in a block with the block aggregate value;
            only the return value of the first lane in the warp is applied as
            the prefix value.  If a non-None value is supplied, `initial_value`
            must be None.

        algorithm: Optionally supplies the algorithm to use for the block-wide
            scan.  Must be one of the following: ``"raking"``,
            ``"raking_memoize"``, or ``"warp_scans"``.  The default is
            ``"raking"``.

        methods: Optionally supplies a dictionary of methods to use for
            user-defined types.  The default is ``None``.

    Returns:
        A callable that can be linked to a CUDA kernel and invoked to perform
        the block-wide exclusive prefix scan.

    """
    return _scan(
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


def inclusive_scan(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: DimType,
    scan_op: ScanOpType,
    initial_value: Any = None,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
    methods: dict = None,
) -> Callable:
    """
    Computes an inclusive block-wide prefix scan using the specified scan
    operator.

    Args:
        dtype: Supplies the data type of the input and output arrays.

        threads_per_block: Supplies the number of threads in the block, either
            as an integer for a 1D block or a tuple of two or three integers
            for a 2D or 3D block, respectively.

        scan_op: Supplies the scan operator to use for the block-wide scan.

        initial_value: Optionally supplies the initial value to use for the
            block-wide scan.  If a non-None value is supplied, `prefix_op`
            must be None.  Only supported for ``items_per_thread > 1``.

        items_per_thread: Optionally supplies the number of items partitioned
            onto each thread.  Defaults to 1.

        prefix_op: Optionally supplies a callable that will be invoked by the
            first warp of threads in a block with the block aggregate value;
            only the return value of the first lane in the warp is applied as
            the prefix value.  If a non-None value is supplied, `initial_value`
            must be None.

        algorithm: Optionally supplies the algorithm to use for the block-wide
            scan.  Must be one of the following: ``"raking"``,
            ``"raking_memoize"``, or ``"warp_scans"``.  The default is
            ``"raking"``.

        methods: Optionally supplies a dictionary of methods to use for
            user-defined types.  The default is ``None``.

    Returns:
        A callable that can be linked to a CUDA kernel and invoked to perform
        the block-wide exclusive prefix scan.

    """
    return _scan(
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
