# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba

from .._common import normalize_dtype_param
from .._scan_op import ScanOp
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentCxxOperator,
    DependentPointerReference,
    DependentPythonOperator,
    DependentReference,
    Invocable,
    TemplateParameter,
    TempStoragePointer,
    Value,
)


def _make_scan_op_param(scan_op):
    if scan_op.is_known or scan_op.is_sum:
        return DependentCxxOperator(
            dep=Dependency("T"),
            cpp=scan_op.op_cpp,
            name="scan_op",
        )
    if scan_op.is_callable:
        return DependentPythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=[Dependency("T"), Dependency("T")],
            op=Dependency("ScanOp"),
            name="scan_op",
        )
    raise RuntimeError("Unsupported scan op for warp scan")


class exclusive_sum(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        threads_in_warp=32,
        warp_aggregate=None,
        unique_id=None,
        temp_storage=None,
    ):
        """
        Computes an exclusive warp-wide prefix sum using addition (+).

        :param dtype: Supplies the scan item dtype.
        :type  dtype: DtypeType

        :param threads_in_warp: Supplies the logical warp size.
        :type  threads_in_warp: int, optional

        :param warp_aggregate: Optionally supplies a single-element output that
            receives the warp aggregate.
        :type  warp_aggregate: Any, optional

        :param temp_storage: Optionally supplies explicit cooperative temporary
            storage (e.g. via ``coop.TempStorage``).
        :type  temp_storage: Any, optional

        Example:
            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
                :language: python
                :dedent:
                :start-after: example-begin exclusive-sum
                :end-before: example-end exclusive-sum
        """
        self.temp_storage = temp_storage
        self.dtype = normalize_dtype_param(dtype)
        self.threads_in_warp = threads_in_warp
        self.scan_op = ScanOp("+")
        self.initial_value = None
        self.warp_aggregate = warp_aggregate

        template = Algorithm(
            "WarpScan",
            "ExclusiveSum",
            "warp_scan",
            ["cub/warp/warp_scan.cuh"],
            [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
            [
                [
                    DependentReference(Dependency("T")),
                    DependentReference(Dependency("T"), True),
                ]
            ],
            self,
            fake_return=True,
            threads=threads_in_warp,
            unique_id=unique_id,
        )
        if warp_aggregate is not None:
            template.parameters[0].append(
                DependentPointerReference(
                    Dependency("T"),
                    name="warp_aggregate",
                    is_array_pointer=True,
                )
            )
        if temp_storage is not None:
            template.parameters[0].insert(
                0,
                TempStoragePointer(
                    numba.types.uint8, is_array_pointer=True, name="temp_storage"
                ),
            )
        self.algorithm = template
        self.specialization = template.specialize(
            {"T": self.dtype, "VIRTUAL_WARP_THREADS": threads_in_warp}
        )

    @classmethod
    def create(cls, dtype, threads_in_warp=32):
        algo = cls(dtype=dtype, threads_in_warp=threads_in_warp)
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class inclusive_sum(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        threads_in_warp=32,
        warp_aggregate=None,
        unique_id=None,
        temp_storage=None,
    ):
        """
        Computes an inclusive warp-wide prefix sum using addition (+).

        :param dtype: Supplies the scan item dtype.
        :type  dtype: DtypeType

        :param threads_in_warp: Supplies the logical warp size.
        :type  threads_in_warp: int, optional

        :param warp_aggregate: Optionally supplies a single-element output that
            receives the warp aggregate.
        :type  warp_aggregate: Any, optional

        :param temp_storage: Optionally supplies explicit cooperative temporary
            storage (e.g. via ``coop.TempStorage``).
        :type  temp_storage: Any, optional

        Example:
            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
                :language: python
                :dedent:
                :start-after: example-begin inclusive-sum
                :end-before: example-end inclusive-sum
        """
        self.temp_storage = temp_storage
        self.dtype = normalize_dtype_param(dtype)
        self.threads_in_warp = threads_in_warp
        self.scan_op = ScanOp("+")
        self.initial_value = None
        self.warp_aggregate = warp_aggregate

        template = Algorithm(
            "WarpScan",
            "InclusiveSum",
            "warp_scan",
            ["cub/warp/warp_scan.cuh"],
            [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
            [
                [
                    DependentReference(Dependency("T")),
                    DependentReference(Dependency("T"), True),
                ]
            ],
            self,
            fake_return=True,
            threads=threads_in_warp,
            unique_id=unique_id,
        )
        if warp_aggregate is not None:
            template.parameters[0].append(
                DependentPointerReference(
                    Dependency("T"),
                    name="warp_aggregate",
                    is_array_pointer=True,
                )
            )
        if temp_storage is not None:
            template.parameters[0].insert(
                0,
                TempStoragePointer(
                    numba.types.uint8, is_array_pointer=True, name="temp_storage"
                ),
            )
        self.algorithm = template
        self.specialization = template.specialize(
            {"T": self.dtype, "VIRTUAL_WARP_THREADS": threads_in_warp}
        )

    @classmethod
    def create(cls, dtype, threads_in_warp=32):
        algo = cls(dtype=dtype, threads_in_warp=threads_in_warp)
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class exclusive_scan(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        scan_op,
        initial_value=None,
        threads_in_warp=32,
        valid_items=None,
        warp_aggregate=None,
        unique_id=None,
        temp_storage=None,
    ):
        """
        Computes an exclusive warp-wide prefix scan.

        :param dtype: Supplies the scan item dtype.
        :type  dtype: DtypeType

        :param scan_op: Supplies the scan operator (known-op string, callable,
            or ``ScanOp`` instance).
        :type  scan_op: ScanOpType

        :param initial_value: Optionally supplies an initial value for
            ``ExclusiveScan`` overloads.
        :type  initial_value: Any, optional

        :param threads_in_warp: Supplies the logical warp size.
        :type  threads_in_warp: int, optional

        :param valid_items: Optionally limits how many items in the warp are
            valid for partial-tile scans.
        :type  valid_items: int, optional

        :param warp_aggregate: Optionally supplies a single-element output that
            receives the warp aggregate.
        :type  warp_aggregate: Any, optional

        :param temp_storage: Optionally supplies explicit cooperative temporary
            storage (e.g. via ``coop.TempStorage``).
        :type  temp_storage: Any, optional

        Example:
            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
                :language: python
                :dedent:
                :start-after: example-begin exclusive-scan
                :end-before: example-end exclusive-scan
        """
        self.temp_storage = temp_storage
        self.dtype = normalize_dtype_param(dtype)
        self.scan_op = scan_op if isinstance(scan_op, ScanOp) else ScanOp(scan_op)
        self.initial_value = initial_value
        self.threads_in_warp = threads_in_warp
        self.valid_items = valid_items
        self.warp_aggregate = warp_aggregate

        parameters = []
        specialization_kwds = {
            "T": self.dtype,
            "VIRTUAL_WARP_THREADS": threads_in_warp,
        }
        use_sum_method = (
            self.scan_op.is_sum and initial_value is None and valid_items is None
        )
        if use_sum_method:
            method_name = "ExclusiveSum"
            parameters = [
                [
                    DependentReference(Dependency("T")),
                    DependentReference(Dependency("T"), True),
                ]
            ]
            if warp_aggregate is not None:
                parameters[0].append(
                    DependentPointerReference(
                        Dependency("T"),
                        name="warp_aggregate",
                        is_array_pointer=True,
                    )
                )
        else:
            method_name = (
                "ExclusiveScanPartial" if valid_items is not None else "ExclusiveScan"
            )
            params = [
                DependentReference(Dependency("T")),
                DependentReference(Dependency("T"), True),
            ]
            if initial_value is not None:
                params.append(DependentReference(Dependency("T")))
            params.append(_make_scan_op_param(self.scan_op))
            if valid_items is not None:
                params.append(Value(numba.types.int32, name="valid_items"))
            if warp_aggregate is not None:
                params.append(
                    DependentPointerReference(
                        Dependency("T"),
                        name="warp_aggregate",
                        is_array_pointer=True,
                    )
                )
            parameters = [params]
            if self.scan_op.is_callable:
                specialization_kwds["ScanOp"] = self.scan_op.op
        if temp_storage is not None:
            parameters[0].insert(
                0,
                TempStoragePointer(
                    numba.types.uint8, is_array_pointer=True, name="temp_storage"
                ),
            )

        template = Algorithm(
            "WarpScan",
            method_name,
            "warp_scan",
            ["cub/warp/warp_scan.cuh"],
            [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
            parameters,
            self,
            fake_return=True,
            threads=threads_in_warp,
            unique_id=unique_id,
        )
        self.algorithm = template
        self.specialization = template.specialize(specialization_kwds)

    @classmethod
    def create(cls, dtype, scan_op, initial_value=None, threads_in_warp=32):
        algo = cls(
            dtype=dtype,
            scan_op=scan_op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class inclusive_scan(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        scan_op,
        initial_value=None,
        threads_in_warp=32,
        valid_items=None,
        warp_aggregate=None,
        unique_id=None,
        temp_storage=None,
    ):
        """
        Computes an inclusive warp-wide prefix scan.

        :param dtype: Supplies the scan item dtype.
        :type  dtype: DtypeType

        :param scan_op: Supplies the scan operator (known-op string, callable,
            or ``ScanOp`` instance).
        :type  scan_op: ScanOpType

        :param initial_value: Optionally supplies an initial value for
            ``InclusiveScan`` overloads.
        :type  initial_value: Any, optional

        :param threads_in_warp: Supplies the logical warp size.
        :type  threads_in_warp: int, optional

        :param valid_items: Optionally limits how many items in the warp are
            valid for partial-tile scans.
        :type  valid_items: int, optional

        :param warp_aggregate: Optionally supplies a single-element output that
            receives the warp aggregate.
        :type  warp_aggregate: Any, optional

        :param temp_storage: Optionally supplies explicit cooperative temporary
            storage (e.g. via ``coop.TempStorage``).
        :type  temp_storage: Any, optional

        Example:
            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
                :language: python
                :dedent:
                :start-after: example-begin inclusive-scan
                :end-before: example-end inclusive-scan
        """
        self.temp_storage = temp_storage
        self.dtype = normalize_dtype_param(dtype)
        self.scan_op = scan_op if isinstance(scan_op, ScanOp) else ScanOp(scan_op)
        self.initial_value = initial_value
        self.threads_in_warp = threads_in_warp
        self.valid_items = valid_items
        self.warp_aggregate = warp_aggregate

        parameters = []
        specialization_kwds = {
            "T": self.dtype,
            "VIRTUAL_WARP_THREADS": threads_in_warp,
        }
        use_sum_method = (
            self.scan_op.is_sum and initial_value is None and valid_items is None
        )
        if use_sum_method:
            method_name = "InclusiveSum"
            parameters = [
                [
                    DependentReference(Dependency("T")),
                    DependentReference(Dependency("T"), True),
                ]
            ]
            if warp_aggregate is not None:
                parameters[0].append(
                    DependentPointerReference(
                        Dependency("T"),
                        name="warp_aggregate",
                        is_array_pointer=True,
                    )
                )
        else:
            method_name = (
                "InclusiveScanPartial" if valid_items is not None else "InclusiveScan"
            )
            params = [
                DependentReference(Dependency("T")),
                DependentReference(Dependency("T"), True),
            ]
            if initial_value is not None:
                params.append(DependentReference(Dependency("T")))
            params.append(_make_scan_op_param(self.scan_op))
            if valid_items is not None:
                params.append(Value(numba.types.int32, name="valid_items"))
            if warp_aggregate is not None:
                params.append(
                    DependentPointerReference(
                        Dependency("T"),
                        name="warp_aggregate",
                        is_array_pointer=True,
                    )
                )
            parameters = [params]
            if self.scan_op.is_callable:
                specialization_kwds["ScanOp"] = self.scan_op.op
        if temp_storage is not None:
            parameters[0].insert(
                0,
                TempStoragePointer(
                    numba.types.uint8, is_array_pointer=True, name="temp_storage"
                ),
            )

        template = Algorithm(
            "WarpScan",
            method_name,
            "warp_scan",
            ["cub/warp/warp_scan.cuh"],
            [TemplateParameter("T"), TemplateParameter("VIRTUAL_WARP_THREADS")],
            parameters,
            self,
            fake_return=True,
            threads=threads_in_warp,
            unique_id=unique_id,
        )
        self.algorithm = template
        self.specialization = template.specialize(specialization_kwds)

    @classmethod
    def create(cls, dtype, scan_op, initial_value=None, threads_in_warp=32):
        algo = cls(
            dtype=dtype,
            scan_op=scan_op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


def _build_exclusive_sum_spec(
    dtype,
    threads_in_warp=32,
    warp_aggregate=None,
):
    return {
        "dtype": dtype,
        "threads_in_warp": threads_in_warp,
        "warp_aggregate": warp_aggregate,
    }


def _build_inclusive_sum_spec(
    dtype,
    threads_in_warp=32,
    warp_aggregate=None,
):
    return {
        "dtype": dtype,
        "threads_in_warp": threads_in_warp,
        "warp_aggregate": warp_aggregate,
    }


def _build_exclusive_scan_spec(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
):
    return {
        "dtype": dtype,
        "scan_op": scan_op,
        "initial_value": initial_value,
        "threads_in_warp": threads_in_warp,
        "valid_items": valid_items,
        "warp_aggregate": warp_aggregate,
    }


def _build_inclusive_scan_spec(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
):
    return {
        "dtype": dtype,
        "scan_op": scan_op,
        "initial_value": initial_value,
        "threads_in_warp": threads_in_warp,
        "valid_items": valid_items,
        "warp_aggregate": warp_aggregate,
    }


def _make_exclusive_sum_two_phase(
    dtype,
    threads_in_warp=32,
):
    return exclusive_sum.create(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
    )


def _make_exclusive_sum_rewrite(
    dtype,
    threads_in_warp=32,
    warp_aggregate=None,
    unique_id=None,
    temp_storage=None,
):
    spec = _build_exclusive_sum_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        warp_aggregate=warp_aggregate,
    )
    spec.update({"unique_id": unique_id, "temp_storage": temp_storage})
    return exclusive_sum(**spec)


def _make_inclusive_sum_two_phase(
    dtype,
    threads_in_warp=32,
):
    return inclusive_sum.create(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
    )


def _make_inclusive_sum_rewrite(
    dtype,
    threads_in_warp=32,
    warp_aggregate=None,
    unique_id=None,
    temp_storage=None,
):
    spec = _build_inclusive_sum_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        warp_aggregate=warp_aggregate,
    )
    spec.update({"unique_id": unique_id, "temp_storage": temp_storage})
    return inclusive_sum(**spec)


def _make_exclusive_scan_two_phase(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
):
    return exclusive_scan.create(
        dtype=dtype,
        scan_op=scan_op,
        initial_value=initial_value,
        threads_in_warp=threads_in_warp,
    )


def _make_exclusive_scan_rewrite(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
    unique_id=None,
    temp_storage=None,
):
    spec = _build_exclusive_scan_spec(
        dtype=dtype,
        scan_op=scan_op,
        initial_value=initial_value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        warp_aggregate=warp_aggregate,
    )
    spec.update({"unique_id": unique_id, "temp_storage": temp_storage})
    return exclusive_scan(**spec)


def _make_inclusive_scan_two_phase(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
):
    return inclusive_scan.create(
        dtype=dtype,
        scan_op=scan_op,
        initial_value=initial_value,
        threads_in_warp=threads_in_warp,
    )


def _make_inclusive_scan_rewrite(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
    unique_id=None,
    temp_storage=None,
):
    spec = _build_inclusive_scan_spec(
        dtype=dtype,
        scan_op=scan_op,
        initial_value=initial_value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        warp_aggregate=warp_aggregate,
    )
    spec.update({"unique_id": unique_id, "temp_storage": temp_storage})
    return inclusive_scan(**spec)
