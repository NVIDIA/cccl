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

        Example (explicit temp storage):
            temp_storage = coop.TempStorage(bytes, alignment)
            out = warp_exclusive_sum(x, temp_storage=temp_storage)
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

        Example (explicit temp storage):
            temp_storage = coop.TempStorage(bytes, alignment)
            out = warp_inclusive_sum(x, temp_storage=temp_storage)
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
        Computes an exclusive warp-wide prefix scan using the specified scan operator.

        Example (explicit temp storage):
            temp_storage = coop.TempStorage(bytes, alignment)
            out = warp_exclusive_scan(x, temp_storage=temp_storage)
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
        Computes an inclusive warp-wide prefix scan using the specified scan operator.

        Example (explicit temp storage):
            temp_storage = coop.TempStorage(bytes, alignment)
            out = warp_inclusive_scan(x, temp_storage=temp_storage)
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
