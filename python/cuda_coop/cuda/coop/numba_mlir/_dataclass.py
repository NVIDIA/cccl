# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import dataclasses
import itertools
import threading
import weakref
from typing import Any

import numba_cuda_mlir.models as mlir_models
from numba_cuda_mlir import types
from numba_cuda_mlir.models import register_model as register_mlir_model
from numba_cuda_mlir.numba_cuda.cudadecl import registry as cuda_registry
from numba_cuda_mlir.numba_cuda.extending import (
    make_attribute_wrapper,
    typeof_impl,
)
from numba_cuda_mlir.numba_cuda.extending import (
    models as cuda_models,
)
from numba_cuda_mlir.numba_cuda.extending import (
    register_model as register_cuda_model,
)
from numba_cuda_mlir.numba_cuda.typing.templates import AttributeTemplate
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof

_CUDA_RECORD_MODEL = getattr(cuda_models, "Struct" + "Model")
_MLIR_RECORD_MODEL = getattr(mlir_models, "Struct" + "Model")

_GPU_DATACLASS_UNIQUE_ID_COUNTER = itertools.count()
_GPU_DATACLASS_TYPES_BY_CLASS = {}
_GPU_DATACLASS_SIGNATURE_REFCOUNTS = {}
_GPU_DATACLASS_TYPES_BY_SIGNATURE = {}
_GPU_DATACLASS_TYPEOF_REGISTERED = set()
_GPU_DATACLASS_LOCK = threading.RLock()


def _planner_field_semantic_key(value):
    """Return the compile-time semantics a planner may read from *value*."""

    from ._types import Invocable, algo_coalesce_key

    if isinstance(value, Invocable):
        return (
            "invocable",
            algo_coalesce_key(value.specialization),
            value.temp_storage_bytes,
            value.temp_storage_alignment,
        )

    return None


def _release_gpu_dataclass_signature(signature_key):
    with _GPU_DATACLASS_LOCK:
        count = _GPU_DATACLASS_SIGNATURE_REFCOUNTS.get(signature_key, 0)
        if count <= 1:
            _GPU_DATACLASS_SIGNATURE_REFCOUNTS.pop(signature_key, None)
            _GPU_DATACLASS_TYPES_BY_SIGNATURE.pop(signature_key, None)
        else:
            _GPU_DATACLASS_SIGNATURE_REFCOUNTS[signature_key] = count - 1


def _remove_registered_gpu_dataclass(cls, dc_id, dc_ref):
    with _GPU_DATACLASS_LOCK:
        instances = _GPU_DATACLASS_TYPES_BY_CLASS.get(cls)
        if instances is None:
            return
        current = instances.get(dc_id)
        if current is not None and current[0] is dc_ref:
            _, _, signature_key = instances.pop(dc_id)
            _release_gpu_dataclass_signature(signature_key)
        if not instances:
            _GPU_DATACLASS_TYPES_BY_CLASS.pop(cls, None)


def gpu_dataclass(dc: Any, *, compute_temp_storage: bool = True) -> Any:
    """Register a Python dataclass instance as a Numba-CUDA-MLIR kernel argument.

    ``gpu_dataclass`` exposes dataclass fields as attributes inside
    ``numba_cuda_mlir.cuda.jit`` kernels and installs the launch-time argument
    hook needed to marshal the Python object as its flattened field values.
    Numba-CUDA-MLIR primitive descriptors contribute compile-time
    specialization identity and aggregate temp-storage metadata. All scalar
    fields remain ordinary by-value runtime data. Primitive fields are
    marshalled as dummy pointer-sized values.

    The dataclass instance must support weak references and remain alive for
    every kernel launch that uses it.
    """
    fields = dataclasses.fields(dc)
    names = [f.name for f in fields]
    objs = [getattr(dc, name) for name in names]

    from ._types import Invocable

    primitives = {
        name: obj for (name, obj) in zip(names, objs) if isinstance(obj, Invocable)
    }
    if compute_temp_storage and primitives:
        from ._types import prepare_ltoir_bundle

        algorithms = []
        seen = set()
        for primitive in primitives.values():
            algo = getattr(primitive, "specialization", None)
            if algo is None:
                continue
            if "_size_and_alignment_info" in algo.__dict__:
                continue
            key = id(algo)
            if key in seen:
                continue
            seen.add(key)
            algorithms.append(algo)

        if algorithms:
            for algo in algorithms:
                if getattr(algo, "unique_id", None) is None:
                    algo.unique_id = next(_GPU_DATACLASS_UNIQUE_ID_COUNTER)

            prepare_ltoir_bundle(
                algorithms,
                bundle_name=f"cuda_coop_numba_mlir_gpu_dataclass_{id(dc)}",
                allow_single=True,
            )

        setattr(
            dc,
            "temp_storage_bytes_sum",
            sum(obj.temp_storage_bytes for obj in primitives.values()),
        )
        setattr(
            dc,
            "temp_storage_bytes_max",
            max(obj.temp_storage_bytes for obj in primitives.values()),
        )
        setattr(
            dc,
            "temp_storage_alignment",
            max(obj.temp_storage_alignment for obj in primitives.values()),
        )
    else:
        setattr(dc, "temp_storage_bytes_sum", 0)
        setattr(dc, "temp_storage_bytes_max", 0)
        setattr(dc, "temp_storage_alignment", 0)

    def field_type(obj):
        if isinstance(obj, Invocable):
            return types.uintp
        return typeof(obj)

    members = [(name, field_type(obj)) for (name, obj) in zip(names, objs)]

    field_signature = tuple((name, str(type_obj)) for name, type_obj in members)
    planner_signature = tuple(
        (name, semantic_key)
        for name, obj in zip(names, objs)
        if (semantic_key := _planner_field_semantic_key(obj)) is not None
    )
    signature_key = (dc.__class__, field_signature, planner_signature)
    with _GPU_DATACLASS_LOCK:
        gpu_dataclass_type = _GPU_DATACLASS_TYPES_BY_SIGNATURE.get(signature_key)
        if gpu_dataclass_type is None:
            dc_class_name = dc.__class__.__qualname__
            field_signature_name = "_".join(
                f"{name}_{type_obj}" for name, type_obj in members
            )

            class GpuDataClassType(types.Type):
                def __init__(self):
                    super().__init__(
                        name=(f"GpuDataClass[{dc_class_name}:{field_signature_name}]")
                    )

            gpu_dataclass_type = GpuDataClassType()
            _GPU_DATACLASS_TYPES_BY_SIGNATURE[signature_key] = gpu_dataclass_type

            @register_cuda_model(GpuDataClassType)
            class GpuDataClassCudaRecordModel(_CUDA_RECORD_MODEL):
                def __init__(self, dmm, fe_type):
                    super().__init__(dmm, fe_type, members)

            @register_mlir_model(GpuDataClassType)
            class GpuDataClassMlirRecordModel(_MLIR_RECORD_MODEL):
                def __init__(self, dmm, fe_type):
                    super().__init__(dmm, fe_type, members)

            class GpuDataClassAttrsTemplate(AttributeTemplate):
                key = gpu_dataclass_type

            for name, type_obj in members:

                def resolver(self, this, type_obj=type_obj):
                    return type_obj

                setattr(GpuDataClassAttrsTemplate, f"resolve_{name}", resolver)

                make_attribute_wrapper(GpuDataClassType, name, name)

            cuda_registry.register_attr(GpuDataClassAttrsTemplate)

            # The compiler contexts may already exist when a descriptor is
            # registered. Synchronize both typing/lowering registries before
            # the next public compilation attempt.
            from numba_cuda_mlir.extending import refresh_registries

            refresh_registries()

    dc_id = id(dc)

    def remove_registered(
        dc_ref,
        cls=dc.__class__,
        dc_id=dc_id,
        lock=_GPU_DATACLASS_LOCK,
        types_by_class=_GPU_DATACLASS_TYPES_BY_CLASS,
        signature_refcounts=_GPU_DATACLASS_SIGNATURE_REFCOUNTS,
        types_by_signature=_GPU_DATACLASS_TYPES_BY_SIGNATURE,
    ):
        with lock:
            instances = types_by_class.get(cls)
            if instances is None:
                return
            current = instances.get(dc_id)
            if current is not None and current[0] is dc_ref:
                _, _, signature_key = instances.pop(dc_id)
                count = signature_refcounts.get(signature_key, 0)
                if count <= 1:
                    signature_refcounts.pop(signature_key, None)
                    types_by_signature.pop(signature_key, None)
                else:
                    signature_refcounts[signature_key] = count - 1
            if not instances:
                types_by_class.pop(cls, None)

    try:
        dc_ref = weakref.ref(dc, remove_registered)
    except TypeError as exc:
        raise TypeError(
            f"{dc.__class__.__qualname__} instances must support weak references"
        ) from exc

    with _GPU_DATACLASS_LOCK:
        class_instances = _GPU_DATACLASS_TYPES_BY_CLASS.setdefault(dc.__class__, {})
        previous = class_instances.get(dc_id)
        if previous is not None:
            _release_gpu_dataclass_signature(previous[2])

        _GPU_DATACLASS_SIGNATURE_REFCOUNTS[signature_key] = (
            _GPU_DATACLASS_SIGNATURE_REFCOUNTS.get(signature_key, 0) + 1
        )
        class_instances[dc_id] = (
            dc_ref,
            gpu_dataclass_type,
            signature_key,
        )

        register_typeof = dc.__class__ not in _GPU_DATACLASS_TYPEOF_REGISTERED
        if register_typeof:
            _GPU_DATACLASS_TYPEOF_REGISTERED.add(dc.__class__)

    if register_typeof:

        @typeof_impl.register(dc.__class__)
        def typeof_gpu_dataclass(val, c):
            try:
                with _GPU_DATACLASS_LOCK:
                    registered_ref, registered_type, _ = _GPU_DATACLASS_TYPES_BY_CLASS[
                        val.__class__
                    ][id(val)]
                registered = registered_ref()
                if registered is not val:
                    raise KeyError
                return registered_type
            except KeyError as exc:
                raise TypeError(
                    f"{val.__class__.__qualname__} instance has not been registered "
                    "with cuda.coop.numba_mlir.gpu_dataclass"
                ) from exc

    def pre_launch_callback(kernel, launch_config):
        _ = kernel
        _ = launch_config

    setattr(dc, "pre_launch_callback", pre_launch_callback)

    def prepare_args(ty, val, *args, **kwds):
        registered = dc_ref()
        if registered is None or val is not registered:
            return (ty, val)

        def coerce_field(field_val: Any):
            if isinstance(field_val, Invocable):
                return 0
            return field_val

        # Keep the dataclass frontend type for compilation. The
        # numba-cuda-mlir launcher flattens this tuple of field values to match
        # the MLIR record-model ABI.
        return (ty, tuple(coerce_field(getattr(val, name)) for name in names))

    setattr(dc, "prepare_args", prepare_args)
    setattr(dc, "__cuda_coop_numba_mlir_gpu_dataclass__", True)

    return dc
