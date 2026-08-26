# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register GPU dataclasses with both Numba-CUDA-MLIR type systems.

This module owns record-model registration, launch argument marshalling, and
the per-class caches needed to keep compiler identities stable.
"""

import dataclasses
import itertools
import threading
import weakref
from typing import Any

import numba_cuda_mlir.models as mlir_models
import numba_cuda_mlir.numba_cuda.extending as cuda_extending
from numba_cuda_mlir import types
from numba_cuda_mlir.extending import ArgumentHandler, typeof_impl
from numba_cuda_mlir.models import register_model as register_mlir_model
from numba_cuda_mlir.numba_cuda.cudadecl import registry as cuda_registry
from numba_cuda_mlir.numba_cuda.typing.templates import AttributeTemplate
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof

_CUDA_RECORD_MODEL = cuda_extending.models.StructModel
_MLIR_RECORD_MODEL = mlir_models.StructModel

_GPU_DATACLASS_UNIQUE_ID_COUNTER = itertools.count()
_GPU_DATACLASS_TYPES_BY_CLASS = {}
_GPU_DATACLASS_SIGNATURE_REFCOUNTS = {}
_GPU_DATACLASS_TYPES_BY_SIGNATURE = {}
_GPU_DATACLASS_TYPEOF_REGISTERED = set()
_GPU_DATACLASS_LOCK = threading.RLock()


class _GpuDataclassArgument(tuple):
    """Flattenable launch value that retains its registered frontend type."""


@typeof_impl.register(_GpuDataclassArgument)
def _typeof_gpu_dataclass_argument(val, c):
    _ = c
    return val._gpu_dataclass_type


class _GpuDataclassArgumentHandler:
    """Marshal registered dataclasses through the public dispatcher hook."""

    def prepare_args(self, ty, val, stream=None, retr=None):
        _ = stream
        _ = retr

        with _GPU_DATACLASS_LOCK:
            instances = _GPU_DATACLASS_TYPES_BY_CLASS.get(val.__class__)
            registration = None if instances is None else instances.get(id(val))

        if registration is None:
            return (ty, val)

        registered_ref, registered_type, _ = registration
        if registered_ref() is not val:
            return (ty, val)

        from ._types import Invocable

        def coerce_field(field_value):
            if isinstance(field_value, Invocable):
                return 0
            return field_value

        values = (
            coerce_field(getattr(val, field.name)) for field in dataclasses.fields(val)
        )
        return (registered_type, registered_type._argument_type(values))


gpu_dataclass_argument_handler: ArgumentHandler = _GpuDataclassArgumentHandler()


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


def gpu_dataclass(dc: Any, *, compute_temp_storage: bool = True) -> Any:
    """Register a Python dataclass instance as a Numba-CUDA-MLIR kernel argument.

    ``gpu_dataclass`` exposes dataclass fields as attributes inside
    ``numba_cuda_mlir.cuda.jit`` kernels. Kernels that accept a registered
    dataclass must include ``gpu_dataclass_argument_handler`` in their
    ``cuda.jit(extensions=[...])`` list so the Python object is flattened at
    launch time.
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

    from ._types import Invocable, prepare_ltoir_bundle

    primitives = {
        name: obj for (name, obj) in zip(names, objs) if isinstance(obj, Invocable)
    }
    if compute_temp_storage and primitives:
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

        dc.temp_storage_bytes_sum = sum(
            obj.temp_storage_bytes for obj in primitives.values()
        )
        dc.temp_storage_bytes_max = max(
            obj.temp_storage_bytes for obj in primitives.values()
        )
        dc.temp_storage_alignment = max(
            obj.temp_storage_alignment for obj in primitives.values()
        )
    else:
        dc.temp_storage_bytes_sum = 0
        dc.temp_storage_bytes_max = 0
        dc.temp_storage_alignment = 0

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

            class GpuDataClassType(types.BaseTuple):
                def __init__(self):
                    self.types = tuple(type_obj for _, type_obj in members)
                    self.count = len(self.types)
                    super().__init__(
                        name=(f"GpuDataClass[{dc_class_name}:{field_signature_name}]")
                    )

                def __getitem__(self, index):
                    return self.types[index]

                def __iter__(self):
                    return iter(self.types)

                def __len__(self):
                    return self.count

            gpu_dataclass_type = GpuDataClassType()

            class GpuDataClassArgument(_GpuDataclassArgument):
                pass

            GpuDataClassArgument._gpu_dataclass_type = gpu_dataclass_type
            gpu_dataclass_type._argument_type = GpuDataClassArgument
            _GPU_DATACLASS_TYPES_BY_SIGNATURE[signature_key] = gpu_dataclass_type

            @cuda_extending.register_model(GpuDataClassType)
            class GpuDataClassCudaRecordModel(_CUDA_RECORD_MODEL):
                def __init__(self, dmm, fe_type):
                    super().__init__(dmm, fe_type, members)

            @register_mlir_model(GpuDataClassType)
            class GpuDataClassMlirRecordModel(_MLIR_RECORD_MODEL):
                def __init__(self, dmm, fe_type):
                    super().__init__(dmm, fe_type, members)

                def _as_abi(self, method_name, builder, value):
                    return tuple(
                        getattr(model, method_name)(
                            builder,
                            mlir_models.llvm.extractvalue(
                                model.get_value_type(), value, [index]
                            ),
                        )
                        for index, model in enumerate(self._models)
                    )

                def _from_abi(self, method_name, builder, values):
                    result = mlir_models.llvm.UndefOp(self.get_value_type()).result
                    for index, (model, value) in enumerate(zip(self._models, values)):
                        converted = getattr(model, method_name)(builder, value)
                        result = mlir_models.llvm.insertvalue(
                            result, converted, [index]
                        )
                    return result

                def get_argument_type(self):
                    return tuple(model.get_argument_type() for model in self._models)

                def get_return_type(self):
                    return tuple(model.get_return_type() for model in self._models)

                def as_argument(self, builder, value):
                    return self._as_abi("as_argument", builder, value)

                def from_argument(self, builder, value):
                    return self._from_abi("from_argument", builder, value)

                def as_return(self, builder, value):
                    return self._as_abi("as_return", builder, value)

                def from_return(self, builder, value):
                    return self._from_abi("from_return", builder, value)

            class GpuDataClassAttrsTemplate(AttributeTemplate):
                key = gpu_dataclass_type

            for name, type_obj in members:

                def resolver(self, this, type_obj=type_obj):
                    return type_obj

                setattr(GpuDataClassAttrsTemplate, f"resolve_{name}", resolver)

                cuda_extending.make_attribute_wrapper(GpuDataClassType, name, name)

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
        same_registration = (
            previous is not None
            and previous[0]() is dc
            and previous[2] == signature_key
        )
        if not same_registration:
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

    dc.__cuda_coop_numba_mlir_gpu_dataclass__ = True

    return dc


__all__ = ["gpu_dataclass", "gpu_dataclass_argument_handler"]
