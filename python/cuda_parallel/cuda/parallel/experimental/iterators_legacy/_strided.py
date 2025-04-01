import ctypes
from functools import lru_cache
from typing import Tuple

import numba
import numpy as np
from numba import types
from numba.core import cgutils
from numba.core.extending import (
    models,
    register_model,
)
from numba.core.typing.templates import AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import registry as cuda_lower_registry

from . import _iterators


@lru_cache
def make_iterator_struct_class(ndim):
    shape_ctype = ctypes.c_int64 * ndim
    strides_ctype = ctypes.c_int64 * ndim

    class StridedArrayView(ctypes.Structure):
        _fields_ = [
            ("linear_id", ctypes.c_int64),
            ("ptr", ctypes.c_void_p),
            ("shape", shape_ctype),
            ("strides", strides_ctype),
            ("ndim", ctypes.c_int32),
        ]

    return StridedArrayView


def iterator_struct_ctype(ptr: int, ndim: int, shape: Tuple[int], strides: Tuple[int]):
    StridedArrayView_cls = make_iterator_struct_class(ndim)

    c_shape = (ctypes.c_int64 * ndim)(*shape)
    c_strides = (ctypes.c_int64 * ndim)(*strides)
    return StridedArrayView_cls(0, ptr, c_shape, c_strides, ndim)


@lru_cache
def strided_view_iterator_numba_type(value_type: types.Type, ndim: int):
    """Returns the numba type that stores a typed pointer
    to record describing strided view into nd-array of
    elements with type `value_type` that has `ndim` dimensions.

    On the host the struct would be accessed using
    ``StridedArrayView`` ctype class defined above.
    """
    # ------
    # Typing
    # ------

    # View into strided device ndarray
    class NdArrayViewType(types.Type):
        def __init__(self):
            super(NdArrayViewType, self).__init__(name="NdArrayView")

    ndarray_view_type = NdArrayViewType()
    ptr_type = types.CPointer(ndarray_view_type)

    int64_numba_t = numba.from_dtype(np.int64)
    shape_arr_numba_t = types.UniTuple(int64_numba_t, ndim)
    strides_arr_numba_t = types.UniTuple(int64_numba_t, ndim)
    ndarray_view_members = [
        ("linear_id", int64_numba_t),
        ("ptr", types.CPointer(value_type)),
        ("shape", shape_arr_numba_t),
        ("strides", strides_arr_numba_t),
        ("ndim", numba.from_dtype(np.int32)),
    ]

    # Typing for accessing attributes of the struct members
    class NdArrayViewAttrsTemplate(AttributeTemplate):
        pass

    def make_attr_resolver(ty):
        """
        Function to capture a copy of **ty** argument in resolve function
        """

        def resolve_fn(self, pp):
            return ty

        return resolve_fn

    for name, typ in ndarray_view_members:
        setattr(NdArrayViewAttrsTemplate, f"resolve_{name}", make_attr_resolver(typ))

    @cuda_registry.register_attr
    class NdArrayViewAttrs(NdArrayViewAttrsTemplate):
        key = ndarray_view_type

    @cuda_registry.register_attr
    class PtrAttrs(AttributeTemplate):
        key = ptr_type

        def resolve_linear_id(self, pp):
            return int64_numba_t

    # -----------
    # Data models
    # -----------

    @register_model(NdArrayViewType)
    class NdArrayViewModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            super().__init__(dmm, fe_type, ndarray_view_members)

    # --------
    # Lowering
    # --------

    @cuda_lower_registry.lower_getattr_generic(ndarray_view_type)
    def ndarray_view_getattr_lowering_fn(context, builder, sig, arg, attr):
        struct_values = cgutils.create_struct_proxy(ndarray_view_type)(
            context, builder, value=arg
        )
        attr_ptr = struct_values._get_ptr_by_name(attr)
        attr_val = builder.load(attr_ptr)
        return attr_val

    @cuda_lower_registry.lower_setattr(ptr_type, "linear_id")
    def ndarray_view_pointer_set_linear_id(context, builder, sig, args):
        data = builder.load(args[0])
        values = cgutils.create_struct_proxy(ndarray_view_type)(
            context, builder, value=data
        )
        setattr(values, "linear_id", args[1])
        return builder.store(values._getvalue(), args[0])

    @cuda_lower_registry.lower_getattr(ptr_type, "linear_id")
    def ndarray_view_pointer_get_linear_id(context, builder, sig, arg):
        data = builder.load(arg)
        values = cgutils.create_struct_proxy(ndarray_view_type)(
            context, builder, value=data
        )
        attr_ptr = values._get_ptr_by_name("linear_id")
        attr_val = builder.load(attr_ptr)
        return attr_val

    return ndarray_view_type


class NdArrayIteratorKind(_iterators.IteratorKind):
    pass


class NdArrayIterator(_iterators.IteratorBase):
    iterator_kind_type = NdArrayIteratorKind

    def __init__(
        self, ptr: int, value_type: types.Type, shape: Tuple[int], strides: Tuple[int]
    ):
        ndim = len(shape)
        if not (len(strides) == ndim):
            raise ValueError

        state_numba_type = strided_view_iterator_numba_type(value_type, ndim)
        # build ctypes struct for state of iterator
        host_sav_cvalue = iterator_struct_ctype(ptr, ndim, shape, strides)
        super().__init__(
            cvalue=host_sav_cvalue, numba_type=state_numba_type, value_type=value_type
        )

    @property
    def ltoirs(self):
        abi_suffix = _iterators._get_abi_suffix(self.kind)
        advance_abi_name = f"{self.prefix}advance_{abi_suffix}"
        deref_abi_name = f"{self.prefix}dereference_{abi_suffix}"
        state_arg_numba_type = types.CPointer(self.numba_type)
        advance_ltoir, _ = _iterators.cached_compile(
            self.__class__.advance,
            (
                state_arg_numba_type,
                types.uint64,  # distance type
            ),
            output="ltoir",
            abi_name=advance_abi_name,
        )

        deref_ltoir, _ = _iterators.cached_compile(
            self.__class__.dereference,
            (state_arg_numba_type,),
            output="ltoir",
            abi_name=deref_abi_name,
        )
        return {advance_abi_name: advance_ltoir, deref_abi_name: deref_ltoir}

    @staticmethod
    def advance(state_ref, distance):
        state_ref.linear_id = state_ref.linear_id + distance

    @staticmethod
    def dereference(state_ref):
        state = state_ref[0]
        id_ = state.linear_id
        # init offset_ to zero of the same type as id_
        offset_ = id_ - id_
        ndim_ = state.ndim
        if ndim_ > 0:
            shape_ = state.shape
            strides_ = state.strides
            one_i32 = numba.int32(1)
            for i in range(one_i32, ndim_):
                bi_ = ndim_ - i
                sh_i = shape_[bi_]
                if sh_i > 0:
                    q_ = id_ // sh_i
                    r_ = id_ - q_ * sh_i
                else:
                    q_ = id_
                    r_ = id_ - id_  # make zero of the right type
                offset_ = offset_ + r_ * strides_[bi_]
                id_ = q_
            zero_i32 = one_i32 - one_i32
            offset_ = offset_ + id_ * strides_[zero_i32]
        val = (state.ptr)[offset_]
        return val


def make_ndarray_iterator(array_like, perm):
    ptr = array_like.data.ptr
    dt = array_like.dtype
    shape_ = array_like.shape
    strides_ = array_like.strides
    itemsize = array_like.itemsize
    perm_shape, perm_strides, rems = zip(
        *tuple(
            (shape_[idx], (strides_[idx] // itemsize), strides_[idx] % itemsize)
            for idx in perm
        )
    )
    assert all(rem == 0 for rem in rems)

    return NdArrayIterator(ptr, numba.from_dtype(dt), perm_shape, perm_strides)
