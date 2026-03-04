# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

# Python signatures are declared in the companion Python stub file _bindings.pyi
# Make sure to update PYI with change to Python API to ensure that Python
# static type checker tools like mypy green-lights cuda.cccl.parallel

from cpython.buffer cimport (
    Py_buffer, PyBUF_FORMAT, PyBUF_ND, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS,
    PyObject_GetBuffer, PyBuffer_Release, PyObject_CheckBuffer
)
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pycapsule cimport (
    PyCapsule_CheckExact, PyCapsule_IsValid, PyCapsule_GetPointer
)
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t, uintptr_t
from libc.string cimport memset, memcpy

import numpy as np

import ctypes
from enum import IntFlag

# ctypes Structure mirrors for the composite mapper callback.
# The C API uses void (*)(stf_pos4*, stf_pos4, stf_dim4, stf_dim4) so ctypes
# can create per-mapper callbacks (no struct return = no ctypes limitation).
class _mapper_pos4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int64), ("y", ctypes.c_int64),
                ("z", ctypes.c_int64), ("t", ctypes.c_int64)]

class _mapper_dim4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64), ("y", ctypes.c_uint64),
                ("z", ctypes.c_uint64), ("t", ctypes.c_uint64)]

_mapper_cfunc_type = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(_mapper_pos4), _mapper_pos4, _mapper_dim4, _mapper_dim4)


def _make_mapper_callback(mapper):
    """Wrap a Python partitioner as a C function pointer for the STF C API.

    Returns (callback_object, c_function_pointer_as_int).
    The caller must prevent GC of callback_object for the lifetime of the
    composite data place.
    """
    def _trampoline(result_ptr, c_coords, c_data_dims, c_grid_dims):
        coords = (c_coords.x, c_coords.y, c_coords.z, c_coords.t)
        data_dims = (c_data_dims.x, c_data_dims.y, c_data_dims.z, c_data_dims.t)
        grid_dims = (c_grid_dims.x, c_grid_dims.y, c_grid_dims.z, c_grid_dims.t)
        rx, ry, rz, rt = mapper(coords, data_dims, grid_dims)
        result_ptr[0].x = int(rx)
        result_ptr[0].y = int(ry)
        result_ptr[0].z = int(rz)
        result_ptr[0].t = int(rt)

    callback = _mapper_cfunc_type(_trampoline)
    c_ptr = ctypes.cast(callback, ctypes.c_void_p).value
    return (callback, c_ptr)

cdef extern from "<cuda.h>":
    cdef struct OpaqueCUstream_st
    cdef struct OpaqueCUkernel_st
    cdef struct OpaqueCUlibrary_st

    ctypedef int CUresult
    ctypedef OpaqueCUstream_st *CUstream
    ctypedef OpaqueCUkernel_st *CUkernel
    ctypedef OpaqueCUlibrary_st *CUlibrary

cdef extern from "cccl/c/experimental/stf/stf.h":
    #
    # Contexts
    #
    ctypedef struct stf_ctx_handle_t
    ctypedef stf_ctx_handle_t* stf_ctx_handle
    void stf_ctx_create(stf_ctx_handle* ctx)
    void stf_ctx_create_graph(stf_ctx_handle* ctx)
    void stf_ctx_finalize(stf_ctx_handle ctx)

    #
    # Exec places
    #
    ctypedef void* stf_exec_place_grid_handle

    ctypedef enum stf_exec_place_kind:
        STF_EXEC_PLACE_DEVICE
        STF_EXEC_PLACE_HOST
        STF_EXEC_PLACE_GRID
        STF_EXEC_PLACE_OPAQUE

    ctypedef struct stf_exec_place_device:
        int dev_id

    ctypedef struct stf_exec_place_host:
        int dummy

    ctypedef union stf_exec_place_u:
        stf_exec_place_device device
        stf_exec_place_host   host
        stf_exec_place_grid_handle grid
        void* opaque

    ctypedef struct stf_exec_place:
        stf_exec_place_kind kind
        stf_exec_place_u    u

    stf_exec_place make_device_place(int  dev_id)
    stf_exec_place make_host_place()
    stf_exec_place make_exec_place_from_grid(stf_exec_place_grid_handle grid)
    stf_exec_place make_opaque_exec_place(void* handle)

    void* stf_exec_place_opaque_wrap(const void* cpp_exec_place)
    void stf_exec_place_opaque_destroy(void* handle)
    void* stf_exec_place_to_opaque(const stf_exec_place* c_place)
    void* stf_exec_place_dummy_create(int dev_id)

    #
    # Data places
    #
    ctypedef struct stf_data_place_device:
        int dev_id

    ctypedef struct stf_data_place_host:
        int dummy

    ctypedef struct stf_data_place_managed:
        int dummy

    ctypedef struct stf_data_place_affine:
        int dummy

    # Composite data place (grid + partition function)
    ctypedef struct stf_pos4:
        int64_t x
        int64_t y
        int64_t z
        int64_t t

    ctypedef struct stf_dim4:
        uint64_t x
        uint64_t y
        uint64_t z
        uint64_t t

    ctypedef void (*stf_get_executor_fn)(stf_pos4* result, stf_pos4 data_coords, stf_dim4 data_dims, stf_dim4 grid_dims)

    ctypedef struct stf_data_place_composite:
        stf_exec_place_grid_handle grid
        stf_get_executor_fn mapper

    ctypedef enum stf_data_place_kind:
        STF_DATA_PLACE_DEVICE
        STF_DATA_PLACE_HOST
        STF_DATA_PLACE_MANAGED
        STF_DATA_PLACE_AFFINE
        STF_DATA_PLACE_COMPOSITE
        STF_DATA_PLACE_OPAQUE

    ctypedef union stf_data_place_u:
        stf_data_place_device device
        stf_data_place_host   host
        stf_data_place_managed   managed
        stf_data_place_affine   affine
        stf_data_place_composite composite
        void* opaque

    ctypedef struct stf_data_place:
        stf_data_place_kind kind
        stf_data_place_u    u

    stf_data_place make_device_data_place(int  dev_id)
    stf_data_place make_host_data_place()
    stf_data_place make_managed_data_place()
    stf_data_place make_affine_data_place()
    stf_data_place make_opaque_data_place(void* handle)

    void* stf_data_place_opaque_wrap(const void* cpp_data_place)
    void stf_data_place_opaque_destroy(void* handle)
    void* stf_data_place_to_opaque(const stf_data_place* c_place)

    ctypedef struct stf_logical_data_handle_t
    ctypedef stf_logical_data_handle_t* stf_logical_data_handle
    void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz)
    void stf_logical_data_with_place(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz, stf_data_place dplace)
    void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
    void stf_logical_data_destroy(stf_logical_data_handle ld)
    void stf_logical_data_empty(stf_ctx_handle ctx, size_t length, stf_logical_data_handle *to)

    void stf_token(stf_ctx_handle ctx, stf_logical_data_handle* ld);

    ctypedef struct stf_task_handle_t
    ctypedef stf_task_handle_t* stf_task_handle
    void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t)
    void stf_task_set_exec_place(stf_task_handle t, stf_exec_place* exec_p)
    void stf_task_set_symbol(stf_task_handle t, const char* symbol)
    void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
    void stf_task_add_dep_with_dplace(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place* data_p)
    void stf_make_composite_data_place(stf_data_place* out, stf_exec_place_grid_handle grid, stf_get_executor_fn mapper)
    stf_exec_place_grid_handle stf_exec_place_grid_from_devices(const int* device_ids, size_t count)
    stf_exec_place_grid_handle stf_exec_place_grid_create(const stf_exec_place* places, size_t count, const stf_dim4* grid_dims)
    void stf_exec_place_grid_destroy(stf_exec_place_grid_handle grid)
    void stf_exec_place_grid_set_affine_data_place(stf_exec_place_grid_handle grid, const stf_data_place* dplace)
    void stf_task_start(stf_task_handle t)
    void stf_task_end(stf_task_handle t)
    void stf_task_enable_capture(stf_task_handle t)
    CUstream stf_task_get_custream(stf_task_handle t)
    int stf_task_get_grid_dims(stf_task_handle t, stf_dim4* out_dims)
    int stf_task_get_custream_at_index(stf_task_handle t, size_t place_index, CUstream* out_stream)
    void* stf_task_get(stf_task_handle t, int submitted_index)
    void stf_task_destroy(stf_task_handle t)

    cdef enum stf_access_mode:
        STF_NONE
        STF_READ
        STF_WRITE
        STF_RW

class AccessMode(IntFlag):
    NONE  = STF_NONE
    READ  = STF_READ
    WRITE = STF_WRITE
    RW    = STF_RW

class stf_cai:
    """
    Wrapper that exposes __cuda_array_interface__ for interop (torch, cupy, etc.).
    Supports dict-style access (e.g. obj['data']) for code that expects a CAI dict.
    """
    def __init__(self, ptr, tuple shape, dtype, stream=0):
        self.ptr = ptr               # integer device pointer
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.stream = stream        # CUDA stream handle (int or 0)
        self.__cuda_array_interface__ = {
            'version': 2,
            'shape': self.shape,
            'typestr': self.dtype.str,     # e.g., '<f4' for float32
            'data': (self.ptr, False),     # (ptr, read-only?)
            'strides': None,               # or tuple of strides in bytes
            'stream': self.stream if self.stream != 0 else None,  # CAI v3: 0 disallowed
        }

    def __getitem__(self, key):
        return self.__cuda_array_interface__[key]

    def get(self, key, default=None):
        return self.__cuda_array_interface__.get(key, default)

cdef class logical_data:
    cdef stf_logical_data_handle _ld
    cdef stf_ctx_handle _ctx

    cdef object _dtype
    cdef tuple  _shape
    cdef int    _ndim
    cdef size_t _len
    cdef str    _symbol  # Store symbol for display purposes
    cdef readonly bint _is_token  # readonly makes it accessible from Python

    def __cinit__(self, context ctx=None, object buf=None, data_place dplace=None, shape=None, dtype=None, str name=None):
        cdef Py_buffer view
        cdef int flags

        if ctx is None or buf is None:
            # allow creation via __new__ (eg. in empty_like)
            self._ld = NULL
            self._ctx = NULL
            self._len = 0
            self._dtype = None
            self._shape = ()
            self._ndim = 0
            self._symbol = None
            self._is_token = False
            return

        self._ctx = ctx._ctx
        self._symbol = None  # Initialize symbol
        self._is_token = False  # Initialize token flag

        # Default to host data place if not specified (matches C++ API)
        if dplace is None:
            dplace = data_place.host()

        # Try CUDA Array Interface first
        if hasattr(buf, '__cuda_array_interface__'):
            cai = buf.__cuda_array_interface__

            # Extract CAI information
            data_ptr, readonly = cai['data']
            original_shape = cai['shape']
            typestr = cai['typestr']

            # Handle vector types (e.g., wp.vec2, wp.vec3)
            # Use structured dtype from descr if available
            if typestr.startswith('|V') and 'descr' in cai:
                # Vector/structured type - use descr field
                self._dtype = np.dtype(cai['descr'])
            else:
                # Regular scalar type or vector without descr - use typestr
                self._dtype = np.dtype(typestr)

            # Shape is always the same regardless of type
            self._shape = original_shape

            self._ndim = len(self._shape)

            # Calculate total size in bytes
            itemsize = self._dtype.itemsize
            total_items = 1
            for dim in self._shape:
                total_items *= dim
            self._len = total_items * itemsize

            # Create STF logical data using the new C API with data place specification
            stf_logical_data_with_place(ctx._ctx, &self._ld, <void*><uintptr_t>data_ptr, self._len, dplace._c_place)

        else:
            # Fallback to Python buffer protocol
            flags = PyBUF_FORMAT | PyBUF_ND          # request dtype + shape

            if PyObject_GetBuffer(buf, &view, flags) != 0:
                raise ValueError("object doesn't support the full buffer protocol or __cuda_array_interface__")

            try:
                self._ndim  = view.ndim
                self._len = view.len
                self._shape = tuple(<Py_ssize_t>view.shape[i] for i in range(view.ndim))
                self._dtype = np.dtype(view.format)
                # For buffer protocol objects, use the specified data place (defaults to host)
                stf_logical_data_with_place(ctx._ctx, &self._ld, view.buf, view.len, dplace._c_place)

            finally:
                PyBuffer_Release(&view)

        # Apply symbol name if provided
        if name is not None:
            self.set_symbol(name)


    def set_symbol(self, str name):
        stf_logical_data_set_symbol(self._ld, name.encode())
        self._symbol = name  # Store locally for retrieval

    @property
    def symbol(self):
        """Get the symbol name of this logical data, if set."""
        return self._symbol

    def __dealloc__(self):
        if self._ld != NULL:
            stf_logical_data_destroy(self._ld)
            self._ld = NULL

    def __repr__(self):
        """Return a detailed string representation of the logical_data object."""
        return (f"logical_data(shape={self._shape}, dtype={self._dtype}, "
                f"is_token={self._is_token}, symbol={self._symbol!r}, "
                f"len={self._len}, ndim={self._ndim})")

    @property
    def dtype(self):
        """Return the dtype of the logical data."""
        return self._dtype

    @property
    def shape(self):
        """Return the shape of the logical data."""
        return self._shape

    def read(self, dplace=None):
        return dep(self, AccessMode.READ.value, dplace)

    def write(self, dplace=None):
        return dep(self, AccessMode.WRITE.value, dplace)

    def rw(self, dplace=None):
        return dep(self, AccessMode.RW.value, dplace)

    def empty_like(self):
        """
        Create a new logical_data with the same shape (and dtype metadata)
        as this object.
        """
        if self._ld == NULL:
            raise RuntimeError("source logical_data handle is NULL")

        cdef logical_data out = logical_data.__new__(logical_data)
        stf_logical_data_empty(self._ctx, self._len, &out._ld)
        out._ctx   = self._ctx
        out._dtype = self._dtype
        out._shape = self._shape
        out._ndim  = self._ndim
        out._len   = self._len
        out._symbol = None  # New object has no symbol initially
        out._is_token = False

        return out

    @staticmethod
    def token(context ctx):
        cdef logical_data out = logical_data.__new__(logical_data)
        out._ctx   = ctx._ctx
        out._dtype = None
        out._shape = None
        out._ndim  = 0
        out._len   = 0
        out._symbol = None  # New object has no symbol initially
        out._is_token = True
        stf_token(ctx._ctx, &out._ld)

        return out

    @staticmethod
    def init_by_shape(context ctx, shape, dtype, str name=None):
        """
        Create a new logical_data from a shape and a dtype.
        """
        cdef logical_data out = logical_data.__new__(logical_data)
        out._ctx   = ctx._ctx
        out._dtype = np.dtype(dtype)
        out._shape = shape
        out._ndim  = len(shape)
        cdef size_t total_items = 1
        for dim in shape:
            total_items *= dim
        out._len   = total_items * out._dtype.itemsize
        out._symbol = None
        out._is_token = False
        stf_logical_data_empty(ctx._ctx, out._len, &out._ld)

        if name is not None:
            out.set_symbol(name)

        return out

    def borrow_ctx_handle(self):
        ctx = context(borrowed=True)
        ctx.borrow_from_handle(self._ctx)
        return ctx

class dep:
    __slots__ = ("ld", "mode", "dplace")
    def __init__(self, logical_data ld, int mode, dplace=None):
        self.ld   = ld
        self.mode = mode
        self.dplace = dplace  # can be None or a data place
    def __iter__(self):      # nice unpacking support
        yield self.ld
        yield self.mode
        yield self.dplace
    def __repr__(self):
        return f"dep({self.ld!r}, {self.mode}, {self.dplace!r})"
    def get_ld(self):
        return self.ld

def read(ld, dplace=None):   return dep(ld, AccessMode.READ.value, dplace)
def write(ld, dplace=None):  return dep(ld, AccessMode.WRITE.value, dplace)
def rw(ld, dplace=None):     return dep(ld, AccessMode.RW.value, dplace)

cdef exec_place _to_exec_place(object obj):
    """Convert an exec_place-like object to a Cython exec_place.

    Accepts:
    - A Cython exec_place (or subclass like exec_place_grid) -- returned as-is.
    - Any object with a _as_stf_exec_place() method -- called to obtain a Cython exec_place.
    Raises TypeError otherwise.
    """
    if isinstance(obj, exec_place):
        return <exec_place>obj
    if hasattr(obj, '_as_stf_exec_place'):
        result = obj._as_stf_exec_place()
        if isinstance(result, exec_place):
            return <exec_place>result
        raise TypeError(
            f"_as_stf_exec_place() must return an exec_place, got {type(result).__name__}"
        )
    raise TypeError(
        f"expected an exec_place or an object with _as_stf_exec_place(), "
        f"got {type(obj).__name__}"
    )

cdef bint _is_exec_place_like(object obj):
    """Check if obj is exec_place-like (for dispatch in context.task)."""
    return isinstance(obj, exec_place) or hasattr(obj, '_as_stf_exec_place')

cdef class exec_place:
    cdef stf_exec_place _c_place
    cdef bint _owns_opaque

    def __cinit__(self):
        self._owns_opaque = False

    def __dealloc__(self):
        if self._owns_opaque and self._c_place.kind == STF_EXEC_PLACE_OPAQUE:
            stf_exec_place_opaque_destroy(self._c_place.u.opaque)

    @staticmethod
    def device(int dev_id):
        cdef exec_place p = exec_place.__new__(exec_place)
        p._c_place = make_device_place(dev_id)
        return p

    @staticmethod
    def host():
        cdef exec_place p = exec_place.__new__(exec_place)
        p._c_place = make_host_place()
        return p

    @staticmethod
    def from_grid(exec_place_grid grid):
        """Return the grid as an exec_place (exec_place_grid is a subclass of exec_place)."""
        return grid

    @staticmethod
    def from_opaque(uintptr_t handle):
        """Wrap an opaque C++ exec_place pointer as a Python exec_place.

        The returned object takes ownership of the handle and will destroy it
        when garbage collected. The handle should come from
        stf_exec_place_opaque_wrap() or stf_exec_place_dummy_create().
        """
        if handle == 0:
            raise ValueError("opaque handle must not be NULL")
        cdef exec_place p = exec_place.__new__(exec_place)
        p._c_place = make_opaque_exec_place(<void*>handle)
        p._owns_opaque = True
        return p

    def _as_stf_exec_place(self):
        """Return self (satisfies the ExecPlaceLike duck-typing protocol)."""
        return self

    @property
    def kind(self) -> str:
        if self._c_place.kind == STF_EXEC_PLACE_DEVICE:
            return "device"
        elif self._c_place.kind == STF_EXEC_PLACE_HOST:
            return "host"
        elif self._c_place.kind == STF_EXEC_PLACE_GRID:
            return "grid"
        elif self._c_place.kind == STF_EXEC_PLACE_OPAQUE:
            return "opaque"
        else:
            return "unknown"

    @property
    def device_id(self) -> int:
        if self._c_place.kind != STF_EXEC_PLACE_DEVICE:
            raise AttributeError("not a device execution place")
        return self._c_place.u.device.dev_id

cdef data_place _to_data_place(object obj):
    """Convert a data_place-like object to a Cython data_place.

    Accepts:
    - A Cython data_place -- returned as-is.
    - Any object with a _as_stf_data_place() method -- called to obtain a Cython data_place.
    Raises TypeError otherwise.
    """
    if isinstance(obj, data_place):
        return <data_place>obj
    if hasattr(obj, '_as_stf_data_place'):
        result = obj._as_stf_data_place()
        if isinstance(result, data_place):
            return <data_place>result
        raise TypeError(
            f"_as_stf_data_place() must return a data_place, got {type(result).__name__}"
        )
    raise TypeError(
        f"expected a data_place or an object with _as_stf_data_place(), "
        f"got {type(obj).__name__}"
    )

cdef class data_place:
    cdef stf_data_place _c_place
    cdef object _mapper_callback  # prevent GC of ctypes callback so the C function pointer stays valid
    cdef bint _owns_opaque

    def __cinit__(self):
        self._owns_opaque = False

    def __dealloc__(self):
        if self._owns_opaque and self._c_place.kind == STF_DATA_PLACE_OPAQUE:
            stf_data_place_opaque_destroy(self._c_place.u.opaque)

    @staticmethod
    def device(int dev_id):
        cdef data_place p = data_place.__new__(data_place)
        p._c_place = make_device_data_place(dev_id)
        return p

    @staticmethod
    def host():
        cdef data_place p = data_place.__new__(data_place)
        p._c_place = make_host_data_place()
        return p

    @staticmethod
    def managed():
        cdef data_place p = data_place.__new__(data_place)
        p._c_place = make_managed_data_place()
        return p

    @staticmethod
    def affine():
        cdef data_place p = data_place.__new__(data_place)
        p._c_place = make_affine_data_place()
        return p

    @staticmethod
    def from_opaque(uintptr_t handle):
        """Wrap an opaque C++ data_place pointer as a Python data_place.

        The returned object takes ownership of the handle and will destroy it
        when garbage collected. The handle should come from
        stf_data_place_opaque_wrap().
        """
        if handle == 0:
            raise ValueError("opaque handle must not be NULL")
        cdef data_place p = data_place.__new__(data_place)
        p._c_place = make_opaque_data_place(<void*>handle)
        p._owns_opaque = True
        return p

    def _as_stf_data_place(self):
        """Return self (satisfies the DataPlaceLike duck-typing protocol)."""
        return self

    @property
    def kind(self) -> str:
        cdef stf_data_place_kind k = self._c_place.kind
        if k == STF_DATA_PLACE_DEVICE:
            return "device"
        elif k == STF_DATA_PLACE_HOST:
            return "host"
        elif k == STF_DATA_PLACE_MANAGED:
            return "managed"
        elif k == STF_DATA_PLACE_AFFINE:
            return "affine"
        elif k == STF_DATA_PLACE_COMPOSITE:
            return "composite"
        elif k == STF_DATA_PLACE_OPAQUE:
            return "opaque"
        else:
            raise ValueError(f"Unknown data place kind: {k}")

    @property
    def device_id(self) -> int:
        if self._c_place.kind != STF_DATA_PLACE_DEVICE:
            raise AttributeError("not a device data place")
        return self._c_place.u.device.dev_id

    @staticmethod
    def composite(exec_place_grid grid, object mapper):
        """
        Create a composite data place: grid of execution places + partition function.

        The partitioner (mapper) is a callable with signature:
            (data_coords, data_dims, grid_dims) -> grid_position

        Each of the four arguments/return is a 4-tuple (x, y, z, t) of integers.
        Only callability is checked here; signature errors surface when the runtime
        first invokes the mapper (duck typing).
        - data_coords: logical position in the data
        - data_dims: full shape of the data
        - grid_dims: shape of the execution place grid
        - return: position in the place grid (which place owns this data)

        Example: blocked partition along first dimension:
            def blocked_1d(data_coords, data_dims, grid_dims):
                n = data_dims[0]
                nplaces = grid_dims[0]
                part_size = (n + nplaces - 1) // nplaces or 1
                place_x = min(data_coords[0] // part_size, nplaces - 1)
                return (place_x, 0, 0, 0)

            grid = exec_place_grid.from_devices([0, 1])
            dplace = data_place.composite(grid, blocked_1d)
        """
        if not callable(mapper):
            raise TypeError("mapper must be callable (data_coords, data_dims, grid_dims) -> (x, y, z, t)")
        callback_obj, c_ptr = _make_mapper_callback(mapper)
        cdef data_place p = data_place.__new__(data_place)
        p._mapper_callback = callback_obj
        cdef uintptr_t ptr_val = c_ptr
        stf_make_composite_data_place(&p._c_place, grid._handle, <stf_get_executor_fn>ptr_val)
        return p


cdef class exec_place_grid(exec_place):
    """
    Grid of execution places (a kind of exec_place). Use wherever an exec_place is
    expected (e.g. ctx.task(grid, ...), set_exec_place(grid)). Create with
    from_devices() or create(). The grid is destroyed automatically when the
    object goes out of scope.
    """
    cdef stf_exec_place_grid_handle _handle

    def __cinit__(self):
        self._handle = NULL

    @staticmethod
    def from_devices(device_ids):
        """
        Create a grid with one place per device.
        device_ids: sequence of int (e.g. [0, 1] or [0, 0, 0] for same device repeated).
        """
        cdef int dev_ids[64]
        cdef size_t n = len(device_ids)
        if n == 0:
            raise ValueError("device_ids must contain at least one device")
        if n > 64:
            raise ValueError("at most 64 devices supported")
        for i in range(n):
            dev_ids[i] = int(device_ids[i])
        cdef exec_place_grid g = exec_place_grid.__new__(exec_place_grid)
        g._handle = stf_exec_place_grid_from_devices(dev_ids, n)
        g._c_place = make_exec_place_from_grid(g._handle)
        return g

    @staticmethod
    def create(places, grid_dims=None):
        """
        Create a grid from a list of exec_place(-like) objects and optional shape.
        places: list of exec_place or objects with _as_stf_exec_place()
        grid_dims: optional (x, y, z, t) tuple; if None, linear grid (len(places), 1, 1, 1).
        """
        cdef stf_exec_place c_places[64]
        cdef stf_dim4 dims
        cdef size_t n = len(places)
        cdef exec_place ep
        cdef exec_place_grid g
        if n == 0:
            raise ValueError("places must contain at least one place")
        if n > 64:
            raise ValueError("at most 64 places supported")
        cdef list converted = []
        for i in range(n):
            ep = _to_exec_place(places[i])
            converted.append(ep)
            c_places[i] = ep._c_place
        g = exec_place_grid.__new__(exec_place_grid)
        if grid_dims is not None:
            dims.x = int(grid_dims[0])
            dims.y = int(grid_dims[1]) if len(grid_dims) > 1 else 1
            dims.z = int(grid_dims[2]) if len(grid_dims) > 2 else 1
            dims.t = int(grid_dims[3]) if len(grid_dims) > 3 else 1
            g._handle = stf_exec_place_grid_create(c_places, n, &dims)
        else:
            g._handle = stf_exec_place_grid_create(c_places, n, NULL)
        g._c_place = make_exec_place_from_grid(g._handle)
        return g

    def set_affine_data_place(self, data_place dplace):
        """
        Set the affine data place for this grid (used when the task uses data_place::affine()).
        Call before passing the grid to set_exec_place() / ctx.task() so that dependencies
        with affine data place resolve to dplace (e.g. a composite data place).
        """
        stf_exec_place_grid_set_affine_data_place(self._handle, &dplace._c_place)

    def destroy(self):
        if self._handle != NULL:
            stf_exec_place_grid_destroy(self._handle)
            self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            stf_exec_place_grid_destroy(self._handle)


cdef class task:
    cdef stf_task_handle _t

    # list of logical data in deps: we need this because we can't exchange
    # dtype/shape easily through the C API of STF
    cdef list _lds_args

    def __cinit__(self, context ctx):
        stf_task_create(ctx._ctx, &self._t)
        self._lds_args = []

    def __dealloc__(self):
        if self._t != NULL:
             stf_task_destroy(self._t)

    def start(self):
        # This is ignored if this is not a graph task
        stf_task_enable_capture(self._t)

        stf_task_start(self._t)

    def end(self):
        stf_task_end(self._t)

    def add_dep(self, object d):
        """
        Accept a `dep` instance created with read(ld), write(ld), or rw(ld).
        """
        if not isinstance(d, dep):
            raise TypeError("add_dep expects read(ld), write(ld) or rw(ld)")

        cdef logical_data ldata = <logical_data> d.ld
        cdef int           mode_int  = int(d.mode)
        cdef stf_access_mode mode_ce = <stf_access_mode> mode_int
        cdef data_place dp

        if d.dplace is None:
            stf_task_add_dep(self._t, ldata._ld, mode_ce)
        else:
            dp = _to_data_place(d.dplace)
            stf_task_add_dep_with_dplace(self._t, ldata._ld, mode_ce, &dp._c_place)

        self._lds_args.append(ldata)

    def set_exec_place(self, object exec_p):
        """Set execution place (exec_place, exec_place_grid, or any ExecPlaceLike object)."""
        cdef exec_place ep = _to_exec_place(exec_p)
        stf_task_set_exec_place(self._t, &ep._c_place)

    def stream_ptr(self) -> int:
        """
        Return the raw CUstream pointer as a Python int
        (memory address).  Suitable for ctypes or PyCUDA.
        """
        cdef CUstream s = stf_task_get_custream(self._t)
        return <uintptr_t> s         # cast pointer -> Py int

    def get_grid_dims(self):
        """
        When the task's exec place is a grid, return (x, y, z, t) shape.
        Call after start(). Returns None if the task is not on a grid.
        """
        cdef stf_dim4 dims
        if stf_task_get_grid_dims(self._t, &dims) != 0:
            return None
        return (dims.x, dims.y, dims.z, dims.t)

    def get_stream_at_index(self, size_t place_index):
        """
        When the task's exec place is a grid, return the CUstream for the given
        linear index (0 to product of grid dims - 1) as a Python int (pointer).
        Call after start(). Raises if not a grid or index invalid.
        """
        cdef CUstream s
        if stf_task_get_custream_at_index(self._t, place_index, &s) != 0:
            raise RuntimeError("task is not on a grid or place_index out of range")
        return <uintptr_t> s

    def get_arg(self, index) -> int:
        if self._lds_args[index]._is_token:
           raise RuntimeError("cannot materialize a token argument")

        cdef void *ptr = stf_task_get(self._t, index)
        return <uintptr_t>ptr

    def get_arg_cai(self, index):
        """Return the argument as an stf_cai object (has __cuda_array_interface__; supports obj['data'] etc.).
        The underlying memory is owned by the task/context; keep the task (or context) alive while using the returned view."""
        ptr = self.get_arg(index)
        return stf_cai(ptr, self._lds_args[index].shape, self._lds_args[index].dtype, stream=self.stream_ptr())

    def args_cai(self):
        """
        Return all non-token buffer arguments as stf_cai objects (have __cuda_array_interface__).
        Returns None, a single object, or a tuple. Use from non-shipped code (e.g. tests) to
        convert to numba/torch/cupy via from_cuda_array_interface or torch.as_tensor(obj).
        Keep the task (or context) alive while any consumer uses the returned view(s).
        """
        non_token_cais = [self.get_arg_cai(i) for i in range(len(self._lds_args))
                          if not self._lds_args[i]._is_token]

        if len(non_token_cais) == 0:
            return None
        elif len(non_token_cais) == 1:
            return non_token_cais[0]
        return tuple(non_token_cais)

    # ---- context‑manager helpers -------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, object exc_type, object exc, object tb):
        """
        Always called, even if an exception occurred inside the block.
        """
        self.end()
        return False

cdef class context:
    cdef stf_ctx_handle _ctx
    # Is this a context that we have borrowed ?
    cdef bint _borrowed

    def __cinit__(self, bint use_graph=False, bint borrowed=False):
        self._ctx = <stf_ctx_handle>NULL
        self._borrowed = borrowed
        if not borrowed:
            if use_graph:
                stf_ctx_create_graph(&self._ctx)
            else:
                stf_ctx_create(&self._ctx)

    cdef borrow_from_handle(self, stf_ctx_handle ctx_handle):
        if self._ctx != NULL:
            raise RuntimeError("context already initialized")

        if not self._borrowed:
            raise RuntimeError("cannot call borrow_from_handle on this context")

        self._ctx = ctx_handle

    def __repr__(self):
        return f"context(handle={<uintptr_t>self._ctx}, borrowed={self._borrowed})"

    def __dealloc__(self):
        if not self._borrowed:
            self.finalize()

    def finalize(self):
        if self._borrowed:
            raise RuntimeError("cannot finalize borrowed context")

        if self._ctx != NULL:
                stf_ctx_finalize(self._ctx)
        self._ctx = NULL

    def logical_data(self, object buf, data_place dplace=None, str name=None):
        """
        Create and return a `logical_data` object bound to this context [PRIMARY API].

        This is the primary function for creating logical data from existing buffers.
        It supports both Python buffer protocol objects and CUDA Array Interface objects,
        with explicit data_place specification for optimal STF data movement strategies.

        Parameters
        ----------
        buf : any buffer‑supporting Python object or __cuda_array_interface__ object
              (NumPy array, Warp array, CuPy array, bytes, bytearray, memoryview, …)
        dplace : data_place, optional
              Specifies where the buffer is located (host, device, managed, affine).
              Defaults to data_place.host() for backward compatibility.
              Essential for GPU arrays - use data_place.device() for optimal performance.
        name : str, optional
              Symbol name for debugging and DOT graph output.

        Examples
        --------
        >>> # Host memory (explicit - recommended)
        >>> host_place = data_place.host()
        >>> ld = ctx.logical_data(numpy_array, host_place)
        >>>
        >>> # GPU device memory (recommended for CUDA arrays)
        >>> device_place = data_place.device(0)
        >>> ld = ctx.logical_data(warp_array, device_place)
        >>>
        >>> # With a symbol name for debugging
        >>> ld = ctx.logical_data(numpy_array, name="X")
        >>>
        >>> # Backward compatibility (defaults to host)
        >>> ld = ctx.logical_data(numpy_array)  # Same as specifying host

        Note
        ----
        For GPU arrays (Warp, CuPy, etc.), always specify data_place.device()
        for zero-copy performance and correct memory management.
        """
        return logical_data(self, buf, dplace, name=name)


    def logical_data_empty(self, shape, dtype=None, str name=None):
        """
        Create logical data with uninitialized values.

        Equivalent to numpy.empty() but for STF logical data.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : numpy.dtype, optional
            Data type. Defaults to np.float64.
        name : str, optional
            Symbol name for debugging and DOT graph output.

        Returns
        -------
        logical_data
            New logical data with uninitialized values

        Examples
        --------
        >>> # Create uninitialized array (fast but contains garbage)
        >>> ld = ctx.logical_data_empty((100, 100), dtype=np.float32)

        >>> # Fast allocation without initialization
        >>> ld = ctx.logical_data_empty((50, 50, 50), name="tmp")
        """
        if dtype is None:
            dtype = np.float64
        return logical_data.init_by_shape(self, shape, dtype, name)

    def logical_data_full(self, shape, fill_value, dtype=None, where=None, exec_place=None, str name=None):
        """
        Create logical data initialized with a constant value.

        Similar to numpy.full(), this creates a new logical data with the given
        shape and fills it with fill_value.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        fill_value : scalar
            Value to fill the array with
        dtype : numpy.dtype, optional
            Data type. If None, infer from fill_value.
        where : data_place, optional
            Data placement for initialization. Defaults to current device.
        exec_place : exec_place, optional
            Execution place for the fill operation. Defaults to current device.
            Note: exec_place.host() is not yet supported.
        name : str, optional
            Symbol name for debugging and DOT graph output.

        Returns
        -------
        logical_data
            New logical data initialized with fill_value

        Examples
        --------
        >>> # Create array filled with epsilon0 on current device
        >>> ld = ctx.logical_data_full((100, 100), 8.85e-12, dtype=np.float64)

        >>> # Create array on host memory
        >>> ld = ctx.logical_data_full((50, 50), 1.0, where=data_place.host())

        >>> # With a symbol name
        >>> ld = ctx.logical_data_full((200, 200), 0.0, name="epsilon")
        """
        # Infer dtype from fill_value if not provided
        if dtype is None:
            dtype = np.array(fill_value).dtype
        else:
            dtype = np.dtype(dtype)

        # Validate exec_place - host execution not yet supported
        if exec_place is not None:
            if hasattr(exec_place, 'kind') and exec_place.kind == "host":
                raise NotImplementedError(
                    "exec_place.host() is not yet supported for logical_data_full. "
                    "Use exec_place.device() or omit exec_place parameter."
                )

        # Create empty logical data
        ld = self.logical_data_empty(shape, dtype, name)

        # Initialize with the specified value (cuda.core.Buffer.fill; CuPy/Numba fallback for 8-byte)
        try:
            from cuda.stf.fill_utils import init_logical_data
            init_logical_data(self, ld, fill_value, where, exec_place)
        except ImportError as e:
            raise RuntimeError("Fill support (cuda.core) is not available for logical_data_full") from e

        return ld

    def logical_data_zeros(self, shape, dtype=None, where=None, exec_place=None, str name=None):
        """
        Create logical data filled with zeros.

        Equivalent to numpy.zeros() but for STF logical data.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : numpy.dtype, optional
            Data type. Defaults to np.float64.
        where : data_place, optional
            Data placement. Defaults to current device.
        exec_place : exec_place, optional
            Execution place for the fill operation. Defaults to current device.
        name : str, optional
            Symbol name for debugging and DOT graph output.

        Returns
        -------
        logical_data
            New logical data filled with zeros

        Examples
        --------
        >>> # Create zero-filled array
        >>> ld = ctx.logical_data_zeros((100, 100), dtype=np.float32)

        >>> # Create on host memory with a name
        >>> ld = ctx.logical_data_zeros((50, 50), where=data_place.host(), name="Z")
        """
        if dtype is None:
            dtype = np.float64
        return self.logical_data_full(shape, 0.0, dtype, where, exec_place, name)

    def logical_data_ones(self, shape, dtype=None, where=None, exec_place=None, str name=None):
        """
        Create logical data filled with ones.

        Equivalent to numpy.ones() but for STF logical data.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : numpy.dtype, optional
            Data type. Defaults to np.float64.
        where : data_place, optional
            Data placement. Defaults to current device.
        exec_place : exec_place, optional
            Execution place for the fill operation. Defaults to current device.
        name : str, optional
            Symbol name for debugging and DOT graph output.

        Returns
        -------
        logical_data
            New logical data filled with ones

        Examples
        --------
        >>> # Create ones-filled array
        >>> ld = ctx.logical_data_ones((100, 100), dtype=np.float32)

        >>> # Create on specific device with a name
        >>> ld = ctx.logical_data_ones((50, 50), name="ones")
        """
        if dtype is None:
            dtype = np.float64
        return self.logical_data_full(shape, 1.0, dtype, where, exec_place, name)

    def token(self):
        return logical_data.token(self)

    def task(self, *args):
        """
        Create a `task`

        Example
        -------
        >>> t = ctx.task(read(lX), rw(lY))
        >>> t.start()
        >>> t.end()
        """
        exec_place_set = False
        t = task(self)          # construct with this context
        for d in args:
            if isinstance(d, dep):
                t.add_dep(d)
            elif _is_exec_place_like(d):
                if exec_place_set:
                      raise ValueError("Only one exec_place can be given")
                t.set_exec_place(d)
                exec_place_set = True
            else:
                raise TypeError(
                    "Arguments must be dependency objects or an exec_place-like object"
                )
        return t
