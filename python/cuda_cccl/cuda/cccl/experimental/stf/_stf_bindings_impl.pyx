# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

# Python signatures are declared in the companion Python stub file _bindings.pyi
# Make sure to update PYI with change to Python API to ensure that Python
# static type checker tools like mypy green-lights cuda.cccl.parallel

from cpython.buffer cimport Py_buffer, PyObject_GetBuffer, PyBuffer_Release
from cpython.buffer cimport Py_buffer, PyBUF_FORMAT, PyBUF_ND, PyObject_GetBuffer, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t, uintptr_t
from libc.stdint cimport uintptr_t
from libc.string cimport memset, memcpy

import numpy as np
from numba import cuda


from cpython.buffer cimport (
    Py_buffer, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS,
    PyBuffer_Release, PyObject_CheckBuffer, PyObject_GetBuffer
)
from cpython.pycapsule cimport (
    PyCapsule_CheckExact, PyCapsule_IsValid, PyCapsule_GetPointer
)

import ctypes
from enum import IntFlag

cdef extern from "<cuda.h>":
    cdef struct OpaqueCUstream_st
    cdef struct OpaqueCUkernel_st
    cdef struct OpaqueCUlibrary_st

    ctypedef int CUresult
    ctypedef OpaqueCUstream_st *CUstream
    ctypedef OpaqueCUkernel_st *CUkernel
    ctypedef OpaqueCUlibrary_st *CUlibrary

#typedef struct CUstream_st* cudaStream_t;


cdef extern from "cccl/c/experimental/stf/stf.h":
    ctypedef struct stf_ctx_handle_t
    ctypedef stf_ctx_handle_t* stf_ctx_handle
    void stf_ctx_create(stf_ctx_handle* ctx)
    void stf_ctx_finalize(stf_ctx_handle ctx)

    ctypedef struct stf_logical_data_handle_t
    ctypedef stf_logical_data_handle_t* stf_logical_data_handle
    void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz)
    void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
    void stf_logical_data_destroy(stf_logical_data_handle ld)

    ctypedef struct stf_task_handle_t
    ctypedef stf_task_handle_t* stf_task_handle
    void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t)
    void stf_task_set_symbol(stf_task_handle t, const char* symbol)
    void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
    void stf_task_start(stf_task_handle t)
    void stf_task_end(stf_task_handle t)
    CUstream stf_task_get_custream(stf_task_handle t)
    # cudaStream_t stf_task_get_stream(stf_task_handle t)
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

class stf_arg_cai:
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
            'stream': self.stream,         # CUDA stream for access
        }

cdef class logical_data:
    cdef stf_logical_data_handle _ld

    cdef object _dtype
    cdef tuple  _shape
    cdef int    _ndim

    def __cinit__(self, context ctx, object buf):
        cdef Py_buffer view
        cdef int flags = PyBUF_FORMAT | PyBUF_ND          # request dtype + shape

        if PyObject_GetBuffer(buf, &view, flags) != 0:
            raise ValueError("object doesn’t support the full buffer protocol")

        try:
            self._ndim  = view.ndim
            self._shape = tuple(<Py_ssize_t>view.shape[i] for i in range(view.ndim))
            self._dtype = np.dtype(view.format)
            stf_logical_data(ctx._ctx, &self._ld, view.buf, view.len)

        finally:
            PyBuffer_Release(&view)

    def set_symbol(self, str name):
        stf_logical_data_set_symbol(self._ld, name.encode())

    def __dealloc__(self):
        if self._ld != NULL:
            stf_logical_data_destroy(self._ld)
            self._ld = NULL

    @property
    def dtype(self):
        """Return the dtype of the logical data."""
        return self._dtype

    @property
    def shape(self):
        """Return the shape of the logical data."""
        return self._shape



class dep:
    __slots__ = ("ld", "mode")
    def __init__(self, logical_data ld, int mode):
        self.ld   = ld
        self.mode = mode
    def __iter__(self):      # nice unpacking support
        yield self.ld
        yield self.mode
    def __repr__(self):
        return f"dep({self.ld!r}, {self.mode})"

# optional sugar
def read(ld):   return dep(ld, AccessMode.READ.value)
def write(ld):  return dep(ld, AccessMode.WRITE.value)
def rw(ld):     return dep(ld, AccessMode.RW.value)

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
#        self._lds_args.clear()

    def start(self):
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

        stf_task_add_dep(self._t, ldata._ld, mode_ce)

        self._lds_args.append(ldata)

    def stream_ptr(self) -> int:
        """
        Return the raw CUstream pointer as a Python int
        (memory address).  Suitable for ctypes or PyCUDA.
        """
        cdef CUstream s = stf_task_get_custream(self._t)
        return <uintptr_t> s         # cast pointer -> Py int

    def get_arg(self, index) -> int:
        cdef void *ptr = stf_task_get(self._t, index)
        return <uintptr_t>ptr

    def get_arg_cai(self, index):
        ptr = self.get_arg(index)
        return stf_arg_cai(ptr, self._lds_args[index].shape, self._lds_args[index].dtype, stream=0).__cuda_array_interface__

    def get_arg_numba(self, index):
        cai = self.get_arg_cai(index)
        return cuda.from_cuda_array_interface(cai, owner=None, sync=False)

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

    def __cinit__(self):
        stf_ctx_create(&self._ctx)

    def __dealloc__(self):
        self.finalize()

    def finalize(self):
        if self._ctx != NULL:
            stf_ctx_finalize(self._ctx)
            self._ctx = NULL

    def logical_data(self, object buf):
        """
        Create and return a `logical_data` object bound to this context.

        Parameters
        ----------
        buf : any buffer‑supporting Python object
              (NumPy array, bytes, bytearray, memoryview, …)
        """
        return logical_data(self, buf)

    def task(self, *deps):
        """
        Create a `task`

        Example
        -------
        >>> t = ctx.task(read(lX), rw(lY))
        >>> t.start()
        >>> t.end()
        """
        t = task(self)          # construct with this context
        for d in deps:
            t.add_dep(d)        # your existing add_dep logic
        return t
