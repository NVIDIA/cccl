# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

# Python signatures are declared in the companion Python stub file _bindings.pyi
# Make sure to update PYI with change to Python API to ensure that Python
# static type checker tools like mypy green-lights cuda.cccl.parallel

from cpython.buffer cimport Py_buffer, PyObject_GetBuffer, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t, uintptr_t
from libc.stdint cimport uintptr_t
from libc.string cimport memset, memcpy

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
    # cudaStream_t stf_task_get_stream(stf_task_handle t)
    void* stf_task_get(stf_task_handle t, size_t submitted_index)
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

cdef class logical_data:
    cdef stf_logical_data_handle _ld

    def __cinit__(self, context ctx, object buf):
        cdef Py_buffer view
        if PyObject_GetBuffer(buf, &view, PyBUF_SIMPLE) != 0:
            raise ValueError("object doesn’t support the buffer protocol")

        try:
            stf_logical_data(ctx._ctx, &self._ld, view.buf, view.len)

        finally:
            PyBuffer_Release(&view)

    def set_symbol(self, str name):
        stf_logical_data_set_symbol(self._ld, name.encode())

    def __dealloc__(self):
        if self._ld != NULL:
            stf_logical_data_destroy(self._ld)
            self._ld = NULL

cdef class task:
    cdef stf_task_handle _t

    def __cinit__(self, context ctx):
        stf_task_create(ctx._ctx, &self._t)

    def __dealloc__(self):
        if self._t != NULL:
             stf_task_destroy(self._t)

    def start(self):
        stf_task_start(self._t)

    def end(self):
        stf_task_end(self._t)

    def add_dep(self, logical_data ld, int mode):
        stf_task_add_dep(self._t, ld._ld, <stf_access_mode> mode)

cdef class context:
    cdef stf_ctx_handle _ctx

    def __cinit__(self):
        stf_ctx_create(&self._ctx)

    def __dealloc__(self):
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

    def task(self):
        return task(self)
