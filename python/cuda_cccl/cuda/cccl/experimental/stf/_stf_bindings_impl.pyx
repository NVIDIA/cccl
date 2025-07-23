# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

# Python signatures are declared in the companion Python stub file _bindings.pyi
# Make sure to update PYI with change to Python API to ensure that Python
# static type checker tools like mypy green-lights cuda.cccl.parallel

from libc.string cimport memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t, uintptr_t
from cpython.bytes cimport PyBytes_FromStringAndSize

from cpython.buffer cimport (
    Py_buffer, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS,
    PyBuffer_Release, PyObject_CheckBuffer, PyObject_GetBuffer
)
from cpython.pycapsule cimport (
    PyCapsule_CheckExact, PyCapsule_IsValid, PyCapsule_GetPointer
)

import ctypes

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

cdef class Ctx:
    cdef stf_ctx_handle _ctx

    def __cinit__(self):
        stf_ctx_create(&self._ctx)

    def __dealloc__(self):
        if self._ctx != NULL:
            stf_ctx_finalize(self._ctx)
            self._ctx = NULL
