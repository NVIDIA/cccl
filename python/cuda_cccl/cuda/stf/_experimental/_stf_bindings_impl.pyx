# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=False


from cpython.buffer cimport (
    Py_buffer, PyBUF_FORMAT, PyBUF_ND, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS,
    PyBUF_WRITABLE, PyBUF_C_CONTIGUOUS,
    PyObject_GetBuffer, PyBuffer_Release, PyObject_CheckBuffer
)
from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pycapsule cimport (
    PyCapsule_CheckExact, PyCapsule_IsValid, PyCapsule_GetPointer
)
from libc.stddef cimport ptrdiff_t
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t, uintptr_t
from libc.string cimport memset, memcpy

import numpy as np

import ctypes
import warnings
from enum import IntFlag

cdef extern from "<cuda.h>":
    cdef struct OpaqueCUstream_st
    cdef struct OpaqueCUctx_st
    cdef struct OpaqueCUkernel_st
    cdef struct OpaqueCUlibrary_st
    cdef struct OpaqueCUfunc_st

    ctypedef int CUresult
    ctypedef int CUdevice
    ctypedef OpaqueCUctx_st *CUcontext
    ctypedef OpaqueCUstream_st *CUstream
    ctypedef OpaqueCUkernel_st *CUkernel
    ctypedef OpaqueCUlibrary_st *CUlibrary
    ctypedef OpaqueCUfunc_st *CUfunction

    CUresult cuInit(unsigned int flags)
    CUresult cuDeviceGetCount(int* count)
    CUresult cuDeviceGet(CUdevice* device, int ordinal)
    CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)
    CUresult cuDevicePrimaryCtxRelease(CUdevice dev)


cdef extern from "<cuda_runtime.h>":
    cdef struct dim3:
        unsigned int x, y, z
    ctypedef OpaqueCUstream_st *cudaStream_t
    cdef struct CUgraphExec_st
    ctypedef CUgraphExec_st *cudaGraphExec_t
    cdef struct CUgraph_st
    ctypedef CUgraph_st *cudaGraph_t

cdef extern from "cccl/c/experimental/stf/stf.h":
    #
    # Contexts
    #
    ctypedef struct stf_ctx_handle_t
    ctypedef stf_ctx_handle_t* stf_ctx_handle
    stf_ctx_handle stf_ctx_create()
    stf_ctx_handle stf_ctx_create_graph()
    void stf_ctx_finalize(stf_ctx_handle ctx) nogil
    CUstream stf_fence(stf_ctx_handle ctx) nogil

    #
    # Shareable async_resources_handle (opaque) and unified ctx factory
    #
    ctypedef struct stf_async_resources_opaque_t
    ctypedef stf_async_resources_opaque_t* stf_async_resources_handle
    stf_async_resources_handle stf_async_resources_create()
    void stf_async_resources_destroy(stf_async_resources_handle h)

    ctypedef enum stf_backend_kind:
        STF_BACKEND_STREAM
        STF_BACKEND_GRAPH

    ctypedef struct stf_ctx_options:
        stf_backend_kind backend
        int has_stream
        cudaStream_t stream
        stf_async_resources_handle handle

    stf_ctx_handle stf_ctx_create_ex(const stf_ctx_options* opts)

    #
    # 4D position/dimensions for partition mapping
    #
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

    # Forward-declare data place handle (needed by stf_exec_place_set_affine_data_place)
    ctypedef struct stf_data_place_opaque_t
    ctypedef stf_data_place_opaque_t* stf_data_place_handle
    ctypedef struct stf_green_context_helper_opaque_t
    ctypedef stf_green_context_helper_opaque_t* stf_green_context_helper_handle

    #
    # Exec places (opaque handles)
    #
    ctypedef struct stf_exec_place_opaque_t
    ctypedef stf_exec_place_opaque_t* stf_exec_place_handle
    stf_exec_place_handle stf_exec_place_host()
    stf_exec_place_handle stf_exec_place_device(int dev_id)
    stf_exec_place_handle stf_exec_place_current_device()
    stf_green_context_helper_handle stf_green_context_helper_create(int sm_count, int dev_id)
    void stf_green_context_helper_destroy(stf_green_context_helper_handle h)
    size_t stf_green_context_helper_get_count(stf_green_context_helper_handle h)
    int stf_green_context_helper_get_device_id(stf_green_context_helper_handle h)
    stf_exec_place_handle stf_exec_place_clone(stf_exec_place_handle h)
    void stf_exec_place_destroy(stf_exec_place_handle h)
    int stf_exec_place_is_host(stf_exec_place_handle h)
    int stf_exec_place_is_device(stf_exec_place_handle h)

    # Grid introspection
    void stf_exec_place_get_dims(stf_exec_place_handle h, stf_dim4* out_dims)
    size_t stf_exec_place_size(stf_exec_place_handle h)
    void stf_exec_place_set_affine_data_place(stf_exec_place_handle h, stf_data_place_handle affine_dplace)

    # Grid factories
    stf_exec_place_handle stf_exec_place_grid_from_devices(const int* device_ids, size_t count)
    stf_exec_place_handle stf_exec_place_grid_create(const stf_exec_place_handle* places, size_t count, const stf_dim4* grid_dims)
    void stf_exec_place_grid_destroy(stf_exec_place_handle grid)

    # exec_place_scope
    ctypedef struct stf_exec_place_scope_opaque_t
    ctypedef stf_exec_place_scope_opaque_t* stf_exec_place_scope_handle
    stf_exec_place_scope_handle stf_exec_place_scope_enter(stf_exec_place_handle place, size_t idx)
    void stf_exec_place_scope_exit(stf_exec_place_scope_handle scope)

    # Place accessors
    stf_data_place_handle stf_exec_place_get_affine_data_place(stf_exec_place_handle h)
    ctypedef struct stf_exec_place_resources_opaque_t
    ctypedef stf_exec_place_resources_opaque_t* stf_exec_place_resources_handle
    stf_exec_place_resources_handle stf_exec_place_resources_create()
    void stf_exec_place_resources_destroy(stf_exec_place_resources_handle h)
    stf_exec_place_resources_handle stf_ctx_get_place_resources(stf_ctx_handle ctx)
    CUstream stf_exec_place_pick_stream(stf_exec_place_resources_handle res, stf_exec_place_handle h, int for_computation)
    stf_exec_place_handle stf_exec_place_get_place(stf_exec_place_handle h, size_t idx)
    stf_exec_place_handle stf_exec_place_green_ctx(stf_green_context_helper_handle helper, size_t idx, int use_green_ctx_data_place)
    void stf_machine_init()

    #
    # Data places (functions using the forward-declared handle)
    #
    stf_data_place_handle stf_data_place_host()
    stf_data_place_handle stf_data_place_device(int dev_id)
    stf_data_place_handle stf_data_place_managed()
    stf_data_place_handle stf_data_place_affine()
    stf_data_place_handle stf_data_place_current_device()
    stf_data_place_handle stf_data_place_composite(stf_exec_place_handle grid, stf_get_executor_fn mapper)
    stf_data_place_handle stf_data_place_green_ctx(stf_green_context_helper_handle helper, size_t idx)
    stf_data_place_handle stf_data_place_clone(stf_data_place_handle h)
    void stf_data_place_destroy(stf_data_place_handle h)
    int stf_data_place_get_device_ordinal(stf_data_place_handle h)
    const char* stf_data_place_to_string(stf_data_place_handle h)
    void* stf_data_place_allocate(stf_data_place_handle h, ptrdiff_t size, cudaStream_t stream)
    void stf_data_place_deallocate(stf_data_place_handle h, void* ptr, size_t size, cudaStream_t stream)
    int stf_data_place_allocation_is_stream_ordered(stf_data_place_handle h)

    #
    # Logical data
    #
    ctypedef struct stf_logical_data_handle_t
    ctypedef stf_logical_data_handle_t* stf_logical_data_handle
    int stf_ctx_wait(stf_ctx_handle ctx, stf_logical_data_handle ld, void* out, size_t size) nogil
    stf_logical_data_handle stf_logical_data(stf_ctx_handle ctx, void* addr, size_t sz)
    stf_logical_data_handle stf_logical_data_with_place(stf_ctx_handle ctx, void* addr, size_t sz, stf_data_place_handle dplace)
    void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
    void stf_logical_data_destroy(stf_logical_data_handle ld)
    stf_logical_data_handle stf_logical_data_empty(stf_ctx_handle ctx, size_t length)
    stf_logical_data_handle stf_token(stf_ctx_handle ctx)

    #
    # Tasks
    #
    ctypedef struct stf_task_handle_t
    ctypedef stf_task_handle_t* stf_task_handle
    stf_task_handle stf_task_create(stf_ctx_handle ctx)
    void stf_task_set_exec_place(stf_task_handle t, stf_exec_place_handle exec_p)
    void stf_task_set_symbol(stf_task_handle t, const char* symbol)
    void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
    void stf_task_add_dep_with_dplace(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle data_p)
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

    #
    # CUDA kernel tasks
    #
    ctypedef struct stf_cuda_kernel_handle_t
    ctypedef stf_cuda_kernel_handle_t* stf_cuda_kernel_handle
    stf_cuda_kernel_handle stf_cuda_kernel_create(stf_ctx_handle ctx)
    void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, stf_exec_place_handle exec_p)
    void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol)
    void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m)
    void stf_cuda_kernel_start(stf_cuda_kernel_handle k)
    void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index)
    void stf_cuda_kernel_add_desc_cufunc(stf_cuda_kernel_handle k, CUfunction cufunc, dim3 grid_dim_, dim3 block_dim_, size_t shared_mem_, int arg_cnt, const void** args)
    void stf_cuda_kernel_end(stf_cuda_kernel_handle k)
    void stf_cuda_kernel_destroy(stf_cuda_kernel_handle k)

    #
    # Host launch
    #
    ctypedef struct stf_host_launch_handle_t
    ctypedef stf_host_launch_handle_t* stf_host_launch_handle
    ctypedef struct stf_host_launch_deps_handle_t
    ctypedef stf_host_launch_deps_handle_t* stf_host_launch_deps_handle
    ctypedef void (*stf_host_callback_fn)(stf_host_launch_deps_handle deps) noexcept

    stf_host_launch_handle stf_host_launch_create(stf_ctx_handle ctx)
    void stf_host_launch_add_dep(stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m)
    void stf_host_launch_set_symbol(stf_host_launch_handle h, const char* symbol)
    void stf_host_launch_set_user_data(stf_host_launch_handle h, const void* data, size_t size, void (*dtor)(void*))
    void stf_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback)
    void stf_host_launch_destroy(stf_host_launch_handle h)
    void* stf_host_launch_deps_get(stf_host_launch_deps_handle deps, size_t index)
    size_t stf_host_launch_deps_get_size(stf_host_launch_deps_handle deps, size_t index)
    size_t stf_host_launch_deps_size(stf_host_launch_deps_handle deps)
    void* stf_host_launch_deps_get_user_data(stf_host_launch_deps_handle deps)

    #
    # Stackable contexts (PR #8165 features ported on top of the opaque-handle
    # C API). All stackable_* entry points reuse existing handle types where
    # appropriate (stf_ctx_handle, stf_logical_data_handle, stf_task_handle,
    # stf_host_launch_handle); only while/repeat scopes have their own opaque
    # handles.
    #
    stf_ctx_handle stf_stackable_ctx_create()
    void stf_stackable_ctx_finalize(stf_ctx_handle ctx) nogil
    CUstream stf_stackable_ctx_fence(stf_ctx_handle ctx) nogil
    void stf_stackable_push_graph(stf_ctx_handle ctx)
    void stf_stackable_pop(stf_ctx_handle ctx)

    ctypedef struct stf_while_scope_handle_t
    ctypedef stf_while_scope_handle_t* stf_while_scope_handle
    ctypedef struct stf_repeat_scope_handle_t
    ctypedef stf_repeat_scope_handle_t* stf_repeat_scope_handle

    stf_while_scope_handle stf_stackable_push_while(stf_ctx_handle ctx)
    void stf_stackable_pop_while(stf_while_scope_handle scope)
    uint64_t stf_while_scope_get_cond_handle(stf_while_scope_handle scope)
    stf_repeat_scope_handle stf_stackable_push_repeat(stf_ctx_handle ctx, size_t count)
    void stf_stackable_pop_repeat(stf_repeat_scope_handle scope)

    ctypedef struct stf_launchable_graph_handle_t
    ctypedef stf_launchable_graph_handle_t* stf_launchable_graph_handle

    stf_launchable_graph_handle stf_stackable_pop_prologue(stf_ctx_handle ctx)
    void stf_stackable_pop_epilogue(stf_ctx_handle ctx)
    void stf_launchable_graph_launch(stf_launchable_graph_handle h) nogil
    cudaGraphExec_t stf_launchable_graph_exec(stf_launchable_graph_handle h)
    cudaStream_t stf_launchable_graph_stream(stf_launchable_graph_handle h)
    cudaGraph_t stf_launchable_graph_graph(stf_launchable_graph_handle h)
    void stf_launchable_graph_destroy(stf_launchable_graph_handle h)

    ctypedef struct stf_launchable_graph_shared_t
    ctypedef stf_launchable_graph_shared_t* stf_launchable_graph_shared

    int stf_stackable_pop_prologue_shared(stf_ctx_handle ctx, stf_launchable_graph_shared* out)
    int stf_launchable_graph_shared_dup(stf_launchable_graph_shared h, stf_launchable_graph_shared* out)
    void stf_launchable_graph_shared_free(stf_launchable_graph_shared h) nogil
    int stf_launchable_graph_shared_valid(stf_launchable_graph_shared h)
    void stf_launchable_graph_shared_launch(stf_launchable_graph_shared h) nogil
    cudaGraphExec_t stf_launchable_graph_shared_exec(stf_launchable_graph_shared h)
    cudaStream_t stf_launchable_graph_shared_stream(stf_launchable_graph_shared h)
    cudaGraph_t stf_launchable_graph_shared_graph(stf_launchable_graph_shared h)

    cdef enum stf_compare_op:
        STF_CMP_GT
        STF_CMP_LT
        STF_CMP_GE
        STF_CMP_LE

    cdef enum stf_dtype:
        STF_DTYPE_FLOAT32
        STF_DTYPE_FLOAT64
        STF_DTYPE_INT32
        STF_DTYPE_INT64

    void stf_stackable_while_cond_scalar(
        stf_ctx_handle ctx,
        stf_while_scope_handle scope,
        stf_logical_data_handle ld,
        stf_compare_op op,
        double threshold,
        stf_dtype dtype)

    stf_logical_data_handle stf_stackable_logical_data_with_place(
        stf_ctx_handle ctx, void* addr, size_t sz, stf_data_place_handle dplace)
    stf_logical_data_handle stf_stackable_logical_data(stf_ctx_handle ctx, void* addr, size_t sz)
    stf_logical_data_handle stf_stackable_logical_data_empty(stf_ctx_handle ctx, size_t length)
    stf_logical_data_handle stf_stackable_logical_data_no_export_empty(stf_ctx_handle ctx, size_t length)
    stf_logical_data_handle stf_stackable_token(stf_ctx_handle ctx)
    void stf_stackable_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
    void stf_stackable_logical_data_set_read_only(stf_logical_data_handle ld)
    void stf_stackable_logical_data_push(
        stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle dplace)
    void stf_stackable_logical_data_destroy(stf_logical_data_handle ld)
    void stf_stackable_token_destroy(stf_logical_data_handle ld)

    stf_task_handle stf_stackable_task_create(stf_ctx_handle ctx)
    void stf_stackable_task_add_dep(
        stf_ctx_handle ctx, stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
    void stf_stackable_task_add_dep_with_dplace(
        stf_ctx_handle ctx, stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle data_p)

    stf_host_launch_handle stf_stackable_host_launch_create(stf_ctx_handle ctx)
    void stf_stackable_host_launch_add_dep(
        stf_ctx_handle ctx, stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m)
    void stf_stackable_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback)
    void stf_stackable_host_launch_destroy(stf_host_launch_handle h)

# ctypes mirror structs for the partition mapper callback.
# The C API uses an out-pointer signature for stf_get_executor_fn:
#   void (*)(stf_pos4* result, stf_pos4 data_coords, stf_dim4 data_dims, stf_dim4 grid_dims)
# This is directly representable as a ctypes CFUNCTYPE.
class _mapper_pos4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int64), ("y", ctypes.c_int64),
                ("z", ctypes.c_int64), ("t", ctypes.c_int64)]

class _mapper_dim4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64), ("y", ctypes.c_uint64),
                ("z", ctypes.c_uint64), ("t", ctypes.c_uint64)]

_mapper_cfunc_type = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(_mapper_pos4), _mapper_pos4, _mapper_dim4, _mapper_dim4)


def _make_mapper_callback(mapper):
    """Wrap a Python partitioner as a C function pointer for stf_data_place_composite.

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

class AccessMode(IntFlag):
    NONE  = STF_NONE
    READ  = STF_READ
    WRITE = STF_WRITE
    RW    = STF_RW


def _logical_data_full(ctx, shape, fill_value, dtype=None, where=None, exec_place=None, name=None, *, no_export=False):
    """Shared implementation for ``context`` and ``stackable_context`` initializers."""
    if dtype is None:
        dtype = np.array(fill_value).dtype
    else:
        dtype = np.dtype(dtype)

    if exec_place is not None:
        if hasattr(exec_place, 'kind') and exec_place.kind == "host":
            raise NotImplementedError(
                "exec_place.host() is not yet supported for logical_data_full. "
                "Use exec_place.device() or omit exec_place parameter."
            )

    if no_export:
        ld = ctx.logical_data_empty(shape, dtype, name, no_export=True)
    else:
        ld = ctx.logical_data_empty(shape, dtype, name)

    try:
        from cuda.stf._experimental.fill_utils import init_logical_data
        init_logical_data(ctx, ld, fill_value, where, exec_place)
    except ImportError as e:
        raise RuntimeError("Fill support (cuda.core) is not available for logical_data_full") from e

    return ld


def _logical_data_default_dtype(dtype):
    return np.float64 if dtype is None else dtype


class stf_cai:
    """
    Wrapper that exposes CUDA Array Interface v3 for interop (torch, cupy, etc.).
    Supports dict-style access (e.g. obj['data']) for code that expects a CAI dict.
    """
    def __init__(self, ptr, tuple shape, dtype, stream=0):
        self.ptr = ptr               # integer device pointer
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.stream = int(stream)    # CUDA stream handle (int or 0)
        self.__cuda_array_interface__ = {
            'version': 3,
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


# Shared "alive" sentinel used to safely no-op a child wrapper's __dealloc__
# when its parent (stackable_)context was already finalized.
#
# Implementation note: we use a ``cdef class`` with a single ``bint`` field
# rather than a Python list ``[True]`` because Python lists are GC-tracked
# mutable containers whose contents pytest's ``gc_collect_harder`` may clear
# (causing ``IndexError`` from an emptied list).  A ``cdef class`` instance
# with no Python-object members is *not* GC-tracked and cannot be mutated by
# the cycle collector, giving us a stable shared flag.
cdef class _AliveFlag:
    cdef bint alive

    def __cinit__(self):
        self.alive = True


# Python-only lifetime guard for framework-owned primary CUDA contexts.
#
# Rationale:
# - STF's C++ core assumes callers keep the CUDA context valid for the whole
#   lifetime of a context.
# - In Python that assumption is frequently violated by third-party frameworks
#   (Numba/PyTorch/CuPy) that share the runtime primary context and may release
#   it between tests or after wrapper objects go out of scope.
# - The pragmatic fix is therefore localized to the Python wrappers: when
#   Python creates an STF context, it also retains every visible primary
#   context and releases those retains only after STF finalize() returns.
#
# This keeps the interop workaround out of the C++ core while still protecting
# Python-created STF contexts against refcount-driven teardown of primary
# contexts by foreign libraries.
cdef class _PrimaryContextPin:
    cdef list _devices
    cdef bint _released

    def __cinit__(self):
        cdef int dev_count = 0
        cdef int ordinal
        cdef CUdevice dev = 0
        cdef CUcontext ctx = <CUcontext>NULL
        cdef CUresult status

        self._devices = []
        self._released = False

        status = cuInit(0)
        if status != 0:
            raise RuntimeError(
                f"failed to initialize CUDA driver for STF Python context pin (CUresult={status})"
            )

        status = cuDeviceGetCount(&dev_count)
        if status != 0:
            raise RuntimeError(
                f"failed to enumerate CUDA devices for STF Python context pin (CUresult={status})"
            )

        for ordinal in range(dev_count):
            status = cuDeviceGet(&dev, ordinal)
            if status != 0:
                self.release()
                raise RuntimeError(
                    f"failed to query CUDA device {ordinal} for STF Python context pin "
                    f"(CUresult={status})"
                )

            status = cuDevicePrimaryCtxRetain(&ctx, dev)
            if status != 0:
                self.release()
                raise RuntimeError(
                    f"failed to retain CUDA primary context for device {ordinal} "
                    f"(CUresult={status})"
                )

            self._devices.append(dev)

    cdef void release(self):
        cdef list devices
        cdef object dev_obj

        if self._released or self._devices is None:
            return

        devices = self._devices
        self._devices = None
        self._released = True

        for dev_obj in devices:
            # Best-effort cleanup: reset/teardown paths may already have touched
            # the primary context; the release is only to balance our retain.
            cuDevicePrimaryCtxRelease(<CUdevice>dev_obj)

    def __dealloc__(self):
        # Explicit-finalize policy: do not make CUDA driver calls from GC.
        # If a context leaks without finalize(), its primary-context retain is
        # intentionally abandoned until process exit.
        pass


cdef class logical_data:
    cdef stf_logical_data_handle _ld
    cdef stf_ctx_handle _ctx

    cdef object _dtype
    cdef tuple  _shape
    cdef int    _ndim
    cdef size_t _len
    cdef str    _symbol  # Store symbol for display purposes
    cdef readonly bint _is_token  # readonly makes it accessible from Python
    # Prevent GC of the Python object whose raw pointer was passed to
    # the C API.  STF may access that pointer asynchronously, so the
    # source object must outlive the logical_data.
    cdef object _source_buf
    # Shared "alive" sentinel from the parent context. See context._alive.
    cdef _AliveFlag _alive

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
            self._source_buf = None
            self._alive = None
            return

        self._ctx = ctx._ctx
        self._alive = ctx._alive
        self._symbol = None  # Initialize symbol
        self._is_token = False  # Initialize token flag
        self._source_buf = buf  # prevent garbage collection in the case of numpy objects

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

            self._ld = stf_logical_data_with_place(ctx._ctx, <void*><uintptr_t>data_ptr, self._len, dplace._h)
            if self._ld == NULL:
                raise RuntimeError("failed to create logical_data from CUDA array interface")

        else:
            # Fallback to Python buffer protocol; require contiguous memory
            # since STF registers view.buf/view.len as a flat byte range.
            flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_ANY_CONTIGUOUS

            if PyObject_GetBuffer(buf, &view, flags) != 0:
                raise ValueError(
                    "object doesn't support the buffer protocol, is not contiguous, "
                    "or doesn't expose __cuda_array_interface__"
                )

            try:
                self._ndim  = view.ndim
                self._len = view.len
                self._shape = tuple(<Py_ssize_t>view.shape[i] for i in range(view.ndim))
                self._dtype = np.dtype(view.format)
                self._ld = stf_logical_data_with_place(ctx._ctx, view.buf, view.len, dplace._h)
                if self._ld == NULL:
                    raise RuntimeError("failed to create logical_data from buffer")

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
        # See stackable_logical_data.__dealloc__ for why _alive may be None
        # here even though it was set in the constructor (Cython's tp_clear
        # resets it to Py_None before tp_dealloc runs when breaking cycles).
        if self._ld != NULL and self._alive is not None and self._alive.alive:
            try:
                stf_logical_data_destroy(self._ld)
            except Exception as e:
                print(f"stf.logical_data: cleanup failed: {e}")
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
        out._ld = stf_logical_data_empty(self._ctx, self._len)
        if out._ld == NULL:
            raise RuntimeError("failed to create empty logical_data")
        out._ctx   = self._ctx
        out._dtype = self._dtype
        out._shape = self._shape
        out._ndim  = self._ndim
        out._len   = self._len
        out._symbol = None
        out._is_token = False
        out._source_buf = None
        out._alive = self._alive

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
        out._source_buf = None
        out._alive = ctx._alive
        out._ld = stf_token(ctx._ctx)
        if out._ld == NULL:
            raise RuntimeError("failed to create STF token")

        return out

    @staticmethod
    def init_by_shape(context ctx, shape, dtype, str name=None):
        """
        Create a new logical_data from a shape and a dtype.
        """
        try:
            shape_tuple = tuple(int(dim) for dim in shape)
        except TypeError:
            raise TypeError("shape must be an iterable of integers")
        if not shape_tuple:
            raise ValueError("shape must contain at least one dimension")
        for dim in shape_tuple:
            if dim <= 0:
                raise ValueError("all shape dimensions must be positive integers")
        cdef logical_data out = logical_data.__new__(logical_data)
        out._ctx   = ctx._ctx
        out._dtype = np.dtype(dtype)
        out._shape = shape_tuple
        out._ndim  = len(shape_tuple)
        cdef size_t total_items = 1
        for dim in shape_tuple:
            total_items *= dim
        out._len   = total_items * out._dtype.itemsize
        out._symbol = None
        out._is_token = False
        out._source_buf = None
        out._alive = ctx._alive
        out._ld = stf_logical_data_empty(ctx._ctx, out._len)
        if out._ld == NULL:
            raise RuntimeError("failed to create logical_data from shape")

        if name is not None:
            out.set_symbol(name)

        return out

    def borrow_ctx_handle(self):
        ctx = context(borrowed=True)
        ctx.borrow_from_handle(self._ctx)
        ctx._alive = self._alive
        return ctx

class dep:
    __slots__ = ("ld", "mode", "dplace")
    # ld may be either a logical_data or a stackable_logical_data; both classes
    # call dep(self, ...) from their .read()/.write()/.rw() helpers.
    def __init__(self, object ld, int mode, dplace=None):
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

def machine_init():
    """Initialize machine topology (P2P access, device memory pools).

    This is done automatically when creating an ``stf.context()``, but must
    be called explicitly when using places without an STF task context
    (e.g. for direct ``exec_place`` / ``exec_place_resources`` /
    ``pick_stream`` usage).

    Safe to call multiple times; only the first invocation has effect.
    """
    stf_machine_init()


class CudaStream(int):
    """An ``int`` subclass that also implements ``__cuda_stream__``.

    Because it **is** an ``int``, it passes ``PyLong_Check`` and works
    everywhere a raw stream pointer was accepted before (PyTorch's
    ``ExternalStream``, Numba's ``external_stream``, CuPy, ctypes, ...).

    The added ``__cuda_stream__`` method satisfies the protocol expected
    by ``cuda.compute`` algorithms, so the object can be passed directly
    as ``stream=`` without a manual wrapper.

    Instances are returned by :meth:`exec_place.pick_stream` and
    :meth:`task.stream_ptr`.
    """

    def __new__(cls, ptr):
        return super().__new__(cls, ptr)

    def __cuda_stream__(self):
        return (0, int(self))

    @property
    def ptr(self) -> int:
        """Raw CUstream pointer as a plain Python ``int``."""
        return int(self)

    def __repr__(self):
        return f"CudaStream(0x{int(self):x})"

cdef class green_ctx_view:
    cdef object _helper
    cdef size_t _idx

    def __cinit__(self):
        self._helper = None
        self._idx = 0

    @property
    def helper(self):
        return self._helper

    @property
    def index(self):
        return self._idx

    @property
    def device_id(self):
        return self._helper.device_id

    def __repr__(self):
        return f"green_ctx_view(device_id={self.device_id}, index={self._idx})"


cdef class green_context_helper:
    cdef stf_green_context_helper_handle _h

    def __cinit__(self, int sm_count, int dev_id=0):
        self._h = NULL
        if sm_count < 1:
            raise ValueError("sm_count must be a positive integer")
        self._h = stf_green_context_helper_create(sm_count, dev_id)
        if self._h == NULL:
            raise RuntimeError(
                f"failed to create green_context_helper(sm_count={sm_count}, dev_id={dev_id})"
            )

    def __dealloc__(self):
        if self._h != NULL:
            try:
                stf_green_context_helper_destroy(self._h)
            except Exception as e:
                print(f"stf.green_context_helper: cleanup failed: {e}")
            self._h = NULL

    def get_count(self):
        return stf_green_context_helper_get_count(self._h)

    def __len__(self):
        return self.get_count()

    @property
    def device_id(self):
        return stf_green_context_helper_get_device_id(self._h)

    def get_view(self, size_t idx):
        if idx >= self.get_count():
            raise IndexError(f"green_ctx index {idx} is out of range")
        cdef green_ctx_view view = green_ctx_view.__new__(green_ctx_view)
        view._helper = self
        view._idx = idx
        return view

    def __repr__(self):
        return (
            f"green_context_helper(device_id={self.device_id}, "
            f"count={self.get_count()})"
        )


cdef class exec_place_resources:
    """Standalone per-place stream-pool registry.

    Owns the CUDA streams it lazily creates the first time
    :meth:`exec_place.pick_stream` is called against a given place. Streams
    are released when this registry is garbage-collected (or when the owning
    STF context is finalized, for borrowed instances).

    There are two ways to obtain one:

    * ``exec_place_resources()`` — construct a fresh, owned registry. Use
      this when working with the ``places`` layer without an STF context.
    * ``ctx.place_resources`` — borrow the registry embedded in an STF
      context's ``async_resources_handle``. The borrowed handle's lifetime
      is bounded by ``ctx``; do not keep references past
      ``ctx.finalize()``.
    """
    cdef stf_exec_place_resources_handle _h
    # Owned registries are destroyed in __dealloc__; borrowed ones are
    # released by the owning STF context when it is finalized.
    cdef bint _owned

    def __cinit__(self, *, bint _borrow=False):
        self._h = NULL
        self._owned = not _borrow

    def __init__(self, *, bint _borrow=False):
        if _borrow:
            return
        self._h = stf_exec_place_resources_create()
        if self._h == NULL:
            raise RuntimeError("failed to create exec_place_resources")

    @staticmethod
    cdef exec_place_resources _borrow_from(stf_exec_place_resources_handle h):
        cdef exec_place_resources r = exec_place_resources.__new__(exec_place_resources, _borrow=True)
        r._owned = False
        r._h = h
        return r

    def __dealloc__(self):
        if self._owned and self._h != NULL:
            stf_exec_place_resources_destroy(self._h)
            self._h = NULL


cdef class exec_place:
    cdef stf_exec_place_handle _h
    cdef stf_exec_place_scope_handle _scope

    def __cinit__(self):
        self._h = NULL
        self._scope = NULL

    def __dealloc__(self):
        if self._scope != NULL:
            stf_exec_place_scope_exit(self._scope)
            self._scope = NULL
        if self._h != NULL:
            try:
                stf_exec_place_destroy(self._h)
            except Exception as e:
                print(f"stf.exec_place: cleanup failed: {e}")
            self._h = NULL

    @staticmethod
    def device(int dev_id):
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_device(dev_id)
        if p._h == NULL:
            raise RuntimeError(f"failed to create exec_place for device {dev_id}")
        return p

    @staticmethod
    def host():
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_host()
        if p._h == NULL:
            raise RuntimeError("failed to create host exec_place")
        return p

    @staticmethod
    def current_device():
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_current_device()
        if p._h == NULL:
            raise RuntimeError("failed to create current_device exec_place")
        return p

    @staticmethod
    def green_ctx(green_ctx_view view, use_green_ctx_data_place=False):
        cdef green_context_helper helper = <green_context_helper>view._helper
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_green_ctx(
            helper._h,
            view._idx,
            1 if use_green_ctx_data_place else 0,
        )
        if p._h == NULL:
            raise RuntimeError(f"failed to create green_ctx exec_place for index {view._idx}")
        return p

    @staticmethod
    def from_handle(uintptr_t handle):
        """Wrap an existing ``stf_exec_place_handle`` (as an integer).

        Takes ownership of the handle: the returned :class:`exec_place`
        frees it via ``stf_exec_place_destroy`` on destruction. Intended
        for extensibility layers (e.g. custom places built through the
        STF C API) that produce a handle out of band and want to hand it
        to the Python runtime.
        """
        if handle == 0:
            raise ValueError("exec_place.from_handle received a null handle")
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = <stf_exec_place_handle>handle
        return p

    @property
    def kind(self) -> str:
        if stf_exec_place_is_host(self._h):
            return "host"
        return "device"

    @property
    def dims(self):
        """Grid dimensions as (x, y, z, t). Scalar places return (1, 1, 1, 1)."""
        cdef stf_dim4 d
        stf_exec_place_get_dims(self._h, &d)
        return (d.x, d.y, d.z, d.t)

    @property
    def size(self):
        """Number of sub-places (1 for scalar places)."""
        return stf_exec_place_size(self._h)

    @property
    def _handle_int(self):
        """Return the opaque C handle as an integer (for FFI / ctypes use)."""
        return <uintptr_t>self._h

    def set_affine_data_place(self, data_place dplace):
        """Set the affine data place for this exec place grid.

        Dependencies using ``data_place.affine()`` will resolve to ``dplace``
        when this exec place is used as the task's execution place.
        """
        stf_exec_place_set_affine_data_place(self._h, dplace._h)

    def __enter__(self):
        if self._h == NULL:
            raise RuntimeError("exec_place handle is null")
        self._scope = stf_exec_place_scope_enter(self._h, 0)
        if self._scope == NULL:
            raise RuntimeError("failed to activate exec_place scope")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._scope != NULL:
            stf_exec_place_scope_exit(self._scope)
            self._scope = NULL
        return False

    @property
    def affine_data_place(self):
        """Return the data_place associated with this exec_place."""
        cdef stf_data_place_handle dh = stf_exec_place_get_affine_data_place(self._h)
        if dh == NULL:
            raise RuntimeError("failed to get affine data_place")
        cdef data_place dp = data_place.__new__(data_place)
        dp._h = dh
        return dp

    def pick_stream(self, exec_place_resources resources, bint for_computation=True):
        """Return a :class:`CudaStream` from this place's stream pool.

        The pool is owned by ``resources`` and the returned stream remains
        valid until that registry is destroyed (or, for a borrowed registry,
        until the owning STF context is finalized). The returned object
        implements ``__cuda_stream__`` for direct use with ``cuda.compute``
        and behaves like an ``int`` (raw pointer).

        Must be called inside a ``with place:`` block (or after manual scope
        entry).

        Parameters
        ----------
        resources : exec_place_resources
            The registry that should hand out the stream. Construct one
            standalone (``exec_place_resources()``) when using the places
            layer without STF, or borrow ``ctx.place_resources`` to share
            the pools embedded in an STF context. There is intentionally no
            no-arg overload: every stream must be tied to a registry whose
            lifetime governs it.
        for_computation : bool, optional (default ``True``)
            Hint selecting between the compute pool (``True``) and the
            data-transfer pool (``False``). Pooled places (``device(N)``,
            ``host()``) honor it; self-contained places ignore it.
        """
        if resources is None:
            raise TypeError("pick_stream requires an exec_place_resources argument")
        cdef CUstream s = stf_exec_place_pick_stream(resources._h, self._h, 1 if for_computation else 0)
        return CudaStream(<uintptr_t>s)

    def get_place(self, size_t idx):
        """Get the sub-place at linear index *idx* (0 for scalar places).

        For grids, returns the sub-place at that index.
        Caller owns the returned exec_place.
        """
        cdef stf_exec_place_handle sub = stf_exec_place_get_place(self._h, idx)
        if sub == NULL:
            raise IndexError(f"sub-place index {idx} is out of range")
        cdef exec_place ep = exec_place.__new__(exec_place)
        ep._h = sub
        return ep

    def __getitem__(self, size_t idx):
        return self.get_place(idx)


cdef class exec_place_grid(exec_place):
    """Grid of execution places (a subclass of exec_place).

    Use wherever an exec_place is expected.  Create with ``from_devices()``
    or ``create()``.
    """
    cdef object _mapper_keep_alive  # prevent GC of ctypes callback if mapper was set

    def __cinit__(self):
        self._mapper_keep_alive = None

    @staticmethod
    def from_devices(device_ids):
        """Create a 1-D grid with one place per device.

        Parameters
        ----------
        device_ids : sequence of int
            Device ordinals (e.g. ``[0, 1]`` for two GPUs, or ``[0, 0]``
            for the same device repeated).
        """
        cdef int c_ids[64]
        cdef size_t n = len(device_ids)
        if n == 0:
            raise ValueError("device_ids must contain at least one device")
        if n > 64:
            raise ValueError("at most 64 devices supported")
        for i in range(n):
            c_ids[i] = int(device_ids[i])
        cdef exec_place_grid g = exec_place_grid.__new__(exec_place_grid)
        g._h = stf_exec_place_grid_from_devices(c_ids, n)
        if g._h == NULL:
            raise RuntimeError("failed to create exec_place grid from devices")
        return g

    @staticmethod
    def create(places, grid_dims=None, mapper=None):
        """Create a grid from a list of exec_place objects.

        Parameters
        ----------
        places : list of exec_place
            Individual execution places that form the grid.
        grid_dims : tuple of int, optional
            Shape of the grid as ``(x, y, z, t)``.  If *None*, a 1-D
            grid of length ``len(places)`` is used.
        mapper : callable, optional
            If provided, a composite data place is created from this
            partitioner and set as the grid's affine data place so that
            dependencies with ``data_place.affine()`` resolve automatically.
            Signature: ``(data_coords, data_dims, grid_dims) -> (x, y, z, t)``.
        """
        cdef size_t n = len(places)
        if n == 0:
            raise ValueError("places must contain at least one place")
        if n > 64:
            raise ValueError("at most 64 places supported")

        cdef stf_exec_place_handle c_places[64]
        cdef stf_dim4 dims
        cdef exec_place ep

        converted = []
        for i in range(n):
            place = places[i]
            if not isinstance(place, exec_place) and hasattr(place, "_as_stf_exec_place"):
                place = place._as_stf_exec_place()
            ep = <exec_place?>place
            converted.append(ep)
            c_places[i] = ep._h

        cdef exec_place_grid g = exec_place_grid.__new__(exec_place_grid)
        if grid_dims is not None:
            dims.x = int(grid_dims[0])
            dims.y = int(grid_dims[1]) if len(grid_dims) > 1 else 1
            dims.z = int(grid_dims[2]) if len(grid_dims) > 2 else 1
            dims.t = int(grid_dims[3]) if len(grid_dims) > 3 else 1
            g._h = stf_exec_place_grid_create(c_places, n, &dims)
        else:
            g._h = stf_exec_place_grid_create(c_places, n, NULL)

        if g._h == NULL:
            raise RuntimeError("failed to create exec_place grid")

        if mapper is not None:
            dplace = data_place.composite(g, mapper)
            g.set_affine_data_place(dplace)
            g._mapper_keep_alive = dplace

        return g


cdef class data_place:
    cdef stf_data_place_handle _h
    cdef object _mapper_callback  # prevent GC of ctypes callback for composite places

    def __cinit__(self):
        self._h = NULL
        self._mapper_callback = None

    def __dealloc__(self):
        if self._h != NULL:
            try:
                stf_data_place_destroy(self._h)
            except Exception as e:
                print(f"stf.data_place: cleanup failed: {e}")
            self._h = NULL

    @staticmethod
    def device(int dev_id):
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_device(dev_id)
        if p._h == NULL:
            raise RuntimeError(f"failed to create data_place for device {dev_id}")
        return p

    @staticmethod
    def host():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_host()
        if p._h == NULL:
            raise RuntimeError("failed to create host data_place")
        return p

    @staticmethod
    def managed():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_managed()
        if p._h == NULL:
            raise RuntimeError("failed to create managed data_place")
        return p

    @staticmethod
    def affine():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_affine()
        if p._h == NULL:
            raise RuntimeError("failed to create affine data_place")
        return p

    @staticmethod
    def current_device():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_current_device()
        if p._h == NULL:
            raise RuntimeError("failed to create current_device data_place")
        return p

    @staticmethod
    def green_ctx(green_ctx_view view):
        cdef green_context_helper helper = <green_context_helper>view._helper
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_green_ctx(helper._h, view._idx)
        if p._h == NULL:
            raise RuntimeError(f"failed to create green_ctx data_place for index {view._idx}")
        return p

    @staticmethod
    def from_handle(uintptr_t handle):
        """Wrap an existing ``stf_data_place_handle`` (as an integer).

        Takes ownership of the handle: the returned :class:`data_place`
        frees it via ``stf_data_place_destroy`` on destruction. Intended
        for extensibility layers that produce a handle through the STF C
        API and want to hand it to the Python runtime.
        """
        if handle == 0:
            raise ValueError("data_place.from_handle received a null handle")
        cdef data_place p = data_place.__new__(data_place)
        p._h = <stf_data_place_handle>handle
        return p

    @staticmethod
    def composite(exec_place grid, object mapper):
        """Create a composite data place: grid of execution places + partition function.

        The partitioner (mapper) is a callable with signature::

            (data_coords, data_dims, grid_dims) -> (x, y, z, t)

        Each argument/return is a 4-tuple of integers:

        - *data_coords*: logical position in the data
        - *data_dims*: full shape of the data
        - *grid_dims*: shape of the execution place grid
        - return: position in the grid (which place owns this data element)

        Example — blocked partition along first dimension::

            def blocked_1d(data_coords, data_dims, grid_dims):
                n = data_dims[0]
                nplaces = grid_dims[0]
                part_size = max((n + nplaces - 1) // nplaces, 1)
                place_x = min(data_coords[0] // part_size, nplaces - 1)
                return (place_x, 0, 0, 0)

            grid = exec_place_grid.from_devices([0, 1])
            dplace = data_place.composite(grid, blocked_1d)
        """
        if not callable(mapper):
            raise TypeError(
                "mapper must be callable: (data_coords, data_dims, grid_dims) -> (x, y, z, t)")
        callback_obj, c_ptr = _make_mapper_callback(mapper)
        cdef data_place p = data_place.__new__(data_place)
        p._mapper_callback = callback_obj
        cdef uintptr_t ptr_val = c_ptr
        p._h = stf_data_place_composite(grid._h, <stf_get_executor_fn>ptr_val)
        if p._h == NULL:
            raise RuntimeError("failed to create composite data_place")
        return p

    @property
    def kind(self) -> str:
        cdef const char* s = stf_data_place_to_string(self._h)
        return s.decode("utf-8") if s != NULL else "unknown"

    @property
    def device_id(self) -> int:
        return stf_data_place_get_device_ordinal(self._h)

    def allocate(self, Py_ssize_t nbytes, stream=None):
        """Allocate *nbytes* on this data place.

        Parameters
        ----------
        nbytes : int
            Number of bytes to allocate.
        stream : optional
            CUDA stream for stream-ordered allocation (int, CudaStream, or
            any object implementing ``__cuda_stream__``).  ``None`` uses the
            default (null) stream.

        Returns
        -------
        int
            Device (or host) pointer as a Python int.

        Raises
        ------
        MemoryError
            If the underlying place cannot allocate (out of memory, or
            the place type does not support allocation).
        """
        cdef uintptr_t s_val = 0
        if stream is not None:
            s_val = <uintptr_t>int(stream)
        cdef cudaStream_t s = <cudaStream_t>s_val
        cdef void* ptr = stf_data_place_allocate(self._h, <ptrdiff_t>nbytes, s)
        if ptr == NULL:
            raise MemoryError(f"data_place.allocate failed for {nbytes} bytes")
        return <uintptr_t>ptr

    def deallocate(self, uintptr_t ptr, size_t nbytes, stream=None):
        """Free memory previously obtained from :meth:`allocate`.

        Parameters
        ----------
        ptr : int
            Pointer returned by :meth:`allocate`.
        nbytes : int
            Size of the original allocation in bytes.
        stream : optional
            CUDA stream for stream-ordered deallocation.
        """
        cdef uintptr_t s_val = 0
        if stream is not None:
            s_val = <uintptr_t>int(stream)
        cdef cudaStream_t s = <cudaStream_t>s_val
        stf_data_place_deallocate(self._h, <void*>ptr, nbytes, s)

    @property
    def allocation_is_stream_ordered(self):
        """Whether allocations on this place are stream-ordered."""
        return bool(stf_data_place_allocation_is_stream_ordered(self._h))


cdef class task:
    cdef stf_task_handle _t

    # list of logical data in deps: we need this because we can't exchange
    # dtype/shape easily through the C API of STF
    cdef list _lds_args
    # Shared "alive" sentinel from the parent context. See context._alive.
    cdef _AliveFlag _alive

    def __cinit__(self, context ctx):
        self._t = stf_task_create(ctx._ctx)
        if self._t == NULL:
            raise RuntimeError("failed to create STF task")
        self._lds_args = []
        self._alive = ctx._alive

    def __dealloc__(self):
        # See stackable_logical_data.__dealloc__ for why a None _alive must
        # be treated as "context already gone" rather than "no parent".
        if self._t != NULL and self._alive is not None and self._alive.alive:
            try:
                stf_task_destroy(self._t)
            except Exception as e:
                print(f"stf.task: cleanup failed: {e}")
        self._t = NULL

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
        if not isinstance(d.ld, logical_data):
            raise TypeError(
                "dep payload must be a logical_data for context.task(); "
                "did you mix stackable and non-stackable deps?"
            )

        cdef logical_data ldata = <logical_data> d.ld
        cdef int           mode_int  = int(d.mode)
        cdef stf_access_mode mode_ce = <stf_access_mode> mode_int
        cdef data_place dp

        # Only the dep *type* is validated here, not its owning context. A dep
        # whose logical_data belongs to a different context is rejected later by
        # the C++ core when the task acquires its deps (it aborts with a
        # context-mismatch error; see cudax .../internal/acquire_release.cuh).
        if d.dplace is None:
            stf_task_add_dep(self._t, ldata._ld, mode_ce)
        else:
            if not isinstance(d.dplace, data_place):
                raise TypeError("dep data_place override must be a data_place")
            dp = <data_place> d.dplace
            stf_task_add_dep_with_dplace(self._t, ldata._ld, mode_ce, dp._h)

        self._lds_args.append(ldata)

    def set_symbol(self, str name):
        stf_task_set_symbol(self._t, name.encode())

    def set_exec_place(self, object exec_p):
        if not isinstance(exec_p, exec_place):
            raise TypeError("set_exec_place expects an exec_place argument")

        cdef exec_place ep = <exec_place> exec_p
        stf_task_set_exec_place(self._t, ep._h)

    def stream_ptr(self):
        """Return a :class:`CudaStream` for this task's CUDA stream.

        The returned object implements ``__cuda_stream__`` for direct use
        with ``cuda.compute`` and behaves like an ``int`` (raw pointer)
        for ctypes or PyCUDA.
        """
        cdef CUstream s = stf_task_get_custream(self._t)
        return CudaStream(<uintptr_t>s)

    def get_grid_dims(self):
        """When the task's exec place is a grid, return (x, y, z, t) shape.

        Call after start(). Returns None if the task is not on a grid.
        """
        cdef stf_dim4 dims
        if stf_task_get_grid_dims(self._t, &dims) != 0:
            return None
        return (dims.x, dims.y, dims.z, dims.t)

    def get_stream_at_index(self, size_t place_index):
        """When the task's exec place is a grid, return the CUstream for the
        given linear index (0 to product of grid dims - 1) as a Python int.

        Call after start(). Raises if not a grid or index invalid.
        """
        cdef CUstream s
        if stf_task_get_custream_at_index(self._t, place_index, &s) != 0:
            raise RuntimeError("task is not on a grid or place_index out of range")
        return <uintptr_t> s

    def get_stream_ptrs(self):
        """Return a list of raw CUstream pointers (as ints), one per place in the grid.

        Convenience for grid tasks. Returns [stream_ptr()] (length 1) for non-grid tasks.
        Call after start().
        """
        dims = self.get_grid_dims()
        if dims is None:
            return [self.stream_ptr()]
        cdef size_t n = dims[0] * dims[1] * dims[2] * dims[3]
        return [self.get_stream_at_index(i) for i in range(n)]

    def get_arg(self, index) -> int:
        if self._lds_args[index]._is_token:
           raise RuntimeError("cannot materialize a token argument")

        cdef void *ptr = stf_task_get(self._t, index)
        return <uintptr_t>ptr

    def get_arg_cai(self, index):
        """Return the argument as a CUDA Array Interface v3 object.
        The returned view is only valid while the task is active, i.e. until stf_task_end()
        or the end of the surrounding ``with ctx.task(...)`` block."""
        ptr = self.get_arg(index)
        return stf_cai(ptr, self._lds_args[index].shape, self._lds_args[index].dtype, stream=self.stream_ptr())

    def args_cai(self):
        """
        Return all non-token buffer arguments as CUDA Array Interface v3 objects.
        Returns None, a single object, or a tuple. Use from non-shipped code (e.g. tests) to
        convert to numba/torch/cupy via from_cuda_array_interface or torch.as_tensor(obj).
        Returned views are only valid while the task is active.
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

cdef dim3 _to_dim3(object val):
    """Convert an int or 1-3 element tuple to a dim3 struct."""
    cdef dim3 d
    cdef tuple t
    cdef int n
    if isinstance(val, int):
        d.x = val; d.y = 1; d.z = 1
        return d
    t = tuple(val)
    n = len(t)
    if n == 1:
        d.x = t[0]; d.y = 1; d.z = 1
    elif n == 2:
        d.x = t[0]; d.y = t[1]; d.z = 1
    elif n == 3:
        d.x = t[0]; d.y = t[1]; d.z = t[2]
    else:
        raise ValueError("grid/block must have 1-3 dimensions")
    return d


cdef class cuda_kernel:
    """Optimized CUDA kernel task with full dependency tracking.

    Unlike a generic ``task`` where the user manually launches work on a
    stream, ``cuda_kernel`` receives the complete kernel description
    (function, grid, block, args) so STF can create native CUDA graph
    kernel nodes, avoiding stream-capture overhead.
    """
    cdef stf_cuda_kernel_handle _k
    cdef list _lds_args
    cdef list _arg_holders  # keep ParamHolder(s) alive until end()
    # Shared "alive" sentinel from the parent context. See context._alive.
    cdef _AliveFlag _alive

    def __cinit__(self, context ctx):
        self._k = stf_cuda_kernel_create(ctx._ctx)
        if self._k == NULL:
            raise RuntimeError("failed to create STF cuda_kernel")
        self._lds_args = []
        self._arg_holders = []
        self._alive = ctx._alive

    def __dealloc__(self):
        # See stackable_logical_data.__dealloc__ for why a None _alive must
        # be treated as "context already gone" rather than "no parent".
        if self._k != NULL and self._alive is not None and self._alive.alive:
            try:
                stf_cuda_kernel_destroy(self._k)
            except Exception as e:
                print(f"stf.cuda_kernel: cleanup failed: {e}")
        self._k = NULL

    def start(self):
        stf_cuda_kernel_start(self._k)

    def end(self):
        stf_cuda_kernel_end(self._k)
        self._arg_holders.clear()

    def add_dep(self, object d):
        if not isinstance(d, dep):
            raise TypeError("add_dep expects read(ld), write(ld) or rw(ld)")
        if not isinstance(d.ld, logical_data):
            raise TypeError(
                "dep payload must be a logical_data for context.cuda_kernel(); "
                "did you mix stackable and non-stackable deps?"
            )
        cdef logical_data ldata = <logical_data>d.ld
        cdef int mode_int = int(d.mode)
        cdef stf_access_mode mode_ce = <stf_access_mode>mode_int
        # Only the dep *type* is validated here, not its owning context. A dep
        # whose logical_data belongs to a different context is rejected later by
        # the C++ core when the task acquires its deps (it aborts with a
        # context-mismatch error; see cudax .../internal/acquire_release.cuh).
        stf_cuda_kernel_add_dep(self._k, ldata._ld, mode_ce)
        self._lds_args.append(ldata)

    def set_symbol(self, str name):
        stf_cuda_kernel_set_symbol(self._k, name.encode())

    def set_exec_place(self, object exec_p):
        if not isinstance(exec_p, exec_place):
            raise TypeError("set_exec_place expects an exec_place argument")
        cdef exec_place ep = <exec_place>exec_p
        stf_cuda_kernel_set_exec_place(self._k, ep._h)

    def get_arg(self, int index) -> int:
        if self._lds_args[index]._is_token:
            raise RuntimeError("cannot materialize a token argument")
        cdef void* ptr = stf_cuda_kernel_get_arg(self._k, index)
        return <uintptr_t>ptr

    def get_arg_cai(self, int index):
        ptr = self.get_arg(index)
        return stf_cai(ptr, self._lds_args[index].shape, self._lds_args[index].dtype)

    def launch(self, kernel, grid, block, args, size_t shmem=0):
        """Launch a CUDA kernel through STF.

        Parameters
        ----------
        kernel : cuda.core.Kernel or int
            Compiled kernel object (``cuda.core.Kernel``) or raw
            ``CUfunction`` handle as an integer.
        grid : int or tuple
            Grid dimensions (up to 3D).
        block : int or tuple
            Block dimensions (up to 3D).
        args : list
            Kernel arguments.  ``int`` values are treated as device
            pointers (matching ``cuda.core.launch`` conventions);
            use ``ctypes`` or ``numpy`` scalars for typed values.
        shmem : int, optional
            Dynamic shared memory in bytes (default 0).
        """
        from cuda.core._kernel_arg_handler import ParamHolder

        cdef uintptr_t func_handle
        if hasattr(kernel, '_handle'):
            handle = kernel._handle
            try:
                from cuda.bindings.driver import CUkernel as _CUkernel
                if isinstance(handle, _CUkernel):
                    from cuda.bindings.driver import cuKernelGetFunction
                    err, cufunc = cuKernelGetFunction(handle)
                    if int(err) != 0:
                        raise RuntimeError(
                            f"cuKernelGetFunction failed with error {err}")
                    func_handle = <uintptr_t>int(cufunc)
                else:
                    func_handle = <uintptr_t>int(handle)
            except ImportError:
                func_handle = <uintptr_t>int(handle)
        else:
            func_handle = <uintptr_t>int(kernel)

        cdef dim3 grid_dim = _to_dim3(grid)
        cdef dim3 block_dim = _to_dim3(block)

        holder = ParamHolder(tuple(args))
        cdef const void** raw_args = <const void**><uintptr_t>(holder.ptr)

        stf_cuda_kernel_add_desc_cufunc(
            self._k, <CUfunction>func_handle,
            grid_dim, block_dim, shmem,
            <int>len(args), raw_args)

        self._arg_holders.append(holder)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, object exc_type, object exc, object tb):
        self.end()
        return False


# ---------------------------------------------------------------------------
# host_launch helpers: C callback trampoline and Python payload destructor
# ---------------------------------------------------------------------------

cdef void _python_payload_destructor(void* data) noexcept with gil:
    """Release the Python payload tuple when C++ destroys the host_launch scope."""
    cdef PyObject* obj = (<PyObject**>data)[0]
    Py_XDECREF(obj)

cdef void _host_launch_trampoline(stf_host_launch_deps_handle deps_h) noexcept with gil:
    """C callback that unpacks deps as numpy arrays and calls the Python fn."""
    cdef PyObject** payload_ptr_ptr = <PyObject**>stf_host_launch_deps_get_user_data(deps_h)
    cdef object payload = <object>(payload_ptr_ptr[0])
    fn, user_args, dep_meta = payload

    cdef size_t ndeps = stf_host_launch_deps_size(deps_h)
    dep_arrays = []
    cdef size_t i
    cdef void* ptr
    cdef size_t nbytes
    for i in range(ndeps):
        ptr = stf_host_launch_deps_get(deps_h, i)
        nbytes = stf_host_launch_deps_get_size(deps_h, i)
        shape, dtype = dep_meta[i]
        dt = np.dtype(dtype)
        cbuf = (ctypes.c_char * nbytes).from_address(<uintptr_t>ptr)
        arr = np.frombuffer(cbuf, dtype=dt).reshape(shape)
        dep_arrays.append(arr)

    fn(*dep_arrays, *user_args)

cdef class async_resources:
    """Shareable ``async_resources_handle`` for STF contexts.

    Wraps the C++ ``async_resources_handle``. Reusing a single instance
    across multiple ``context`` constructions lets the graph backend amortize
    graph-instantiation cost and lets every context share the same per-place
    stream pools. Mirrors the C++ pattern::

        async_resources_handle h;
        for (...) {
            graph_ctx ctx(stream, h);   // or stream_ctx(stream, h)
            // ...
            ctx.finalize();
        }

    The handle must outlive every context it was passed to: ``finalize()``
    those contexts before dropping the Python handle.
    """
    cdef stf_async_resources_handle _h

    def __cinit__(self):
        self._h = stf_async_resources_create()
        if self._h == NULL:
            raise RuntimeError("failed to create stf async_resources handle")

    def __dealloc__(self):
        if self._h != NULL:
            stf_async_resources_destroy(self._h)
            self._h = <stf_async_resources_handle>NULL

    def __repr__(self):
        return f"async_resources(handle={<uintptr_t>self._h})"


cdef class context:
    cdef stf_ctx_handle _ctx
    # Is this a context that we have borrowed ?
    cdef bint _borrowed
    # Python-only primary-context retain. This is intentionally kept out of the
    # C++ core and exists only to shield Python interop with frameworks that
    # share and later release CUDA primary contexts.
    cdef _PrimaryContextPin _pin
    # Shared "alive" sentinel: an _AliveFlag whose .alive bint is flipped to
    # False by finalize(). Every child wrapper (logical_data, task, ...)
    # created from this context holds the same _AliveFlag by reference and
    # consults it before calling its C destroy in __dealloc__. This prevents
    # use-after-free when a child outlives its context (e.g. a logical_data
    # still on the Python stack when the next test's context creation /
    # numba kernel rebinds the CUDA primary context).
    cdef _AliveFlag _alive
    # Keep-alive reference to a caller-provided async_resources, if any,
    # so Python-side GC cannot destroy it while this context still uses it.
    cdef async_resources _handle_ref

    def __cinit__(self, bint use_graph=False, bint borrowed=False,
                  stream=None, async_resources handle=None):
        """Create an STF context.

        Parameters
        ----------
        use_graph : bool, default False
            If ``True``, use the CUDA-graph backend (equivalent to C++
            ``graph_ctx``). Otherwise the default stream backend.
        borrowed : bool, default False
            Internal: wrap an externally-owned ``stf_ctx_handle``.
        stream : optional
            CUDA stream to inherit (any object implementing the
            ``__cuda_stream__`` protocol, or a pointer-valued int).
            When provided, STF emits its work on top of this stream
            instead of picking one from its internal pool. Mirrors the
            C++ ``stream_ctx ctx(stream)`` / ``graph_ctx ctx(stream)``.
        handle : async_resources, optional
            Shareable resources handle. Reusing one across many contexts
            lets the graph backend cache instantiated graphs and lets the
            stream backend reuse its stream pools. Mirrors the C++
            ``stream_ctx ctx(stream, handle)`` / ``graph_ctx ctx(stream,
            handle)``.
        """
        self._ctx = <stf_ctx_handle>NULL
        self._borrowed = borrowed
        self._pin = None
        self._alive = _AliveFlag()
        self._handle_ref = None
        if borrowed:
            return

        self._pin = _PrimaryContextPin()

        cdef bint has_overrides = (stream is not None) or (handle is not None)
        cdef stf_ctx_options opts
        cdef uintptr_t stream_val = 0

        if has_overrides:
            opts.backend = STF_BACKEND_GRAPH if use_graph else STF_BACKEND_STREAM
            # has_stream distinguishes "user explicitly passed a stream" from
            # "user omitted stream" (unlike nullptr, which is a valid NULL stream).
            if stream is not None:
                stream_val = <uintptr_t>int(stream)
                opts.has_stream = 1
            else:
                opts.has_stream = 0
            opts.stream = <cudaStream_t>stream_val
            if handle is not None:
                opts.handle = handle._h
                self._handle_ref = handle
            else:
                opts.handle = <stf_async_resources_handle>NULL
            self._ctx = stf_ctx_create_ex(&opts)
        elif use_graph:
            self._ctx = stf_ctx_create_graph()
        else:
            self._ctx = stf_ctx_create()

        if self._ctx == NULL:
            self._handle_ref = None
            self._pin.release()
            self._pin = None
            raise RuntimeError("failed to create STF context")

    cdef borrow_from_handle(self, stf_ctx_handle ctx_handle):
        if self._ctx != NULL:
            raise RuntimeError("context already initialized")

        if not self._borrowed:
            raise RuntimeError("cannot call borrow_from_handle on this context")

        self._ctx = ctx_handle

    def __repr__(self):
        return f"context(handle={<uintptr_t>self._ctx}, borrowed={self._borrowed})"

    def __dealloc__(self):
        if self._borrowed:
            self._ctx = <stf_ctx_handle>NULL
            return

        if self._ctx != NULL:
            if self._alive is not None:
                self._alive.alive = False
            try:
                warnings.warn(
                    "cuda.stf._experimental.context was garbage-collected without an explicit finalize(); "
                    "STF/CUDA resources were abandoned. Call finalize() explicitly or use "
                    "'with cuda.stf._experimental.context(...) as ctx:'.",
                    ResourceWarning,
                )
            except Exception:
                pass
            self._ctx = <stf_ctx_handle>NULL

    def finalize(self):
        cdef _PrimaryContextPin pin = self._pin

        if self._borrowed:
            raise RuntimeError("cannot finalize borrowed context")

        # Flip the shared sentinel first so every surviving child wrapper
        # turns its __dealloc__ into a no-op. Idempotent: safe to call twice
        # (e.g. explicit finalize() then implicit one via __dealloc__).
        if self._alive is not None:
            self._alive.alive = False

        cdef stf_ctx_handle h = self._ctx
        self._pin = None
        if h != NULL:
            self._ctx = NULL
            with nogil:
                stf_ctx_finalize(h)
        else:
            self._ctx = NULL

        # Drop the keep-alive on the shared async_resources only after the
        # context has been finalized -- until then the C++ ctx holds a copy
        # that the underlying shared state must still back.
        self._handle_ref = None

        if pin is not None:
            pin.release()

    def __enter__(self):
        return self

    def __exit__(self, object exc_type, object exc, object tb):
        if not self._borrowed:
            self.finalize()
        return False

    @property
    def place_resources(self):
        """Borrowed reference to this context's per-place stream-pool registry.

        The returned :class:`exec_place_resources` is owned by the context;
        do **not** keep references to it (or to streams it has handed out)
        past :meth:`finalize`. Useful for sharing pools between STF code and
        standalone places-layer calls within one context's lifetime.
        """
        if self._ctx == NULL:
            raise RuntimeError("context has been finalized")
        cdef stf_exec_place_resources_handle h = stf_ctx_get_place_resources(self._ctx)
        return exec_place_resources._borrow_from(h)

    def fence(self):
        """Return a CUDA stream that completes when all pending tasks finish.

        Provides a non-blocking synchronization point: the returned stream
        will be signaled once every task submitted so far has completed.
        Unlike ``finalize()``, this does **not** destroy the context, so
        more tasks can be submitted afterwards.

        Returns
        -------
        int
            Raw ``CUstream`` handle as a Python integer (suitable for
            ``cudaStreamSynchronize`` via ctypes, PyCUDA, etc.).

        Examples
        --------
        >>> ctx = stf.context()
        >>> ld = ctx.logical_data(np.zeros(8, dtype=np.float32))
        >>> with ctx.task(ld.rw()):
        ...     pass
        >>> stream = ctx.fence()
        >>> # cudaStreamSynchronize(stream) to wait for completion
        >>> ctx.finalize()
        """
        if self._ctx == NULL:
            raise RuntimeError("context handle is NULL")
        cdef CUstream s
        with nogil:
            s = stf_fence(self._ctx)
        return <uintptr_t>s

    def wait(self, ld not None):
        """Synchronize and return a logical data's contents as a numpy array.

        Like C++ ``ctx.wait(ldata)``, this blocks until the data is
        available and returns a host copy.  The context remains usable
        afterwards, unlike ``finalize()``.

        Parameters
        ----------
        ld : logical_data
            The logical data whose contents should be retrieved.

        Returns
        -------
        numpy.ndarray
            A new numpy array with the same shape and dtype as ``ld``.

        Examples
        --------
        >>> ctx = stf.context()
        >>> lSum = ctx.logical_data(np.zeros(1, dtype=np.float64))
        >>> # ... submit tasks that write to lSum ...
        >>> result = ctx.wait(lSum)   # blocks, returns numpy array
        >>> print(result[0])          # context still usable
        >>> ctx.finalize()
        """
        if self._ctx == NULL:
            raise RuntimeError("context handle is NULL")
        if not isinstance(ld, logical_data):
            raise TypeError("wait() requires a logical_data object")
        cdef logical_data ldata = <logical_data>ld
        import numpy as np
        cdef object buf = np.empty(ldata._shape, dtype=ldata._dtype)
        cdef Py_buffer pybuf
        PyObject_GetBuffer(buf, &pybuf, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)
        cdef void* ptr = pybuf.buf
        cdef size_t sz = <size_t>pybuf.len
        cdef int rc
        try:
            with nogil:
                rc = stf_ctx_wait(self._ctx, ldata._ld, ptr, sz)
        finally:
            PyBuffer_Release(&pybuf)
        if rc != 0:
            raise RuntimeError("stf_ctx_wait failed")
        return buf

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
        return _logical_data_full(self, shape, fill_value, dtype, where, exec_place, name)

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
        dtype = _logical_data_default_dtype(dtype)
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
        dtype = _logical_data_default_dtype(dtype)
        return self.logical_data_full(shape, 1.0, dtype, where, exec_place, name)

    def token(self):
        return logical_data.token(self)

    def task(self, *args, symbol=None):
        """
        Create a `task`

        Example
        -------
        >>> t = ctx.task(read(lX), rw(lY), symbol="axpy")
        >>> t.start()
        >>> t.end()
        """
        exec_place_set = False
        t = task(self)          # construct with this context
        if symbol is not None:
            t.set_symbol(symbol)
        for d in args:
            if isinstance(d, dep):
                t.add_dep(d)
            elif isinstance(d, exec_place):
                if exec_place_set:
                      raise ValueError("Only one exec_place can be given")
                t.set_exec_place(d)
                exec_place_set = True
            elif hasattr(d, "_as_stf_exec_place"):
                if exec_place_set:
                    raise ValueError("Only one exec_place can be given")
                converted = d._as_stf_exec_place()
                if not isinstance(converted, exec_place):
                    raise TypeError(
                        "_as_stf_exec_place() must return a cuda.stf exec_place"
                    )
                t.set_exec_place(converted)
                exec_place_set = True
            else:
                raise TypeError(
                    "Arguments must be dependency objects or an exec_place"
                )
        return t

    def cuda_kernel(self, *args, symbol=None):
        """Create an optimized CUDA kernel task.

        Accepts the same positional dep/exec_place arguments as
        ``ctx.task()``, but the resulting object exposes a ``launch()``
        method that describes a kernel to STF directly (enabling native
        graph-kernel nodes instead of stream capture).

        Example
        -------
        >>> with ctx.cuda_kernel(lX.read(), lY.rw(), symbol="axpy") as k:
        ...     dX, dY = k.get_arg(0), k.get_arg(1)
        ...     k.launch(kernel, grid=(4,), block=(256,),
        ...              args=[ctypes.c_int(N), ctypes.c_double(alpha), dX, dY])
        """
        exec_place_set = False
        k = cuda_kernel(self)
        if symbol is not None:
            k.set_symbol(symbol)
        for d in args:
            if isinstance(d, dep):
                k.add_dep(d)
            elif isinstance(d, exec_place):
                if exec_place_set:
                    raise ValueError("Only one exec_place can be given")
                k.set_exec_place(d)
                exec_place_set = True
            elif hasattr(d, "_as_stf_exec_place"):
                if exec_place_set:
                    raise ValueError("Only one exec_place can be given")
                converted = d._as_stf_exec_place()
                if not isinstance(converted, exec_place):
                    raise TypeError(
                        "_as_stf_exec_place() must return a cuda.stf exec_place"
                    )
                k.set_exec_place(converted)
                exec_place_set = True
            else:
                raise TypeError(
                    "Arguments must be dependency objects or an exec_place"
                )
        return k

    def host_launch(self, *deps, fn, args=None, symbol=None):
        """Schedule a host callback with dependency tracking.

        Deps (positional) are auto-unpacked as numpy arrays and passed as
        the first N arguments to ``fn``.  Extra user data goes through
        ``args`` and is appended after the dep arrays.

        Example::

            ctx.host_launch(lX.read(), fn=lambda x: print(x.sum()))
            ctx.host_launch(lX.read(), lY.read(), fn=check, args=[result])
        """
        if args is None:
            user_args = ()
        else:
            user_args = tuple(args)

        cdef logical_data ldata
        dep_meta = []
        # Only the dep *type* is validated here, not its owning context. A dep
        # whose logical_data belongs to a different context is rejected later by
        # the C++ core when the host launch acquires its deps (it aborts with a
        # context-mismatch error; see cudax .../internal/acquire_release.cuh).
        for d in deps:
            if not isinstance(d, dep):
                raise TypeError(
                    "Positional arguments must be dep objects "
                    "(use ld.read(), ld.write(), or ld.rw())")
            if not isinstance(d.ld, logical_data):
                raise TypeError(
                    "host_launch deps must come from logical_data "
                    "(non-stackable context)"
                )
            ldata = <logical_data>d.ld
            dep_meta.append((ldata._shape, ldata._dtype))

        payload = (fn, user_args, dep_meta)
        Py_INCREF(payload)
        cdef PyObject* payload_ptr = <PyObject*>payload

        cdef stf_host_launch_handle h
        cdef int mode_ce
        h = stf_host_launch_create(self._ctx)
        if h == NULL:
            Py_XDECREF(<PyObject*>payload)
            raise RuntimeError("failed to create STF host_launch")
        try:
            if symbol is not None:
                sym_bytes = symbol.encode("utf-8")
                stf_host_launch_set_symbol(h, sym_bytes)
            for d in deps:
                ldata = <logical_data>d.ld
                mode_ce = <int>d.mode
                stf_host_launch_add_dep(h, ldata._ld, <stf_access_mode>mode_ce)
            stf_host_launch_set_user_data(
                h, &payload_ptr, sizeof(PyObject*), _python_payload_destructor)
            stf_host_launch_submit(h, _host_launch_trampoline)
        finally:
            stf_host_launch_destroy(h)


# ===========================================================================
# Stackable bindings (PR #8165 features ported on top of the opaque-handle
# C API). The classes below mirror the non-stackable surface, but every C
# call goes through the stf_stackable_* entry points so logical data is
# auto-pushed across nested scopes (push_graph / push_while / push_repeat).
# ===========================================================================

cdef class stackable_logical_data:
    cdef stf_logical_data_handle _ld
    cdef stf_ctx_handle _ctx

    cdef object _dtype
    cdef tuple  _shape
    cdef int    _ndim
    cdef size_t _len
    cdef str    _symbol
    cdef readonly bint _is_token
    cdef object _source_buf
    # Shared "alive" sentinel from the parent stackable_context. See
    # context._alive for the rationale.
    cdef _AliveFlag _alive

    def __cinit__(self):
        self._ld = NULL
        self._ctx = NULL
        self._len = 0
        self._dtype = None
        self._shape = ()
        self._ndim = 0
        self._symbol = None
        self._is_token = False
        self._source_buf = None
        self._alive = None

    def __dealloc__(self):
        # We must only call into STF when the parent context is still alive.
        # Note: a None _alive may be seen here even though it was set in the
        # constructor: Cython's tp_clear is called before tp_dealloc when
        # breaking reference cycles, and it resets _alive to Py_None. In that
        # case the safe action is to skip the destroy call (the parent will
        # tear down the underlying handle, or the GC will reclaim everything
        # at interpreter shutdown).
        if self._ld != NULL and self._alive is not None and self._alive.alive:
            try:
                if self._is_token:
                    stf_stackable_token_destroy(self._ld)
                else:
                    stf_stackable_logical_data_destroy(self._ld)
            except Exception as e:
                print(f"stf.stackable_logical_data: cleanup failed: {e}")
        self._ld = NULL

    def set_symbol(self, str name):
        stf_stackable_logical_data_set_symbol(self._ld, name.encode())
        self._symbol = name

    @property
    def symbol(self):
        return self._symbol

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def set_read_only(self):
        """Mark this logical data as read-only (enables concurrent reads across scopes)."""
        stf_stackable_logical_data_set_read_only(self._ld)

    def push(self, mode, data_place dplace=None):
        """Explicitly import this logical data into the current stackable scope.

        Must be called while a ``graph_scope`` / ``while_loop`` / ``repeat``
        scope is open on the parent context. By default, the first access to a
        logical data from a nested scope auto-pushes it with a conservative
        (read-write) mode, which serialises sibling scopes that only need to
        read it. Calling ``ld.push(AccessMode.READ)`` inside each sibling scope
        lets them execute concurrently without having to mark the data
        globally read-only via :meth:`set_read_only`.

        Parameters
        ----------
        mode : AccessMode or int
            Desired access mode for the data inside the current scope
            (typically ``AccessMode.READ``).
        dplace : data_place, optional
            Data placement for the imported view. ``None`` (default) uses the
            default placement.
        """
        cdef int m = int(mode)
        cdef stf_data_place_handle dh = NULL
        if dplace is not None:
            dh = dplace._h
        stf_stackable_logical_data_push(self._ld, <stf_access_mode>m, dh)

    def read(self, dplace=None):
        return dep(self, AccessMode.READ.value, dplace)

    def write(self, dplace=None):
        return dep(self, AccessMode.WRITE.value, dplace)

    def rw(self, dplace=None):
        return dep(self, AccessMode.RW.value, dplace)

    def empty_like(self):
        cdef stackable_logical_data out = stackable_logical_data.__new__(stackable_logical_data)
        out._ld = stf_stackable_logical_data_empty(self._ctx, self._len)
        if out._ld == NULL:
            raise RuntimeError("failed to create empty stackable_logical_data")
        out._ctx   = self._ctx
        out._dtype = self._dtype
        out._shape = self._shape
        out._ndim  = self._ndim
        out._len   = self._len
        out._symbol = None
        out._is_token = False
        out._alive = self._alive
        return out

    def __repr__(self):
        return (f"stackable_logical_data(shape={self._shape}, dtype={self._dtype}, "
                f"is_token={self._is_token}, symbol={self._symbol!r})")


cdef class stackable_task:
    cdef stf_task_handle _t
    cdef stf_ctx_handle _ctx
    cdef list _lds_args
    # Shared "alive" sentinel from the parent stackable_context. See
    # context._alive for the rationale.
    cdef _AliveFlag _alive

    def __cinit__(self, stackable_context ctx):
        self._t = stf_stackable_task_create(ctx._ctx)
        if self._t == NULL:
            raise RuntimeError("failed to create STF stackable task")
        self._ctx = ctx._ctx
        self._lds_args = []
        self._alive = ctx._alive

    def __dealloc__(self):
        # See stackable_logical_data.__dealloc__ for why a None _alive must
        # be treated as "context already gone" rather than "no parent".
        if self._t != NULL and self._alive is not None and self._alive.alive:
            try:
                stf_task_destroy(self._t)
            except Exception as e:
                print(f"stf.stackable_task: cleanup failed: {e}")
        self._t = NULL

    def start(self):
        stf_task_enable_capture(self._t)
        stf_task_start(self._t)

    def end(self):
        stf_task_end(self._t)

    def add_dep(self, object d):
        if not isinstance(d, dep):
            raise TypeError("add_dep expects read(ld), write(ld) or rw(ld)")
        if not isinstance(d.ld, stackable_logical_data):
            raise TypeError(
                "dep payload must be a stackable_logical_data for stackable_context.task(); "
                "did you mix stackable and non-stackable deps?"
            )

        cdef stackable_logical_data ldata = <stackable_logical_data> d.ld
        cdef int mode_int = int(d.mode)
        cdef stf_access_mode mode_ce = <stf_access_mode> mode_int
        cdef data_place dp

        # Only the dep *type* is validated here, not its owning context. A dep
        # whose logical_data belongs to a different context is rejected later by
        # the C++ core when the task acquires its deps (it aborts with a
        # context-mismatch error; see cudax .../internal/acquire_release.cuh).
        if d.dplace is None:
            stf_stackable_task_add_dep(self._ctx, self._t, ldata._ld, mode_ce)
        else:
            if not isinstance(d.dplace, data_place):
                raise TypeError("dep data_place override must be a data_place")
            dp = <data_place> d.dplace
            stf_stackable_task_add_dep_with_dplace(
                self._ctx, self._t, ldata._ld, mode_ce, dp._h)

        self._lds_args.append(ldata)

    def set_symbol(self, str name):
        stf_task_set_symbol(self._t, name.encode())

    def set_exec_place(self, object exec_p):
        if not isinstance(exec_p, exec_place):
            raise TypeError("set_exec_place expects an exec_place argument")
        cdef exec_place ep = <exec_place> exec_p
        stf_task_set_exec_place(self._t, ep._h)

    def stream_ptr(self):
        cdef CUstream s = stf_task_get_custream(self._t)
        return CudaStream(<uintptr_t>s)

    def get_arg(self, index) -> int:
        if self._lds_args[index]._is_token:
            raise RuntimeError("cannot materialize a token argument")
        cdef void *ptr = stf_task_get(self._t, index)
        return <uintptr_t>ptr

    def get_arg_cai(self, index):
        ptr = self.get_arg(index)
        return stf_cai(
            ptr, self._lds_args[index].shape, self._lds_args[index].dtype,
            stream=self.stream_ptr())

    def args_cai(self):
        non_token_cais = [self.get_arg_cai(i) for i in range(len(self._lds_args))
                          if not self._lds_args[i]._is_token]
        if len(non_token_cais) == 0:
            return None
        elif len(non_token_cais) == 1:
            return non_token_cais[0]
        return tuple(non_token_cais)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, object exc_type, object exc, object tb):
        self.end()
        return False


# Small cdef helpers so we can keep the C-typed locals out of the Python
# context-manager classes below.

cdef uintptr_t _push_while_impl(stf_ctx_handle ctx) except? 0:
    cdef stf_while_scope_handle scope = stf_stackable_push_while(ctx)
    if scope == NULL:
        raise RuntimeError("stf_stackable_push_while failed")
    return <uintptr_t>scope

cdef uint64_t _get_cond_handle_impl(uintptr_t scope_ptr):
    return stf_while_scope_get_cond_handle(<stf_while_scope_handle>scope_ptr)

cdef _pop_while_impl(uintptr_t scope_ptr):
    stf_stackable_pop_while(<stf_while_scope_handle>scope_ptr)

cdef uintptr_t _push_repeat_impl(stf_ctx_handle ctx, size_t count) except? 0:
    cdef stf_repeat_scope_handle scope = stf_stackable_push_repeat(ctx, count)
    if scope == NULL:
        raise RuntimeError("stf_stackable_push_repeat failed")
    return <uintptr_t>scope

cdef _pop_repeat_impl(uintptr_t scope_ptr):
    stf_stackable_pop_repeat(<stf_repeat_scope_handle>scope_ptr)

cdef _while_cond_scalar_impl(stf_ctx_handle ctx, uintptr_t scope_ptr,
                              stf_logical_data_handle ld,
                              int op, double threshold, int dtype_code):
    stf_stackable_while_cond_scalar(
        ctx,
        <stf_while_scope_handle>scope_ptr,
        ld,
        <stf_compare_op>op,
        threshold,
        <stf_dtype>dtype_code)


cdef uintptr_t _pop_prologue_impl(stf_ctx_handle ctx) except? 0:
    cdef stf_launchable_graph_handle h = stf_stackable_pop_prologue(ctx)
    if h == NULL:
        raise RuntimeError("stf_stackable_pop_prologue failed")
    return <uintptr_t>h

cdef _pop_epilogue_impl(stf_ctx_handle ctx):
    stf_stackable_pop_epilogue(ctx)

cdef _launchable_launch_impl(uintptr_t h):
    cdef stf_launchable_graph_handle handle = <stf_launchable_graph_handle>h
    with nogil:
        stf_launchable_graph_launch(handle)

cdef uintptr_t _launchable_exec_impl(uintptr_t h):
    return <uintptr_t>stf_launchable_graph_exec(<stf_launchable_graph_handle>h)

cdef uintptr_t _launchable_stream_impl(uintptr_t h):
    return <uintptr_t>stf_launchable_graph_stream(<stf_launchable_graph_handle>h)

cdef uintptr_t _launchable_graph_impl(uintptr_t h):
    return <uintptr_t>stf_launchable_graph_graph(<stf_launchable_graph_handle>h)

cdef _launchable_destroy_impl(uintptr_t h):
    stf_launchable_graph_destroy(<stf_launchable_graph_handle>h)


# ---- Shared-ownership flavor -----------------------------------------------

cdef uintptr_t _pop_prologue_shared_impl(stf_ctx_handle ctx) except? 0:
    cdef stf_launchable_graph_shared h = NULL
    cdef int rc = stf_stackable_pop_prologue_shared(ctx, &h)
    if rc != 0 or h == NULL:
        raise RuntimeError("stf_stackable_pop_prologue_shared failed")
    return <uintptr_t>h

cdef uintptr_t _launchable_shared_dup_impl(uintptr_t h) except? 0:
    cdef stf_launchable_graph_shared out = NULL
    cdef int rc = stf_launchable_graph_shared_dup(<stf_launchable_graph_shared>h, &out)
    if rc != 0 or out == NULL:
        raise RuntimeError("stf_launchable_graph_shared_dup failed")
    return <uintptr_t>out

cdef int _launchable_shared_valid_impl(uintptr_t h):
    return stf_launchable_graph_shared_valid(<stf_launchable_graph_shared>h)

cdef _launchable_shared_launch_impl(uintptr_t h):
    cdef stf_launchable_graph_shared handle = <stf_launchable_graph_shared>h
    with nogil:
        stf_launchable_graph_shared_launch(handle)

cdef uintptr_t _launchable_shared_exec_impl(uintptr_t h):
    return <uintptr_t>stf_launchable_graph_shared_exec(<stf_launchable_graph_shared>h)

cdef uintptr_t _launchable_shared_stream_impl(uintptr_t h):
    return <uintptr_t>stf_launchable_graph_shared_stream(<stf_launchable_graph_shared>h)

cdef uintptr_t _launchable_shared_graph_impl(uintptr_t h):
    return <uintptr_t>stf_launchable_graph_shared_graph(<stf_launchable_graph_shared>h)


cdef class LaunchableGraph:
    """Shared-ownership, storable handle for a re-launchable stackable graph.

    Returned by :py:meth:`stackable_context.pop_prologue_shared`. Unlike the
    ``launchable_graph_scope`` context manager, a :class:`LaunchableGraph`
    can be stashed as a data member, placed in a ``list``/``dict``, or
    returned from a factory function -- making it the natural fit for a
    classic "build once, launch many times, release later" graph cache.

    Each Python :class:`LaunchableGraph` holds a single C-level shared
    reference. When the last Python reference dies (via normal refcounting
    or an explicit :py:meth:`reset`) the underlying STF ``pop_epilogue``
    runs automatically.

    Examples
    --------
    Stash graphs in a dict, launch at will::

        class Engine:
            def __init__(self):
                self.ctx = stf.stackable_context()
                self.graphs = {}

            def build(self, name, n, alpha):
                self.ctx.push()
                la = self.ctx.logical_data(np.zeros(n, dtype=np.float64))
                # ... submit parallel_for / task blocks ...
                self.graphs[name] = self.ctx.pop_prologue_shared()

            def step(self, name):
                self.graphs[name].launch()

            def drop(self, name):
                del self.graphs[name]   # last ref -> pop_epilogue

    Explicit ``reset()`` semantics::

        g = ctx.pop_prologue_shared()
        h = g                        # shares the same Python object
        g.reset()                    # no-op here (same object)
        assert h.valid

    Context-manager shorthand (distinct from
    :py:meth:`stackable_context.launchable_graph_scope`: the latter also
    runs ``push()`` for you and cannot be moved or stored)::

        with ctx.pop_prologue_shared() as g:
            for _ in range(100):
                g.launch()
    """
    cdef uintptr_t _h

    def __cinit__(self):
        self._h = 0

    def __dealloc__(self):
        cdef uintptr_t h = self._h
        self._h = 0
        if h != 0:
            with nogil:
                stf_launchable_graph_shared_free(<stf_launchable_graph_shared>h)

    def reset(self):
        """Drop this shared reference eagerly.

        When this was the last live reference to the underlying graph,
        ``stf_stackable_pop_epilogue`` runs now instead of at destruction
        time. Subsequent accessors / :py:meth:`launch` raise.
        Idempotent.
        """
        cdef uintptr_t h = self._h
        self._h = 0
        if h != 0:
            with nogil:
                stf_launchable_graph_shared_free(<stf_launchable_graph_shared>h)

    def _check_valid(self):
        if self._h == 0:
            raise RuntimeError("LaunchableGraph has been reset")

    @property
    def valid(self) -> bool:
        """True iff this handle still refers to a live graph.

        Returns ``False`` after :py:meth:`reset`, or after some other code
        path (e.g. a manual ``ctx.pop_epilogue()`` behind STF's back) has
        released the underlying state.
        """
        if self._h == 0:
            return False
        return bool(_launchable_shared_valid_impl(self._h))

    def launch(self):
        """Launch the graph once on its support stream."""
        self._check_valid()
        _launchable_shared_launch_impl(self._h)

    @property
    def exec_graph(self) -> int:
        """Raw ``cudaGraphExec_t`` as a plain Python ``int``."""
        self._check_valid()
        return _launchable_shared_exec_impl(self._h)

    @property
    def stream(self) -> int:
        """Raw ``cudaStream_t`` as a plain Python ``int``."""
        self._check_valid()
        return _launchable_shared_stream_impl(self._h)

    @property
    def graph(self) -> int:
        """Raw (non-executable) ``cudaGraph_t`` as a plain Python ``int``.

        Intended for embedding the nested graph as a child node into another
        graph (``cudaGraphAddChildGraphNode``). Unlike :py:attr:`exec_graph`,
        this property does NOT force ``cudaGraphInstantiate``.
        """
        self._check_valid()
        return _launchable_shared_graph_impl(self._h)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()
        return False


class _GraphScope:
    """Context manager wrapping ``stf_stackable_push_graph`` / ``_pop``."""
    def __init__(self, ctx):
        self._ctx = ctx

    def __enter__(self):
        stf_stackable_push_graph((<stackable_context>self._ctx)._ctx)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stf_stackable_pop((<stackable_context>self._ctx)._ctx)
        return False


class _LaunchableGraphScope:
    """Context manager exposing the re-launchable ``pop_prologue`` API.

    On ``__enter__`` pushes a graph scope. ``stf_stackable_pop_prologue`` is
    called lazily on the first call to any of :py:meth:`launch`,
    :py:attr:`exec_graph`, :py:attr:`stream` or :py:attr:`graph`; that step
    only finalizes the nested ``cudaGraph_t``. Actual
    ``cudaGraphInstantiate`` is deferred until :py:meth:`launch` or
    :py:attr:`exec_graph` is used, so callers that only want the graph
    topology (via :py:attr:`graph`) pay no instantiation cost. ``__exit__``
    always runs ``stf_stackable_pop_epilogue`` so that the context unfreezes
    cleanly even when the user never launched the graph.

    Usage::

        with ctx.launchable_graph_scope() as scope:
            ctx.parallel_for(...)
            for _ in range(N):
                scope.launch()
    """
    def __init__(self, ctx):
        self._ctx = ctx
        self._h = 0

    def __enter__(self):
        stf_stackable_push_graph((<stackable_context>self._ctx)._ctx)
        return self

    def _ensure_prepared(self):
        if self._h == 0:
            self._h = _pop_prologue_impl((<stackable_context>self._ctx)._ctx)

    def launch(self):
        """Launch the instantiated graph once on its support stream."""
        self._ensure_prepared()
        _launchable_launch_impl(self._h)

    @property
    def exec_graph(self) -> int:
        """Raw ``cudaGraphExec_t`` as a plain Python ``int``."""
        self._ensure_prepared()
        return _launchable_exec_impl(self._h)

    @property
    def stream(self) -> int:
        """Raw ``cudaStream_t`` as a plain Python ``int``."""
        self._ensure_prepared()
        return _launchable_stream_impl(self._h)

    @property
    def graph(self) -> int:
        """Raw (non-executable) ``cudaGraph_t`` as a plain Python ``int``.

        Intended for embedding the nested graph as a child node into another
        graph (``cudaGraphAddChildGraphNode``). Unlike :py:attr:`exec_graph`,
        this property does NOT force ``cudaGraphInstantiate``. The graph
        stays valid only until the scope's ``__exit__`` runs; clone it with
        ``cudaGraphClone`` if you need a longer lifetime.
        """
        self._ensure_prepared()
        return _launchable_graph_impl(self._h)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If the user never touched launch/exec/stream we still need to run
        # the prologue+epilogue pair so that data pushed in the scope gets
        # unfrozen (matches the default ``pop()`` semantics).
        if self._h == 0:
            self._h = _pop_prologue_impl((<stackable_context>self._ctx)._ctx)
        try:
            _pop_epilogue_impl((<stackable_context>self._ctx)._ctx)
        finally:
            _launchable_destroy_impl(self._h)
            self._h = 0
        return False


class _WhileLoop:
    """Context manager for a CUDA 12.4+ conditional while loop."""
    def __init__(self, ctx):
        self._ctx = ctx
        self._scope = 0
        self._cond_handle = 0

    def __enter__(self):
        self._scope = _push_while_impl((<stackable_context>self._ctx)._ctx)
        self._cond_handle = _get_cond_handle_impl(self._scope)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_while_impl(self._scope)
        return False

    @property
    def cond_handle(self):
        """Raw ``cudaGraphConditionalHandle`` as ``uint64_t`` (custom kernels)."""
        return self._cond_handle

    def continue_while(self, *args):
        """Set a built-in ``continue while (ld <op> threshold)`` condition.

        Usage: ``loop.continue_while(ld, ">", threshold)``
        """
        if len(args) != 3:
            raise ValueError(
                "continue_while expects (logical_data, op_string, threshold)")
        ld_obj, op_str, threshold = args
        self._set_scalar_condition(ld_obj, op_str, float(threshold))

    def _set_scalar_condition(self, ld_obj, str op_str, double threshold):
        cdef int op
        cdef int dtype_code
        if op_str == ">":
            op = <int>STF_CMP_GT
        elif op_str == "<":
            op = <int>STF_CMP_LT
        elif op_str == ">=":
            op = <int>STF_CMP_GE
        elif op_str == "<=":
            op = <int>STF_CMP_LE
        else:
            raise ValueError(f"Unsupported comparison operator: {op_str}")

        dt = ld_obj.dtype
        if dt == np.float32:
            dtype_code = <int>STF_DTYPE_FLOAT32
        elif dt == np.float64:
            dtype_code = <int>STF_DTYPE_FLOAT64
        elif dt == np.int32:
            dtype_code = <int>STF_DTYPE_INT32
        elif dt == np.int64:
            dtype_code = <int>STF_DTYPE_INT64
        else:
            raise ValueError(f"Unsupported dtype for while condition: {dt}")

        _while_cond_scalar_impl(
            (<stackable_context>self._ctx)._ctx,
            self._scope,
            (<stackable_logical_data>ld_obj)._ld,
            op,
            threshold,
            dtype_code)

    def condition_task(self, *args):
        """Return a ``stackable_task`` for manual condition setting (advanced)."""
        return self._ctx.task(*args)


class _RepeatScope:
    """Context manager for a fixed-iteration repeat scope (CUDA 12.4+)."""
    def __init__(self, ctx, count):
        self._ctx = ctx
        self._count = count
        self._scope = 0

    def __enter__(self):
        self._scope = _push_repeat_impl(
            (<stackable_context>self._ctx)._ctx, self._count)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_repeat_impl(self._scope)
        return False


cdef class stackable_context:
    cdef stf_ctx_handle _ctx
    cdef _PrimaryContextPin _pin
    # Shared "alive" sentinel. See context._alive for the rationale.
    cdef _AliveFlag _alive

    def __cinit__(self):
        cdef stf_ctx_handle h

        self._pin = None
        self._pin = _PrimaryContextPin()
        self._ctx = stf_stackable_ctx_create()
        if self._ctx == NULL:
            self._pin.release()
            self._pin = None
            raise RuntimeError("failed to create STF stackable context")
        self._alive = _AliveFlag()

    def __dealloc__(self):
        if self._ctx != NULL:
            if self._alive is not None:
                self._alive.alive = False
            try:
                warnings.warn(
                    "cuda.stf._experimental.stackable_context was garbage-collected without an explicit finalize(); "
                    "STF/CUDA resources were abandoned. Call finalize() explicitly or use "
                    "'with cuda.stf._experimental.stackable_context() as ctx:'.",
                    ResourceWarning,
                )
            except Exception:
                pass
            self._ctx = <stf_ctx_handle>NULL

    def __repr__(self):
        return f"stackable_context(handle={<uintptr_t>self._ctx})"

    def finalize(self):
        cdef _PrimaryContextPin pin = self._pin

        # Flip the shared sentinel first so every surviving child wrapper
        # turns its __dealloc__ into a no-op. Idempotent.
        if self._alive is not None:
            self._alive.alive = False

        cdef stf_ctx_handle h = self._ctx
        self._pin = None
        self._ctx = NULL
        if h != NULL:
            with nogil:
                stf_stackable_ctx_finalize(h)

        if pin is not None:
            pin.release()

    def __enter__(self):
        return self

    def __exit__(self, object exc_type, object exc, object tb):
        self.finalize()
        return False

    def fence(self):
        """Return the fence CUDA stream as a Python int. Must be at root level."""
        if self._ctx == NULL:
            raise RuntimeError("stackable_context handle is NULL")
        cdef CUstream s
        with nogil:
            s = stf_stackable_ctx_fence(self._ctx)
        return <uintptr_t>s

    def logical_data(self, object buf, data_place dplace=None, str name=None):
        """Create stackable logical data from an existing buffer."""
        cdef stackable_logical_data out = stackable_logical_data.__new__(stackable_logical_data)
        out._ctx = self._ctx
        out._alive = self._alive
        out._source_buf = buf
        cdef Py_buffer view
        cdef int flags

        if dplace is None:
            dplace = data_place.host()

        if hasattr(buf, '__cuda_array_interface__'):
            cai = buf.__cuda_array_interface__
            data_ptr, readonly = cai['data']
            original_shape = cai['shape']
            typestr = cai['typestr']
            if typestr.startswith('|V') and 'descr' in cai:
                out._dtype = np.dtype(cai['descr'])
            else:
                out._dtype = np.dtype(typestr)
            out._shape = original_shape
            out._ndim = len(out._shape)
            itemsize = out._dtype.itemsize
            total_items = 1
            for dim in out._shape:
                total_items *= dim
            out._len = total_items * itemsize
            out._ld = stf_stackable_logical_data_with_place(
                self._ctx, <void*><uintptr_t>data_ptr, out._len, dplace._h)
        else:
            flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_ANY_CONTIGUOUS
            if PyObject_GetBuffer(buf, &view, flags) != 0:
                raise ValueError(
                    "object doesn't support the buffer protocol, is not contiguous, "
                    "or doesn't expose __cuda_array_interface__")
            try:
                out._ndim = view.ndim
                out._len = view.len
                out._shape = tuple(<Py_ssize_t>view.shape[i] for i in range(view.ndim))
                out._dtype = np.dtype(view.format)
                out._ld = stf_stackable_logical_data_with_place(
                    self._ctx, view.buf, view.len, dplace._h)
            finally:
                PyBuffer_Release(&view)

        if out._ld == NULL:
            raise RuntimeError("failed to create stackable_logical_data")

        if name is not None:
            out.set_symbol(name)
        return out

    def logical_data_empty(self, shape, dtype=None, str name=None, *, bint no_export=False):
        """Create stackable logical data with uninitialized values.

        If ``no_export=True``, the logical data is local to the current
        stackable scope (head context) and is not exported to parent scopes.
        Useful for temporaries inside ``while_loop`` / ``repeat_scope`` bodies
        so each iteration gets its own buffer instead of reusing one that
        escapes into the enclosing graph.
        """
        if dtype is None:
            dtype = np.float64

        cdef stackable_logical_data out = stackable_logical_data.__new__(stackable_logical_data)
        out._ctx = self._ctx
        out._alive = self._alive
        out._dtype = np.dtype(dtype)
        out._shape = tuple(shape) if not isinstance(shape, tuple) else shape
        out._ndim = len(out._shape)
        cdef size_t total_items = 1
        for dim in out._shape:
            total_items *= dim
        out._len = total_items * out._dtype.itemsize
        if no_export:
            out._ld = stf_stackable_logical_data_no_export_empty(self._ctx, out._len)
        else:
            out._ld = stf_stackable_logical_data_empty(self._ctx, out._len)
        if out._ld == NULL:
            raise RuntimeError("failed to create empty stackable_logical_data")

        if name is not None:
            out.set_symbol(name)
        return out

    def logical_data_full(
        self,
        shape,
        fill_value,
        dtype=None,
        where=None,
        exec_place=None,
        str name=None,
        *,
        bint no_export=False,
    ):
        """Create stackable logical data initialized with a constant value.

        This mirrors :meth:`context.logical_data_full` for stackable contexts.
        The allocation is created as stackable logical data, then initialized
        by an STF task in the current stackable scope. If ``no_export=True``,
        the logical data remains local to the head scope.
        """
        return _logical_data_full(self, shape, fill_value, dtype, where, exec_place, name, no_export=no_export)

    def logical_data_zeros(self, shape, dtype=None, where=None, exec_place=None, str name=None, *, bint no_export=False):
        """Create stackable logical data filled with zeros."""
        dtype = _logical_data_default_dtype(dtype)
        return self.logical_data_full(shape, 0.0, dtype, where, exec_place, name, no_export=no_export)

    def logical_data_ones(self, shape, dtype=None, where=None, exec_place=None, str name=None, *, bint no_export=False):
        """Create stackable logical data filled with ones."""
        dtype = _logical_data_default_dtype(dtype)
        return self.logical_data_full(shape, 1.0, dtype, where, exec_place, name, no_export=no_export)

    def token(self):
        """Create a synchronization token."""
        cdef stackable_logical_data out = stackable_logical_data.__new__(stackable_logical_data)
        out._ctx = self._ctx
        out._alive = self._alive
        out._dtype = None
        out._shape = None
        out._ndim = 0
        out._len = 0
        out._is_token = True
        out._ld = stf_stackable_token(self._ctx)
        if out._ld == NULL:
            raise RuntimeError("failed to create stackable token")
        return out

    def task(self, *args, symbol=None):
        """Create a task on the head (innermost) scope of this context."""
        exec_place_set = False
        t = stackable_task(self)
        if symbol is not None:
            t.set_symbol(symbol)
        for d in args:
            if isinstance(d, dep):
                t.add_dep(d)
            elif isinstance(d, exec_place):
                if exec_place_set:
                    raise ValueError("Only one exec_place can be given")
                t.set_exec_place(d)
                exec_place_set = True
            elif hasattr(d, "_as_stf_exec_place"):
                if exec_place_set:
                    raise ValueError("Only one exec_place can be given")
                converted = d._as_stf_exec_place()
                if not isinstance(converted, exec_place):
                    raise TypeError("_as_stf_exec_place() must return a cuda.stf exec_place")
                t.set_exec_place(converted)
                exec_place_set = True
            else:
                raise TypeError("Arguments must be dependency objects or an exec_place")
        return t

    def graph_scope(self):
        """Return a context manager that pushes/pops a nested graph scope."""
        return _GraphScope(self)

    def push(self):
        """Push a nested graph scope (decoupled from :meth:`pop`).

        Prefer :meth:`graph_scope` (RAII) whenever possible. This raw form
        only exists so that callers who want to build a graph and return a
        :class:`LaunchableGraph` from :meth:`pop_prologue_shared` can
        decouple the push from the final release.
        """
        stf_stackable_push_graph(self._ctx)

    def pop(self):
        """Pop the innermost graph scope (matches an unmatched :meth:`push`)."""
        stf_stackable_pop(self._ctx)

    def launchable_graph_scope(self):
        """Return a context manager exposing the re-launchable graph API.

        The returned scope behaves like :meth:`graph_scope` but instantiates
        the nested graph into a reusable ``cudaGraphExec_t`` that can be
        launched one or more times via :py:meth:`_LaunchableGraphScope.launch`
        (or directly via :py:attr:`_LaunchableGraphScope.exec_graph` /
        :py:attr:`_LaunchableGraphScope.stream`) before the scope exits.
        """
        return _LaunchableGraphScope(self)

    def pop_prologue_shared(self) -> LaunchableGraph:
        """Shared-ownership flavor of ``pop_prologue``.

        Runs the same prologue as :meth:`pop` on the innermost graph scope,
        but returns a :class:`LaunchableGraph` whose destructor runs
        ``pop_epilogue`` when the last shared reference dies. Use this when
        you want to **build a graph, store it** (as a data member, in a
        ``list`` / ``dict`` or returned across function boundaries) and
        **launch it many times before releasing**. For a purely lexical
        scope, prefer :meth:`launchable_graph_scope` which also handles the
        matching ``push`` for you.

        Usage::

            self.ctx.push()
            # ... submit tasks ...
            self.step_graph = self.ctx.pop_prologue_shared()

            for _ in range(1000):
                self.step_graph.launch()   # outside the originating scope
            # self.step_graph drops its last ref (or is explicitly .reset())
            # -> pop_epilogue runs automatically.
        """
        cdef LaunchableGraph g = LaunchableGraph.__new__(LaunchableGraph)
        g._h = _pop_prologue_shared_impl(self._ctx)
        return g

    def while_loop(self):
        """Return a context manager for a while loop (CUDA 12.4+)."""
        return _WhileLoop(self)

    def repeat(self, size_t count):
        """Return a context manager that repeats the body ``count`` times (CUDA 12.4+)."""
        return _RepeatScope(self, count)

    def host_launch(self, *deps, fn, args=None, symbol=None):
        """Schedule a host callback inside the current stackable scope.

        Mirrors :meth:`context.host_launch` but auto-pushes stackable
        logical data through ``stf_stackable_host_launch_add_dep``.
        """
        if args is None:
            user_args = ()
        else:
            user_args = tuple(args)

        cdef stackable_logical_data sldata
        dep_meta = []
        # Only the dep *type* is validated here, not its owning context. A dep
        # whose logical_data belongs to a different context is rejected later by
        # the C++ core when the host launch acquires its deps (it aborts with a
        # context-mismatch error; see cudax .../internal/acquire_release.cuh).
        for d in deps:
            if not isinstance(d, dep):
                raise TypeError(
                    "Positional arguments must be dep objects "
                    "(use ld.read(), ld.write(), or ld.rw())")
            if not isinstance(d.ld, stackable_logical_data):
                raise TypeError(
                    "host_launch deps must come from stackable_logical_data "
                    "(stackable_context)"
                )
            sldata = <stackable_logical_data>d.ld
            dep_meta.append((sldata._shape, sldata._dtype))

        payload = (fn, user_args, dep_meta)
        Py_INCREF(payload)
        cdef PyObject* payload_ptr = <PyObject*>payload

        cdef stf_host_launch_handle h = stf_stackable_host_launch_create(self._ctx)
        if h == NULL:
            Py_XDECREF(<PyObject*>payload)
            raise RuntimeError("failed to create stackable host_launch")

        cdef int mode_ce
        try:
            if symbol is not None:
                sym_bytes = symbol.encode("utf-8")
                stf_host_launch_set_symbol(h, sym_bytes)
            for d in deps:
                sldata = <stackable_logical_data>d.ld
                mode_ce = <int>d.mode
                stf_stackable_host_launch_add_dep(
                    self._ctx, h, sldata._ld, <stf_access_mode>mode_ce)
            stf_host_launch_set_user_data(
                h, &payload_ptr, sizeof(PyObject*), _python_payload_destructor)
            stf_stackable_host_launch_submit(h, _host_launch_trampoline)
        finally:
            stf_stackable_host_launch_destroy(h)
