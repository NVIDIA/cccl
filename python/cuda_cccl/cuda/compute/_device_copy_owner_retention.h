// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CUDA_COMPUTE_DEVICE_COPY_OWNER_RETENTION_H
#define CUDA_COMPUTE_DEVICE_COPY_OWNER_RETENTION_H

#include <cuda.h>
#include <Python.h>

#if defined(__cplusplus)
#  define CUDA_COMPUTE_DEVICE_COPY_NOEXCEPT noexcept
extern "C" {
#else
#  define CUDA_COMPUTE_DEVICE_COPY_NOEXCEPT
#endif

typedef struct cccl_device_copy_owner_retention cccl_device_copy_owner_retention;

cccl_device_copy_owner_retention*
cccl_device_copy_create_owner_retention(PyObject* source, PyObject* destination) CUDA_COMPUTE_DEVICE_COPY_NOEXCEPT;

void cccl_device_copy_release_owner_retention(cccl_device_copy_owner_retention* owners)
  CUDA_COMPUTE_DEVICE_COPY_NOEXCEPT;

CUresult cccl_device_copy_schedule_owner_release(CUstream stream, cccl_device_copy_owner_retention* owners)
  CUDA_COMPUTE_DEVICE_COPY_NOEXCEPT;

Py_ssize_t cccl_device_copy_drain_completed_owners_impl(void) CUDA_COMPUTE_DEVICE_COPY_NOEXCEPT;

#if defined(__cplusplus)
}
#endif

#undef CUDA_COMPUTE_DEVICE_COPY_NOEXCEPT

#endif // CUDA_COMPUTE_DEVICE_COPY_OWNER_RETENTION_H
