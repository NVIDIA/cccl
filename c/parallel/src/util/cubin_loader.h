//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

// C-compatible declaration of the generic cubin loader.
// The C++ implementation lives in reduce.cu (uses new char[] / delete[] to
// match the allocator expected by the cleanup functions).
// This header intentionally contains no C++ includes so that Cython-generated
// C code can safely include it.

#include <cuda.h>
#include <stddef.h>

#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

// Load cubin bytes and extract N kernels by lowered (mangled) name.
//
// On success:
//   - A heap copy of the cubin (allocated with new char[]) is stored in *cubin_copy_out.
//     The caller is responsible for freeing it (via delete[]).
//   - The loaded CUlibrary handle is stored in *library_out.
//   - kernel_handles[i] is set to the CUkernel for kernel_names[i].
//
// Optional kernels: if kernel_names[i] is nullptr, kernel_handles[i] is set to
// nullptr and no cuLibraryGetKernel call is made for that slot.
//
// On failure (non-CUDA_SUCCESS return):
//   - *cubin_copy_out is set to nullptr (nothing to free).
//   - *library_out is set to nullptr (nothing to unload).
CCCL_C_API CUresult cccl_load_cubin_and_get_kernels(
  const void* cubin_in,
  size_t cubin_size,
  void** cubin_copy_out,
  CUlibrary* library_out,
  const char** kernel_names,
  CUkernel* kernel_handles,
  int num_kernels);

CCCL_C_EXTERN_C_END

#ifdef __cplusplus
#  include "errors.h"

// Try to load *cubin into *library and resolve *first_kernel by lowered name.
// If the cubin targets a different CC than the current device
// (CUDA_ERROR_NO_BINARY_FOR_GPU from cuLibraryLoadData or cuLibraryGetKernel),
// leaves all handles null and returns false — the caller stores the cubin for
// AoT cross-CC use.  On any other error, throws via check().
inline bool
cccl_try_load_for_device(CUlibrary* library, const void* cubin, CUkernel* first_kernel, const char* first_kernel_name)
{
  *library      = nullptr;
  *first_kernel = nullptr;

  const CUresult load_status = cuLibraryLoadData(library, cubin, nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (load_status == CUDA_ERROR_NO_BINARY_FOR_GPU)
  {
    return false;
  }
  check(load_status);

  const CUresult k_status = cuLibraryGetKernel(first_kernel, *library, first_kernel_name);
  if (k_status == CUDA_ERROR_NO_BINARY_FOR_GPU)
  {
    cuLibraryUnload(*library);
    *library      = nullptr;
    *first_kernel = nullptr;
    return false;
  }
  check(k_status);
  return true;
}
#endif // __cplusplus
