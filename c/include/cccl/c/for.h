//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef CCCL_C_EXPERIMENTAL
#  warning "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this warning."
#else // ^^^ !CCCL_C_EXPERIMENTAL ^^^ / vvv CCCL_C_EXPERIMENTAL vvv

#  include <cuda.h>

#  include <cccl/c/types.h>

struct cccl_device_for_build_result_t
{
  int cc;
  void* cubin;
  size_t cubin_size;
  CUlibrary library;
  CUkernel static_kernel;
};

extern "C" CCCL_C_API CUresult cccl_device_for_build(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path) noexcept;

extern "C" CCCL_C_API CUresult cccl_device_for(
  cccl_device_for_build_result_t build,
  cccl_iterator_t d_data,
  int64_t num_items,
  cccl_op_t op,
  CUstream stream) noexcept;

extern "C" CCCL_C_API CUresult cccl_device_for_cleanup(cccl_device_for_build_result_t* bld_ptr);

#endif // CCCL_C_EXPERIMENTAL
