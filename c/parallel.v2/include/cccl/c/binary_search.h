//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once
// NOLINTBEGIN(modernize-use-using)

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>
#include <stdint.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

typedef struct cccl_device_binary_search_build_result_t
{
  int cc;
  int cudart_version; // CUDART_VERSION of the toolkit the payload was built against (0 = unknown)
  void* payload;
  size_t payload_size;
  void* jit_compiler; // hostjit::JITCompiler*
#if defined(_WIN32)
  // Opaque state for serializing CUB's lazy first-call initialization.
  void* first_call_state;
#endif // _WIN32
  void* binary_search_fn; // int(*)(void*, ull, void*, ull, void*, void*, void*)
} cccl_device_binary_search_build_result_t;

CCCL_C_API CUresult cccl_device_binary_search_build(
  cccl_device_binary_search_build_result_t* build,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_binary_search_build_ex(
  cccl_device_binary_search_build_result_t* build,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  const cccl_build_config* config);

// compile / load / serialize / deserialize: see the AoT contract in serialization.h.
CCCL_C_API CUresult cccl_device_binary_search_compile(
  cccl_device_binary_search_build_result_t* build,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  const cccl_build_config* config);

CCCL_C_API CUresult
cccl_device_binary_search_load(cccl_device_binary_search_build_result_t* build, const char* cudart_path);

CCCL_C_API CUresult cccl_device_binary_search_serialize(
  const cccl_device_binary_search_build_result_t* build, void** out_buf, size_t* out_size);

CCCL_C_API CUresult
cccl_device_binary_search_deserialize(cccl_device_binary_search_build_result_t* build, const void* buf, size_t size);

CCCL_C_API CUresult cccl_device_binary_search(
  cccl_device_binary_search_build_result_t build,
  cccl_iterator_t d_data,
  uint64_t num_items,
  cccl_iterator_t d_values,
  uint64_t num_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  CUstream stream);

CCCL_C_API CUresult cccl_device_binary_search_cleanup(cccl_device_binary_search_build_result_t* bld_ptr);

CCCL_C_EXTERN_C_END
// NOLINTEND(modernize-use-using)
