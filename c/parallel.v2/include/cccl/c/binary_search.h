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
  cccl_build_config* config);

// Compiles and links the build result, without loading it. build->jit_compiler
// and build->binary_search_fn remain null until cccl_device_binary_search_load
// is called.
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
  cccl_build_config* config);

// Loads a build_result populated by cccl_device_binary_search_compile or
// cccl_device_binary_search_deserialize, populating build->jit_compiler and
// build->binary_search_fn. ctk_path locates the CUDA Toolkit on this
// machine; pass NULL to auto-detect. Call at most once per build_result.
CCCL_C_API CUresult
cccl_device_binary_search_load(cccl_device_binary_search_build_result_t* build, const char* ctk_path);

// Serializes a populated build_result into a self-describing byte buffer.
// On success *out_buf points to a heap allocation the caller must free
// with cccl_serialization_v2_buffer_free, and *out_size holds its length.
// Requires build->payload to be populated.
CCCL_C_API CUresult cccl_device_binary_search_serialize(
  const cccl_device_binary_search_build_result_t* build, void** out_buf, size_t* out_size);

// Reconstructs a build_result from a buffer produced by
// cccl_device_binary_search_serialize. On success, build->jit_compiler and
// build->binary_search_fn remain null until cccl_device_binary_search_load
// is called; on failure build is left unchanged. Rejects a blob built for a
// different OS/architecture than this machine; does not check compute
// capability against a live device (a mismatch instead surfaces later, at
// cccl_device_binary_search).
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
