//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <format>
#include <string>

#include <cccl/c/radix_sort.h>
#include <hostjit/codegen/types.hpp>
#include <hostjit/jit_compiler.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

static bool is_null_it(cccl_iterator_t it)
{
  return it.type == CCCL_POINTER && it.state == nullptr;
}

static bool is_null_op(cccl_op_t op)
{
  return op.name == nullptr || op.name[0] == '\0';
}

// ---------------------------------------------------------------------------
// JIT source generation
// ---------------------------------------------------------------------------
// For keys-only sort, the JIT function takes:
//   (temp, bytes, keys_in, keys_out, num_items, begin_bit, end_bit, selector_out, stream)
// For pairs sort, the JIT function takes:
//   (temp, bytes, keys_in, keys_out, values_in, values_out, num_items, begin_bit, end_bit, selector_out, stream)
//
// The copy-based (non-DoubleBuffer) CUB API is used. The result is always in
// the *_out buffer (selector=0 from the caller's perspective).
// is_overwrite_okay is accepted by the C wrapper but ignored on this path.
//
// Decomposer: only identity (null decomposer) is supported.

static const char* k_export_macro = R"(
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
)";

static std::string make_keys_only_source(const std::string& key_type, bool ascending)
{
  return std::format(
    R"SRC(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/device/device_radix_sort.cuh>
{0}
extern "C" EXPORT int cccl_jit_radix_sort(
    void* d_temp_storage, size_t* temp_storage_bytes,
    void* d_keys_in_ptr, void* d_keys_out_ptr,
    unsigned long long num_items,
    int begin_bit, int end_bit,
    void* stream)
{{
    using key_t = {1};
    cudaError_t err = cub::DeviceRadixSort::{2}(
        d_temp_storage, *temp_storage_bytes,
        static_cast<const key_t*>(d_keys_in_ptr),
        static_cast<key_t*>(d_keys_out_ptr),
        static_cast<unsigned long long>(num_items),
        begin_bit, end_bit,
        static_cast<cudaStream_t>(stream));
    return static_cast<int>(err);
}}
)SRC",
    k_export_macro,
    key_type,
    ascending ? "SortKeys" : "SortKeysDescending");
}

static std::string make_pairs_source(const std::string& key_type, const std::string& value_type, bool ascending)
{
  return std::format(
    R"SRC(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/device/device_radix_sort.cuh>
{0}
extern "C" EXPORT int cccl_jit_radix_sort(
    void* d_temp_storage, size_t* temp_storage_bytes,
    void* d_keys_in_ptr, void* d_keys_out_ptr,
    void* d_values_in_ptr, void* d_values_out_ptr,
    unsigned long long num_items,
    int begin_bit, int end_bit,
    void* stream)
{{
    using key_t   = {1};
    using value_t = {2};
    cudaError_t err = cub::DeviceRadixSort::{3}(
        d_temp_storage, *temp_storage_bytes,
        static_cast<const key_t*>(d_keys_in_ptr),
        static_cast<key_t*>(d_keys_out_ptr),
        static_cast<const value_t*>(d_values_in_ptr),
        static_cast<value_t*>(d_values_out_ptr),
        static_cast<unsigned long long>(num_items),
        begin_bit, end_bit,
        static_cast<cudaStream_t>(stream));
    return static_cast<int>(err);
}}
)SRC",
    k_export_macro,
    key_type,
    value_type,
    ascending ? "SortPairs" : "SortPairsDescending");
}

// ---------------------------------------------------------------------------
// Runtime function typedefs
// ---------------------------------------------------------------------------

// Keys-only: (temp, bytes, keys_in, keys_out, num_items, begin_bit, end_bit, stream)
using radix_sort_keys_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, int, int, void*);

// Pairs: (temp, bytes, keys_in, keys_out, values_in, values_out, num_items, begin_bit, end_bit, stream)
using radix_sort_pairs_fn_t = int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, int, int, void*);

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_radix_sort_build_ex(
  cccl_device_radix_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* /*decomposer_return_type*/,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (!is_null_op(decomposer))
  {
    fprintf(stderr,
            "\nERROR in cccl_device_radix_sort_build(): custom radix decomposers are not supported "
            "in the ClangJIT path. Use standard integer/float key types.\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const bool keys_only = is_null_it(input_values_it);
  const bool ascending = (sort_order == CCCL_ASCENDING);

  std::string key_type = get_type_name(input_keys_it.value_type.type);
  if (key_type.empty())
  {
    fprintf(stderr, "\nERROR in cccl_device_radix_sort_build(): unsupported key type\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string source;
  if (keys_only)
  {
    source = make_keys_only_source(key_type, ascending);
  }
  else
  {
    std::string value_type = get_type_name(input_values_it.value_type.type);
    if (value_type.empty())
    {
      fprintf(stderr, "\nERROR in cccl_device_radix_sort_build(): unsupported value type\n");
      return CUDA_ERROR_UNKNOWN;
    }
    source = make_pairs_source(key_type, value_type, ascending);
  }

  auto jit = cccl::detail::compile_jit_source(
    source, "cccl_jit_radix_sort", cc_major, cc_minor, ctk_root, cccl_include_path, merged.get());
  if (!jit.compiler)
  {
    return CUDA_ERROR_UNKNOWN;
  }

  build_ptr->cc           = cc_major * 10 + cc_minor;
  build_ptr->cubin        = cccl::detail::copy_cubin(jit.cubin, &build_ptr->cubin_size);
  build_ptr->jit_compiler = jit.compiler.release();
  build_ptr->sort_fn      = jit.fn_ptr;
  build_ptr->key_type     = input_keys_it.value_type;
  build_ptr->value_type   = input_values_it.value_type;
  build_ptr->order        = sort_order;
  build_ptr->keys_only    = keys_only ? 1 : 0;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_build(
  cccl_device_radix_sort_build_result_t* build,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* decomposer_return_type,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_radix_sort_build_ex(
    build,
    sort_order,
    input_keys_it,
    input_values_it,
    decomposer,
    decomposer_return_type,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

// ---------------------------------------------------------------------------
// Run
// The JIT function uses the copy-based CUB API so the result is always in the
// *_out buffers. selector is always set to 0. is_overwrite_okay is accepted
// but ignored. decomposer is accepted but must be null (identity).
// ---------------------------------------------------------------------------

CUresult cccl_device_radix_sort(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t /*decomposer*/,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  try
  {
    if (!build.sort_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    int status;
    if (build.keys_only)
    {
      auto fn = reinterpret_cast<radix_sort_keys_fn_t>(build.sort_fn);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.state,
        d_keys_out.state,
        static_cast<unsigned long long>(num_items),
        begin_bit,
        end_bit,
        reinterpret_cast<void*>(stream));
    }
    else
    {
      auto fn = reinterpret_cast<radix_sort_pairs_fn_t>(build.sort_fn);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.state,
        d_keys_out.state,
        d_values_in.state,
        d_values_out.state,
        static_cast<unsigned long long>(num_items),
        begin_bit,
        end_bit,
        reinterpret_cast<void*>(stream));
    }

    if (selector)
    {
      // Copy variant always writes to d_keys_out (= d_buffers[1] in DoubleBuffer mode).
      // When is_overwrite_okay (DoubleBuffer mode), the caller interprets selector as an
      // index into d_buffers, so 1 means "result is in the other/output buffer".
      *selector = is_overwrite_okay ? 1 : 0;
    }

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_radix_sort_cleanup(cccl_device_radix_sort_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (build_ptr->jit_compiler)
  {
    delete static_cast<hostjit::JITCompiler*>(build_ptr->jit_compiler);
    build_ptr->jit_compiler = nullptr;
  }
  if (build_ptr->cubin)
  {
    delete[] static_cast<char*>(build_ptr->cubin);
    build_ptr->cubin = nullptr;
  }
  build_ptr->cubin_size = 0;
  build_ptr->sort_fn    = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
