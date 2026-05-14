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

#include <cccl/c/segmented_sort.h>
#include <hostjit/codegen/types.hpp>
#include <hostjit/jit_compiler.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

static bool is_null_it(cccl_iterator_t it)
{
  return it.type == CCCL_POINTER && it.state == nullptr;
}

// ---------------------------------------------------------------------------
// JIT source generation
// ---------------------------------------------------------------------------
// Note: offset iterators must be raw device pointers to long long.
// The copy-only CUB API is used, so is_overwrite_okay has no effect and
// the result is always in d_keys_out / d_values_out (selector=0).

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
#include <cub/device/device_segmented_sort.cuh>
{0}
extern "C" EXPORT int cccl_jit_segmented_sort(
    void* d_temp_storage, size_t* temp_storage_bytes,
    void* d_keys_in_ptr, void* d_keys_out_ptr,
    unsigned long long num_items, unsigned long long num_segments,
    const long long* d_begin_offsets, const long long* d_end_offsets,
    void* stream)
{{
    using key_t = {1};
    cudaError_t err = cub::DeviceSegmentedSort::{2}(
        d_temp_storage, *temp_storage_bytes,
        static_cast<const key_t*>(d_keys_in_ptr),
        static_cast<key_t*>(d_keys_out_ptr),
        static_cast<long long>(num_items),
        static_cast<long long>(num_segments),
        d_begin_offsets, d_end_offsets,
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
#include <cub/device/device_segmented_sort.cuh>
{0}
extern "C" EXPORT int cccl_jit_segmented_sort(
    void* d_temp_storage, size_t* temp_storage_bytes,
    void* d_keys_in_ptr, void* d_keys_out_ptr,
    void* d_values_in_ptr, void* d_values_out_ptr,
    unsigned long long num_items, unsigned long long num_segments,
    const long long* d_begin_offsets, const long long* d_end_offsets,
    void* stream)
{{
    using key_t   = {1};
    using value_t = {2};
    cudaError_t err = cub::DeviceSegmentedSort::{3}(
        d_temp_storage, *temp_storage_bytes,
        static_cast<const key_t*>(d_keys_in_ptr),
        static_cast<key_t*>(d_keys_out_ptr),
        static_cast<const value_t*>(d_values_in_ptr),
        static_cast<value_t*>(d_values_out_ptr),
        static_cast<long long>(num_items),
        static_cast<long long>(num_segments),
        d_begin_offsets, d_end_offsets,
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

// Keys-only
using segmented_sort_keys_fn_t = int (*)(
  void*, size_t*, void*, void*, unsigned long long, unsigned long long, const long long*, const long long*, void*);

// Pairs
using segmented_sort_pairs_fn_t = int (*)(
  void*,
  size_t*,
  void*,
  void*,
  void*,
  void*,
  unsigned long long,
  unsigned long long,
  const long long*,
  const long long*,
  void*);

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_segmented_sort_build_ex(
  cccl_device_segmented_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t /*begin_offset_in*/,
  cccl_iterator_t /*end_offset_in*/,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const bool keys_only = is_null_it(d_values_in);
  const bool ascending = (sort_order == CCCL_ASCENDING);

  std::string key_type = get_type_name(d_keys_in.value_type.type);
  if (key_type.empty())
  {
    fprintf(stderr, "\nERROR in cccl_device_segmented_sort_build(): unsupported key type\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string source;
  if (keys_only)
  {
    source = make_keys_only_source(key_type, ascending);
  }
  else
  {
    std::string value_type = get_type_name(d_values_in.value_type.type);
    if (value_type.empty())
    {
      fprintf(stderr, "\nERROR in cccl_device_segmented_sort_build(): unsupported value type\n");
      return CUDA_ERROR_UNKNOWN;
    }
    source = make_pairs_source(key_type, value_type, ascending);
  }

  auto jit = cccl::detail::compile_jit_source(
    source, "cccl_jit_segmented_sort", cc_major, cc_minor, ctk_root, cccl_include_path, merged.get());
  if (!jit.compiler)
  {
    return CUDA_ERROR_UNKNOWN;
  }

  build_ptr->cc           = cc_major * 10 + cc_minor;
  build_ptr->cubin        = cccl::detail::copy_cubin(jit.cubin, &build_ptr->cubin_size);
  build_ptr->jit_compiler = jit.compiler;
  build_ptr->sort_fn      = jit.fn_ptr;
  build_ptr->key_type     = d_keys_in.value_type;
  build_ptr->value_type   = d_values_in.value_type;
  build_ptr->order        = sort_order;
  build_ptr->keys_only    = keys_only ? 1 : 0;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_sort_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_sort_build(
  cccl_device_segmented_sort_build_result_t* build,
  cccl_sort_order_t sort_order,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t begin_offset_in,
  cccl_iterator_t end_offset_in,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_segmented_sort_build_ex(
    build,
    sort_order,
    d_keys_in,
    d_values_in,
    begin_offset_in,
    end_offset_in,
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
// The JIT function uses the copy variant of DeviceSegmentedSort so the result
// is always in d_keys_out / d_values_out. selector is always set to 0.
// is_overwrite_okay is accepted but ignored on this path.
// Offset iterators must be raw device pointers to long long.
// ---------------------------------------------------------------------------

CUresult cccl_device_segmented_sort(
  cccl_device_segmented_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  uint64_t num_items,
  uint64_t num_segments,
  cccl_iterator_t start_offset_in,
  cccl_iterator_t end_offset_in,
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
      auto fn = reinterpret_cast<segmented_sort_keys_fn_t>(build.sort_fn);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.state,
        d_keys_out.state,
        static_cast<unsigned long long>(num_items),
        static_cast<unsigned long long>(num_segments),
        static_cast<const long long*>(start_offset_in.state),
        static_cast<const long long*>(end_offset_in.state),
        reinterpret_cast<void*>(stream));
    }
    else
    {
      auto fn = reinterpret_cast<segmented_sort_pairs_fn_t>(build.sort_fn);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.state,
        d_keys_out.state,
        d_values_in.state,
        d_values_out.state,
        static_cast<unsigned long long>(num_items),
        static_cast<unsigned long long>(num_segments),
        static_cast<const long long*>(start_offset_in.state),
        static_cast<const long long*>(end_offset_in.state),
        reinterpret_cast<void*>(stream));
    }

    if (selector)
    {
      *selector = is_overwrite_okay ? 1 : 0;
    }

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_sort(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_segmented_sort_cleanup(cccl_device_segmented_sort_build_result_t* build_ptr)
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
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_sort_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
