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

#include <cccl/c/merge_sort.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// Keys-only: (temp, temp_bytes, in_keys, out_keys, num_items, cmp_state, stream)
using keys_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*);
// Key-value pairs: (temp, temp_bytes, in_keys, in_items, out_keys, out_items, num_items, cmp_state, stream)
using pairs_fn_t = int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, void*, void*);

static bool is_null_items(cccl_iterator_t it)
{
  return it.type == CCCL_POINTER && it.state == nullptr;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort_build_ex(
  cccl_device_merge_sort_build_result_t* build_ptr,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (d_out_keys.type == CCCL_ITERATOR || d_out_items.type == CCCL_ITERATOR)
  {
    fprintf(stderr, "\nERROR in cccl_device_merge_sort_build(): merge sort output cannot be an iterator\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const bool has_items = !is_null_items(d_in_items);

  CubCallResult result = [&] {
    if (has_items)
    {
      return CubCall::from("cub/device/device_merge_sort.cuh")
        .run("cub::DeviceMergeSort::SortPairsCopy")
        .name("cccl_jit_merge_sort")
        .with(temp_storage,
              temp_bytes,
              in(d_in_keys),
              in(d_in_items),
              out(d_out_keys),
              out(d_out_items),
              num_items,
              cmp(op),
              stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else
    {
      return CubCall::from("cub/device/device_merge_sort.cuh")
        .run("cub::DeviceMergeSort::SortKeysCopy")
        .name("cccl_jit_merge_sort")
        .with(temp_storage, temp_bytes, in(d_in_keys), out(d_out_keys), num_items, cmp(op), stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
  }();

  build_ptr->cc         = cc_major * 10 + cc_minor;
  build_ptr->cubin      = nullptr;
  build_ptr->cubin_size = 0;
  if (!result.cubin.empty())
  {
    auto* cubin_copy = new char[result.cubin.size()];
    std::memcpy(cubin_copy, result.cubin.data(), result.cubin.size());
    build_ptr->cubin      = cubin_copy;
    build_ptr->cubin_size = result.cubin.size();
  }
  build_ptr->jit_compiler = result.compiler;
  build_ptr->sort_fn      = result.fn_ptr;
  build_ptr->key_type     = d_in_keys.value_type;
  build_ptr->item_type    = d_in_items.value_type;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_merge_sort_build(
  cccl_device_merge_sort_build_result_t* build,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_merge_sort_build_ex(
    build,
    d_in_keys,
    d_in_items,
    d_out_keys,
    d_out_items,
    op,
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
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort(
  cccl_device_merge_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  try
  {
    if (!build.sort_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    int status;
    // Dispatch to the correct function arity based on whether the current call
    // has items.  The build function compiles either SortKeysCopy (7-arg) or
    // SortPairsCopy (9-arg); both the build and the run must agree on which
    // variant is being used (null items → keys, non-null → pairs).
    const bool has_items = !(d_in_items.type == CCCL_POINTER && d_in_items.state == nullptr);
    if (has_items)
    {
      // Pairs build: (temp, temp_bytes, in_keys, in_items, out_keys, out_items, num_items, cmp_state, stream)
      auto fn = reinterpret_cast<pairs_fn_t>(build.sort_fn);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_in_keys.state,
        d_in_items.state,
        d_out_keys.state,
        d_out_items.state,
        num_items,
        op.state,
        reinterpret_cast<void*>(stream));
    }
    else
    {
      // Keys-only build: (temp, temp_bytes, in_keys, out_keys, num_items, cmp_state, stream)
      auto fn = reinterpret_cast<keys_fn_t>(build.sort_fn);
      status =
        fn(d_temp_storage,
           temp_storage_bytes,
           d_in_keys.state,
           d_out_keys.state,
           num_items,
           op.state,
           reinterpret_cast<void*>(stream));
    }

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort_cleanup(cccl_device_merge_sort_build_result_t* build_ptr)
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
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
