//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

#include <cccl/c/segmented_sort.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

static bool is_null_it(cccl_iterator_t it)
{
  return it.type == CCCL_POINTER && it.state == nullptr;
}

// JIT wrappers produced by CubCall:
//   keys-only:  fn(temp, temp_bytes, keys_in, keys_out, num_items, num_segments,
//                  begin_offsets, end_offsets, stream)
//   pairs:      fn(temp, temp_bytes, keys_in, keys_out, values_in, values_out,
//                  num_items, num_segments, begin_offsets, end_offsets, stream)
//
// The copy variant of DeviceSegmentedSort is used, so the result is always in
// d_keys_out / d_values_out (selector=0). is_overwrite_okay is accepted at the
// run API but has no effect.
using segmented_sort_keys_fn_t =
  int (*)(void*, size_t*, void*, void*, unsigned long long, unsigned long long, void*, void*, void*);
using segmented_sort_pairs_fn_t =
  int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, unsigned long long, void*, void*, void*);

CUresult cccl_device_segmented_sort_build_ex(
  cccl_device_segmented_sort_build_result_t* build_ptr,
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

  // DeviceSegmentedSort writes to caller-provided device pointers at run time.
  // Build needs an iterator descriptor for the outputs; synthesize raw-pointer
  // ones with the same value_type as the matching inputs.
  cccl_iterator_t d_keys_out = d_keys_in;
  d_keys_out.type            = CCCL_POINTER;
  d_keys_out.state           = nullptr;
  cccl_iterator_t d_values_out{};
  d_values_out.type       = CCCL_POINTER;
  d_values_out.state      = nullptr;
  d_values_out.value_type = d_values_in.value_type;

  const char* cub_algo;
  if (keys_only)
  {
    cub_algo = ascending ? "cub::DeviceSegmentedSort::SortKeys" : "cub::DeviceSegmentedSort::SortKeysDescending";
  }
  else
  {
    cub_algo = ascending ? "cub::DeviceSegmentedSort::SortPairs" : "cub::DeviceSegmentedSort::SortPairsDescending";
  }

  CubCallResult result = [&] {
    if (keys_only)
    {
      return CubCall::from("cub/device/device_segmented_sort.cuh")
        .run(cub_algo)
        .name("cccl_jit_segmented_sort")
        .with(temp_storage,
              temp_bytes,
              in(d_keys_in),
              out(d_keys_out),
              num_items,
              num_segments,
              in(begin_offset_in),
              in(end_offset_in),
              stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    return CubCall::from("cub/device/device_segmented_sort.cuh")
      .run(cub_algo)
      .name("cccl_jit_segmented_sort")
      .with(temp_storage,
            temp_bytes,
            in(d_keys_in),
            out(d_keys_out),
            in(d_values_in),
            out(d_values_out),
            num_items,
            num_segments,
            in(begin_offset_in),
            in(end_offset_in),
            stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
  }();

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_cubin(result.cubin, build_ptr->cubin, build_ptr->cubin_size);
  build_ptr->jit_compiler = result.compiler;
  build_ptr->sort_fn      = result.fn_ptr;
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
        start_offset_in.state,
        end_offset_in.state,
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
        start_offset_in.state,
        end_offset_in.state,
        reinterpret_cast<void*>(stream));
    }

    if (selector)
    {
      // Copy variant always writes to d_keys_out (= d_buffers[1] in DoubleBuffer mode).
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

CUresult cccl_device_segmented_sort_cleanup(cccl_device_segmented_sort_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->sort_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_sort_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
