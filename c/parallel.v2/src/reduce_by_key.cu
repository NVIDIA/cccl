//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//
//
// cuda.compute reduce-by-key, HostJIT/v2 backend.
//
// The public CUB overload takes neither an equality operator nor an initial value: keys compare with
// their own ==, and each run's representative key is its *last* key. Outputs are unique keys,
// aggregates, and a device-side run count.

#include <cstdio>
#include <cstring>

#include <cccl/c/reduce_by_key.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// ReduceByKey: 10 args
// (temp, temp_bytes, keys_in, unique_out, values_in, aggregates_out, num_runs_out, op, num_items, stream)
using reduce_by_key_fn_t =
  int (*)(void*, size_t*, void*, void*, void*, void*, void*, void*, unsigned long long, void*);

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_reduce_by_key_build_ex(
  cccl_device_reduce_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
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
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  CubCallResult result = [&] {
    auto base = CubCall::from("cub/device/device_reduce.cuh").name("cccl_jit_reduce_by_key");
    // force_accum_type is required: the accumulator is otherwise resolved from the first input
    // iterator, which here is the keys, so a floating-point aggregate would be truncated to the key
    // type.
    return base.run("cub::DeviceReduce::ReduceByKey")
      .with(force_accum_type(d_values_in.value_type),
            temp_storage,
            temp_bytes,
            in(d_keys_in),
            out(d_unique_out),
            in(d_values_in),
            out(d_aggregates_out),
            out(d_num_runs_out),
            op,
            num_items,
            stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
  }();

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_cubin(result.cubin, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler     = result.compiler;
  build_ptr->reduce_by_key_fn = result.fn_ptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_by_key_build_ex(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_by_key_build(
  cccl_device_reduce_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_reduce_by_key_build_ex(
    build_ptr,
    d_keys_in,
    d_values_in,
    d_unique_out,
    d_aggregates_out,
    d_num_runs_out,
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

CUresult cccl_device_reduce_by_key(
  cccl_device_reduce_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
try
{
  if (!build.reduce_by_key_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto fn          = reinterpret_cast<reduce_by_key_fn_t>(build.reduce_by_key_fn);
  const int status = fn(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in.state,
    d_unique_out.state,
    d_values_in.state,
    d_aggregates_out.state,
    d_num_runs_out.state,
    op.state,
    static_cast<unsigned long long>(num_items),
    reinterpret_cast<void*>(stream));
  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_by_key(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_reduce_by_key_cleanup(cccl_device_reduce_by_key_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->reduce_by_key_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_by_key_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
