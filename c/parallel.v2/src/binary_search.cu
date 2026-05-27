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
#include <cstring>

#include <cccl/c/binary_search.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// (d_in_0, num_items, d_in_1, num_values, d_out_0, op_0_state, stream)
using binary_search_fn_t = int (*)(void*, unsigned long long, void*, unsigned long long, void*, void*, void*);

CUresult cccl_device_binary_search_build_ex(
  cccl_device_binary_search_build_result_t* build_ptr,
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
  cccl_build_config* config)
try
{
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const char* find_fn =
    (mode == CCCL_BINARY_SEARCH_LOWER_BOUND) ? "cub::DeviceFind::LowerBound" : "cub::DeviceFind::UpperBound";

  // env_stream uses the env-based DeviceFind overload so CUB manages its own
  // temp storage via the env's memory_resource — no caller-managed buffer.
  auto result =
    CubCall::from("cub/device/device_find.cuh")
      .run(find_fn)
      .name("cccl_jit_binary_search")
      .with(in(d_data), num_haystack, in(d_values), num_needles, out(d_out), cmp(op), env_stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_cubin(result.cubin, build_ptr->cubin, build_ptr->cubin_size);
  build_ptr->jit_compiler     = result.compiler;
  build_ptr->binary_search_fn = result.fn_ptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_search(
  cccl_device_binary_search_build_result_t build,
  cccl_iterator_t d_data,
  uint64_t num_items,
  cccl_iterator_t d_values,
  uint64_t num_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  CUstream stream)
{
  try
  {
    auto fn = reinterpret_cast<binary_search_fn_t>(build.binary_search_fn);
    if (!fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    int status =
      fn(d_data.state, num_items, d_values.state, num_values, d_out.state, op.state, reinterpret_cast<void*>(stream));
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_binary_search_build(
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
  const char* ctk_path)
{
  return cccl_device_binary_search_build_ex(
    build,
    mode,
    d_data,
    d_values,
    d_out,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_binary_search_cleanup(cccl_device_binary_search_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->binary_search_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
