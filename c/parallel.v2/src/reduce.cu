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
#include <filesystem>
#include <string>

#include <cccl/c/reduce.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// (temp_storage, temp_bytes, d_in, d_out, num_items, op_state, init_state, stream)
using reduce_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*, void*);

CUresult cccl_device_reduce_build_ex(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config)
try
{
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto result =
    CubCall::from("cub/device/device_reduce.cuh")
      .run("cub::DeviceReduce::Reduce")
      .name("cccl_jit_reduce")
      .with(temp_storage, temp_bytes, in(input_it), out(output_it), num_items, op, init, stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_cubin(result.cubin, build->cubin, build->cubin_size);
  build->jit_compiler     = result.compiler;
  build->reduce_fn        = reinterpret_cast<void*>(result.fn_ptr);
  build->accumulator_size = init.type.size;
  build->determinism      = determinism;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  try
  {
    if (!build.reduce_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    auto reduce_fn = reinterpret_cast<reduce_fn_t>(build.reduce_fn);

    // Parameter order matches CubCall::with() order: ..., num_items, op.state, init.state, stream
    const int status = reduce_fn(
      d_temp_storage,
      temp_storage_bytes,
      d_in.state,
      d_out.state,
      num_items,
      op.state,
      init.state,
      reinterpret_cast<void*>(stream));

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_reduce(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_reduce_nondeterministic(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  return cccl_device_reduce(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, stream);
}

CUresult cccl_device_reduce_cleanup(cccl_device_reduce_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->reduce_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_build(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_reduce_build_ex(
    build,
    d_in,
    d_out,
    op,
    init,
    determinism,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}
