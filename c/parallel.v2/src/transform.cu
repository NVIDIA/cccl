//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>

#include <cccl/c/transform.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// (d_in, d_out, num_items, op_state, stream)
using unary_transform_fn_t = int (*)(void*, void*, unsigned long long, void*, void*);
// (d_in1, d_in2, d_out, num_items, op_state, stream)
using binary_transform_fn_t = int (*)(void*, void*, void*, unsigned long long, void*, void*);

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_unary_transform_build_ex(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in,
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

  auto result =
    CubCall::from("cub/device/device_transform.cuh")
      .run("cub::DeviceTransform::Transform")
      .name("cccl_jit_unary_transform")
      .with(in(d_in), out(d_out), num_items, unary_op(op, d_in.value_type, d_out.value_type), stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

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
  build_ptr->transform_fn = result.fn_ptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_unary_transform_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_transform_build_ex(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
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

  // Use the output type as the accumulator type (same as the previous raw JIT
  // implementation) so the binary op functor uses the correct result type.
  auto result =
    CubCall::from("cub/device/device_transform.cuh")
      .run("cub::DeviceTransform::Transform")
      .name("cccl_jit_binary_transform")
      .use_tuple_inputs()
      .with(force_accum_type(d_out.value_type), in(d_in1), in(d_in2), out(d_out), num_items, op, stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

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
  build_ptr->transform_fn = result.fn_ptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_transform_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

// ---------------------------------------------------------------------------
// Non-ex wrappers (call _ex with nullptr config)
// ---------------------------------------------------------------------------

CUresult cccl_device_unary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_unary_transform_build_ex(
    build_ptr, d_in, d_out, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_binary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_binary_transform_build_ex(
    build_ptr, d_in1, d_in2, d_out, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

// ---------------------------------------------------------------------------
// Runtime functions
// ---------------------------------------------------------------------------

CUresult cccl_device_unary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  try
  {
    auto fn = reinterpret_cast<unary_transform_fn_t>(build.transform_fn);
    if (!fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    int status = fn(d_in.state, d_out.state, num_items, op.state, reinterpret_cast<void*>(stream));
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_unary_transform(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_binary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  try
  {
    auto fn = reinterpret_cast<binary_transform_fn_t>(build.transform_fn);
    if (!fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    int status = fn(d_in1.state, d_in2.state, d_out.state, num_items, op.state, reinterpret_cast<void*>(stream));
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_binary_transform(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_transform_cleanup(cccl_device_transform_build_result_t* build_ptr)
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
  build_ptr->cubin_size   = 0;
  build_ptr->transform_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_transform_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
