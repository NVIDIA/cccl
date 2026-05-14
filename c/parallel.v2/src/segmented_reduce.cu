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
#include <string>

#include <cccl/c/segmented_reduce.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

using segmented_reduce_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*, void*, void*);

CUresult cccl_device_segmented_reduce_build_ex(
  cccl_device_segmented_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config)
try
{
  const std::string cccl_include_str = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  const char* cccl_include_path      = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();

  const std::string ctk_root_str = cccl::detail::parse_ctk_root(ctk_path);
  const char* ctk_root           = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto result =
    CubCall::from("cub/device/device_segmented_reduce.cuh")
      .run("cub::DeviceSegmentedReduce::Reduce")
      .name("cccl_jit_segmented_reduce")
      .with(temp_storage, temp_bytes, in(d_in), out(d_out), num_items, in(start_offset_it), in(end_offset_it), op, init)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build->cc         = cc_major * 10 + cc_minor;
  build->cubin      = nullptr;
  build->cubin_size = 0;
  if (!result.cubin.empty())
  {
    auto* cubin_copy = new char[result.cubin.size()];
    std::memcpy(cubin_copy, result.cubin.data(), result.cubin.size());
    build->cubin      = cubin_copy;
    build->cubin_size = result.cubin.size();
  }
  build->jit_compiler        = result.compiler;
  build->segmented_reduce_fn = reinterpret_cast<void*>(result.fn_ptr);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce(
  cccl_device_segmented_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_segments,
  cccl_iterator_t start_offset,
  cccl_iterator_t end_offset,
  cccl_op_t op,
  cccl_value_t init,
  CUstream /*stream*/)
{
  try
  {
    auto segmented_reduce_fn = reinterpret_cast<segmented_reduce_fn_t>(build.segmented_reduce_fn);

    if (!segmented_reduce_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    // Parameter order matches CubCall::with() order:
    // temp_storage, temp_bytes, d_in, d_out, num_items, begin_offsets, end_offsets, op, init
    int status = segmented_reduce_fn(
      d_temp_storage,
      temp_storage_bytes,
      d_in.state,
      d_out.state,
      num_segments,
      start_offset.state,
      end_offset.state,
      op.state,
      init.state);

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_segmented_reduce_build(
  cccl_device_segmented_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_iterator_t begin_offset_in,
  cccl_iterator_t end_offset_in,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_segmented_reduce_build_ex(
    build,
    d_in,
    d_out,
    begin_offset_in,
    end_offset_in,
    op,
    init,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_segmented_reduce_cleanup(cccl_device_segmented_reduce_build_result_t* build_ptr)
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
  build_ptr->cubin_size          = 0;
  build_ptr->segmented_reduce_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
