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

#include <cccl/c/three_way_partition.h>
#include <hostjit/codegen/cub_call.hpp>
#include <hostjit/jit_compiler.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// CUB DevicePartition::If (three-way) generated signature:
// (temp, bytes, d_in, first_out, second_out, unselected_out, num_selected_out,
//  num_items, first_op_state, second_op_state, stream)
using three_way_partition_fn_t =
  int (*)(void*, size_t*, void*, void*, void*, void*, void*, unsigned long long, void*, void*, void*);

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_three_way_partition_build_ex(
  cccl_device_three_way_partition_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_first_part_out,
  cccl_iterator_t d_second_part_out,
  cccl_iterator_t d_unselected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_first_part_op,
  cccl_op_t select_second_part_op,
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

  // DevicePartition::If (three-way):
  // (temp, bytes, d_in, d_first_part_out, d_second_part_out, d_unselected_out,
  //  d_num_selected_out, num_items, select_first_op, select_second_op, stream)
  auto result =
    CubCall::from("cub/device/device_partition.cuh")
      .run("cub::DevicePartition::If")
      .name("cccl_jit_three_way_partition")
      .with(temp_storage,
            temp_bytes,
            in(d_in),
            out(d_first_part_out),
            out(d_second_part_out),
            out(d_unselected_out),
            out(d_num_selected_out),
            num_items,
            pred(select_first_part_op, d_in.value_type),
            pred(select_second_part_op, d_in.value_type),
            stream)
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
  build_ptr->jit_compiler           = result.compiler;
  build_ptr->three_way_partition_fn = result.fn_ptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_three_way_partition_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_three_way_partition_build(
  cccl_device_three_way_partition_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_first_part_out,
  cccl_iterator_t d_second_part_out,
  cccl_iterator_t d_unselected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_first_part_op,
  cccl_op_t select_second_part_op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_three_way_partition_build_ex(
    build_ptr,
    d_in,
    d_first_part_out,
    d_second_part_out,
    d_unselected_out,
    d_num_selected_out,
    select_first_part_op,
    select_second_part_op,
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

CUresult cccl_device_three_way_partition(
  cccl_device_three_way_partition_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_first_part_out,
  cccl_iterator_t d_second_part_out,
  cccl_iterator_t d_unselected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_first_part_op,
  cccl_op_t select_second_part_op,
  uint64_t num_items,
  CUstream stream)
{
  try
  {
    if (!build.three_way_partition_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    auto fn    = reinterpret_cast<three_way_partition_fn_t>(build.three_way_partition_fn);
    int status = fn(
      d_temp_storage,
      temp_storage_bytes,
      d_in.state,
      d_first_part_out.state,
      d_second_part_out.state,
      d_unselected_out.state,
      d_num_selected_out.state,
      static_cast<unsigned long long>(num_items),
      select_first_part_op.state,
      select_second_part_op.state,
      reinterpret_cast<void*>(stream));

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_three_way_partition(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_three_way_partition_cleanup(cccl_device_three_way_partition_build_result_t* build_ptr)
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
  build_ptr->cubin_size             = 0;
  build_ptr->three_way_partition_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_three_way_partition_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
