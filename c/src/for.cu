//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_device.cuh>

#include <format>
#include <iostream>
#include <type_traits>

#include <cccl/c/for.h>
#include <cccl/c/types.h>
#include <for/for_op_helper.h>
#include <nvJitLink.h>
#include <nvrtc.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/types.h>

struct op_wrapper;
struct device_reduce_policy;

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

static cudaError_t
Invoke(cccl_iterator_t d_in, size_t num_items, cccl_op_t op, int cc, CUfunction static_kernel, CUstream stream)
{
  cudaError error = cudaSuccess;

  if (num_items == 0)
  {
    return error;
  }

  auto for_kernel_state = make_for_kernel_state(op, d_in);

  void* args[] = {&num_items, for_kernel_state.get()};

  int thread_count = 256;
  int block_count  = (num_items + 511) / 512;
  check(cuLaunchKernel(static_kernel, block_count, 1, 1, thread_count, 1, 1, 0, stream, args, 0));

  // Check for failure to launch
  error = CubDebug(cudaPeekAtLastError());

  return error;
}

struct for_each_wrapper;

static std::string get_device_for_kernel_name()
{
  std::string offset_t;
  std::string function_op_t;
  check(nvrtcGetTypeName<for_each_wrapper>(&function_op_t));
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  return std::format("cub::detail::for_each::static_kernel<device_for_policy, {0}, {1}>", offset_t, function_op_t);
}

extern "C" CCCL_C_API CUresult cccl_device_for_build(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path) noexcept
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    if (d_data.type == cccl_iterator_kind_t::iterator)
    {
      throw std::runtime_error(std::string("Iterators are unsupported in for_each currently"));
    }

    nvrtcProgram prog{};
    const char* name = "test";

    const int cc                     = cc_major * 10 + cc_minor;
    const std::string d_data_value_t = cccl_type_enum_to_string(d_data.value_type.type);
    const std::string offset_t       = cccl_type_enum_to_string(cccl_type_enum::UINT64);

    const std::string for_kernel_name   = get_device_for_kernel_name();
    const std::string device_for_kernel = get_for_kernel(op, d_data);

    check(nvrtcCreateProgram(&prog, device_for_kernel.c_str(), name, 0, nullptr, nullptr));

    check(nvrtcAddNameExpression(prog, for_kernel_name.c_str()));

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    constexpr int num_args = 7;
    const char* args[]     = {arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto"};

    std::size_t log_size{};
    nvrtcResult compile_result = nvrtcCompileProgram(prog, num_args, args);

    check(nvrtcGetProgramLogSize(prog, &log_size));
    std::unique_ptr<char[]> log{new char[log_size]};
    check(nvrtcGetProgramLog(prog, log.get()));

    if (log_size > 1)
    {
      std::cerr << log.get() << std::endl;
    }

    std::string for_kernel_lowered_name;
    {
      const char* for_kernel_lowered_name_temp;
      check(nvrtcGetLoweredName(prog, for_kernel_name.c_str(), &for_kernel_lowered_name_temp));
      for_kernel_lowered_name = for_kernel_lowered_name_temp;
    }

    check(compile_result);

    std::size_t ltoir_size{};
    check(nvrtcGetLTOIRSize(prog, &ltoir_size));
    std::unique_ptr<char[]> ltoir{new char[ltoir_size]};
    check(nvrtcGetLTOIR(prog, ltoir.get()));
    check(nvrtcDestroyProgram(&prog));

    nvJitLinkHandle handle;
    const char* lopts[] = {"-lto", arch.c_str()};

    check(nvJitLinkCreate(&handle, 2, lopts));
    check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoir.get(), ltoir_size, name));
    check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, op.ltoir, op.ltoir_size, name));
    if (cccl_iterator_kind_t::iterator == d_data.type)
    {
      check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, d_data.advance.ltoir, d_data.advance.ltoir_size, name));
      check(
        nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, d_data.dereference.ltoir, d_data.dereference.ltoir_size, name));
    }

    auto jitlink_error = nvJitLinkComplete(handle);

    check(nvJitLinkGetErrorLogSize(handle, &log_size));
    std::unique_ptr<char[]> jitlinklog{new char[log_size]};
    check(nvJitLinkGetErrorLog(handle, jitlinklog.get()));

    if (log_size > 1)
    {
      std::cerr << jitlinklog.get() << std::endl;
    }

    check(jitlink_error);

    std::size_t cubin_size{};
    check(nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
    std::unique_ptr<char[]> cubin{new char[cubin_size]};
    check(nvJitLinkGetLinkedCubin(handle, cubin.get()));
    check(nvJitLinkDestroy(&handle));

    cuLibraryLoadData(&build->library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build->static_kernel, build->library, for_kernel_lowered_name.c_str()));

    build->cc         = cc;
    build->cubin      = cubin.release();
    build->cubin_size = cubin_size;
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

extern "C" CCCL_C_API CUresult cccl_device_for(
  cccl_device_for_build_result_t build,
  cccl_iterator_t d_data,
  int64_t num_items,
  cccl_op_t op,
  CUstream stream) noexcept
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;

  try
  {
    pushed = try_push_context();
    Invoke(d_data, num_items, op, build.cc, (CUfunction) build.static_kernel, stream);
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

extern "C" CCCL_C_API CUresult cccl_device_for_cleanup(cccl_device_for_build_result_t* bld_ptr)
{
  try
  {
    if (bld_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(bld_ptr->cubin));
    check(cuLibraryUnload(bld_ptr->library));
  }
  catch (...)
  {
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
