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
#include <type_traits>

#include <cccl/c/for.h>
#include <cccl/c/types.h>
#include <for/for_op_helper.h>
#include <nvrtc/command_list.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/types.h>

struct op_wrapper;
struct device_reduce_policy;

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

static cudaError_t
Invoke(cccl_iterator_t d_in, size_t num_items, cccl_op_t op, int /*cc*/, CUfunction static_kernel, CUstream stream)
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

    const char* name = "test";

    const int cc                     = cc_major * 10 + cc_minor;
    const std::string d_data_value_t = cccl_type_enum_to_string(d_data.value_type.type);
    const std::string offset_t       = cccl_type_enum_to_string(cccl_type_enum::UINT64);

    const std::string for_kernel_name   = get_device_for_kernel_name();
    const std::string device_for_kernel = get_for_kernel(op, d_data);

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    constexpr size_t num_args  = 7;
    const char* args[num_args] = {arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    std::string lowered_name;

    auto cl =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{device_for_kernel, name})
        .add_expression({for_kernel_name})
        .compile_program({args, num_args})
        .get_name({for_kernel_name, lowered_name})
        .cleanup_program()
        .add_link({op.ltoir, op.ltoir_size});

    nvrtc_cubin result{};

    if (cccl_iterator_kind_t::iterator == d_data.type)
    {
      result = cl.add_link({d_data.advance.ltoir, d_data.advance.ltoir_size})
                 .add_link({d_data.dereference.ltoir, d_data.dereference.ltoir_size})
                 .finalize_program(num_lto_args, lopts);
    }
    else
    {
      result = cl.finalize_program(num_lto_args, lopts);
    }

    cuLibraryLoadData(&build->library, result.cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build->static_kernel, build->library, lowered_name.c_str()));

    build->cc         = cc;
    build->cubin      = (void*) result.cubin.release();
    build->cubin_size = result.size;
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
