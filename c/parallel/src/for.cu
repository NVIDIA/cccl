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
#include <vector>

#include <cccl/c/for.h>
#include <cccl/c/types.h>
#include <for/for_op_helper.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
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

CUresult cccl_device_for_build_ex(
  cccl_device_for_build_result_t* build_ptr,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    if (d_data.type == cccl_iterator_kind_t::CCCL_ITERATOR)
    {
      throw std::runtime_error(std::string("Iterators are unsupported in for_each currently"));
    }

    const char* name = "test";

    const int cc = cc_major * 10 + cc_minor;

    const std::string for_kernel_name   = get_device_for_kernel_name();
    const std::string device_for_kernel = get_for_kernel(op, d_data);

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    std::vector<const char*> args = {
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

    cccl::detail::extend_args_with_build_config(args, config);

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    std::string lowered_name;

    // Collect all LTO-IRs to be linked
    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};

    // Add operation if it's LTO-IR (C++ source not yet supported in for)
    appender.append_operation(op);

    // Add iterator definitions if present
    if (cccl_iterator_kind_t::CCCL_ITERATOR == d_data.type)
    {
      appender.append_operation(d_data.advance);
      appender.append_operation(d_data.dereference);
    }

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{device_for_kernel, name})
        ->add_expression({for_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({for_kernel_name, lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
        ->finalize_program();

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->static_kernel, build_ptr->library, lowered_name.c_str()));

    build_ptr->cc         = cc;
    build_ptr->cubin      = (void*) result.data.release();
    build_ptr->cubin_size = result.size;
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }
  return error;
}

CUresult cccl_device_for(
  cccl_device_for_build_result_t build, cccl_iterator_t d_data, uint64_t num_items, cccl_op_t op, CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;

  try
  {
    pushed           = try_push_context();
    auto exec_status = Invoke(d_data, num_items, op, build.cc, (CUfunction) build.static_kernel, stream);
    error            = static_cast<CUresult>(exec_status);
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

CUresult cccl_device_for_build(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_for_build_ex(
    build, d_data, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_for_cleanup(cccl_device_for_build_result_t* build_ptr)
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
    check(cuLibraryUnload(build_ptr->library));
  }
  catch (...)
  {
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
