//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/dispatch/dispatch_transform.cuh>
#include <cub/device/dispatch/tuning/tuning_transform.cuh> // cub::detail::transform::Algorithm
#include <cub/util_arch.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <format>
#include <string>
#include <type_traits>

#include "kernels/iterators.h"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/transform.h>
#include <cccl/c/types.h> // cccl_type_info
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <stdio.h> // printf

struct op_wrapper;
struct device_transform_policy;

using OffsetT = long;
static_assert(std::is_same_v<cub::detail::choose_signed_offset_t<OffsetT>, OffsetT>,
              "OffsetT must be signed int32 or int64");

struct input_storage_t;
struct input1_storage_t;
struct input2_storage_t;
struct output_storage_t;

namespace transform
{

constexpr auto input_iterator_name  = "input_iterator_t";
constexpr auto input1_iterator_name = "input1_iterator_t";
constexpr auto input2_iterator_name = "input2_iterator_t";
constexpr auto output_iterator_name = "output_iterator_t";

struct transform_runtime_tuning_policy
{
  int block_threads;
  int items_per_thread_no_input;
  int min_items_per_thread;
  int max_items_per_thread;

  // Note: when we extend transform to support UBLKCP, we may no longer
  // be able to keep this constexpr:
  static constexpr cub::detail::transform::Algorithm GetAlgorithm()
  {
    return cub::detail::transform::Algorithm::prefetch;
  }

  int BlockThreads()
  {
    return block_threads;
  }

  int ItemsPerThreadNoInput()
  {
    return items_per_thread_no_input;
  }

  int MinItemsPerThread()
  {
    return min_items_per_thread;
  }

  int MaxItemsPerThread()
  {
    return max_items_per_thread;
  }
  static constexpr int min_bif = 1024 * 12;
};

transform_runtime_tuning_policy get_policy()
{
  // return prefetch policy defaults:
  return {256, 2, 1, 32};
}

template <typename StorageT>
const std::string get_iterator_name(cccl_iterator_t iterator, const std::string& name)
{
  if (iterator.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return cccl_type_enum_to_name<StorageT>(iterator.value_type.type, true);
  }
  return name;
}

std::string get_kernel_name(cccl_iterator_t input_it, cccl_iterator_t output_it, cccl_op_t /*op*/)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_transform_policy>(&chained_policy_t));

  const std::string input_iterator_t  = get_iterator_name<input_storage_t>(input_it, input_iterator_name);
  const std::string output_iterator_t = get_iterator_name<output_storage_t>(output_it, output_iterator_name);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&transform_op_t));

  return std::format(
    "cub::detail::transform::transform_kernel<{0}, {1}, {2}, {3}, {4}>",
    chained_policy_t, // 0
    offset_t, // 1
    transform_op_t, // 2
    output_iterator_t, // 3
    input_iterator_t); // 4
}

std::string
get_kernel_name(cccl_iterator_t input1_it, cccl_iterator_t input2_it, cccl_iterator_t output_it, cccl_op_t /*op*/)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_transform_policy>(&chained_policy_t));

  const std::string input1_iterator_t = get_iterator_name<input1_storage_t>(input1_it, input1_iterator_name);
  const std::string input2_iterator_t = get_iterator_name<input2_storage_t>(input2_it, input2_iterator_name);
  const std::string output_iterator_t = get_iterator_name<output_storage_t>(output_it, output_iterator_name);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&transform_op_t));

  return std::format(
    "cub::detail::transform::transform_kernel<{0}, {1}, {2}, {3}, {4}, {5}>",
    chained_policy_t, // 0
    offset_t, // 1
    transform_op_t, // 2
    output_iterator_t, // 3
    input1_iterator_t, // 4
    input2_iterator_t); // 5
}

template <auto* GetPolicy>
struct dynamic_transform_policy_t
{
  using max_policy = dynamic_transform_policy_t;

  template <typename F>
  cudaError_t Invoke(int /*device_ptx_version*/, F& op)
  {
    return op.template Invoke<transform_runtime_tuning_policy>(GetPolicy());
  }
};

struct transform_kernel_source
{
  cccl_device_transform_build_result_t& build;

  CUkernel TransformKernel() const
  {
    return build.transform_kernel;
  }

  int LoadedBytesPerIteration()
  {
    return build.loaded_bytes_per_iteration;
  }

  template <typename It>
  constexpr It MakeIteratorKernelArg(It it)
  {
    return it;
  }
};

} // namespace transform

CUresult cccl_device_unary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "test";

    const int cc                 = cc_major * 10 + cc_minor;
    const auto policy            = transform::get_policy();
    const auto input_it_value_t  = cccl_type_enum_to_name<input_storage_t>(input_it.value_type.type);
    const auto output_it_value_t = cccl_type_enum_to_name<output_storage_t>(output_it.value_type.type);
    const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT64);
    const std::string input_iterator_src =
      make_kernel_input_iterator(offset_t, transform::input_iterator_name, input_it_value_t, input_it);
    const std::string output_iterator_src =
      make_kernel_output_iterator(offset_t, transform::output_iterator_name, output_it_value_t, output_it);
    const std::string op_src = make_kernel_user_unary_operator(input_it_value_t, output_it_value_t, op);

    constexpr std::string_view src_template = R"XXX(
#define _CUB_HAS_TRANSFORM_UBLKCP 0
#include <cub/device/dispatch/kernels/transform.cuh>
struct __align__({1}) input_storage_t {{
  char data[{0}];
}};
struct __align__({3}) output_storage_t {{
  char data[{2}];
}};
{8}
{9}
struct prefetch_policy_t {{
  static constexpr int block_threads = {4};
  static constexpr int items_per_thread_no_input = {5};
  static constexpr int min_items_per_thread      = {6};
  static constexpr int max_items_per_thread      = {7};
}};
struct device_transform_policy {{
  struct ActivePolicy {{
    static constexpr auto algorithm = cub::detail::transform::Algorithm::prefetch;
    using algo_policy = prefetch_policy_t;
  }};
}};
{10}
)XXX";

    const std::string& src = std::format(
      src_template,
      input_it.value_type.size, // 0
      input_it.value_type.alignment, // 1
      output_it.value_type.size, // 2
      output_it.value_type.alignment, // 3
      policy.block_threads, // 4
      policy.items_per_thread_no_input, // 5
      policy.min_items_per_thread, // 6
      policy.max_items_per_thread, // 7
      input_iterator_src, // 8
      output_iterator_src, // 9
      op_src); // 10

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    std::string kernel_name = transform::get_kernel_name(input_it, output_it, op);
    std::string kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    // Note: `-default-device` is needed because of the use of lambdas
    // in the transform kernel code. Qualifying those explicitly with
    // `__device__` seems not to be supported by NVRTC.
    constexpr size_t num_args  = 9;
    const char* args[num_args] = {
      arch.c_str(),
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path,
      "-rdc=true",
      "-dlto",
      "-default-device",
      "-DCUB_DISABLE_CDP"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    nvrtc_ltoir_list_appender appender{ltoir_list};

    appender.append({op.ltoir, op.ltoir_size});
    appender.add_iterator_definition(input_it);
    appender.add_iterator_definition(output_it);

    nvrtc_link_result result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{src.c_str(), name})
        .add_expression({kernel_name})
        .compile_program({args, num_args})
        .get_name({kernel_name, kernel_lowered_name})
        .cleanup_program()
        .add_link_list(ltoir_list)
        .finalize_program(num_lto_args, lopts);

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->transform_kernel, build_ptr->library, kernel_lowered_name.c_str()));

    build_ptr->loaded_bytes_per_iteration = input_it.value_type.size;
    build_ptr->cc                         = cc;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_unary_transform_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

CUresult cccl_device_unary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));
    auto cuda_error = cub::detail::transform::dispatch_t<
      cub::detail::transform::requires_stable_address::no, // TODO implement yes
      OffsetT,
      ::cuda::std::tuple<indirect_arg_t>,
      indirect_arg_t,
      indirect_arg_t,
      transform::dynamic_transform_policy_t<&transform::get_policy>,
      transform::transform_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      dispatch(d_in, d_out, num_items, op, stream, {build}, cub::detail::CudaDriverLauncherFactory{cu_device, build.cc});
    if (cuda_error != cudaSuccess)
    {
      const char* errorString = cudaGetErrorString(cuda_error); // Get the error string
      std::cerr << "CUDA error: " << errorString << std::endl;
    }
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_unary_transform(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }
  if (pushed)
  {
    CUcontext cu_context;
    cuCtxPopCurrent(&cu_context);
  }
  return error;
}

CUresult cccl_device_binary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t input1_it,
  cccl_iterator_t input2_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "test";

    const int cc                 = cc_major * 10 + cc_minor;
    const auto policy            = transform::get_policy();
    const auto input1_it_value_t = cccl_type_enum_to_name<input1_storage_t>(input1_it.value_type.type);
    const auto input2_it_value_t = cccl_type_enum_to_name<input2_storage_t>(input2_it.value_type.type);

    const auto output_it_value_t = cccl_type_enum_to_name<output_storage_t>(output_it.value_type.type);
    const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT64);
    const std::string input1_iterator_src =
      make_kernel_input_iterator(offset_t, transform::input1_iterator_name, input1_it_value_t, input1_it);
    const std::string input2_iterator_src =
      make_kernel_input_iterator(offset_t, transform::input2_iterator_name, input2_it_value_t, input2_it);

    const std::string output_iterator_src =
      make_kernel_output_iterator(offset_t, transform::output_iterator_name, output_it_value_t, output_it);
    const std::string op_src =
      make_kernel_user_binary_operator(input1_it_value_t, input2_it_value_t, output_it_value_t, op);

    constexpr std::string_view src_template = R"XXX(
#define _CUB_HAS_TRANSFORM_UBLKCP 0
#include <cub/device/dispatch/kernels/transform.cuh>
struct __align__({1}) input1_storage_t {{
  char data[{0}];
}};
struct __align__({3}) input2_storage_t {{
  char data[{2}];
}};

struct __align__({5}) output_storage_t {{
  char data[{4}];
}};

{10}
{11}
{12}

struct prefetch_policy_t {{
  static constexpr int block_threads = {6};
  static constexpr int items_per_thread_no_input = {7};
  static constexpr int min_items_per_thread      = {8};
  static constexpr int max_items_per_thread      = {9};
}};

struct device_transform_policy {{
  struct ActivePolicy {{
    static constexpr auto algorithm = cub::detail::transform::Algorithm::prefetch;
    using algo_policy = prefetch_policy_t;
  }};
}};

{13}
)XXX";
    const std::string& src                  = std::format(
      src_template,
      input1_it.value_type.size, // 0
      input1_it.value_type.alignment, // 1
      input2_it.value_type.size, // 2
      input2_it.value_type.alignment, // 3
      output_it.value_type.size, // 4
      output_it.value_type.alignment, // 5
      policy.block_threads, // 6
      policy.items_per_thread_no_input, // 7
      policy.min_items_per_thread, // 8
      policy.max_items_per_thread, // 9
      input1_iterator_src, // 10
      input2_iterator_src, // 11
      output_iterator_src, // 12
      op_src); // 13

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    std::string kernel_name = transform::get_kernel_name(input1_it, input2_it, output_it, op);
    std::string kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    constexpr size_t num_args  = 8;
    const char* args[num_args] = {
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-default-device"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    nvrtc_ltoir_list_appender appender{ltoir_list};

    appender.append({op.ltoir, op.ltoir_size});
    appender.add_iterator_definition(input1_it);
    appender.add_iterator_definition(input2_it);
    appender.add_iterator_definition(output_it);

    nvrtc_link_result result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{src.c_str(), name})
        .add_expression({kernel_name})
        .compile_program({args, num_args})
        .get_name({kernel_name, kernel_lowered_name})
        .cleanup_program()
        .add_link_list(ltoir_list)
        .finalize_program(num_lto_args, lopts);

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->transform_kernel, build_ptr->library, kernel_lowered_name.c_str()));

    build_ptr->loaded_bytes_per_iteration = (input1_it.value_type.size + input2_it.value_type.size);
    build_ptr->cc                         = cc;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_binary_transform_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
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
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::detail::transform::dispatch_t<
      cub::detail::transform::requires_stable_address::no, // TODO implement yes
      OffsetT,
      ::cuda::std::tuple<indirect_arg_t, indirect_arg_t>,
      indirect_arg_t,
      indirect_arg_t,
      transform::dynamic_transform_policy_t<&transform::get_policy>,
      transform::transform_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      dispatch(::cuda::std::make_tuple<indirect_arg_t, indirect_arg_t>(d_in1, d_in2),
               d_out,
               num_items,
               op,
               stream,
               {build},
               cub::detail::CudaDriverLauncherFactory{cu_device, build.cc});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_binary_transform(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }
  if (pushed)
  {
    CUcontext cu_context;
    cuCtxPopCurrent(&cu_context);
  }
  return error;
}

CUresult cccl_device_transform_cleanup(cccl_device_transform_build_result_t* build_ptr)
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
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_transform_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
