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
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/__algorithm_>
#include <cuda/std/cstdint>
#include <cuda/std/functional> // ::cuda::std::identity
#include <cuda/std/variant>

#include <format>
#include <memory>
#include <vector>

#include "jit_templates/templates/input_iterator.h"
#include "jit_templates/templates/operation.h"
#include "jit_templates/templates/output_iterator.h"
#include "jit_templates/traits.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/runtime_policy.h"
#include "util/types.h"
#include <cccl/c/reduce.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

struct device_reduce_policy;
using TransformOpT = ::cuda::std::identity;
using OffsetT      = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

namespace reduce
{

struct reduce_runtime_tuning_policy
{
  cub::detail::RuntimeReduceAgentPolicy single_tile;
  cub::detail::RuntimeReduceAgentPolicy reduce;

  auto SingleTile() const
  {
    return single_tile;
  }
  auto Reduce() const
  {
    return reduce;
  }

  using MaxPolicy = reduce_runtime_tuning_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<reduce_runtime_tuning_policy>(*this);
  }
};

static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_value_t init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init.type;
}

std::string get_single_tile_kernel_name(
  std::string_view input_iterator_t,
  std::string_view output_iterator_t,
  std::string_view reduction_op_t,
  cccl_value_t init,
  std::string_view accum_cpp_t,
  bool is_second_kernel)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_reduce_policy>(&chained_policy_t));

  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  std::string offset_t;
  if (is_second_kernel)
  {
    // Second kernel is always invoked with an int offset.
    // See the definition of the local variable `reduce_grid_size`
    // in DispatchReduce::InvokePasses.
    check(nvrtcGetTypeName<int>(&offset_t));
  }
  else
  {
    check(nvrtcGetTypeName<OffsetT>(&offset_t));
  }

  return std::format(
    "cub::detail::reduce::DeviceReduceSingleTileKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}>",
    chained_policy_t,
    input_iterator_t,
    output_iterator_t,
    offset_t,
    reduction_op_t,
    init_t,
    accum_cpp_t);
}

std::string get_device_reduce_kernel_name(
  std::string_view reduction_op_t, std::string_view input_iterator_t, std::string_view accum_t)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_reduce_policy>(&chained_policy_t));

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(nvrtcGetTypeName<cuda::std::identity>(&transform_op_t));

  return std::format(
    "cub::detail::reduce::DeviceReduceKernel<{0}, {1}, {2}, {3}, {4}, {5}>",
    chained_policy_t,
    input_iterator_t,
    offset_t,
    reduction_op_t,
    accum_t,
    transform_op_t);
}

struct reduce_kernel_source
{
  cccl_device_reduce_build_result_t& build;

  std::size_t AccumSize() const
  {
    return build.accumulator_size;
  }
  CUkernel SingleTileKernel() const
  {
    return build.single_tile_kernel;
  }
  CUkernel SingleTileSecondKernel() const
  {
    return build.single_tile_second_kernel;
  }
  CUkernel ReductionKernel() const
  {
    return build.reduction_kernel;
  }
};
} // namespace reduce

struct reduce_iterator_tag;
struct reduction_operation_tag;

CUresult cccl_device_reduce_build_ex(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
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
    const char* name = "device_reduce";

    const int cc                 = cc_major * 10 + cc_minor;
    const cccl_type_info accum_t = reduce::get_accumulator_type(op, input_it, init);
    const auto accum_cpp         = cccl_type_enum_to_name(accum_t.type);

    const auto [input_iterator_name, input_iterator_src] =
      get_specialization<reduce_iterator_tag>(template_id<input_iterator_traits>(), input_it);
    const auto [output_iterator_name, output_iterator_src] =
      get_specialization<reduce_iterator_tag>(template_id<output_iterator_traits>(), output_it, accum_t);

    const auto [op_name, op_src] =
      get_specialization<reduction_operation_tag>(template_id<binary_user_operation_traits>(), op, accum_t);

    const std::string ptx_arch = std::format("-arch=compute_{}{}", cc_major, cc_minor);

    constexpr size_t ptx_num_args      = 6;
    const char* ptx_args[ptx_num_args] = {
      ptx_arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true"};

    const std::string src = std::format(
      R"XXX(
#include <cub/block/block_reduce.cuh>
{5}
struct __align__({1}) storage_t {{
  char data[{0}];
}};
{2}
{3}
{4}
)XXX",
      input_it.value_type.size, // 0
      input_it.value_type.alignment, // 1
      input_iterator_src, // 2
      output_iterator_src, // 3
      op_src, // 4
      jit_template_header_contents); // 5

    const auto offset_t           = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);
    nlohmann::json runtime_policy = get_policy(
      std::format("cub::detail::reduce::MakeReducePolicyWrapper(cub::detail::reduce::policy_hub<{}, {}, "
                  "{}>::MaxPolicy::ActivePolicy{{}})",
                  accum_cpp,
                  offset_t,
                  op_name),
      "#include <cub/device/dispatch/tuning/tuning_reduce.cuh>\n" + src,
      ptx_args);

    using cub::detail::RuntimeReduceAgentPolicy;
    auto [reduce_policy, reduce_policy_str] = RuntimeReduceAgentPolicy::from_json(runtime_policy, "ReducePolicy");
    auto [st_policy, st_policy_str]         = RuntimeReduceAgentPolicy::from_json(runtime_policy, "SingleTilePolicy");

    std::string final_src = std::format(
      "#include <cub/device/dispatch/kernels/reduce.cuh>\n"
      "{0}\n"
      "struct device_reduce_policy {{\n"
      "  struct ActivePolicy {{\n"
      "    {1}\n"
      "    {2}\n"
      "  }};\n"
      "}};",
      src,
      reduce_policy_str,
      st_policy_str);

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
    fflush(stdout);
#endif

    std::string single_tile_kernel_name =
      reduce::get_single_tile_kernel_name(input_iterator_name, output_iterator_name, op_name, init, accum_cpp, false);
    std::string single_tile_second_kernel_name = reduce::get_single_tile_kernel_name(
      cccl_type_enum_to_name(accum_t.type, true), output_iterator_name, op_name, init, accum_cpp, true);
    std::string reduction_kernel_name = reduce::get_device_reduce_kernel_name(op_name, input_iterator_name, accum_cpp);
    std::string single_tile_kernel_lowered_name;
    std::string single_tile_second_kernel_lowered_name;
    std::string reduction_kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    // Build compilation arguments
    std::vector<const char*> args = {
      arch.c_str(),
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path,
      "-rdc=true",
      "-dlto",
      "-DCUB_DISABLE_CDP",
      "-std=c++20"};

    // Add user's extra flags if config is provided
    cccl::detail::extend_args_with_build_config(args, config);

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};

    appender.append_operation(op);
    appender.add_iterator_definition(input_it);
    appender.add_iterator_definition(output_it);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
        ->add_expression({single_tile_kernel_name})
        ->add_expression({single_tile_second_kernel_name})
        ->add_expression({reduction_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({single_tile_kernel_name, single_tile_kernel_lowered_name})
        ->get_name({single_tile_second_kernel_name, single_tile_second_kernel_lowered_name})
        ->get_name({reduction_kernel_name, reduction_kernel_lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
        ->finalize_program();

    cuLibraryLoadData(&build->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build->single_tile_kernel, build->library, single_tile_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build->single_tile_second_kernel, build->library, single_tile_second_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build->reduction_kernel, build->library, reduction_kernel_lowered_name.c_str()));

    build->cc               = cc;
    build->cubin            = (void*) result.data.release();
    build->cubin_size       = result.size;
    build->accumulator_size = accum_t.size;
    build->runtime_policy   = new reduce::reduce_runtime_tuning_policy{st_policy, reduce_policy};
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_reduce_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
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
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::DispatchReduce<
      indirect_arg_t, // InputIteratorT
      indirect_arg_t, // OutputIteratorT
      ::cuda::std::size_t, // OffsetT
      indirect_arg_t, // ReductionOpT
      indirect_arg_t, // InitT
      void, // AccumT
      ::cuda::std::identity, // TransformOpT
      reduce::reduce_runtime_tuning_policy, // PolicyHub
      reduce::reduce_kernel_source, // KernelSource
      cub::detail::CudaDriverLauncherFactory>:: // KernelLauncherFactory
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        op,
        init,
        stream,
        {},
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        *reinterpret_cast<reduce::reduce_runtime_tuning_policy*>(build.runtime_policy));

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_reduce(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

CUresult cccl_device_reduce_cleanup(cccl_device_reduce_build_result_t* build_ptr)
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
    std::unique_ptr<char[]> policy(reinterpret_cast<char*>(build_ptr->runtime_policy));
    check(cuLibraryUnload(build_ptr->library));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_reduce_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}

// Backward compatibility wrapper
CUresult cccl_device_reduce_build(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_reduce_build_ex(
    build, d_in, d_out, op, init, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}
