//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh> // cub::detail::choose_offset_t
#include <cub/detail/launcher/cuda_driver.cuh> // cub::detail::CudaDriverLauncherFactory
#include <cub/device/dispatch/dispatch_reduce.cuh> // cub::DispatchSegmentedReduce
#include <cub/thread/thread_load.cuh> // cub::LoadModifier

#include <exception> // std::exception
#include <format>
#include <string> // std::string
#include <string_view> // std::string_view
#include <type_traits> // std::is_same_v
#include <vector> // std::format

#include <stdio.h> // printf

#include "jit_templates/templates/input_iterator.h"
#include "jit_templates/templates/operation.h"
#include "jit_templates/templates/output_iterator.h"
#include "jit_templates/traits.h"
#include <cccl/c/segmented_reduce.h>
#include <cccl/c/types.h> // cccl_type_info
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/indirect_arg.h>
#include <util/runtime_policy.h>
#include <util/types.h>

struct device_segmented_reduce_policy;
using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

// check we can map OffsetT to ::cuda::std::uint64_t
static_assert(std::is_unsigned_v<OffsetT>);
static_assert(sizeof(OffsetT) == sizeof(::cuda::std::uint64_t));

namespace segmented_reduce
{

struct segmented_reduce_runtime_tuning_policy
{
  cub::detail::RuntimeReduceAgentPolicy segmented_reduce;

  auto SegmentedReduce() const
  {
    return segmented_reduce;
  }

  using MaxPolicy = segmented_reduce_runtime_tuning_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<segmented_reduce_runtime_tuning_policy>(*this);
  }
};

static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_value_t init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init.type;
}

std::string get_device_segmented_reduce_kernel_name(
  std::string_view reduction_op_t,
  std::string_view input_iterator_t,
  std::string_view output_iterator_t,
  std::string_view start_offset_iterator_t,
  std::string_view end_offset_iterator_t,
  cccl_value_t init,
  std::string_view accum_t)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_segmented_reduce_policy>(&chained_policy_t));

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  /*
  template <typename ChainedPolicyT,       // 0
            typename InputIteratorT,       // 1
            typename OutputIteratorT,      // 2
            typename BeginOffsetIteratorT, // 3
            typename EndOffsetIteratorT,   // 4
            typename OffsetT,              // 5
            typename ReductionOpT,         // 6
            typename InitT,                // 7
            typename AccumT>               // 8
   DeviceSegmentedReduceKernel(...);
  */
  return std::format(
    "cub::detail::reduce::DeviceSegmentedReduceKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}>",
    chained_policy_t, // 0
    input_iterator_t, // 1
    output_iterator_t, // 2
    start_offset_iterator_t, // 3
    end_offset_iterator_t, // 4
    offset_t, // 5
    reduction_op_t, // 6
    init_t, // 7
    accum_t); // 8
}

struct segmented_reduce_kernel_source
{
  cccl_device_segmented_reduce_build_result_t& build;

  std::size_t AccumSize() const
  {
    return build.accumulator_size;
  }
  CUkernel SegmentedReduceKernel() const
  {
    return build.segmented_reduce_kernel;
  }
};
} // namespace segmented_reduce

struct segmented_reduce_input_iterator_tag;
struct segmented_reduce_output_iterator_tag;
struct segmented_reduce_start_offset_iterator_tag;
struct segmented_reduce_end_offset_iterator_tag;
struct segmented_reduce_operation_tag;

CUresult cccl_device_segmented_reduce_build_ex(
  cccl_device_segmented_reduce_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
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
  cccl_build_config* config)
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "device_segmented_reduce";

    const int cc                 = cc_major * 10 + cc_minor;
    const cccl_type_info accum_t = segmented_reduce::get_accumulator_type(op, input_it, init);
    const auto accum_cpp         = cccl_type_enum_to_name(accum_t.type);

    const auto [input_iterator_name, input_iterator_src] =
      get_specialization<segmented_reduce_input_iterator_tag>(template_id<input_iterator_traits>(), input_it);

    const auto [output_iterator_name, output_iterator_src] = get_specialization<segmented_reduce_output_iterator_tag>(
      template_id<output_iterator_traits>(), output_it, accum_t);

    const auto [start_offset_iterator_name, start_offset_iterator_src] =
      get_specialization<segmented_reduce_start_offset_iterator_tag>(
        template_id<input_iterator_traits>(), start_offset_it);

    const auto [end_offset_iterator_name, end_offset_iterator_src] =
      get_specialization<segmented_reduce_end_offset_iterator_tag>(template_id<input_iterator_traits>(), end_offset_it);

    const auto [op_name, op_src] =
      get_specialization<segmented_reduce_operation_tag>(template_id<binary_user_operation_traits>(), op, accum_t);

    // OffsetT is checked to match have 64-bit size
    const auto offset_t = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

    const std::string dependent_definitions_src = std::format(
      R"XXX(
struct __align__({1}) storage_t {{
  char data[{0}];
}};
{2}
{3}
{4}
{5}
{6}
)XXX",
      input_it.value_type.size, // 0
      input_it.value_type.alignment, // 1
      input_iterator_src, // 2
      output_iterator_src, // 3
      op_src, // 4
      start_offset_iterator_src, // 5
      end_offset_iterator_src); // 6

    // Runtime parameter tuning

    const std::string ptx_arch = std::format("-arch=compute_{}{}", cc_major, cc_minor);

    constexpr size_t ptx_num_args      = 6;
    const char* ptx_args[ptx_num_args] = {
      ptx_arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true"};

    static constexpr std::string_view policy_wrapper_expr_tmpl =
      R"XXXX(cub::detail::reduce::MakeReducePolicyWrapper(cub::detail::reduce::policy_hub<{0}, {1}, {2}>::MaxPolicy::ActivePolicy{{}}))XXXX";

    const auto policy_wrapper_expr = std::format(
      policy_wrapper_expr_tmpl,
      accum_cpp, // 0
      offset_t, // 1
      op_name); // 2

    static constexpr std::string_view ptx_query_tu_src_tmpl = R"XXXX(
#include <cub/block/block_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
{0}
{1}
)XXXX";

    const auto ptx_query_tu_src =
      std::format(ptx_query_tu_src_tmpl, jit_template_header_contents, dependent_definitions_src);

    nlohmann::json runtime_policy = get_policy(policy_wrapper_expr, ptx_query_tu_src, ptx_args);

    using cub::detail::RuntimeReduceAgentPolicy;
    auto [segmented_reduce_policy,
          segmented_reduce_policy_str] = RuntimeReduceAgentPolicy::from_json(runtime_policy, "ReducePolicy");

    // agent_policy_t is to specify parameters like policy_hub does in dispatch_reduce.cuh
    constexpr std::string_view program_preamble_template = R"XXX(
#include <cub/block/block_reduce.cuh>
#include <cub/device/dispatch/kernels/segmented_reduce.cuh>
{0}
{1}
struct device_segmented_reduce_policy {{
  struct ActivePolicy {{
    {2}
  }};
}};
)XXX";

    std::string final_src = std::format(
      program_preamble_template,
      jit_template_header_contents, // 0
      dependent_definitions_src, // 1
      segmented_reduce_policy_str); // 2

    std::string segmented_reduce_kernel_name = segmented_reduce::get_device_segmented_reduce_kernel_name(
      op_name,
      input_iterator_name,
      output_iterator_name,
      start_offset_iterator_name,
      end_offset_iterator_name,
      init,
      accum_cpp);
    std::string segmented_reduce_kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

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

    cccl::detail::extend_args_with_build_config(args, config);

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};

    // add definition of binary operation op
    appender.append_operation(op);
    // add iterator definitions
    appender.add_iterator_definition(input_it);
    appender.add_iterator_definition(output_it);
    appender.add_iterator_definition(start_offset_it);
    appender.add_iterator_definition(end_offset_it);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
        ->add_expression({segmented_reduce_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({segmented_reduce_kernel_name, segmented_reduce_kernel_lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
        ->finalize_program();

    // populate build struct members
    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(
      &build_ptr->segmented_reduce_kernel, build_ptr->library, segmented_reduce_kernel_lowered_name.c_str()));

    build_ptr->cc               = cc;
    build_ptr->cubin            = (void*) result.data.release();
    build_ptr->cubin_size       = result.size;
    build_ptr->accumulator_size = accum_t.size;
    build_ptr->runtime_policy   = new segmented_reduce::segmented_reduce_runtime_tuning_policy{segmented_reduce_policy};
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_segmented_reduce_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
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
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::DispatchSegmentedReduce<
      indirect_arg_t, // InputIteratorT
      indirect_iterator_t, // OutputIteratorT
      indirect_iterator_t, // BeginSegmentIteratorT
      indirect_iterator_t, // EndSegmentIteratorT
      OffsetT, // OffsetT
      indirect_arg_t, // ReductionOpT
      indirect_arg_t, // InitT
      void, // AccumT
      segmented_reduce::segmented_reduce_runtime_tuning_policy, // PolicHub
      segmented_reduce::segmented_reduce_kernel_source, // KernelSource
      cub::detail::CudaDriverLauncherFactory>:: // KernelLaunchFactory
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_in,
        indirect_iterator_t{d_out},
        num_segments,
        indirect_iterator_t{start_offset},
        indirect_iterator_t{end_offset},
        op,
        init,
        stream,
        /* kernel_source */ {build},
        /* launcher_factory &*/ cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        /* policy */ *reinterpret_cast<segmented_reduce::segmented_reduce_runtime_tuning_policy*>(build.runtime_policy));

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
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    // allocation behind cubin is owned by unique_ptr with delete[] deleter now
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
};
