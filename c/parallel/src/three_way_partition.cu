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
#include <cub/device/dispatch/dispatch_three_way_partition.cuh> // cub::DispatchThreeWayPartitionIf
#include <cub/device/dispatch/kernels/three_way_partition.cuh> // DeviceThreeWayPartition kernels
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh> // policy_hub

#include <exception>
#include <format>
#include <string>
#include <string_view>
#include <type_traits> // std::is_same_v
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
#include <cccl/c/three_way_partition.h>
#include <cccl/c/types.h>
#include <nlohmann/json.hpp>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

struct device_three_way_partition_policy;
using OffsetT = ptrdiff_t;
static_assert(std::is_same_v<cub::detail::choose_signed_offset<OffsetT>::type, OffsetT>, "OffsetT must be long");

// check we can map OffsetT to cuda::std::int64_t
static_assert(std::is_signed_v<OffsetT>);
static_assert(sizeof(OffsetT) == sizeof(cuda::std::int64_t));

namespace three_way_partition
{

struct three_way_partition_runtime_tuning_policy
{
  cub::detail::RuntimeThreeWayPartitionAgentPolicy three_way_partition;

  auto ThreeWayPartition() const
  {
    return three_way_partition;
  }

  using MaxPolicy = three_way_partition_runtime_tuning_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<three_way_partition_runtime_tuning_policy>(*this);
  }
};

struct three_way_partition_kernel_source
{
  cccl_device_three_way_partition_build_result_t& build;

  CUkernel ThreeWayPartitionInitKernel() const
  {
    return build.three_way_partition_init_kernel;
  }

  CUkernel ThreeWayPartitionKernel() const
  {
    return build.three_way_partition_kernel;
  }
};

std::string get_three_way_partition_init_kernel_name(std::string_view num_selected_out_iterator_name)
{
  constexpr std::string_view scan_tile_state_t = "cub::detail::three_way_partition::ScanTileStateT";
  return std::format("cub::detail::three_way_partition::DeviceThreeWayPartitionInitKernel<{0}, {1}>",
                     scan_tile_state_t, // 0
                     num_selected_out_iterator_name); // 1
}

std::string get_three_way_partition_kernel_name(
  std::string_view d_in_iterator_name,
  std::string_view d_first_part_out_iterator_name,
  std::string_view d_second_part_out_iterator_name,
  std::string_view d_unselected_out_iterator_name,
  std::string_view d_num_selected_out_iterator_name,
  std::string_view select_first_part_op_name,
  std::string_view select_second_part_op_name)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_three_way_partition_policy>(&chained_policy_t));

  constexpr std::string_view scan_tile_state_t = "cub::detail::three_way_partition::ScanTileStateT";

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  const std::string streaming_context_t =
    std::format("cub::detail::three_way_partition::streaming_context_t<{0}>", offset_t);

  return std::format(
    "cub::detail::three_way_partition::DeviceThreeWayPartitionKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, "
    "{10}>",
    chained_policy_t, // 0
    d_in_iterator_name, // 1
    d_first_part_out_iterator_name, // 2
    d_second_part_out_iterator_name, // 3
    d_unselected_out_iterator_name, // 4
    d_num_selected_out_iterator_name, // 5
    scan_tile_state_t, // 6
    select_first_part_op_name, // 7
    select_second_part_op_name, // 8
    "cub::detail::three_way_partition::per_partition_offset_t", // 9
    streaming_context_t // 10
  );
}

std::string get_three_way_partition_policy_delay_constructor(const nlohmann::json& partition_policy)
{
  auto delay_ctor_info = partition_policy["DelayConstructor"];

  std::string delay_ctor_params;
  for (auto&& param : delay_ctor_info["params"])
  {
    delay_ctor_params.append(to_string(param) + ", ");
  }
  delay_ctor_params.erase(delay_ctor_params.size() - 2); // remove last ", "

  return std::format("cub::detail::{}<{}>", delay_ctor_info["name"].get<std::string>(), delay_ctor_params);
}

std::string inject_delay_constructor_into_three_way_policy(
  const std::string& three_way_partition_policy_str, const std::string& delay_constructor_type)
{
  // Insert before the final closing of the struct (right before the sequence "};")
  const std::string needle = "};";
  const auto pos           = three_way_partition_policy_str.rfind(needle);
  if (pos == std::string::npos)
  {
    return three_way_partition_policy_str; // unexpected; return as-is
  }
  const std::string insertion =
    std::format("\n  struct detail {{ using delay_constructor_t = {}; }}; \n", delay_constructor_type);
  std::string out = three_way_partition_policy_str;
  out.insert(pos, insertion);
  return out;
}
} // namespace three_way_partition

struct three_way_partition_input_iterator_tag;
struct three_way_partition_first_part_output_iterator_tag;
struct three_way_partition_second_part_output_iterator_tag;
struct three_way_partition_unselected_output_iterator_tag;
struct three_way_partition_num_selected_output_iterator_tag;
struct three_way_partition_select_first_part_operation_tag;
struct three_way_partition_select_second_part_operation_tag;

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
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "device_three_way_partition";

    const int cc = cc_major * 10 + cc_minor;

    const auto [d_in_iterator_name, d_in_iterator_src] =
      get_specialization<three_way_partition_input_iterator_tag>(template_id<input_iterator_traits>(), d_in);
    const auto [d_first_part_out_iterator_name, d_first_part_out_iterator_src] =
      get_specialization<three_way_partition_first_part_output_iterator_tag>(
        template_id<output_iterator_traits>(), d_first_part_out, d_first_part_out.value_type);
    const auto [d_second_part_out_iterator_name, d_second_part_out_iterator_src] =
      get_specialization<three_way_partition_second_part_output_iterator_tag>(
        template_id<output_iterator_traits>(), d_second_part_out, d_second_part_out.value_type);
    const auto [d_unselected_out_iterator_name, d_unselected_out_iterator_src] =
      get_specialization<three_way_partition_unselected_output_iterator_tag>(
        template_id<output_iterator_traits>(), d_unselected_out, d_unselected_out.value_type);
    const auto [d_num_selected_out_iterator_name, d_num_selected_out_iterator_src] =
      get_specialization<three_way_partition_num_selected_output_iterator_tag>(
        template_id<output_iterator_traits>(), d_num_selected_out, d_num_selected_out.value_type);

    cccl_type_info selector_result_t{sizeof(bool), alignof(bool), cccl_type_enum::CCCL_BOOLEAN};

    const auto [select_first_part_op_name, select_first_part_op_src] =
      get_specialization<three_way_partition_select_first_part_operation_tag>(
        template_id<user_operation_traits>(), select_first_part_op, selector_result_t, d_in.value_type);
    const auto [select_second_part_op_name, select_second_part_op_src] =
      get_specialization<three_way_partition_select_second_part_operation_tag>(
        template_id<user_operation_traits>(), select_second_part_op, selector_result_t, d_in.value_type);

    const auto offset_t = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT64);

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
{7}
{8}
)XXX",
      d_in.value_type.size, // 0
      d_in.value_type.alignment, // 1
      d_in_iterator_src, // 2
      d_first_part_out_iterator_src, // 3
      d_second_part_out_iterator_src, // 4
      d_unselected_out_iterator_src, // 5
      d_num_selected_out_iterator_src, // 6
      select_first_part_op_src, // 7
      select_second_part_op_src); // 8

    const std::string ptx_arch = std::format("-arch=compute_{}{}", cc_major, cc_minor);

    constexpr size_t ptx_num_args      = 6;
    const char* ptx_args[ptx_num_args] = {
      ptx_arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true"};

    static constexpr std::string_view policy_wrapper_expr_tmpl =
      R"XXXX(cub::detail::three_way_partition::MakeThreeWayPartitionPolicyWrapper(cub::detail::three_way_partition::policy_hub<{0}, {1}>::MaxPolicy::ActivePolicy{{}}))XXXX";

    const std::string key_t = cccl_type_enum_to_name(d_in.value_type.type);

    const auto policy_wrapper_expr = std::format(
      policy_wrapper_expr_tmpl,
      key_t, // 0
      offset_t); // 1

    static constexpr std::string_view ptx_query_tu_src_tmpl = R"XXXX(
#include <cub/device/dispatch/kernels/three_way_partition.cuh>
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh>
{0}
{1}
)XXXX";

    const auto ptx_query_tu_src =
      std::format(ptx_query_tu_src_tmpl, jit_template_header_contents, dependent_definitions_src);

    nlohmann::json runtime_policy = get_policy(policy_wrapper_expr, ptx_query_tu_src, ptx_args);

    using cub::detail::RuntimeThreeWayPartitionAgentPolicy;
    auto [three_way_partition_policy, three_way_partition_policy_str] =
      RuntimeThreeWayPartitionAgentPolicy::from_json(runtime_policy, "ThreeWayPartitionPolicy");

    const std::string three_way_partition_policy_delay_constructor =
      three_way_partition::get_three_way_partition_policy_delay_constructor(runtime_policy);

    const std::string injected_three_way_partition_policy_str =
      three_way_partition::inject_delay_constructor_into_three_way_policy(
        three_way_partition_policy_str, three_way_partition_policy_delay_constructor);
    constexpr std::string_view program_preamble_template = R"XXX(
#include <cub/device/dispatch/kernels/three_way_partition.cuh>
{0}
{1}
struct device_three_way_partition_policy {{
  struct ActivePolicy {{
    {2}
  }};
}};
)XXX";

    std::string final_src = std::format(
      program_preamble_template,
      jit_template_header_contents, // 0
      dependent_definitions_src, // 1
      injected_three_way_partition_policy_str); // 2

    std::string three_way_partition_init_kernel_name =
      three_way_partition::get_three_way_partition_init_kernel_name(d_num_selected_out_iterator_name);
    std::string three_way_partition_kernel_name = three_way_partition::get_three_way_partition_kernel_name(
      d_in_iterator_name,
      d_first_part_out_iterator_name,
      d_second_part_out_iterator_name,
      d_unselected_out_iterator_name,
      d_num_selected_out_iterator_name,
      select_first_part_op_name,
      select_second_part_op_name);
    std::string three_way_partition_init_kernel_lowered_name;
    std::string three_way_partition_kernel_lowered_name;

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

    appender.append_operation(select_first_part_op);
    appender.append_operation(select_second_part_op);
    appender.add_iterator_definition(d_in);
    appender.add_iterator_definition(d_first_part_out);
    appender.add_iterator_definition(d_second_part_out);
    appender.add_iterator_definition(d_unselected_out);
    appender.add_iterator_definition(d_num_selected_out);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
        ->add_expression({three_way_partition_init_kernel_name})
        ->add_expression({three_way_partition_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({three_way_partition_init_kernel_name, three_way_partition_init_kernel_lowered_name})
        ->get_name({three_way_partition_kernel_name, three_way_partition_kernel_lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
        ->finalize_program();

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->three_way_partition_init_kernel,
                             build_ptr->library,
                             three_way_partition_init_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build_ptr->three_way_partition_kernel, build_ptr->library, three_way_partition_kernel_lowered_name.c_str()));

    build_ptr->cc         = cc;
    build_ptr->cubin      = (void*) result.data.release();
    build_ptr->cubin_size = result.size;
    build_ptr->runtime_policy =
      new three_way_partition::three_way_partition_runtime_tuning_policy{three_way_partition_policy};
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_three_way_partition_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

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
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::DispatchThreeWayPartitionIf<
      indirect_arg_t, // InputIteratorT
      indirect_arg_t, // FirstOutputIteratorT
      indirect_arg_t, // SecondOutputIteratorT
      indirect_arg_t, // UnselectedOutputIteratorT
      indirect_arg_t, // NumSelectedIteratorT
      indirect_arg_t, // SelectFirstPartOp
      indirect_arg_t, // SelectSecondPartOp
      OffsetT, // OffsetT
      three_way_partition::three_way_partition_runtime_tuning_policy, // PolicyHub
      three_way_partition::three_way_partition_kernel_source, // KernelSource
      cub::detail::CudaDriverLauncherFactory>::
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_in,
        d_first_part_out,
        d_second_part_out,
        d_unselected_out,
        d_num_selected_out,
        select_first_part_op,
        select_second_part_op,
        num_items,
        stream,
        /* kernel_source */ {build},
        /* launcher_factory */ cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        /* policy */
        *reinterpret_cast<three_way_partition::three_way_partition_runtime_tuning_policy*>(build.runtime_policy));

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_three_way_partition(): %s\n", exc.what());
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

CUresult cccl_device_three_way_partition_cleanup(cccl_device_three_way_partition_build_result_t* bld_ptr)
{
  try
  {
    if (bld_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(bld_ptr->cubin));
    std::unique_ptr<char[]> policy(reinterpret_cast<char*>(bld_ptr->runtime_policy));
    check(cuLibraryUnload(bld_ptr->library));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_three_way_partition_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
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
