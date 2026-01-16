//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/detail/ptx-json-parser.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <format>
#include <vector>

#include "cccl/c/types.h"
#include "cub/util_type.cuh"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/radix_sort.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

namespace radix_sort
{
using namespace cub::detail::radix_sort_runtime_policies;

struct radix_sort_runtime_tuning_policy
{
  RuntimeRadixSortHistogramAgentPolicy histogram;
  RuntimeRadixSortExclusiveSumAgentPolicy exclusive_sum;
  RuntimeRadixSortOnesweepAgentPolicy onesweep;
  cub::detail::RuntimeScanAgentPolicy scan;
  cub::detail::RuntimeRadixSortDownsweepAgentPolicy downsweep;
  cub::detail::RuntimeRadixSortDownsweepAgentPolicy alt_downsweep;
  RuntimeRadixSortUpsweepAgentPolicy upsweep;
  RuntimeRadixSortUpsweepAgentPolicy alt_upsweep;
  cub::detail::RuntimeRadixSortDownsweepAgentPolicy single_tile;
  bool is_onesweep;

  auto Histogram() const
  {
    return histogram;
  }

  auto ExclusiveSum() const
  {
    return exclusive_sum;
  }

  auto Onesweep() const
  {
    return onesweep;
  }

  auto Scan() const
  {
    return scan;
  }

  auto Downsweep() const
  {
    return downsweep;
  }

  auto AltDownsweep() const
  {
    return alt_downsweep;
  }

  auto Upsweep() const
  {
    return upsweep;
  }

  auto AltUpsweep() const
  {
    return alt_upsweep;
  }

  auto SingleTile() const
  {
    return single_tile;
  }

  bool IsOnesweep() const
  {
    return is_onesweep;
  }

  template <typename PolicyT>
  CUB_RUNTIME_FUNCTION static constexpr int RadixBits(PolicyT policy)
  {
    return policy.RadixBits();
  }

  template <typename PolicyT>
  CUB_RUNTIME_FUNCTION static constexpr int BlockThreads(PolicyT policy)
  {
    return policy.BlockThreads();
  }

  using MaxPolicy = radix_sort_runtime_tuning_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<radix_sort_runtime_tuning_policy>(*this);
  }
};

std::string get_single_tile_kernel_name(
  std::string_view chained_policy_t,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortSingleTileKernel<{0}, {1}, {2}, {3}, {4}, {5}>",
    chained_policy_t,
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    value_t,
    offset_t,
    "op_wrapper");
}

std::string get_upsweep_kernel_name(
  std::string_view chained_policy_t,
  bool alt_digit_bits,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortUpsweepKernel<{0}, {1}, {2}, {3}, {4}, {5}>",
    chained_policy_t,
    alt_digit_bits ? "true" : "false",
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    offset_t,
    "op_wrapper");
}

std::string get_scan_bins_kernel_name(std::string_view chained_policy_t, std::string_view offset_t)
{
  return std::format("cub::detail::radix_sort::RadixSortScanBinsKernel<{0}, {1}>", chained_policy_t, offset_t);
}

std::string get_downsweep_kernel_name(
  std::string_view chained_policy_t,
  bool alt_digit_bits,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortDownsweepKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}>",
    chained_policy_t,
    alt_digit_bits ? "true" : "false",
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    value_t,
    offset_t,
    "op_wrapper");
}

std::string get_histogram_kernel_name(
  std::string_view chained_policy_t, cccl_sort_order_t sort_order, std::string_view key_t, std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortHistogramKernel<{0}, {1}, {2}, {3}, {4}>",
    chained_policy_t,
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    offset_t,
    "op_wrapper");
}

std::string get_exclusive_sum_kernel_name(std::string_view chained_policy_t, std::string_view offset_t)
{
  return std::format("cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<{0}, {1}>", chained_policy_t, offset_t);
}

std::string get_onesweep_kernel_name(
  std::string_view chained_policy_t,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<{0}, {1}, {2}, {3}, {4}, int, int, {5}>",
    chained_policy_t,
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    value_t,
    offset_t,
    "op_wrapper");
}

struct radix_sort_kernel_source
{
  cccl_device_radix_sort_build_result_t& build;

  CUkernel RadixSortSingleTileKernel() const
  {
    return build.single_tile_kernel;
  }

  CUkernel RadixSortUpsweepKernel() const
  {
    return build.upsweep_kernel;
  }

  CUkernel RadixSortAltUpsweepKernel() const
  {
    return build.alt_upsweep_kernel;
  }

  CUkernel DeviceRadixSortScanBinsKernel() const
  {
    return build.scan_bins_kernel;
  }

  CUkernel RadixSortDownsweepKernel() const
  {
    return build.downsweep_kernel;
  }

  CUkernel RadixSortAltDownsweepKernel() const
  {
    return build.alt_downsweep_kernel;
  }

  CUkernel RadixSortHistogramKernel() const
  {
    return build.histogram_kernel;
  }

  CUkernel RadixSortExclusiveSumKernel() const
  {
    return build.exclusive_sum_kernel;
  }

  CUkernel RadixSortOnesweepKernel() const
  {
    return build.onesweep_kernel;
  }

  std::size_t KeySize() const
  {
    return build.key_type.size;
  }

  std::size_t ValueSize() const
  {
    return build.value_type.size;
  }
};
} // namespace radix_sort

CUresult cccl_device_radix_sort_build_ex(
  cccl_device_radix_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* decomposer_return_type,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  const char* name = "test";

  const int cc       = cc_major * 10 + cc_minor;
  const auto key_cpp = cccl_type_enum_to_name(input_keys_it.value_type.type);
  const auto value_cpp =
    input_values_it.type == cccl_iterator_kind_t::CCCL_POINTER && input_values_it.state == nullptr
      ? "cub::NullType"
      : cccl_type_enum_to_name(input_values_it.value_type.type);
  const std::string op_src =
    (decomposer.name == nullptr || (decomposer.name != nullptr && decomposer.name[0] == '\0'))
      ? "using op_wrapper = cub::detail::identity_decomposer_t;"
      : make_kernel_user_unary_operator(key_cpp, decomposer_return_type, decomposer);
  constexpr std::string_view chained_policy_t = "device_radix_sort_policy";

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  const auto policy_hub_expr =
    std::format("cub::detail::radix_sort::policy_hub<{}, {}, {}>", key_cpp, value_cpp, offset_t);

  const std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/tuning/tuning_radix_sort.cuh>
#include <cub/device/dispatch/kernels/kernel_radix_sort.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>

struct __align__({1}) storage_t {{
  char data[{0}];
}};
struct __align__({3}) values_storage_t {{
  char data[{2}];
}};
{4}
using {5} = {6}::MaxPolicy;

#include <cub/detail/ptx-json/json.cuh>
__device__ consteval auto& policy_generator() {{
  return ptx_json::id<ptx_json::string("device_radix_sort_policy")>()
    = cub::detail::radix_sort::RadixSortPolicyWrapper<{5}::ActivePolicy>::EncodedPolicy();
}}
)XXX",
    input_keys_it.value_type.size, // 0
    input_keys_it.value_type.alignment, // 1
    input_values_it.value_type.size, // 2
    input_values_it.value_type.alignment, // 3
    op_src, // 4
    chained_policy_t, // 5
    policy_hub_expr); // 6

#if false // CCCL_DEBUGGING_SWITCH
  fflush(stderr);
  printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
  fflush(stdout);
#endif

  std::string single_tile_kernel_name =
    radix_sort::get_single_tile_kernel_name(chained_policy_t, sort_order, key_cpp, value_cpp, offset_t);
  std::string upsweep_kernel_name =
    radix_sort::get_upsweep_kernel_name(chained_policy_t, false, sort_order, key_cpp, offset_t);
  std::string alt_upsweep_kernel_name =
    radix_sort::get_upsweep_kernel_name(chained_policy_t, true, sort_order, key_cpp, offset_t);
  std::string scan_bins_kernel_name = radix_sort::get_scan_bins_kernel_name(chained_policy_t, offset_t);
  std::string downsweep_kernel_name =
    radix_sort::get_downsweep_kernel_name(chained_policy_t, false, sort_order, key_cpp, value_cpp, offset_t);
  std::string alt_downsweep_kernel_name =
    radix_sort::get_downsweep_kernel_name(chained_policy_t, true, sort_order, key_cpp, value_cpp, offset_t);
  std::string histogram_kernel_name =
    radix_sort::get_histogram_kernel_name(chained_policy_t, sort_order, key_cpp, offset_t);
  std::string exclusive_sum_kernel_name = radix_sort::get_exclusive_sum_kernel_name(chained_policy_t, offset_t);
  std::string onesweep_kernel_name =
    radix_sort::get_onesweep_kernel_name(chained_policy_t, sort_order, key_cpp, value_cpp, offset_t);
  std::string single_tile_kernel_lowered_name;
  std::string upsweep_kernel_lowered_name;
  std::string alt_upsweep_kernel_lowered_name;
  std::string scan_bins_kernel_lowered_name;
  std::string downsweep_kernel_lowered_name;
  std::string alt_downsweep_kernel_lowered_name;
  std::string histogram_kernel_lowered_name;
  std::string exclusive_sum_kernel_lowered_name;
  std::string onesweep_kernel_lowered_name;

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

  std::vector<const char*> args = {
    arch.c_str(),
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    "-rdc=true",
    "-dlto",
    "-default-device",
    "-DCUB_DISABLE_CDP",
    "-DCUB_ENABLE_POLICY_PTX_JSON",
    "-std=c++20"};

  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  // Collect all LTO-IRs to be linked.
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};
  appender.append_operation(decomposer);

  nvrtc_link_result result =
    begin_linking_nvrtc_program(num_lto_args, lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({single_tile_kernel_name})
      ->add_expression({upsweep_kernel_name})
      ->add_expression({alt_upsweep_kernel_name})
      ->add_expression({scan_bins_kernel_name})
      ->add_expression({downsweep_kernel_name})
      ->add_expression({alt_downsweep_kernel_name})
      ->add_expression({histogram_kernel_name})
      ->add_expression({exclusive_sum_kernel_name})
      ->add_expression({onesweep_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({single_tile_kernel_name, single_tile_kernel_lowered_name})
      ->get_name({upsweep_kernel_name, upsweep_kernel_lowered_name})
      ->get_name({alt_upsweep_kernel_name, alt_upsweep_kernel_lowered_name})
      ->get_name({scan_bins_kernel_name, scan_bins_kernel_lowered_name})
      ->get_name({downsweep_kernel_name, downsweep_kernel_lowered_name})
      ->get_name({alt_downsweep_kernel_name, alt_downsweep_kernel_lowered_name})
      ->get_name({histogram_kernel_name, histogram_kernel_lowered_name})
      ->get_name({exclusive_sum_kernel_name, exclusive_sum_kernel_lowered_name})
      ->get_name({onesweep_kernel_name, onesweep_kernel_lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(
    cuLibraryGetKernel(&build_ptr->single_tile_kernel, build_ptr->library, single_tile_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build_ptr->upsweep_kernel, build_ptr->library, upsweep_kernel_lowered_name.c_str()));
  check(
    cuLibraryGetKernel(&build_ptr->alt_upsweep_kernel, build_ptr->library, alt_upsweep_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build_ptr->scan_bins_kernel, build_ptr->library, scan_bins_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build_ptr->downsweep_kernel, build_ptr->library, downsweep_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(
    &build_ptr->alt_downsweep_kernel, build_ptr->library, alt_downsweep_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build_ptr->histogram_kernel, build_ptr->library, histogram_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(
    &build_ptr->exclusive_sum_kernel, build_ptr->library, exclusive_sum_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build_ptr->onesweep_kernel, build_ptr->library, onesweep_kernel_lowered_name.c_str()));

  nlohmann::json runtime_policy =
    cub::detail::ptx_json::parse("device_radix_sort_policy", {result.data.get(), result.size});

  using namespace cub::detail::radix_sort_runtime_policies;
  using cub::detail::RuntimeScanAgentPolicy;
  auto single_tile_policy =
    cub::detail::RuntimeRadixSortDownsweepAgentPolicy::from_json(runtime_policy, "SingleTilePolicy");
  auto onesweep_policy    = RuntimeRadixSortOnesweepAgentPolicy::from_json(runtime_policy, "OnesweepPolicy");
  auto upsweep_policy     = RuntimeRadixSortUpsweepAgentPolicy::from_json(runtime_policy, "UpsweepPolicy");
  auto alt_upsweep_policy = RuntimeRadixSortUpsweepAgentPolicy::from_json(runtime_policy, "AltUpsweepPolicy");
  auto downsweep_policy =
    cub::detail::RuntimeRadixSortDownsweepAgentPolicy::from_json(runtime_policy, "DownsweepPolicy");
  auto alt_downsweep_policy =
    cub::detail::RuntimeRadixSortDownsweepAgentPolicy::from_json(runtime_policy, "AltDownsweepPolicy");
  auto histogram_policy     = RuntimeRadixSortHistogramAgentPolicy::from_json(runtime_policy, "HistogramPolicy");
  auto exclusive_sum_policy = RuntimeRadixSortExclusiveSumAgentPolicy::from_json(runtime_policy, "ExclusiveSumPolicy");
  auto scan_policy          = RuntimeScanAgentPolicy::from_json(runtime_policy, "ScanPolicy");
  auto is_onesweep          = runtime_policy["Onesweep"].get<bool>();

  build_ptr->cc             = cc;
  build_ptr->cubin          = (void*) result.data.release();
  build_ptr->cubin_size     = result.size;
  build_ptr->key_type       = input_keys_it.value_type;
  build_ptr->value_type     = input_values_it.value_type;
  build_ptr->order          = sort_order;
  build_ptr->runtime_policy = new radix_sort::radix_sort_runtime_tuning_policy{
    histogram_policy,
    exclusive_sum_policy,
    onesweep_policy,
    scan_policy,
    downsweep_policy,
    alt_downsweep_policy,
    upsweep_policy,
    alt_upsweep_policy,
    single_tile_policy,
    is_onesweep};

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_radix_sort_build(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

template <cub::SortOrder Order>
CUresult cccl_device_radix_sort_impl(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t decomposer,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  if (cccl_iterator_kind_t::CCCL_POINTER != d_keys_in.type || cccl_iterator_kind_t::CCCL_POINTER != d_values_in.type
      || cccl_iterator_kind_t::CCCL_POINTER != d_keys_out.type
      || cccl_iterator_kind_t::CCCL_POINTER != d_values_out.type)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_radix_sort(): radix sort input must be a pointer\n");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  CUresult error = CUDA_SUCCESS;
  bool pushed    = false;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    indirect_arg_t key_arg_in{d_keys_in};
    indirect_arg_t key_arg_out{d_keys_out};
    cub::DoubleBuffer<indirect_arg_t> d_keys_buffer(
      *static_cast<indirect_arg_t**>(&key_arg_in), *static_cast<indirect_arg_t**>(&key_arg_out));

    indirect_arg_t val_arg_in{d_values_in};
    indirect_arg_t val_arg_out{d_values_out};
    cub::DoubleBuffer<indirect_arg_t> d_values_buffer(
      *static_cast<indirect_arg_t**>(&val_arg_in), *static_cast<indirect_arg_t**>(&val_arg_out));

    auto exec_status = cub::DispatchRadixSort<
      Order,
      indirect_arg_t,
      indirect_arg_t,
      OffsetT,
      indirect_arg_t,
      radix_sort::radix_sort_runtime_tuning_policy,
      radix_sort::radix_sort_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_keys_buffer,
        d_values_buffer,
        num_items,
        begin_bit,
        end_bit,
        is_overwrite_okay,
        stream,
        decomposer,
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        *reinterpret_cast<radix_sort::radix_sort_runtime_tuning_policy*>(build.runtime_policy));

    *selector = d_keys_buffer.selector;
    error     = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort(): %s\n", exc.what());
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

CUresult cccl_device_radix_sort(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t decomposer,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  auto radix_sort_impl =
    (build.order == CCCL_ASCENDING)
      ? cccl_device_radix_sort_impl<cub::SortOrder::Ascending>
      : cccl_device_radix_sort_impl<cub::SortOrder::Descending>;
  return radix_sort_impl(
    build,
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    decomposer,
    num_items,
    begin_bit,
    end_bit,
    is_overwrite_okay,
    selector,
    stream);
}

CUresult cccl_device_radix_sort_build(
  cccl_device_radix_sort_build_result_t* build,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* decomposer_return_type,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_radix_sort_build_ex(
    build,
    sort_order,
    input_keys_it,
    input_values_it,
    decomposer,
    decomposer_return_type,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_radix_sort_cleanup(cccl_device_radix_sort_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
  std::unique_ptr<char[]> runtime_policy(reinterpret_cast<char*>(build_ptr->runtime_policy));
  check(cuLibraryUnload(build_ptr->library));

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_radix_sort_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}
