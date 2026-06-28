//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cuda/__type_traits/is_trivially_copyable.h>

#include <cstdlib>
#include <cstring>
#include <format>
#include <mutex>
#include <vector>

#include "cccl/c/types.h"
#include "cub/util_type.cuh"
#include "kernels/operators.h"
#include "util/aot_serialize.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/nvjitlink.h"
#include "util/types.h"
#include <cccl/c/aot.h>
#include <cccl/c/radix_sort.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

namespace radix_sort
{
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

std::string get_init_kernel_name(std::string_view chained_policy_t, std::string_view init_t0, std::string_view init_t1)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortInitKernel<{0}, {1}, {2}>", chained_policy_t, init_t0, init_t1);
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

  CUkernel RadixSortInitBinsAndCountersKernel() const
  {
    return build.init_bins_and_counters_kernel;
  }

  CUkernel RadixSortInitLookbackKernel() const
  {
    return build.init_lookback_kernel;
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

  indirect_arg_t* AdvanceKeys(indirect_arg_t* ptr, OffsetT offset) const
  {
    return reinterpret_cast<indirect_arg_t*>(reinterpret_cast<char*>(ptr) + offset * build.key_type.size);
  }

  indirect_arg_t* AdvanceValues(indirect_arg_t* ptr, OffsetT offset) const
  {
    return reinterpret_cast<indirect_arg_t*>(reinterpret_cast<char*>(ptr) + offset * build.value_type.size);
  }
};
} // namespace radix_sort

CUresult cccl_device_radix_sort_compile(
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

  const auto key_cpp   = cccl_type_enum_to_name(input_keys_it.value_type.type);
  const auto keys_only = input_values_it.type == cccl_iterator_kind_t::CCCL_POINTER && input_values_it.state == nullptr;
  const auto value_cpp = keys_only ? "cub::NullType" : cccl_type_enum_to_name(input_values_it.value_type.type);
  const std::string op_src =
    (decomposer.name == nullptr || (decomposer.name != nullptr && decomposer.name[0] == '\0'))
      ? "using op_wrapper = cub::detail::identity_decomposer_t;"
      : make_kernel_user_unary_operator(key_cpp, decomposer_return_type, decomposer);
  constexpr std::string_view chained_policy_t = "device_radix_sort_policy";

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  const auto key_type = cccl_type_enum_to_cub_type(input_keys_it.value_type.type);

  const auto policy_sel = cub::detail::radix_sort::policy_selector{
    static_cast<int>(input_keys_it.value_type.size),
    // FIXME(bgruber): input_values_it.value_type.size is 4 when it represents cub::NullType, which is very odd
    // TODO(bgruber): instead of 0 we should probably use int{sizeof(cub::NullType)}
    keys_only ? 0 : static_cast<int>(input_values_it.value_type.size),
    int{sizeof(OffsetT)},
    key_type};

  // TODO(bgruber): drop this if tuning policies become formattable
  std::stringstream policy_sel_str;
  policy_sel_str << policy_sel(cuda::compute_capability{cc_major, cc_minor});

  auto policy_hub_expr =
    std::format("cub::detail::radix_sort::policy_selector_from_types<{}, {}, {}>", key_cpp, value_cpp, offset_t);

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
using device_radix_sort_policy = {5};
using namespace cub;
using namespace cub::detail;
using namespace cub::detail::radix_sort;
using cub::LookbackDelayPolicy;
using cub::LookbackDelayAlgorithm;
static_assert(device_radix_sort_policy()(current_tuning_cc()) == {6}, "Host generated and JIT compiled policy mismatch");
)XXX",
    input_keys_it.value_type.size, // 0
    input_keys_it.value_type.alignment, // 1
    input_values_it.value_type.size, // 2
    input_values_it.value_type.alignment, // 3
    op_src, // 4
    policy_hub_expr, // 5
    policy_sel_str.view()); // 6

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
  std::string init_bins_and_counters_kernel_name = radix_sort::get_init_kernel_name(chained_policy_t, "int", offset_t);
  std::string init_lookback_kernel_name          = radix_sort::get_init_kernel_name(chained_policy_t, "int", "int");
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
  std::string init_bins_and_counters_kernel_lowered_name;
  std::string init_lookback_kernel_lowered_name;
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
    "-std=c++20"};

  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  const bool kernel_only = is_custom_op(decomposer);

  // Collect all LTO-IRs to be linked (empty in kernel-only mode).
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};
  appender.append_operation(decomposer);

  auto post_build =
    begin_linking_nvrtc_program(kernel_only ? 0 : num_lto_args, kernel_only ? nullptr : lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({single_tile_kernel_name})
      ->add_expression({upsweep_kernel_name})
      ->add_expression({alt_upsweep_kernel_name})
      ->add_expression({scan_bins_kernel_name})
      ->add_expression({downsweep_kernel_name})
      ->add_expression({alt_downsweep_kernel_name})
      ->add_expression({histogram_kernel_name})
      ->add_expression({exclusive_sum_kernel_name})
      ->add_expression({init_bins_and_counters_kernel_name})
      ->add_expression({init_lookback_kernel_name})
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
      ->get_name({init_bins_and_counters_kernel_name, init_bins_and_counters_kernel_lowered_name})
      ->get_name({init_lookback_kernel_name, init_lookback_kernel_lowered_name})
      ->get_name({onesweep_kernel_name, onesweep_kernel_lowered_name});

  struct free_deleter
  {
    void operator()(void* p) const
    {
      std::free(p);
    }
  };
  static_assert(::cuda::is_trivially_copyable_v<cub::detail::radix_sort::policy_selector>);
  const size_t policy_size = sizeof(policy_sel);
  std::unique_ptr<void, free_deleter> policy_ptr(std::malloc(policy_size));
  if (!policy_ptr)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(policy_ptr.get(), &policy_sel, sizeof(policy_sel));
  auto single_tile_name   = std::unique_ptr<char[]>(duplicate_c_string(single_tile_kernel_lowered_name));
  auto upsweep_name       = std::unique_ptr<char[]>(duplicate_c_string(upsweep_kernel_lowered_name));
  auto alt_upsweep_name   = std::unique_ptr<char[]>(duplicate_c_string(alt_upsweep_kernel_lowered_name));
  auto scan_bins_name     = std::unique_ptr<char[]>(duplicate_c_string(scan_bins_kernel_lowered_name));
  auto downsweep_name     = std::unique_ptr<char[]>(duplicate_c_string(downsweep_kernel_lowered_name));
  auto alt_downsweep_name = std::unique_ptr<char[]>(duplicate_c_string(alt_downsweep_kernel_lowered_name));
  auto histogram_name     = std::unique_ptr<char[]>(duplicate_c_string(histogram_kernel_lowered_name));
  auto exclusive_sum_name = std::unique_ptr<char[]>(duplicate_c_string(exclusive_sum_kernel_lowered_name));
  auto init_bins_and_counters_name =
    std::unique_ptr<char[]>(duplicate_c_string(init_bins_and_counters_kernel_lowered_name));
  auto init_lookback_name = std::unique_ptr<char[]>(duplicate_c_string(init_lookback_kernel_lowered_name));
  auto onesweep_name      = std::unique_ptr<char[]>(duplicate_c_string(onesweep_kernel_lowered_name));

  build_ptr->cc         = cc_major * 10 + cc_minor;
  build_ptr->key_type   = input_keys_it.value_type;
  build_ptr->value_type = input_values_it.value_type;
  build_ptr->order      = sort_order;
  // Zero-init fields set by _load, not _compile.
  build_ptr->library                       = nullptr;
  build_ptr->single_tile_kernel            = nullptr;
  build_ptr->upsweep_kernel                = nullptr;
  build_ptr->alt_upsweep_kernel            = nullptr;
  build_ptr->scan_bins_kernel              = nullptr;
  build_ptr->downsweep_kernel              = nullptr;
  build_ptr->alt_downsweep_kernel          = nullptr;
  build_ptr->histogram_kernel              = nullptr;
  build_ptr->exclusive_sum_kernel          = nullptr;
  build_ptr->init_bins_and_counters_kernel = nullptr;
  build_ptr->init_lookback_kernel          = nullptr;
  build_ptr->onesweep_kernel               = nullptr;

  // All potentially-throwing operations come before any release() calls so that
  // unique_ptrs automatically clean up on exception.
  if (kernel_only)
  {
    auto [ltoir_size, ltoir_data] = post_build->get_program_ltoir();
    build_ptr->payload            = ltoir_data.release();
    build_ptr->payload_size       = ltoir_size;
    build_ptr->payload_kind       = CCCL_PAYLOAD_LTOIR;
  }
  else
  {
    nvrtc_link_result result = post_build->link_program()->add_link_list(linkable_list)->finalize_program();
    build_ptr->payload       = (void*) result.data.release();
    build_ptr->payload_size  = result.size;
    build_ptr->payload_kind  = CCCL_PAYLOAD_CUBIN;
  }

  build_ptr->runtime_policy                             = policy_ptr.release();
  build_ptr->runtime_policy_size                        = policy_size;
  build_ptr->single_tile_kernel_lowered_name            = single_tile_name.release();
  build_ptr->upsweep_kernel_lowered_name                = upsweep_name.release();
  build_ptr->alt_upsweep_kernel_lowered_name            = alt_upsweep_name.release();
  build_ptr->scan_bins_kernel_lowered_name              = scan_bins_name.release();
  build_ptr->downsweep_kernel_lowered_name              = downsweep_name.release();
  build_ptr->alt_downsweep_kernel_lowered_name          = alt_downsweep_name.release();
  build_ptr->histogram_kernel_lowered_name              = histogram_name.release();
  build_ptr->exclusive_sum_kernel_lowered_name          = exclusive_sum_name.release();
  build_ptr->init_bins_and_counters_kernel_lowered_name = init_bins_and_counters_name.release();
  build_ptr->init_lookback_kernel_lowered_name          = init_lookback_name.release();
  build_ptr->onesweep_kernel_lowered_name               = onesweep_name.release();

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_radix_sort_compile(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_load(cccl_device_radix_sort_build_result_t* build_ptr)
try
{
  auto invalid_name = [](const char* n) {
    return n == nullptr || n[0] == '\0';
  };
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0
      || build_ptr->payload_kind != CCCL_PAYLOAD_CUBIN || invalid_name(build_ptr->single_tile_kernel_lowered_name)
      || invalid_name(build_ptr->upsweep_kernel_lowered_name)
      || invalid_name(build_ptr->alt_upsweep_kernel_lowered_name)
      || invalid_name(build_ptr->scan_bins_kernel_lowered_name)
      || invalid_name(build_ptr->downsweep_kernel_lowered_name)
      || invalid_name(build_ptr->alt_downsweep_kernel_lowered_name)
      || invalid_name(build_ptr->histogram_kernel_lowered_name)
      || invalid_name(build_ptr->exclusive_sum_kernel_lowered_name)
      || invalid_name(build_ptr->init_bins_and_counters_kernel_lowered_name)
      || invalid_name(build_ptr->init_lookback_kernel_lowered_name)
      || invalid_name(build_ptr->onesweep_kernel_lowered_name))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
    cuLibraryLoadData(&build_ptr->library, build_ptr->payload, nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (status != CUDA_SUCCESS)
  {
    return status;
  }
  try
  {
    check(cuLibraryGetKernel(
      &build_ptr->single_tile_kernel, build_ptr->library, build_ptr->single_tile_kernel_lowered_name));
    check(cuLibraryGetKernel(&build_ptr->upsweep_kernel, build_ptr->library, build_ptr->upsweep_kernel_lowered_name));
    check(cuLibraryGetKernel(
      &build_ptr->alt_upsweep_kernel, build_ptr->library, build_ptr->alt_upsweep_kernel_lowered_name));
    check(
      cuLibraryGetKernel(&build_ptr->scan_bins_kernel, build_ptr->library, build_ptr->scan_bins_kernel_lowered_name));
    check(
      cuLibraryGetKernel(&build_ptr->downsweep_kernel, build_ptr->library, build_ptr->downsweep_kernel_lowered_name));
    check(cuLibraryGetKernel(
      &build_ptr->alt_downsweep_kernel, build_ptr->library, build_ptr->alt_downsweep_kernel_lowered_name));
    check(
      cuLibraryGetKernel(&build_ptr->histogram_kernel, build_ptr->library, build_ptr->histogram_kernel_lowered_name));
    check(cuLibraryGetKernel(
      &build_ptr->exclusive_sum_kernel, build_ptr->library, build_ptr->exclusive_sum_kernel_lowered_name));
    check(cuLibraryGetKernel(&build_ptr->init_bins_and_counters_kernel,
                             build_ptr->library,
                             build_ptr->init_bins_and_counters_kernel_lowered_name));
    check(cuLibraryGetKernel(
      &build_ptr->init_lookback_kernel, build_ptr->library, build_ptr->init_lookback_kernel_lowered_name));
    check(cuLibraryGetKernel(&build_ptr->onesweep_kernel, build_ptr->library, build_ptr->onesweep_kernel_lowered_name));
  }
  catch (...)
  {
    cuLibraryUnload(build_ptr->library);
    build_ptr->library = nullptr;
    throw;
  }
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_radix_sort_load(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

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
{
  CUresult r = cccl_device_radix_sort_compile(
    build_ptr,
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
    config);
  if (r != CUDA_SUCCESS)
  {
    return r;
  }
  CUresult load_r = cccl_device_radix_sort_load(build_ptr);
  if (load_r != CUDA_SUCCESS)
  {
    cccl_device_radix_sort_cleanup(build_ptr);
  }
  return load_r;
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

    auto exec_status = cub::detail::radix_sort::dispatch<Order>(
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
      *static_cast<cub::detail::radix_sort::policy_selector*>(build.runtime_policy),
      radix_sort::radix_sort_kernel_source{build},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc});

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

  using namespace cub::detail::radix_sort;
  std::unique_ptr<char[]> payload(reinterpret_cast<char*>(build_ptr->payload));
  std::free(build_ptr->runtime_policy);
  for (char* p :
       {build_ptr->single_tile_kernel_lowered_name,
        build_ptr->upsweep_kernel_lowered_name,
        build_ptr->alt_upsweep_kernel_lowered_name,
        build_ptr->scan_bins_kernel_lowered_name,
        build_ptr->downsweep_kernel_lowered_name,
        build_ptr->alt_downsweep_kernel_lowered_name,
        build_ptr->histogram_kernel_lowered_name,
        build_ptr->exclusive_sum_kernel_lowered_name,
        build_ptr->init_bins_and_counters_kernel_lowered_name,
        build_ptr->init_lookback_kernel_lowered_name,
        build_ptr->onesweep_kernel_lowered_name})
  {
    delete[] p;
  }
  if (build_ptr->library != nullptr)
  {
    check(cuLibraryUnload(build_ptr->library));
  }

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_radix_sort_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_link_ltoir(
  cccl_device_radix_sort_build_result_t* build_ptr,
  const void** input_blobs,
  const size_t* input_sizes,
  size_t num_inputs)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0
      || build_ptr->payload_kind != CCCL_PAYLOAD_LTOIR)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int cc_major = build_ptr->cc / 10;
  const int cc_minor = build_ptr->cc % 10;
  std::vector<const void*> all_blobs;
  std::vector<size_t> all_sizes;
  all_blobs.push_back(build_ptr->payload);
  all_sizes.push_back(build_ptr->payload_size);
  if (num_inputs > 0 && (input_blobs == nullptr || input_sizes == nullptr))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (size_t i = 0; i < num_inputs; ++i)
  {
    if (input_blobs[i] == nullptr || input_sizes[i] == 0)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    all_blobs.push_back(input_blobs[i]);
    all_sizes.push_back(input_sizes[i]);
  }
  auto [cubin, cubin_size] = nvjitlink_link(all_blobs.data(), all_sizes.data(), all_blobs.size(), cc_major, cc_minor);
  delete[] static_cast<char*>(build_ptr->payload);
  build_ptr->payload      = (void*) cubin.release();
  build_ptr->payload_size = cubin_size;
  build_ptr->payload_kind = CCCL_PAYLOAD_CUBIN;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  printf("\nEXCEPTION in cccl_device_radix_sort_link_ltoir(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_serialize(
  const cccl_device_radix_sort_build_result_t* build_ptr, void** out_buf, size_t* out_size)
try
{
  if (build_ptr == nullptr || out_buf == nullptr || out_size == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (build_ptr->payload == nullptr || build_ptr->payload_size == 0 || build_ptr->runtime_policy == nullptr
      || build_ptr->runtime_policy_size == 0)
  {
    *out_buf  = nullptr;
    *out_size = 0;
    return CUDA_ERROR_INVALID_VALUE;
  }

  using namespace cccl::aot;
  buffer_writer w;
  write_header(w, CCCL_AOT_ALGO_RADIX_SORT, build_ptr->payload_kind, build_ptr->cc);
  write_type_info(w, build_ptr->key_type);
  write_type_info(w, build_ptr->value_type);
  w.write_pod<uint32_t>(static_cast<uint32_t>(build_ptr->order));
  w.write_blob(build_ptr->payload, build_ptr->payload_size);
  w.write_blob(build_ptr->runtime_policy, build_ptr->runtime_policy_size);
  w.write_cstring(build_ptr->single_tile_kernel_lowered_name);
  w.write_cstring(build_ptr->upsweep_kernel_lowered_name);
  w.write_cstring(build_ptr->alt_upsweep_kernel_lowered_name);
  w.write_cstring(build_ptr->scan_bins_kernel_lowered_name);
  w.write_cstring(build_ptr->downsweep_kernel_lowered_name);
  w.write_cstring(build_ptr->alt_downsweep_kernel_lowered_name);
  w.write_cstring(build_ptr->histogram_kernel_lowered_name);
  w.write_cstring(build_ptr->exclusive_sum_kernel_lowered_name);
  w.write_cstring(build_ptr->init_bins_and_counters_kernel_lowered_name);
  w.write_cstring(build_ptr->init_lookback_kernel_lowered_name);
  w.write_cstring(build_ptr->onesweep_kernel_lowered_name);
  w.release(out_buf, out_size);
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_radix_sort_serialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult
cccl_device_radix_sort_deserialize(cccl_device_radix_sort_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  using namespace cccl::aot;
  buffer_reader r{buf, size};
  const auto h = read_and_validate_header(r, CCCL_AOT_ALGO_RADIX_SORT);

  const auto key_t   = read_type_info(r);
  const auto value_t = read_type_info(r);
  const auto order   = static_cast<cccl_sort_order_t>(r.read_pod<uint32_t>());
  if (order != CCCL_ASCENDING && order != CCCL_DESCENDING)
  {
    throw std::runtime_error(std::format("aot blob: invalid sort order ({})", static_cast<uint32_t>(order)));
  }

  std::unique_ptr<char[]> payload_owner;
  size_t payload_size = 0;
  {
    void* p = nullptr;
    r.read_blob_new(&p, &payload_size);
    payload_owner.reset(static_cast<char*>(p));
  }
  if (payload_size == 0)
  {
    throw std::runtime_error("aot blob: empty payload");
  }

  std::unique_ptr<cub::detail::radix_sort::policy_selector, decltype(&std::free)> policy(
    static_cast<cub::detail::radix_sort::policy_selector*>(
      std::malloc(sizeof(cub::detail::radix_sort::policy_selector))),
    std::free);
  if (!policy)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  r.read_into(policy.get(), sizeof(cub::detail::radix_sort::policy_selector));

  std::unique_ptr<char[]> n_st{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_us{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_aus{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_sb{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_ds{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_ads{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_hist{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_es{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_ibc{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_il{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_os{r.read_cstring_dup()};

  cccl_device_radix_sort_build_result_t result{};
  result.cc                                         = static_cast<int>(h.cc);
  result.payload_kind                               = static_cast<cccl_payload_kind_t>(h.payload_kind);
  result.key_type                                   = key_t;
  result.value_type                                 = value_t;
  result.order                                      = order;
  result.payload                                    = payload_owner.release();
  result.payload_size                               = payload_size;
  result.runtime_policy                             = policy.release();
  result.runtime_policy_size                        = sizeof(cub::detail::radix_sort::policy_selector);
  result.single_tile_kernel_lowered_name            = n_st.release();
  result.upsweep_kernel_lowered_name                = n_us.release();
  result.alt_upsweep_kernel_lowered_name            = n_aus.release();
  result.scan_bins_kernel_lowered_name              = n_sb.release();
  result.downsweep_kernel_lowered_name              = n_ds.release();
  result.alt_downsweep_kernel_lowered_name          = n_ads.release();
  result.histogram_kernel_lowered_name              = n_hist.release();
  result.exclusive_sum_kernel_lowered_name          = n_es.release();
  result.init_bins_and_counters_kernel_lowered_name = n_ibc.release();
  result.init_lookback_kernel_lowered_name          = n_il.release();
  result.onesweep_kernel_lowered_name               = n_os.release();
  *build_ptr                                        = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_radix_sort_deserialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}
