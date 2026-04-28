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
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_merge_sort.cuh>

#include <format>
#include <sstream>
#include <vector>

#include "kernels/iterators.h"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/indirect_arg.h"
#include "util/tuning.h"
#include "util/types.h"
#include <cccl/c/merge_sort.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

struct op_wrapper;
struct device_merge_sort_policy;
using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

struct input_keys_iterator_state_t;
struct input_items_iterator_state_t;
struct output_keys_iterator_t;
struct output_items_iterator_t;

namespace merge_sort
{
enum class merge_sort_iterator_t
{
  input_keys   = 0,
  input_items  = 1,
  output_keys  = 2,
  output_items = 3
};

template <typename StorageT = storage_t>
std::string get_iterator_name(cccl_iterator_t iterator, merge_sort_iterator_t which_iterator)
{
  if (iterator.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    if (iterator.state == nullptr)
    {
      return "cub::NullType*";
    }
    else
    {
      return cccl_type_enum_to_name<StorageT>(iterator.value_type.type, true);
    }
  }
  else
  {
    std::string iterator_t;
    switch (which_iterator)
    {
      case merge_sort_iterator_t::input_keys: {
        check(cccl_type_name_from_nvrtc<input_keys_iterator_state_t>(&iterator_t));
        break;
      }
      case merge_sort_iterator_t::input_items: {
        check(cccl_type_name_from_nvrtc<input_items_iterator_state_t>(&iterator_t));
        break;
      }
      case merge_sort_iterator_t::output_keys: {
        check(cccl_type_name_from_nvrtc<output_keys_iterator_t>(&iterator_t));
        break;
      }
      case merge_sort_iterator_t::output_items: {
        check(cccl_type_name_from_nvrtc<output_items_iterator_t>(&iterator_t));
        break;
      }
    }

    return iterator_t;
  }
}

std::string get_merge_sort_kernel_name(
  std::string_view kernel_name,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_items_it,
  cccl_iterator_t output_keys_it,
  cccl_iterator_t output_items_it)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_merge_sort_policy>(&chained_policy_t));

  const std::string input_keys_iterator_t = get_iterator_name(input_keys_it, merge_sort_iterator_t::input_keys);
  const std::string input_items_iterator_t =
    get_iterator_name<items_storage_t>(input_items_it, merge_sort_iterator_t::input_items);
  const std::string output_keys_iterator_t = get_iterator_name(output_keys_it, merge_sort_iterator_t::output_keys);
  const std::string output_items_iterator_t =
    get_iterator_name<items_storage_t>(output_items_it, merge_sort_iterator_t::output_items);

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  std::string compare_op_t;
  check(cccl_type_name_from_nvrtc<op_wrapper>(&compare_op_t));

  const std::string key_t = cccl_type_enum_to_name(output_keys_it.value_type.type);
  const std::string value_t =
    output_items_it.type == cccl_iterator_kind_t::CCCL_POINTER && output_items_it.state == nullptr
      ? "cub::NullType"
      : cccl_type_enum_to_name<items_storage_t>(output_items_it.value_type.type);

  return std::format(
    "cub::detail::merge_sort::{0}<{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}>",
    kernel_name,
    chained_policy_t,
    input_keys_iterator_t,
    input_items_iterator_t,
    output_keys_iterator_t,
    output_items_iterator_t,
    offset_t,
    compare_op_t,
    key_t,
    value_t);
}

std::string get_partition_kernel_name(cccl_iterator_t output_keys_it)
{
  const std::string output_keys_iterator_t = get_iterator_name(output_keys_it, merge_sort_iterator_t::output_keys);

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  std::string compare_op_t;
  check(cccl_type_name_from_nvrtc<op_wrapper>(&compare_op_t));

  std::string key_t = cccl_type_enum_to_name(output_keys_it.value_type.type);

  return std::format(
    "cub::detail::merge_sort::DeviceMergeSortPartitionKernel<{0}, {1}, {2}, {3}>",
    output_keys_iterator_t,
    offset_t,
    compare_op_t,
    key_t);
}

struct merge_sort_kernel_source
{
  cccl_device_merge_sort_build_result_t& build;

  CUkernel MergeSortBlockSortKernel() const
  {
    return build.block_sort_kernel;
  }

  CUkernel MergeSortPartitionKernel() const
  {
    return build.partition_kernel;
  }

  CUkernel MergeSortMergeKernel() const
  {
    return build.merge_kernel;
  }

  std::size_t KeySize() const
  {
    return build.key_type.size;
  }

  std::size_t ValueSize() const
  {
    return build.item_type.size;
  }
};
} // namespace merge_sort

CUresult cccl_device_merge_sort_build_ex(
  cccl_device_merge_sort_build_result_t* build_ptr,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_items_it,
  cccl_iterator_t output_keys_it,
  cccl_iterator_t output_items_it,
  cccl_op_t op,
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

  const int cc = cc_major * 10 + cc_minor;

  const auto input_keys_it_value_t   = cccl_type_enum_to_name(input_keys_it.value_type.type);
  const auto input_items_it_value_t  = cccl_type_enum_to_name(input_items_it.value_type.type);
  const auto output_keys_it_value_t  = cccl_type_enum_to_name(output_keys_it.value_type.type);
  const auto output_items_it_value_t = cccl_type_enum_to_name(output_items_it.value_type.type);
  const auto offset_t                = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

  const std::string input_keys_iterator_src = make_kernel_input_iterator(
    offset_t,
    get_iterator_name(input_keys_it, merge_sort::merge_sort_iterator_t::input_keys),
    input_keys_it_value_t,
    input_keys_it);
  const std::string input_items_iterator_src = make_kernel_input_iterator(
    offset_t,
    get_iterator_name(input_items_it, merge_sort::merge_sort_iterator_t::input_items),
    input_items_it_value_t,
    input_items_it);
  const std::string output_keys_iterator_src = make_kernel_output_iterator(
    offset_t,
    get_iterator_name(output_keys_it, merge_sort::merge_sort_iterator_t::output_keys),
    output_keys_it_value_t,
    output_keys_it);
  const std::string output_items_iterator_src = make_kernel_output_iterator(
    offset_t,
    get_iterator_name(output_items_it, merge_sort::merge_sort_iterator_t::output_items),
    output_items_it_value_t,
    output_items_it);

  const std::string op_src = make_kernel_user_comparison_operator(input_keys_it_value_t, op);

  const auto policy_sel = cub::detail::merge_sort::policy_selector{static_cast<int>(input_keys_it.value_type.size)};

  // TODO(bgruber): drop this if tuning policies become formattable
  std::stringstream policy_sel_str;
  policy_sel_str << policy_sel(cuda::to_arch_id(cuda::compute_capability{cc_major, cc_minor}));

  auto policy_selector_expr =
    std::format("cub::detail::merge_sort::policy_selector_from_types<{}>",
                get_iterator_name(input_keys_it, merge_sort::merge_sort_iterator_t::input_keys));

  std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/tuning/tuning_merge_sort.cuh>
#include <cub/device/dispatch/kernels/kernel_merge_sort.cuh>
#include <cub/util_type.cuh> // needed for cub::NullType
struct __align__({1}) storage_t {{
  char data[{0}];
}};
struct __align__({3}) items_storage_t {{
  char data[{2}];
}};
{4}
{5}
{6}
{7}
{8}
using device_merge_sort_policy = {9};
using namespace cub;
using namespace cub::detail::merge_sort;
static_assert(device_merge_sort_policy()(detail::current_tuning_arch()) == {10}, "Host generated and JIT compiled policy mismatch");
)XXX",
    input_keys_it.value_type.size, // 0
    input_keys_it.value_type.alignment, // 1
    input_items_it.value_type.size, // 2
    input_items_it.value_type.alignment, // 3
    input_keys_iterator_src, // 4
    input_items_iterator_src, // 5
    output_keys_iterator_src, // 6
    output_items_iterator_src, // 7
    op_src, // 8
    policy_selector_expr, // 9
    policy_sel_str.view()); // 10

#if false // CCCL_DEBUGGING_SWITCH
  fflush(stderr);
  printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
  fflush(stdout);
#endif

  std::string block_sort_kernel_name = merge_sort::get_merge_sort_kernel_name(
    "DeviceMergeSortBlockSortKernel", input_keys_it, input_items_it, output_keys_it, output_items_it);
  std::string partition_kernel_name = merge_sort::get_partition_kernel_name(output_keys_it);
  std::string merge_kernel_name     = merge_sort::get_merge_sort_kernel_name(
    "DeviceMergeSortMergeKernel", input_keys_it, input_items_it, output_keys_it, output_items_it);
  std::string block_sort_kernel_lowered_name;
  std::string partition_kernel_lowered_name;
  std::string merge_kernel_lowered_name;

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
  nvrtc_linkable_list_appender list_appender{linkable_list};

  list_appender.append_operation(op);
  list_appender.add_iterator_definition(input_keys_it);
  list_appender.add_iterator_definition(input_items_it);
  list_appender.add_iterator_definition(output_keys_it);
  list_appender.add_iterator_definition(output_items_it);

  nvrtc_link_result result =
    begin_linking_nvrtc_program(num_lto_args, lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({block_sort_kernel_name})
      ->add_expression({partition_kernel_name})
      ->add_expression({merge_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({block_sort_kernel_name, block_sort_kernel_lowered_name})
      ->get_name({partition_kernel_name, partition_kernel_lowered_name})
      ->get_name({merge_kernel_name, merge_kernel_lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(cuLibraryGetKernel(&build_ptr->block_sort_kernel, build_ptr->library, block_sort_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build_ptr->partition_kernel, build_ptr->library, partition_kernel_lowered_name.c_str()));
  check(cuLibraryGetKernel(&build_ptr->merge_kernel, build_ptr->library, merge_kernel_lowered_name.c_str()));

  build_ptr->cc             = cc;
  build_ptr->cubin          = (void*) result.data.release();
  build_ptr->cubin_size     = result.size;
  build_ptr->key_type       = input_keys_it.value_type;
  build_ptr->item_type      = input_items_it.value_type;
  build_ptr->runtime_policy = new cub::detail::merge_sort::policy_selector{policy_sel};

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_merge_sort_build(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_merge_sort(
  cccl_device_merge_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  if (cccl_iterator_kind_t::CCCL_ITERATOR == d_out_keys.type || cccl_iterator_kind_t::CCCL_ITERATOR == d_out_items.type)
  {
    // See https://github.com/NVIDIA/cccl/issues/3722
    fflush(stderr);
    printf("\nERROR in cccl_device_merge_sort(): merge sort output cannot be an iterator\n");
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

    auto exec_status = cub::detail::merge_sort::dispatch(
      d_temp_storage,
      *temp_storage_bytes,
      indirect_iterator_t{d_in_keys},
      indirect_iterator_t{d_in_items},
      indirect_iterator_t{d_out_keys},
      indirect_iterator_t{d_out_items},
      static_cast<OffsetT>(num_items),
      indirect_arg_t{op},
      stream,
      *static_cast<cub::detail::merge_sort::policy_selector*>(build.runtime_policy),
      merge_sort::merge_sort_kernel_source{build},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
      static_cast<indirect_arg_t*>(nullptr),
      static_cast<indirect_arg_t*>(nullptr));

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_merge_sort(): %s\n", exc.what());
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

CUresult cccl_device_merge_sort_build(
  cccl_device_merge_sort_build_result_t* build,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_merge_sort_build_ex(
    build,
    d_in_keys,
    d_in_items,
    d_out_keys,
    d_out_items,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_merge_sort_cleanup(cccl_device_merge_sort_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
  std::unique_ptr<cub::detail::merge_sort::policy_selector> policy(
    static_cast<cub::detail::merge_sort::policy_selector*>(build_ptr->runtime_policy));
  check(cuLibraryUnload(build_ptr->library));

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_merge_sort_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}
