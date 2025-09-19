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

#include <format>
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
struct merge_sort_runtime_tuning_policy
{
  int block_size;
  int items_per_thread;
  int items_per_tile;
  merge_sort_runtime_tuning_policy MergeSort() const
  {
    return *this;
  }

  using MergeSortPolicy = merge_sort_runtime_tuning_policy;
};

struct merge_sort_tuning_t
{
  int cc;
  int block_size;
  int items_per_thread;
};

template <typename Tuning, int N>
Tuning find_tuning(int cc, const Tuning (&tunings)[N])
{
  for (const Tuning& tuning : tunings)
  {
    if (cc >= tuning.cc)
    {
      return tuning;
    }
  }

  return tunings[N - 1];
}

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
        check(nvrtcGetTypeName<input_keys_iterator_state_t>(&iterator_t));
        break;
      }
      case merge_sort_iterator_t::input_items: {
        check(nvrtcGetTypeName<input_items_iterator_state_t>(&iterator_t));
        break;
      }
      case merge_sort_iterator_t::output_keys: {
        check(nvrtcGetTypeName<output_keys_iterator_t>(&iterator_t));
        break;
      }
      case merge_sort_iterator_t::output_items: {
        check(nvrtcGetTypeName<output_items_iterator_t>(&iterator_t));
        break;
      }
    }

    return iterator_t;
  }
}

merge_sort_runtime_tuning_policy get_policy(int cc, int key_size)
{
  merge_sort_tuning_t chain[] = {
    {60, 256, nominal_4b_items_to_items(17, key_size)}, {35, 256, nominal_4b_items_to_items(11, key_size)}};
  auto [_, block_size, items_per_thread] = find_tuning(cc, chain);
  // TODO: we hardcode this value in order to make sure that the merge_sort test does not fail due to the memory op
  // assertions. This currently happens when we pass in items and keys of type uint8_t or int16_t, and for the custom
  // types test as well. This will be fixed after https://github.com/NVIDIA/cccl/issues/3570 is resolved.
  items_per_thread = 1;

  return {block_size, items_per_thread, block_size * items_per_thread};
}

std::string get_merge_sort_kernel_name(
  std::string_view kernel_name,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_items_it,
  cccl_iterator_t output_keys_it,
  cccl_iterator_t output_items_it)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_merge_sort_policy>(&chained_policy_t));

  const std::string input_keys_iterator_t = get_iterator_name(input_keys_it, merge_sort_iterator_t::input_keys);
  const std::string input_items_iterator_t =
    get_iterator_name<items_storage_t>(input_items_it, merge_sort_iterator_t::input_items);
  const std::string output_keys_iterator_t = get_iterator_name(output_keys_it, merge_sort_iterator_t::output_keys);
  const std::string output_items_iterator_t =
    get_iterator_name<items_storage_t>(output_items_it, merge_sort_iterator_t::output_items);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string compare_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&compare_op_t));

  const std::string key_t = cccl_type_enum_to_name(output_keys_it.value_type.type);
  const std::string value_t =
    output_items_it.type == cccl_iterator_kind_t::CCCL_POINTER && output_items_it.state == nullptr
      ? "cub::NullType"
      : cccl_type_enum_to_name<items_storage_t>(output_items_it.value_type.type);

  return std::format(
    "cub::detail::merge_sort::{0}<{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, device_merge_sort_vsmem_helper>",
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
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string compare_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&compare_op_t));

  std::string key_t = cccl_type_enum_to_name(output_keys_it.value_type.type);

  return std::format(
    "cub::detail::merge_sort::DeviceMergeSortPartitionKernel<{0}, {1}, {2}, {3}>",
    output_keys_iterator_t,
    offset_t,
    compare_op_t,
    key_t);
}

template <auto* GetPolicy>
struct dynamic_merge_sort_policy_t
{
  using MaxPolicy = dynamic_merge_sort_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<merge_sort_runtime_tuning_policy>(GetPolicy(device_ptx_version, key_size));
  }

  uint64_t key_size;
};

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

struct dynamic_vsmem_helper_t
{
  template <typename PolicyT, typename... Ts>
  static ::cuda::std::size_t BlockSortVSMemPerBlock(PolicyT /*policy*/)
  {
    return 0;
  }

  template <typename PolicyT, typename... Ts>
  static ::cuda::std::size_t MergeVSMemPerBlock(PolicyT /*policy*/)
  {
    return 0;
  }

  template <typename PolicyT, typename... Ts>
  static int BlockThreads(PolicyT policy)
  {
    return policy.block_size;
  }

  template <typename PolicyT, typename... Ts>
  static int ItemsPerTile(PolicyT policy)
  {
    return policy.items_per_tile;
  }

private:
  merge_sort_runtime_tuning_policy fallback_policy = {64, 1, 64};
  bool uses_fallback_policy() const
  {
    return false;
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
{
  CUresult error = CUDA_SUCCESS;
  try
  {
    const char* name = "test";

    const int cc      = cc_major * 10 + cc_minor;
    const auto policy = merge_sort::get_policy(cc, output_keys_it.value_type.size);

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

    constexpr std::string_view src_template = R"XXX(
#include <cub/device/dispatch/kernels/merge_sort.cuh>
#include <cub/util_type.cuh> // needed for cub::NullType
struct __align__({1}) storage_t {{
  char data[{0}];
}};
struct __align__({3}) items_storage_t {{
  char data[{2}];
}};
{7}
{8}
{9}
{10}
struct agent_policy_t {{
  static constexpr int ITEMS_PER_TILE = {6};
  static constexpr int ITEMS_PER_THREAD = {5};
  static constexpr int BLOCK_THREADS = {4};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = cub::BLOCK_STORE_WARP_TRANSPOSE;
}};
struct device_merge_sort_policy {{
  struct ActivePolicy {{
    using MergeSortPolicy = agent_policy_t;
  }};
}};
struct device_merge_sort_vsmem_helper {{
  template<typename ActivePolicyT, typename KeyInputIteratorT, typename ValueInputIteratorT, typename... Ts>
  struct MergeSortVSMemHelperT {{
    using policy_t = agent_policy_t;
    using block_sort_agent_t = cub::detail::merge_sort::AgentBlockSort<agent_policy_t, KeyInputIteratorT, ValueInputIteratorT, Ts...>;
    using merge_agent_t = cub::detail::merge_sort::AgentMerge<agent_policy_t, Ts...>;
  }};
  template <typename AgentT>
  struct VSmemHelperT {{
    using static_temp_storage_t = typename AgentT::TempStorage;
    static _CCCL_DEVICE _CCCL_FORCEINLINE static_temp_storage_t& get_temp_storage(
      static_temp_storage_t& static_temp_storage, cub::detail::vsmem_t& vsmem)
    {{
        return static_temp_storage;
    }}
    template <bool needs_vsmem_ = false, ::cuda::std::enable_if_t<!needs_vsmem_, int> = 0>
    static _CCCL_DEVICE _CCCL_FORCEINLINE bool discard_temp_storage(static_temp_storage_t& temp_storage)
    {{
      return false;
    }}
  }};
}};
{11};
)XXX";

    const std::string src = std::format(
      src_template,
      input_keys_it.value_type.size, // 0
      input_keys_it.value_type.alignment, // 1
      input_items_it.value_type.size, // 2
      input_items_it.value_type.alignment, // 3
      policy.block_size, // 4
      policy.items_per_thread, // 5
      policy.items_per_tile, // 6
      input_keys_iterator_src, // 7
      input_items_iterator_src, // 8
      output_keys_iterator_src, // 9
      output_items_iterator_src, // 10
      op_src); // 11

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
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
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

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
        ->add_program(nvrtc_translation_unit{src.c_str(), name})
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
    check(
      cuLibraryGetKernel(&build_ptr->block_sort_kernel, build_ptr->library, block_sort_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->partition_kernel, build_ptr->library, partition_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->merge_kernel, build_ptr->library, merge_kernel_lowered_name.c_str()));

    build_ptr->cc         = cc;
    build_ptr->cubin      = (void*) result.data.release();
    build_ptr->cubin_size = result.size;
    build_ptr->key_type   = input_keys_it.value_type;
    build_ptr->item_type  = input_items_it.value_type;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_merge_sort_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
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

    auto exec_status = cub::DispatchMergeSort<
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      OffsetT,
      indirect_arg_t,
      merge_sort::dynamic_merge_sort_policy_t<&merge_sort::get_policy>,
      merge_sort::merge_sort_kernel_source,
      cub::detail::CudaDriverLauncherFactory,
      merge_sort::dynamic_vsmem_helper_t,
      indirect_arg_t,
      indirect_arg_t>::Dispatch(d_temp_storage,
                                *temp_storage_bytes,
                                d_in_keys,
                                d_in_items,
                                d_out_keys,
                                d_out_items,
                                num_items,
                                op,
                                stream,
                                {build},
                                cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
                                {d_out_keys.value_type.size});

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
    printf("\nEXCEPTION in cccl_device_merge_sort_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
