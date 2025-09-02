//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/block/block_scan.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/device_select.cuh>

#include <format>
#include <vector>

#include <cccl/c/unique_by_key.h>
#include <kernels/iterators.h>
#include <kernels/operators.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/indirect_arg.h>
#include <util/scan_tile_state.h>
#include <util/tuning.h>
#include <util/types.h>

struct op_wrapper;
struct device_unique_by_key_policy;
using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

struct num_selected_storage_t;

namespace unique_by_key
{
struct unique_by_key_runtime_tuning_policy
{
  int block_size;
  int items_per_thread;
  cub::BlockLoadAlgorithm load_algorithm;
  cub::CacheLoadModifier load_modifier;
  cub::BlockScanAlgorithm scan_algorithm;

  unique_by_key_runtime_tuning_policy UniqueByKey() const
  {
    return *this;
  }

  using UniqueByKeyPolicyT = unique_by_key_runtime_tuning_policy;
};

struct unique_by_key_tuning_t
{
  int cc;
  int block_size;
  int items_per_thread;
};

unique_by_key_runtime_tuning_policy get_policy(int /*cc*/, int /*key_size*/)
{
  // TODO: we should update this once we figure out a way to reuse
  // tuning logic from C++. Alternately, we should implement
  // something better than a hardcoded default:
  return {128, 4, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS};
}

enum class unique_by_key_iterator_t
{
  input_keys    = 0,
  input_values  = 1,
  output_keys   = 2,
  output_values = 3,
  num_selected  = 4
};

template <typename StorageT = storage_t>
std::string get_iterator_name(cccl_iterator_t iterator, unique_by_key_iterator_t which_iterator)
{
  if (iterator.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return cccl_type_enum_to_name<StorageT>(iterator.value_type.type, true);
  }
  else
  {
    std::string iterator_t;
    switch (which_iterator)
    {
      case unique_by_key_iterator_t::input_keys:
        return "input_keys_iterator_state_t";
        break;
      case unique_by_key_iterator_t::input_values:
        return "input_values_iterator_state_t";
        break;
      case unique_by_key_iterator_t::output_keys:
        return "output_keys_iterator_t";
        break;
      case unique_by_key_iterator_t::output_values:
        return "output_values_iterator_t";
        break;
      case unique_by_key_iterator_t::num_selected:
        return "output_num_selected_iterator_t";
        break;
    }

    return iterator_t;
  }
}

std::string get_compact_init_kernel_name(cccl_iterator_t output_num_selected_it)
{
  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  const std::string num_selected_iterator_t =
    get_iterator_name(output_num_selected_it, unique_by_key_iterator_t::num_selected);

  return std::format(
    "cub::detail::scan::DeviceCompactInitKernel<cub::ScanTileState<{0}>, {1}>", offset_t, num_selected_iterator_t);
}

std::string get_sweep_kernel_name(
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_iterator_t output_keys_it,
  cccl_iterator_t output_values_it,
  cccl_iterator_t output_num_selected_it)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_unique_by_key_policy>(&chained_policy_t));

  const std::string input_keys_iterator_t = get_iterator_name(input_keys_it, unique_by_key_iterator_t::input_keys);
  const std::string input_values_iterator_t =
    get_iterator_name<items_storage_t>(input_values_it, unique_by_key_iterator_t::input_values);
  const std::string output_keys_iterator_t = get_iterator_name(output_keys_it, unique_by_key_iterator_t::output_keys);
  const std::string output_values_iterator_t =
    get_iterator_name<items_storage_t>(output_values_it, unique_by_key_iterator_t::output_values);
  const std::string output_num_selected_iterator_t =
    get_iterator_name<num_selected_storage_t>(output_num_selected_it, unique_by_key_iterator_t::num_selected);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  auto tile_state_t = std::format("cub::ScanTileState<{0}>", offset_t);

  std::string equality_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&equality_op_t));

  return std::format(
    "cub::detail::unique_by_key::DeviceUniqueByKeySweepKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, "
    "device_unique_by_key_vsmem_helper>",
    chained_policy_t,
    input_keys_iterator_t,
    input_values_iterator_t,
    output_keys_iterator_t,
    output_values_iterator_t,
    output_num_selected_iterator_t,
    tile_state_t,
    equality_op_t,
    offset_t);
}

template <auto* GetPolicy>
struct dynamic_unique_by_key_policy_t
{
  using MaxPolicy = dynamic_unique_by_key_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<unique_by_key_runtime_tuning_policy>(GetPolicy(device_ptx_version, key_size));
  }

  uint64_t key_size;
};

struct unique_by_key_kernel_source
{
  cccl_device_unique_by_key_build_result_t& build;

  CUkernel UniqueByKeySweepKernel() const
  {
    return build.sweep_kernel;
  }

  CUkernel CompactInitKernel() const
  {
    return build.compact_init_kernel;
  }

  scan_tile_state TileState()
  {
    return {build.description_bytes_per_tile, build.payload_bytes_per_tile};
  }
};

struct dynamic_vsmem_helper_t
{
  template <typename PolicyT, typename... Ts>
  static int BlockThreads(PolicyT policy)
  {
    return policy.block_size;
  }

  template <typename PolicyT, typename... Ts>
  static int ItemsPerThread(PolicyT policy)
  {
    return policy.items_per_thread;
  }

  template <typename PolicyT, typename... Ts>
  static ::cuda::std::size_t VSMemPerBlock(PolicyT /*policy*/)
  {
    return 0;
  }
};

} // namespace unique_by_key

CUresult cccl_device_unique_by_key_build_ex(
  cccl_device_unique_by_key_build_result_t* build_ptr,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_iterator_t output_keys_it,
  cccl_iterator_t output_values_it,
  cccl_iterator_t output_num_selected_it,
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
    const auto policy = unique_by_key::get_policy(cc, input_keys_it.value_type.size);

    const auto input_keys_it_value_t          = cccl_type_enum_to_name(input_keys_it.value_type.type);
    const auto input_values_it_value_t        = cccl_type_enum_to_name(input_values_it.value_type.type);
    const auto output_keys_it_value_t         = cccl_type_enum_to_name(output_keys_it.value_type.type);
    const auto output_values_it_value_t       = cccl_type_enum_to_name(output_values_it.value_type.type);
    const auto output_num_selected_it_value_t = cccl_type_enum_to_name(output_num_selected_it.value_type.type);
    const auto offset_cpp                     = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);
    const cccl_type_info offset_t{sizeof(OffsetT), alignof(OffsetT), cccl_type_enum::CCCL_UINT64};

    const std::string input_keys_iterator_src = make_kernel_input_iterator(
      offset_cpp,
      get_iterator_name(input_keys_it, unique_by_key::unique_by_key_iterator_t::input_keys),
      input_keys_it_value_t,
      input_keys_it);
    const std::string input_values_iterator_src = make_kernel_input_iterator(
      offset_cpp,
      get_iterator_name(input_values_it, unique_by_key::unique_by_key_iterator_t::input_values),
      input_values_it_value_t,
      input_values_it);
    const std::string output_keys_iterator_src = make_kernel_output_iterator(
      offset_cpp,
      get_iterator_name(output_keys_it, unique_by_key::unique_by_key_iterator_t::output_keys),
      output_keys_it_value_t,
      output_keys_it);
    const std::string output_values_iterator_src = make_kernel_output_iterator(
      offset_cpp,
      get_iterator_name(output_values_it, unique_by_key::unique_by_key_iterator_t::output_values),
      output_values_it_value_t,
      output_values_it);
    const std::string output_num_selected_iterator_src = make_kernel_output_iterator(
      offset_cpp,
      get_iterator_name(output_num_selected_it, unique_by_key::unique_by_key_iterator_t::num_selected),
      output_num_selected_it_value_t,
      output_num_selected_it);

    const std::string op_src = make_kernel_user_comparison_operator(input_keys_it_value_t, op);

    constexpr std::string_view src_template = R"XXX(
#include <cub/device/dispatch/kernels/scan.cuh>
#include <cub/device/dispatch/kernels/unique_by_key.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
struct __align__({1}) storage_t {{
  char data[{0}];
}};
struct __align__({3}) items_storage_t {{
  char data[{2}];
}};
struct __align__({5}) num_out_storage_t {{
  char data[{4}];
}};
{8}
{9}
{10}
{11}
{12}
struct agent_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {7};
  static constexpr int BLOCK_THREADS = {6};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;
  struct detail {{
    using delay_constructor_t = cub::detail::default_delay_constructor_t<unsigned long long>;
  }};
}};
struct device_unique_by_key_policy {{
  struct ActivePolicy {{
    using UniqueByKeyPolicyT = agent_policy_t;
  }};
}};
struct device_unique_by_key_vsmem_helper {{
  template<typename ActivePolicyT, typename... Ts>
  struct VSMemHelperDefaultFallbackPolicyT {{
    using agent_policy_t = agent_policy_t;
    using agent_t = cub::detail::unique_by_key::AgentUniqueByKey<agent_policy_t, Ts...>;
    using static_temp_storage_t = typename cub::detail::unique_by_key::AgentUniqueByKey<agent_policy_t, Ts...>::TempStorage;
    static _CCCL_DEVICE _CCCL_FORCEINLINE static_temp_storage_t& get_temp_storage(
      static_temp_storage_t& static_temp_storage, cub::detail::vsmem_t& vsmem, ::cuda::std::size_t linear_block_id)
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
{13}
)XXX";

    const std::string src = std::format(
      src_template,
      input_keys_it.value_type.size, // 0
      input_keys_it.value_type.alignment, // 1
      input_values_it.value_type.size, // 2
      input_values_it.value_type.alignment, // 3
      output_values_it.value_type.size, // 4
      output_values_it.value_type.alignment, // 5
      policy.block_size, // 6
      policy.items_per_thread, // 7
      input_keys_iterator_src, // 8
      input_values_iterator_src, // 9
      output_keys_iterator_src, // 10
      output_values_iterator_src, // 11
      output_num_selected_iterator_src, // 12
      op_src); // 13

#if false // CCCL_DEBUGGING_SWITCH
      fflush(stderr);
      printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
      fflush(stdout);
#endif

    std::string compact_init_kernel_name = unique_by_key::get_compact_init_kernel_name(output_num_selected_it);
    std::string sweep_kernel_name        = unique_by_key::get_sweep_kernel_name(
      input_keys_it, input_values_it, output_keys_it, output_values_it, output_num_selected_it);
    std::string compact_init_kernel_lowered_name;
    std::string sweep_kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    std::vector<const char*> args = {
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

    cccl::detail::extend_args_with_build_config(args, config);

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};

    appender.append_operation(op);
    appender.add_iterator_definition(input_keys_it);
    appender.add_iterator_definition(input_values_it);
    appender.add_iterator_definition(output_keys_it);
    appender.add_iterator_definition(output_values_it);
    appender.add_iterator_definition(output_num_selected_it);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{src.c_str(), name})
        ->add_expression({compact_init_kernel_name})
        ->add_expression({sweep_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({compact_init_kernel_name, compact_init_kernel_lowered_name})
        ->get_name({sweep_kernel_name, sweep_kernel_lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
        ->finalize_program();

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(
      &build_ptr->compact_init_kernel, build_ptr->library, compact_init_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->sweep_kernel, build_ptr->library, sweep_kernel_lowered_name.c_str()));

    auto [description_bytes_per_tile,
          payload_bytes_per_tile] = get_tile_state_bytes_per_tile(offset_t, offset_cpp, args.data(), args.size(), arch);

    build_ptr->cc                         = cc;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
    build_ptr->description_bytes_per_tile = description_bytes_per_tile;
    build_ptr->payload_bytes_per_tile     = payload_bytes_per_tile;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_unique_by_key_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

CUresult cccl_device_unique_by_key(
  cccl_device_unique_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t op,
  uint64_t num_items,
  CUstream stream)
{
  CUresult error = CUDA_SUCCESS;
  bool pushed    = false;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::DispatchUniqueByKey<
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      OffsetT,
      unique_by_key::dynamic_unique_by_key_policy_t<&unique_by_key::get_policy>,
      unique_by_key::unique_by_key_kernel_source,
      cub::detail::CudaDriverLauncherFactory,
      unique_by_key::dynamic_vsmem_helper_t,
      indirect_arg_t,
      indirect_arg_t>::Dispatch(d_temp_storage,
                                *temp_storage_bytes,
                                d_keys_in,
                                d_values_in,
                                d_keys_out,
                                d_values_out,
                                d_num_selected_out,
                                op,
                                num_items,
                                stream,
                                {build},
                                cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
                                {d_keys_in.value_type.size});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_unique_by_key(): %s\n", exc.what());
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

CUresult cccl_device_unique_by_key_build(
  cccl_device_unique_by_key_build_result_t* build,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_unique_by_key_build_ex(
    build,
    d_keys_in,
    d_values_in,
    d_keys_out,
    d_values_out,
    d_num_selected_out,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_unique_by_key_cleanup(cccl_device_unique_by_key_build_result_t* build_ptr)
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
    printf("\nEXCEPTION in cccl_device_unique_by_key_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
