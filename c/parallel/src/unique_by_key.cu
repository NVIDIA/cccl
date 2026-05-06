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
#include <mutex>
#include <sstream>
#include <vector>

#include "util/nvjitlink.h"
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
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

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
  check(cccl_type_name_from_nvrtc<device_unique_by_key_policy>(&chained_policy_t));

  const std::string input_keys_iterator_t = get_iterator_name(input_keys_it, unique_by_key_iterator_t::input_keys);
  const std::string input_values_iterator_t =
    get_iterator_name<items_storage_t>(input_values_it, unique_by_key_iterator_t::input_values);
  const std::string output_keys_iterator_t = get_iterator_name(output_keys_it, unique_by_key_iterator_t::output_keys);
  const std::string output_values_iterator_t =
    get_iterator_name<items_storage_t>(output_values_it, unique_by_key_iterator_t::output_values);
  const std::string output_num_selected_iterator_t =
    get_iterator_name<num_selected_storage_t>(output_num_selected_it, unique_by_key_iterator_t::num_selected);

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  auto tile_state_t = std::format("cub::ScanTileState<{0}>", offset_t);

  std::string equality_op_t;
  check(cccl_type_name_from_nvrtc<op_wrapper>(&equality_op_t));

  return std::format(
    "cub::detail::unique_by_key::DeviceUniqueByKeySweepKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}>",
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
} // namespace unique_by_key

CUresult cccl_device_unique_by_key_compile(
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
try
{
  const char* name = "test";

  const cuda::compute_capability cc{cc_major, cc_minor};
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

  const std::string input_keys_iterator_t =
    get_iterator_name(input_keys_it, unique_by_key::unique_by_key_iterator_t::input_keys);
  const std::string input_values_iterator_t =
    get_iterator_name<items_storage_t>(input_values_it, unique_by_key::unique_by_key_iterator_t::input_values);
  const std::string output_keys_iterator_t =
    get_iterator_name(output_keys_it, unique_by_key::unique_by_key_iterator_t::output_keys);
  const std::string output_values_iterator_t =
    get_iterator_name<items_storage_t>(output_values_it, unique_by_key::unique_by_key_iterator_t::output_values);

  const std::string op_src = make_kernel_user_comparison_operator(input_keys_it_value_t, op);

  const auto policy_sel = cub::detail::unique_by_key::policy_selector{
    static_cast<int>(input_keys_it.value_type.size),
    static_cast<int>(input_values_it.value_type.size),
    input_keys_it.value_type.type != CCCL_STORAGE && input_keys_it.value_type.size <= 8,
    input_values_it.value_type.type != CCCL_STORAGE && input_values_it.value_type.size <= 8};

  const auto active_policy = policy_sel(cc);

  std::stringstream policy_sel_str;
  policy_sel_str << active_policy;

  std::string policy_selector_expr = std::format(
    "cub::detail::unique_by_key::policy_selector_from_types<{}, {}>", input_keys_it_value_t, input_values_it_value_t);

  std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/tuning/tuning_unique_by_key.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_unique_by_key.cuh>
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
{6}
{7}
{8}
{9}
{10}
{11}
using device_unique_by_key_policy = {12};
using namespace cub;
using namespace cub::detail::unique_by_key;
using cub::LookbackDelayPolicy;
using cub::LookbackDelayAlgorithm;
static_assert(device_unique_by_key_policy()(detail::current_tuning_cc()) == {13}, "Host generated and JIT compiled policy mismatch");
static_assert(
  cub::detail::unique_by_key::unique_by_key_vsmem_helper_t<
    cub::detail::policy_getter<device_unique_by_key_policy, detail::current_tuning_cc().get()>,
    {14},
    {15},
    {16},
    {17},
    op_wrapper,
    {18}>::selected_policy_fits_smem,
  "CCCL.C DeviceSelect::UniqueByKey does not support VSMEM-backed kernels");
using device_unique_by_key_vsmem = cub::detail::unique_by_key::unique_by_key_vsmem_helper_t<
  cub::detail::policy_getter<device_unique_by_key_policy, detail::current_tuning_cc().get()>,
  {14},
  {15},
  {16},
  {17},
  op_wrapper,
  {18}>;
static_assert(
  cub::detail::vsmem_helper_impl<typename device_unique_by_key_vsmem::agent_t>::vsmem_per_block == 0,
  "CCCL.C DeviceSelect::UniqueByKey does not support VSMEM-backed kernels");
)XXX",
    input_keys_it.value_type.size, // 0
    input_keys_it.value_type.alignment, // 1
    input_values_it.value_type.size, // 2
    input_values_it.value_type.alignment, // 3
    output_values_it.value_type.size, // 4
    output_values_it.value_type.alignment, // 5
    input_keys_iterator_src, // 6
    input_values_iterator_src, // 7
    output_keys_iterator_src, // 8
    output_values_iterator_src, // 9
    output_num_selected_iterator_src, // 10
    op_src, // 11
    policy_selector_expr, // 12
    policy_sel_str.view(), // 13
    input_keys_iterator_t, // 14
    input_values_iterator_t, // 15
    output_keys_iterator_t, // 16
    output_values_iterator_t, // 17
    offset_cpp); // 18

#if false // CCCL_DEBUGGING_SWITCH
      fflush(stderr);
      printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
      fflush(stdout);
#endif

  std::string compact_init_kernel_name = unique_by_key::get_compact_init_kernel_name(output_num_selected_it);
  std::string sweep_kernel_name        = unique_by_key::get_sweep_kernel_name(
    input_keys_it, input_values_it, output_keys_it, output_values_it, output_num_selected_it);
  std::string compact_init_kernel_lowered_name;
  std::string sweep_kernel_lowered_name;

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

  const bool kernel_only = (op.code_size == 0) && (op.name != nullptr) && (op.name[0] != '\0');

  // Collect all LTO-IRs to be linked (empty when op.code_size == 0 — kernel-only mode).
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(op);
  appender.add_iterator_definition(input_keys_it);
  appender.add_iterator_definition(input_values_it);
  appender.add_iterator_definition(output_keys_it);
  appender.add_iterator_definition(output_values_it);
  appender.add_iterator_definition(output_num_selected_it);

  auto post_build =
    begin_linking_nvrtc_program(kernel_only ? 0 : num_lto_args, kernel_only ? nullptr : lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({compact_init_kernel_name})
      ->add_expression({sweep_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({compact_init_kernel_name, compact_init_kernel_lowered_name})
      ->get_name({sweep_kernel_name, sweep_kernel_lowered_name});

  auto [description_bytes_per_tile,
        payload_bytes_per_tile] = get_tile_state_bytes_per_tile(offset_t, offset_cpp, args.data(), args.size(), arch);

  auto policy     = std::make_unique<cub::detail::unique_by_key::policy_selector>(policy_sel);
  auto init_name  = std::unique_ptr<char[]>(duplicate_c_string(compact_init_kernel_lowered_name));
  auto sweep_name = std::unique_ptr<char[]>(duplicate_c_string(sweep_kernel_lowered_name));

  build_ptr->cc                               = cc.get();
  build_ptr->description_bytes_per_tile       = description_bytes_per_tile;
  build_ptr->payload_bytes_per_tile           = payload_bytes_per_tile;
  build_ptr->runtime_policy                   = policy.release();
  build_ptr->runtime_policy_size              = sizeof(cub::detail::unique_by_key::policy_selector);
  build_ptr->compact_init_kernel_lowered_name = init_name.release();
  build_ptr->sweep_kernel_lowered_name        = sweep_name.release();

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

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_unique_by_key_compile(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_unique_by_key_load(cccl_device_unique_by_key_build_result_t* build)
try
{
  if (build == nullptr || build->payload == nullptr || build->payload_size == 0
      || build->payload_kind != CCCL_PAYLOAD_CUBIN || build->compact_init_kernel_lowered_name == nullptr
      || build->compact_init_kernel_lowered_name[0] == '\0' || build->sweep_kernel_lowered_name == nullptr
      || build->sweep_kernel_lowered_name[0] == '\0')
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status = cuLibraryLoadData(&build->library, build->payload, nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (status != CUDA_SUCCESS)
  {
    return status;
  }
  try
  {
    check(cuLibraryGetKernel(&build->compact_init_kernel, build->library, build->compact_init_kernel_lowered_name));
    check(cuLibraryGetKernel(&build->sweep_kernel, build->library, build->sweep_kernel_lowered_name));
  }
  catch (...)
  {
    cuLibraryUnload(build->library);
    build->library = nullptr;
    throw;
  }
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_unique_by_key_load(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

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
  CUresult result = cccl_device_unique_by_key_compile(
    build_ptr,
    input_keys_it,
    input_values_it,
    output_keys_it,
    output_values_it,
    output_num_selected_it,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    config);
  if (result != CUDA_SUCCESS)
  {
    return result;
  }
  return cccl_device_unique_by_key_load(build_ptr);
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

    auto launcher_factory = cub::detail::CudaDriverLauncherFactory{cu_device, build.cc};
    auto exec_status      = cub::detail::unique_by_key::dispatch(
      d_temp_storage,
      *temp_storage_bytes,
      indirect_arg_t{d_keys_in},
      indirect_arg_t{d_values_in},
      indirect_arg_t{d_keys_out},
      indirect_arg_t{d_values_out},
      indirect_arg_t{d_num_selected_out},
      indirect_arg_t{op},
      static_cast<OffsetT>(num_items),
      stream,
      *static_cast<cub::detail::unique_by_key::policy_selector*>(build.runtime_policy),
      unique_by_key::unique_by_key_kernel_source{build},
      launcher_factory,
      static_cast<indirect_arg_t*>(nullptr),
      static_cast<indirect_arg_t*>(nullptr));

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
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> payload(reinterpret_cast<char*>(build_ptr->payload));
  std::unique_ptr<cub::detail::unique_by_key::policy_selector> policy(
    static_cast<cub::detail::unique_by_key::policy_selector*>(build_ptr->runtime_policy));
  std::unique_ptr<char[]> init_name(build_ptr->compact_init_kernel_lowered_name);
  std::unique_ptr<char[]> sweep_name(build_ptr->sweep_kernel_lowered_name);
  if (build_ptr->library != nullptr)
  {
    check(cuLibraryUnload(build_ptr->library));
  }

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_unique_by_key_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_unique_by_key_link_ltoir(
  cccl_device_unique_by_key_build_result_t* build_ptr,
  const void** input_blobs,
  const size_t* input_sizes,
  size_t num_inputs)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int cc_major = build_ptr->cc / 10;
  const int cc_minor = build_ptr->cc % 10;
  std::vector<const void*> all_blobs;
  std::vector<size_t> all_sizes;
  if (build_ptr->payload != nullptr && build_ptr->payload_size > 0 && build_ptr->payload_kind == CCCL_PAYLOAD_LTOIR)
  {
    all_blobs.push_back(build_ptr->payload);
    all_sizes.push_back(build_ptr->payload_size);
  }
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
  build_ptr->payload      = nullptr;
  build_ptr->payload_size = 0;
  build_ptr->payload_kind = CCCL_PAYLOAD_LTOIR;
  build_ptr->payload      = (void*) cubin.release();
  build_ptr->payload_size = cubin_size;
  build_ptr->payload_kind = CCCL_PAYLOAD_CUBIN;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  printf("\nEXCEPTION in cccl_device_unique_by_key_link_ltoir(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
