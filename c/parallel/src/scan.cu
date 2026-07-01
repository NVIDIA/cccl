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
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <cuda/__type_traits/is_trivially_copyable.h>

#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <nvrtc.h>

#include "util/aot_serialize.h"
#include "util/nvjitlink.h"
#include <cccl/c/aot.h>
#include <cccl/c/scan.h>
#include <kernels/iterators.h>
#include <kernels/operators.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/indirect_arg.h>
#include <util/scan_tile_state.h>
#include <util/types.h>

struct op_wrapper;
struct device_scan_policy;
using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

struct input_iterator_state_t;
struct output_iterator_t;

namespace scan
{
enum class InitKind
{
  Value,
  FutureValue,
  NoInit,
};

static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_type_info init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init;
}

std::string get_input_iterator_name()
{
  std::string iterator_t;
  check(cccl_type_name_from_nvrtc<input_iterator_state_t>(&iterator_t));
  return iterator_t;
}

std::string get_output_iterator_name()
{
  std::string iterator_t;
  check(cccl_type_name_from_nvrtc<output_iterator_t>(&iterator_t));
  return iterator_t;
}

std::string get_init_kernel_name(cccl_iterator_t input_it, cccl_iterator_t output_it, cccl_op_t op, cccl_type_info init)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_scan_policy>(&chained_policy_t));

  const cccl_type_info accum_t  = scan::get_accumulator_type(op, input_it, init);
  const std::string accum_cpp_t = cccl_type_enum_to_name(accum_t.type);
  const std::string input_iterator_t =
    (input_it.type == cccl_iterator_kind_t::CCCL_POINTER //
       ? cccl_type_enum_to_name(input_it.value_type.type, true) //
       : scan::get_input_iterator_name());
  const std::string output_iterator_t =
    output_it.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(output_it.value_type.type, true) //
      : scan::get_output_iterator_name();
  return std::format(
    "cub::detail::scan::DeviceScanInitKernel<{0}, {1}, {2}, cub::ScanTileState<{3}>, {3}>",
    chained_policy_t,
    input_iterator_t,
    output_iterator_t,
    accum_cpp_t);
}

std::string get_scan_kernel_name(
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_type_info init,
  bool force_inclusive,
  cccl_init_kind_t init_kind)
{
  std::string policy_selector_t;
  check(cccl_type_name_from_nvrtc<device_scan_policy>(&policy_selector_t));

  const cccl_type_info accum_t  = scan::get_accumulator_type(op, input_it, init);
  const std::string accum_cpp_t = cccl_type_enum_to_name(accum_t.type);
  const std::string input_iterator_t =
    (input_it.type == cccl_iterator_kind_t::CCCL_POINTER //
       ? cccl_type_enum_to_name(input_it.value_type.type, true) //
       : scan::get_input_iterator_name());
  const std::string output_iterator_t =
    output_it.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(output_it.value_type.type, true) //
      : scan::get_output_iterator_name();

  std::string init_t;
  std::string init_value_t;
  switch (init_kind)
  {
    case cccl_init_kind_t::CCCL_NO_INIT:
      init_t       = "cub::NullType";
      init_value_t = "cub::NullType";
      break;
    case cccl_init_kind_t::CCCL_FUTURE_VALUE_INIT:
      init_t       = cccl_type_enum_to_name(init.type);
      init_value_t = std::format("cub::FutureValue<{}>", init_t);
      break;
    case cccl_init_kind_t::CCCL_VALUE_INIT:
    default:
      init_t       = cccl_type_enum_to_name(init.type);
      init_value_t = init_t;
      break;
  }

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  std::string scan_op_t;
  check(cccl_type_name_from_nvrtc<op_wrapper>(&scan_op_t));

  auto tile_state_t = std::format("cub::ScanTileState<{0}>", accum_cpp_t);
  return std::format(
    "cub::detail::scan::DeviceScanKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, false, {9}>",
    policy_selector_t, // 0
    input_iterator_t, // 1
    output_iterator_t, // 2
    tile_state_t, // 3
    scan_op_t, // 4
    init_value_t, // 5
    offset_t, // 6
    accum_cpp_t, // 7
    force_inclusive ? "true" : "false", // 8
    init_t); // 9
}

struct scan_kernel_source
{
  cccl_device_scan_build_result_t& build;

  std::size_t InputSize() const
  {
    return build.input_type.size;
  }

  std::size_t InputAlign() const
  {
    return build.input_type.alignment;
  }

  std::size_t OutputSize() const
  {
    return build.output_type.size;
  }

  std::size_t OutputAlign() const
  {
    return build.output_type.alignment;
  }

  std::size_t AccumSize() const
  {
    return build.accumulator_type.size;
  }

  std::size_t AccumAlign() const
  {
    return build.accumulator_type.alignment;
  }

  CUkernel InitKernel() const
  {
    return build.init_kernel;
  }
  CUkernel ScanKernel() const
  {
    return build.scan_kernel;
  }
  scan_tile_state TileState() const
  {
    return {build.description_bytes_per_tile, build.payload_bytes_per_tile};
  }

  std::size_t lookahead_tile_state_size() const
  {
    return lookahead_tile_state_alignment();
  }

  std::size_t lookahead_tile_state_alignment() const
  {
    constexpr int state_size = alignof(cub::detail::warpspeed::scan_state);
    return ::cuda::next_power_of_two(
      ::cuda::round_up(state_size, build.accumulator_type.alignment) + build.accumulator_type.size);
  }

  static auto make_tile_state_kernel_arg(scan_tile_state ts)
  {
    cub::detail::scan::tile_state_kernel_arg_t<scan_tile_state, char> arg;
    ::cuda::std::__construct_at(&arg.lookback, ::cuda::std::move(ts));
    return arg;
  }

  static auto lookahead_make_tile_state_kernel_arg(void* ts)
  {
    // we can ignore passing a wrong AccumT, since we only store a pointer, and the kernel will have the right type
    cub::detail::scan::tile_state_kernel_arg_t<scan_tile_state, char> arg;
    ::cuda::std::__construct_at(&arg.lookahead, static_cast<cub::detail::warpspeed::tile_state_t<char>*>(ts));
    return arg;
  }
};
} // namespace scan

CUresult cccl_device_scan_compile(
  cccl_device_scan_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_type_info init,
  bool force_inclusive,
  cccl_init_kind_t init_kind,
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
  const cccl_type_info accum_t = scan::get_accumulator_type(op, input_it, init);
  const auto accum_cpp         = cccl_type_enum_to_name(accum_t.type);
  const auto input_it_value_t  = cccl_type_enum_to_name(input_it.value_type.type);
  const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

  const std::string input_iterator_t =
    (input_it.type == cccl_iterator_kind_t::CCCL_POINTER //
       ? cccl_type_enum_to_name(input_it.value_type.type, true) //
       : scan::get_input_iterator_name());
  const std::string output_iterator_t =
    output_it.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(output_it.value_type.type, true) //
      : scan::get_output_iterator_name();

  const std::string input_iterator_src =
    make_kernel_input_iterator(offset_t, "input_iterator_state_t", input_it_value_t, input_it);
  const std::string output_iterator_src =
    make_kernel_output_iterator(offset_t, "output_iterator_t", accum_cpp, output_it);

  const std::string op_src = make_kernel_user_binary_operator(accum_cpp, accum_cpp, accum_cpp, op);

  const auto policy_sel = [&] {
    using cub::detail::scan::policy_selector;
    using cub::detail::scan::primitive_accum;
    using cub::detail::scan::primitive_op;

    const auto is_trivial_type = [](cccl_type_enum /* type */) {
      // TODO: implement actual logic here when nontrivial custom types become supported
      return true;
    };

    const auto accum_type   = cccl_type_enum_to_cub_type(accum_t.type);
    const auto operation_t  = cccl_op_kind_to_cub_op(op.type);
    const auto input_type   = input_it.value_type.type;
    const auto input_type_t = cccl_type_enum_to_cub_type(input_type);

    const auto output_type = output_it.value_type.type;
    const bool types_match = input_type == output_type && input_type == accum_t.type;

    const bool input_contiguous             = input_it.type == cccl_iterator_kind_t::CCCL_POINTER;
    const bool output_contiguous            = output_it.type == cccl_iterator_kind_t::CCCL_POINTER;
    const bool input_trivially_copyable     = is_trivial_type(input_it.value_type.type);
    const bool output_trivially_copyable    = is_trivial_type(output_it.value_type.type);
    const bool output_default_constructible = output_trivially_copyable;
    const bool accum_is_primitive_or_trivially_copy_constructible = true;

    const bool benchmark_match =
      operation_t != cub::detail::op_kind_t::other && types_match && input_type != CCCL_STORAGE;

    return policy_selector{
      static_cast<int>(input_it.value_type.size),
      static_cast<int>(input_it.value_type.alignment),
      static_cast<int>(output_it.value_type.size),
      static_cast<int>(output_it.value_type.alignment),
      static_cast<int>(accum_t.size),
      static_cast<int>(accum_t.alignment),
      int{sizeof(OffsetT)},
      input_type_t,
      accum_type,
      operation_t,
      input_contiguous,
      output_contiguous,
      input_trivially_copyable,
      output_trivially_copyable,
      output_default_constructible,
      accum_is_primitive_or_trivially_copy_constructible,
      benchmark_match};
  }();

  const auto active_policy = policy_sel(cc);

  // TODO(bgruber): drop this if tuning policies become formattable
  std::stringstream policy_sel_str;
  policy_sel_str << active_policy;

  std::string policy_selector_expr = std::format(
    "cub::detail::scan::policy_selector_from_types<{}, {}, {}, {}, {}>",
    input_iterator_t,
    output_iterator_t,
    accum_cpp,
    offset_t,
    "op_wrapper");

  std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
struct __align__({1}) storage_t {{
  char data[{0}];
}};
{2}
{3}
{4}
using device_scan_policy = {5};
using namespace cub;
using namespace cub::detail::scan;
using cub::LookbackDelayPolicy;
using cub::LookbackDelayAlgorithm;
static_assert(device_scan_policy()(detail::current_tuning_cc()) == {6}, "Host generated and JIT compiled policy mismatch");
)XXX",
    input_it.value_type.size, // 0
    input_it.value_type.alignment, // 1
    input_iterator_src, // 2
    output_iterator_src, // 3
    op_src, // 4
    policy_selector_expr, // 5
    policy_sel_str.view()); // 6

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
    fflush(stdout);
#endif

  std::string init_kernel_name = scan::get_init_kernel_name(input_it, output_it, op, init);
  std::string scan_kernel_name = scan::get_scan_kernel_name(input_it, output_it, op, init, force_inclusive, init_kind);
  std::string init_kernel_lowered_name;
  std::string scan_kernel_lowered_name;

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

  const bool kernel_only = is_custom_op(op);

  // Collect all LTO-IRs to be linked (empty in kernel-only mode).
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(op);
  appender.add_iterator_definition(input_it);
  appender.add_iterator_definition(output_it);

  auto post_build =
    begin_linking_nvrtc_program(kernel_only ? 0 : num_lto_args, kernel_only ? nullptr : lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({init_kernel_name})
      ->add_expression({scan_kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({init_kernel_name, init_kernel_lowered_name})
      ->get_name({scan_kernel_name, scan_kernel_lowered_name});

  auto [description_bytes_per_tile,
        payload_bytes_per_tile] = get_tile_state_bytes_per_tile(accum_t, accum_cpp, args.data(), args.size(), arch);

  struct free_deleter
  {
    void operator()(void* p) const
    {
      std::free(p);
    }
  };
  static_assert(::cuda::is_trivially_copyable_v<cub::detail::scan::policy_selector>);
  const size_t policy_size = sizeof(policy_sel);
  std::unique_ptr<void, free_deleter> policy_ptr(std::malloc(policy_size));
  if (!policy_ptr)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(policy_ptr.get(), &policy_sel, sizeof(policy_sel));
  auto init_name = std::unique_ptr<char[]>(duplicate_c_string(init_kernel_lowered_name));
  auto scan_name = std::unique_ptr<char[]>(duplicate_c_string(scan_kernel_lowered_name));

  build_ptr->cc                         = cc.get();
  build_ptr->input_type                 = input_it.value_type;
  build_ptr->output_type                = output_it.value_type;
  build_ptr->accumulator_type           = accum_t;
  build_ptr->force_inclusive            = force_inclusive;
  build_ptr->init_kind                  = init_kind;
  build_ptr->description_bytes_per_tile = description_bytes_per_tile;
  build_ptr->payload_bytes_per_tile     = payload_bytes_per_tile;
  // Zero-init fields set by _load, not _compile.
  build_ptr->library     = nullptr;
  build_ptr->init_kernel = nullptr;
  build_ptr->scan_kernel = nullptr;

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

  build_ptr->runtime_policy           = policy_ptr.release();
  build_ptr->runtime_policy_size      = policy_size;
  build_ptr->init_kernel_lowered_name = init_name.release();
  build_ptr->scan_kernel_lowered_name = scan_name.release();

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_scan_compile(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_load(cccl_device_scan_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0
      || build_ptr->payload_kind != CCCL_PAYLOAD_CUBIN || build_ptr->init_kernel_lowered_name == nullptr
      || build_ptr->init_kernel_lowered_name[0] == '\0' || build_ptr->scan_kernel_lowered_name == nullptr
      || build_ptr->scan_kernel_lowered_name[0] == '\0')
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
    check(cuLibraryGetKernel(&build_ptr->init_kernel, build_ptr->library, build_ptr->init_kernel_lowered_name));
    check(cuLibraryGetKernel(&build_ptr->scan_kernel, build_ptr->library, build_ptr->scan_kernel_lowered_name));
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
  printf("\nEXCEPTION in cccl_device_scan_load(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

template <cub::ForceInclusive EnforceInclusive, typename InitValueT>
CUresult cccl_device_scan(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  InitValueT init,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::detail::scan::dispatch_with_accum<void, EnforceInclusive>(
      d_temp_storage,
      *temp_storage_bytes,
      indirect_arg_t{d_in},
      indirect_arg_t{d_out},
      indirect_arg_t{op},
      std::conditional_t<std::is_same_v<InitValueT, cub::NullType>, cub::NullType, indirect_arg_t>{init},
      static_cast<OffsetT>(num_items),
      stream,
      *static_cast<cub::detail::scan::policy_selector*>(build.runtime_policy),
      scan::scan_kernel_source{build},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc});
    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_scan(): %s\n", exc.what());
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

CUresult cccl_device_exclusive_scan(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  assert(!build.force_inclusive);
  assert(build.init_kind == cccl_init_kind_t::CCCL_VALUE_INIT);
  return cccl_device_scan<cub::ForceInclusive::No>(
    build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, stream);
}

CUresult cccl_device_inclusive_scan(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  assert(build.force_inclusive);
  assert(build.init_kind == cccl_init_kind_t::CCCL_VALUE_INIT);
  return cccl_device_scan<cub::ForceInclusive::Yes>(
    build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, stream);
}

CUresult cccl_device_exclusive_scan_future_value(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_iterator_t init,
  CUstream stream)
{
  assert(!build.force_inclusive);
  assert(build.init_kind == cccl_init_kind_t::CCCL_FUTURE_VALUE_INIT);
  return cccl_device_scan<cub::ForceInclusive::No>(
    build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, stream);
}

CUresult cccl_device_inclusive_scan_future_value(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_iterator_t init,
  CUstream stream)
{
  assert(build.force_inclusive);
  assert(build.init_kind == cccl_init_kind_t::CCCL_FUTURE_VALUE_INIT);
  return cccl_device_scan<cub::ForceInclusive::Yes>(
    build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, stream);
}

CUresult cccl_device_inclusive_scan_no_init(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  assert(build.force_inclusive);
  assert(build.init_kind == cccl_init_kind_t::CCCL_NO_INIT);
  return cccl_device_scan<cub::ForceInclusive::Yes, cub::NullType>(
    build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, cub::NullType{}, stream);
}

CUresult cccl_device_scan_build_ex(
  cccl_device_scan_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_type_info init,
  bool force_inclusive,
  cccl_init_kind_t init_kind,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
{
  CUresult r = cccl_device_scan_compile(
    build_ptr,
    d_in,
    d_out,
    op,
    init,
    force_inclusive,
    init_kind,
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
  CUresult load_r = cccl_device_scan_load(build_ptr);
  if (load_r != CUDA_SUCCESS)
  {
    cccl_device_scan_cleanup(build_ptr);
  }
  return load_r;
}

CUresult cccl_device_scan_build(
  cccl_device_scan_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_type_info init,
  bool force_inclusive,
  cccl_init_kind_t init_kind,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_scan_build_ex(
    build_ptr,
    d_in,
    d_out,
    op,
    init,
    force_inclusive,
    init_kind,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_scan_cleanup(cccl_device_scan_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::unique_ptr<char[]> payload(reinterpret_cast<char*>(build_ptr->payload));
  std::free(build_ptr->runtime_policy);
  std::unique_ptr<char[]> init_name(build_ptr->init_kernel_lowered_name);
  std::unique_ptr<char[]> scan_name(build_ptr->scan_kernel_lowered_name);
  if (build_ptr->library != nullptr)
  {
    check(cuLibraryUnload(build_ptr->library));
  }

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_scan_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_link_ltoir(
  cccl_device_scan_build_result_t* build_ptr, const void** input_blobs, const size_t* input_sizes, size_t num_inputs)
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
  printf("\nEXCEPTION in cccl_device_scan_link_ltoir(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_serialize(const cccl_device_scan_build_result_t* build_ptr, void** out_buf, size_t* out_size)
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

  *out_buf  = nullptr;
  *out_size = 0;

  using namespace cccl::aot;
  buffer_writer w;
  write_header(w, CCCL_AOT_ALGO_SCAN, build_ptr->payload_kind, build_ptr->cc);
  write_type_info(w, build_ptr->input_type);
  write_type_info(w, build_ptr->output_type);
  write_type_info(w, build_ptr->accumulator_type);
  w.write_pod<uint8_t>(build_ptr->force_inclusive ? 1 : 0);
  w.write_pod<uint32_t>(static_cast<uint32_t>(build_ptr->init_kind));
  w.write_pod<uint64_t>(build_ptr->description_bytes_per_tile);
  w.write_pod<uint64_t>(build_ptr->payload_bytes_per_tile);
  w.write_blob(build_ptr->payload, build_ptr->payload_size);
  w.write_blob(build_ptr->runtime_policy, build_ptr->runtime_policy_size);
  w.write_cstring(build_ptr->init_kernel_lowered_name);
  w.write_cstring(build_ptr->scan_kernel_lowered_name);
  w.release(out_buf, out_size);
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_scan_serialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_deserialize(cccl_device_scan_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  using namespace cccl::aot;
  buffer_reader r{buf, size};
  const auto h = read_and_validate_header(r, CCCL_AOT_ALGO_SCAN);

  const auto in_type     = read_type_info(r);
  const auto out_type    = read_type_info(r);
  const auto accum_type  = read_type_info(r);
  const bool force_inc   = r.read_pod<uint8_t>() != 0;
  const auto init_kind_v = r.read_pod<uint32_t>();
  if (init_kind_v > static_cast<uint32_t>(CCCL_NO_INIT))
  {
    throw std::runtime_error(std::format("aot blob: invalid init kind ({})", init_kind_v));
  }
  const auto init_kind  = static_cast<cccl_init_kind_t>(init_kind_v);
  const auto desc_bytes = r.read_pod<uint64_t>();
  const auto pay_bytes  = r.read_pod<uint64_t>();

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

  std::unique_ptr<cub::detail::scan::policy_selector, decltype(&std::free)> policy(
    static_cast<cub::detail::scan::policy_selector*>(std::malloc(sizeof(cub::detail::scan::policy_selector))),
    std::free);
  if (!policy)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  r.read_into(policy.get(), sizeof(cub::detail::scan::policy_selector));

  std::unique_ptr<char[]> n_init{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_scan{r.read_cstring_dup()};

  cccl_device_scan_build_result_t result{};
  result.cc                         = static_cast<int>(h.cc);
  result.payload_kind               = static_cast<cccl_payload_kind_t>(h.payload_kind);
  result.input_type                 = in_type;
  result.output_type                = out_type;
  result.accumulator_type           = accum_type;
  result.force_inclusive            = force_inc;
  result.init_kind                  = init_kind;
  result.description_bytes_per_tile = desc_bytes;
  result.payload_bytes_per_tile     = pay_bytes;
  result.payload                    = payload_owner.release();
  result.payload_size               = payload_size;
  result.runtime_policy             = policy.release();
  result.runtime_policy_size        = sizeof(cub::detail::scan::policy_selector);
  result.init_kernel_lowered_name   = n_init.release();
  result.scan_kernel_lowered_name   = n_scan.release();
  *build_ptr                        = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_scan_deserialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}
