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
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <nvrtc.h>

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

struct scan_runtime_tuning_policy
{
  int block_size;
  int items_per_thread;
  cub::CacheLoadModifier load_modifier;

  scan_runtime_tuning_policy Scan() const
  {
    return *this;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }

  int BlockThreads() const
  {
    return block_size;
  }

  cub::CacheLoadModifier LoadModifier() const
  {
    return load_modifier;
  }

  void CheckLoadModifier() const
  {
    if (LoadModifier() == cub::CacheLoadModifier::LOAD_LDG)
    {
      throw std::runtime_error("The memory consistency model does not apply to texture "
                               "accesses");
    }
  }
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

scan_runtime_tuning_policy get_policy(int /*cc*/, cccl_type_info /*accumulator_type*/)
{
  // TODO: we should update this once we figure out a way to reuse
  // tuning logic from C++. Alternately, we should implement
  // something better than a hardcoded default:
  return {128, 4, cub::LOAD_DEFAULT};
}

static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_value_t init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init.type;
}

std::string get_input_iterator_name()
{
  std::string iterator_t;
  check(nvrtcGetTypeName<input_iterator_state_t>(&iterator_t));
  return iterator_t;
}

std::string get_output_iterator_name()
{
  std::string iterator_t;
  check(nvrtcGetTypeName<output_iterator_t>(&iterator_t));
  return iterator_t;
}

std::string
get_init_kernel_name(cccl_iterator_t input_it, cccl_iterator_t /*output_it*/, cccl_op_t op, cccl_value_t init)
{
  const cccl_type_info accum_t  = scan::get_accumulator_type(op, input_it, init);
  const std::string accum_cpp_t = cccl_type_enum_to_name(accum_t.type);
  return std::format("cub::detail::scan::DeviceScanInitKernel<cub::ScanTileState<{0}>>", accum_cpp_t);
}

std::string get_scan_kernel_name(
  cccl_iterator_t input_it, cccl_iterator_t output_it, cccl_op_t op, cccl_value_t init, bool force_inclusive)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_scan_policy>(&chained_policy_t));

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
  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string scan_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&scan_op_t));

  auto tile_state_t = std::format("cub::ScanTileState<{0}>", accum_cpp_t);
  return std::format(
    "cub::detail::scan::DeviceScanKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}>",
    chained_policy_t, // 0
    input_iterator_t, // 1
    output_iterator_t, // 2
    tile_state_t, // 3
    scan_op_t, // 4
    init_t, // 5
    offset_t, // 6
    accum_cpp_t, // 7
    force_inclusive ? "true" : "false", // 8
    init_t); // 9
}

template <auto* GetPolicy>
struct dynamic_scan_policy_t
{
  using MaxPolicy = dynamic_scan_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<scan_runtime_tuning_policy>(GetPolicy(device_ptx_version, accumulator_type));
  }

  cccl_type_info accumulator_type;
};

struct scan_kernel_source
{
  cccl_device_scan_build_result_t& build;

  std::size_t AccumSize() const
  {
    return build.accumulator_type.size;
  }
  CUkernel InitKernel() const
  {
    return build.init_kernel;
  }
  CUkernel ScanKernel() const
  {
    return build.scan_kernel;
  }
  scan_tile_state TileState()
  {
    return {build.description_bytes_per_tile, build.payload_bytes_per_tile};
  }
};
} // namespace scan

CUresult cccl_device_scan_build_ex(
  cccl_device_scan_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
  bool force_inclusive,
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

    const int cc                 = cc_major * 10 + cc_minor;
    const cccl_type_info accum_t = scan::get_accumulator_type(op, input_it, init);
    const auto policy            = scan::get_policy(cc, accum_t);
    const auto accum_cpp         = cccl_type_enum_to_name(accum_t.type);
    const auto input_it_value_t  = cccl_type_enum_to_name(input_it.value_type.type);
    const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

    const std::string input_iterator_src =
      make_kernel_input_iterator(offset_t, "input_iterator_state_t", input_it_value_t, input_it);
    const std::string output_iterator_src =
      make_kernel_output_iterator(offset_t, "output_iterator_t", accum_cpp, output_it);

    const std::string op_src = make_kernel_user_binary_operator(accum_cpp, accum_cpp, accum_cpp, op);

    constexpr std::string_view src_template = R"XXX(
#include <cub/block/block_scan.cuh>
#include <cub/device/dispatch/kernels/scan.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
struct __align__({1}) storage_t {{
  char data[{0}];
}};
{4}
{5}
struct agent_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {2};
  static constexpr int BLOCK_THREADS = {3};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = cub::BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;
  struct detail {{
    using delay_constructor_t = cub::detail::default_delay_constructor_t<{7}>;
  }};
}};
struct device_scan_policy {{
  struct ActivePolicy {{
    using ScanPolicyT = agent_policy_t;
  }};
}};
{6}
)XXX";

    const std::string& src = std::format(
      src_template,
      input_it.value_type.size, // 0
      input_it.value_type.alignment, // 1
      policy.items_per_thread, // 2
      policy.block_size, // 3
      input_iterator_src, // 4
      output_iterator_src, // 5
      op_src, // 6
      accum_cpp); // 7

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    std::string init_kernel_name = scan::get_init_kernel_name(input_it, output_it, op, init);
    std::string scan_kernel_name = scan::get_scan_kernel_name(input_it, output_it, op, init, force_inclusive);
    std::string init_kernel_lowered_name;
    std::string scan_kernel_lowered_name;

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
    appender.add_iterator_definition(input_it);
    appender.add_iterator_definition(output_it);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{src.c_str(), name})
        ->add_expression({init_kernel_name})
        ->add_expression({scan_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({init_kernel_name, init_kernel_lowered_name})
        ->get_name({scan_kernel_name, scan_kernel_lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
        ->finalize_program();

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->init_kernel, build_ptr->library, init_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->scan_kernel, build_ptr->library, scan_kernel_lowered_name.c_str()));

    auto [description_bytes_per_tile,
          payload_bytes_per_tile] = get_tile_state_bytes_per_tile(accum_t, accum_cpp, args.data(), args.size(), arch);

    build_ptr->cc                         = cc;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
    build_ptr->accumulator_type           = accum_t;
    build_ptr->force_inclusive            = force_inclusive;
    build_ptr->description_bytes_per_tile = description_bytes_per_tile;
    build_ptr->payload_bytes_per_tile     = payload_bytes_per_tile;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_scan_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

template <cub::ForceInclusive EnforceInclusive>
CUresult cccl_device_scan(
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
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto exec_status = cub::DispatchScan<
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      indirect_arg_t,
      ::cuda::std::size_t,
      void,
      EnforceInclusive,
      scan::dynamic_scan_policy_t<&scan::get_policy>,
      scan::scan_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_in,
        d_out,
        op,
        init,
        num_items,
        stream,
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        {scan::get_accumulator_type(op, d_in, init)});

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
  return cccl_device_scan<cub::ForceInclusive::Yes>(
    build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, stream);
}

CUresult cccl_device_scan_build(
  cccl_device_scan_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_value_t init,
  bool force_inclusive,
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
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_scan_cleanup(cccl_device_scan_build_result_t* build_ptr)
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
    printf("\nEXCEPTION in cccl_device_scan_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
