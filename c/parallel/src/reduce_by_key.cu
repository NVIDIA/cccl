//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nvrtc.h>

#include "util/nvjitlink.h"
#include "util/serialization.h"
#include <cccl/c/reduce_by_key.h>
#include <cccl/c/serialization.h>
#include <kernels/iterators.h>
#include <kernels/operators.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/indirect_arg.h>
#include <util/reduce_by_key_scan_tile_state.h>
#include <util/scan_tile_state.h>
#include <util/types.h>

struct op_wrapper;
struct device_reduce_by_key_policy;
using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

struct input_keys_iterator_state_t;
struct input_iterator_state_t;
struct output_unique_iterator_t;
struct output_aggregates_iterator_t;
struct num_runs_output_iterator_t;

// CUB's ReduceByKey compares keys with the key type's own `operator==`, and the C API exposes no equality
// operator. The JIT-compiled source therefore aliases this tag to `::cuda::std::equal_to<KeyT>`. Both sides
// are empty types, which keeps the kernel's parameter block identical on the host and in the JIT compilation.
struct equality_op_wrapper
{};

namespace reduce_by_key
{
std::string get_input_keys_iterator_name()
{
  std::string iterator_t;
  check(cccl_type_name_from_nvrtc<input_keys_iterator_state_t>(&iterator_t));
  return iterator_t;
}

std::string get_input_iterator_name()
{
  std::string iterator_t;
  check(cccl_type_name_from_nvrtc<input_iterator_state_t>(&iterator_t));
  return iterator_t;
}

std::string get_output_unique_iterator_name()
{
  std::string iterator_t;
  check(cccl_type_name_from_nvrtc<output_unique_iterator_t>(&iterator_t));
  return iterator_t;
}

std::string get_output_aggregates_iterator_name()
{
  std::string iterator_t;
  check(cccl_type_name_from_nvrtc<output_aggregates_iterator_t>(&iterator_t));
  return iterator_t;
}

std::string get_num_runs_output_iterator_name()
{
  std::string iterator_t;
  check(cccl_type_name_from_nvrtc<num_runs_output_iterator_t>(&iterator_t));
  return iterator_t;
}

std::string get_iterator_name(cccl_iterator_t it, std::string (*named_iterator)())
{
  return it.type == cccl_iterator_kind_t::CCCL_POINTER ? cccl_type_enum_to_name(it.value_type.type, true)
                                                       : named_iterator();
}

// The run index CUB's agent pairs with every accumulator is an OffsetT, so the two of them together decide
// whether the tile state collapses into a single machine word.
constexpr int run_index_size = static_cast<int>(sizeof(OffsetT));

std::string get_tile_state_name(cccl_type_info accum_t)
{
  const bool single_word = reduce_by_key_is_single_word(run_index_size, static_cast<int>(accum_t.size));
  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));
  return std::format(
    "cub::ReduceByKeyScanTileState<{0}, {1}, {2}>",
    cccl_type_enum_to_name(accum_t.type),
    offset_t,
    single_word ? "true" : "false");
}

std::string get_init_kernel_name(cccl_iterator_t d_num_runs_out, cccl_type_info accum_t)
{
  const std::string num_runs_output_iterator_t = get_iterator_name(d_num_runs_out, get_num_runs_output_iterator_name);
  return std::format(
    "cub::detail::scan::DeviceCompactInitKernel<{0}, {1}>", get_tile_state_name(accum_t), num_runs_output_iterator_t);
}

std::string get_sweep_kernel_name(
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  cccl_type_info accum_t)
{
  std::string policy_selector_t;
  check(cccl_type_name_from_nvrtc<device_reduce_by_key_policy>(&policy_selector_t));

  const std::string input_keys_iterator_t        = get_iterator_name(d_keys_in, get_input_keys_iterator_name);
  const std::string unique_output_iterator_t     = get_iterator_name(d_unique_out, get_output_unique_iterator_name);
  const std::string input_values_iterator_t      = get_iterator_name(d_values_in, get_input_iterator_name);
  const std::string aggregates_output_iterator_t = get_iterator_name(d_aggregates_out, get_output_aggregates_iterator_name);
  const std::string num_runs_output_iterator_t   = get_iterator_name(d_num_runs_out, get_num_runs_output_iterator_name);

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  std::string reduction_op_t;
  check(cccl_type_name_from_nvrtc<op_wrapper>(&reduction_op_t));

  std::string equality_op_t;
  check(cccl_type_name_from_nvrtc<equality_op_wrapper>(&equality_op_t));

  return std::format(
    "cub::detail::reduce_by_key::DeviceReduceByKeyKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, "
    "cub::NullType>",
    policy_selector_t, // 0
    input_keys_iterator_t, // 1
    unique_output_iterator_t, // 2
    input_values_iterator_t, // 3
    aggregates_output_iterator_t, // 4
    num_runs_output_iterator_t, // 5
    get_tile_state_name(accum_t), // 6
    equality_op_t, // 7
    reduction_op_t, // 8
    offset_t, // 9
    cccl_type_enum_to_name(accum_t.type)); // 10
}

template <int TxnWordSize>
struct reduce_by_key_single_word_kernel_source
{
  cccl_device_reduce_by_key_build_result_t& build;

  using ReduceByKeyTileStateT = reduce_by_key_single_word_tile_state<TxnWordSize>;

  CUkernel CompactInitKernel() const
  {
    return build.init_kernel;
  }
  CUkernel ReduceByKeySweepKernel() const
  {
    return build.sweep_kernel;
  }
  ReduceByKeyTileStateT TileState() const
  {
    return {};
  }
};

struct reduce_by_key_multi_word_kernel_source
{
  cccl_device_reduce_by_key_build_result_t& build;

  using ReduceByKeyTileStateT = scan_tile_state;

  CUkernel CompactInitKernel() const
  {
    return build.init_kernel;
  }
  CUkernel ReduceByKeySweepKernel() const
  {
    return build.sweep_kernel;
  }
  scan_tile_state TileState() const
  {
    return {build.description_bytes_per_tile, build.payload_bytes_per_tile};
  }
};
} // namespace reduce_by_key

CUresult cccl_device_reduce_by_key_compile(
  cccl_device_reduce_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
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
  const cccl_type_info key_t   = d_keys_in.value_type;
  const cccl_type_info value_t = d_values_in.value_type;
  // ReduceByKey's accumulator is the reduction operator's result type; the driver launcher infers it from
  // the value type, exactly like `cub::DeviceReduce::ReduceByKey` does.
  const cccl_type_info accum_t = value_t;
  const auto accum_cpp         = cccl_type_enum_to_name(accum_t.type);
  const auto key_cpp           = cccl_type_enum_to_name(key_t.type);
  const auto value_cpp         = cccl_type_enum_to_name(value_t.type);
  const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

  const bool single_word = reduce_by_key_is_single_word(reduce_by_key::run_index_size, static_cast<int>(accum_t.size));

  const std::string input_keys_iterator_src =
    make_kernel_input_iterator(offset_t, "input_keys_iterator_state_t", key_cpp, d_keys_in);
  const std::string input_iterator_src =
    make_kernel_input_iterator(offset_t, "input_iterator_state_t", value_cpp, d_values_in);
  const std::string unique_output_iterator_src =
    make_kernel_output_iterator(offset_t, "output_unique_iterator_t", key_cpp, d_unique_out);
  const std::string aggregates_output_iterator_src =
    make_kernel_output_iterator(offset_t, "output_aggregates_iterator_t", accum_cpp, d_aggregates_out);
  const std::string num_runs_output_iterator_src = make_kernel_output_iterator(
    offset_t, "num_runs_output_iterator_t", cccl_type_enum_to_name(d_num_runs_out.value_type.type), d_num_runs_out);

  const std::string op_src = make_kernel_user_binary_operator(accum_cpp, accum_cpp, accum_cpp, op);
  const std::string equality_op_src =
    std::format("using equality_op_wrapper = ::cuda::std::equal_to<{}>;", key_cpp);

  const auto policy_sel = cub::detail::reduce_by_key::policy_selector{
    static_cast<int>(key_t.size),
    static_cast<int>(accum_t.size),
    cccl_type_enum_to_cub_type(accum_t.type),
    true, // key_is_primitive: every type the C API accepts today is primitive
    true, // key_is_trivially_copyable
    true, // accum_is_primitive
    false}; // op_is_primitive: the JIT raises the user op into `op_wrapper`, which is never a builtin

  const auto active_policy = policy_sel(cc);

  std::stringstream policy_sel_str;
  policy_sel_str << active_policy;

  const std::string policy_selector_expr = std::format(
    "cub::detail::reduce_by_key::policy_selector_from_types<{}, {}, {}>", "op_wrapper", accum_cpp, key_cpp);

  std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/tuning/tuning_reduce_by_key.cuh>
#include <cub/device/dispatch/kernels/kernel_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce_by_key.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
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
using device_reduce_by_key_policy = {9};
using namespace cub;
using namespace cub::detail::reduce_by_key;
static_assert(device_reduce_by_key_policy()(detail::current_tuning_cc()) == {10}, "Host generated and JIT compiled policy mismatch");
)XXX",
    value_t.size, // 0
    value_t.alignment, // 1
    input_keys_iterator_src, // 2
    input_iterator_src, // 3
    unique_output_iterator_src, // 4
    aggregates_output_iterator_src, // 5
    num_runs_output_iterator_src, // 6
    op_src, // 7
    equality_op_src, // 8
    policy_selector_expr, // 9
    policy_sel_str.view()); // 10

#if false // CCCL_DEBUGGING_SWITCH
  fflush(stderr);
  printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
  fflush(stdout);
#endif

  const std::string init_kernel_name  = reduce_by_key::get_init_kernel_name(d_num_runs_out, accum_t);
  const std::string sweep_kernel_name = reduce_by_key::get_sweep_kernel_name(
    d_keys_in, d_unique_out, d_values_in, d_aggregates_out, d_num_runs_out, accum_t);
  std::string init_kernel_lowered_name;
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
    "-default-device",
    "-DCUB_DISABLE_CDP",
    "-std=c++20"};

  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  const bool kernel_only = is_custom_op(op);

  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(op);
  appender.add_iterator_definition(d_keys_in);
  appender.add_iterator_definition(d_values_in);
  appender.add_iterator_definition(d_unique_out);
  appender.add_iterator_definition(d_aggregates_out);
  appender.add_iterator_definition(d_num_runs_out);

  auto post_build = begin_linking_nvrtc_program(kernel_only ? 0 : num_lto_args, kernel_only ? nullptr : lopts)
                      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
                      ->add_expression({init_kernel_name})
                      ->add_expression({sweep_kernel_name})
                      ->compile_program({args.data(), args.size()})
                      ->get_name({init_kernel_name, init_kernel_lowered_name})
                      ->get_name({sweep_kernel_name, sweep_kernel_lowered_name});

  // The multi-word tile state sizes its payload by the accumulator paired with the run index, so the probe
  // has to ask for that pair rather than the bare accumulator. No type the C API accepts today is large
  // enough to reach this branch.
  std::pair<size_t, size_t> tile_state_bytes{0, 0};
  if (!single_word)
  {
    tile_state_bytes = get_tile_state_bytes_per_tile(
      accum_t, std::format("cub::KeyValuePair<{}, {}>", offset_t, accum_cpp), args.data(), args.size(), arch);
  }

  struct free_deleter
  {
    void operator()(void* p) const
    {
      std::free(p);
    }
  };
  static_assert(::cuda::is_trivially_copyable_v<cub::detail::reduce_by_key::policy_selector>);
  const size_t policy_size = sizeof(policy_sel);
  std::unique_ptr<void, free_deleter> policy_ptr(std::malloc(policy_size));
  if (!policy_ptr)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(policy_ptr.get(), &policy_sel, sizeof(policy_sel));
  auto init_name  = std::unique_ptr<char[]>(duplicate_c_string(init_kernel_lowered_name));
  auto sweep_name = std::unique_ptr<char[]>(duplicate_c_string(sweep_kernel_lowered_name));

  build_ptr->cc                         = cc.get();
  build_ptr->key_type                   = key_t;
  build_ptr->value_type                 = value_t;
  build_ptr->aggregate_type             = accum_t;
  build_ptr->description_bytes_per_tile = tile_state_bytes.first;
  build_ptr->payload_bytes_per_tile     = tile_state_bytes.second;
  build_ptr->library                    = nullptr;
  build_ptr->init_kernel                = nullptr;
  build_ptr->sweep_kernel               = nullptr;

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
  build_ptr->sweep_kernel_lowered_name = sweep_name.release();

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_reduce_by_key_compile(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_by_key_load(cccl_device_reduce_by_key_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0
      || build_ptr->payload_kind != CCCL_PAYLOAD_CUBIN || build_ptr->init_kernel_lowered_name == nullptr
      || build_ptr->init_kernel_lowered_name[0] == '\0' || build_ptr->sweep_kernel_lowered_name == nullptr
      || build_ptr->sweep_kernel_lowered_name[0] == '\0')
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
    check(cuLibraryGetKernel(&build_ptr->sweep_kernel, build_ptr->library, build_ptr->sweep_kernel_lowered_name));
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
  printf("\nEXCEPTION in cccl_device_reduce_by_key_load(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

static CUresult cccl_device_reduce_by_key_launch(
  cccl_device_reduce_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    auto launcher_factory = cub::detail::CudaDriverLauncherFactory{cu_device, build.cc};
    auto policy_selector  = *static_cast<cub::detail::reduce_by_key::policy_selector*>(build.runtime_policy);

    // The iterators are passed type erased, so the value types the dispatch would otherwise deduce are
    // placeholders; the real ones only live in the JIT-compiled kernel and in the tile state the kernel
    // source hands back.
    auto launch = [&](auto kernel_source) {
      using KernelSourceT = decltype(kernel_source);
      return cub::detail::reduce_by_key::dispatch<
        indirect_arg_t,
        indirect_arg_t,
        indirect_arg_t,
        indirect_arg_t,
        indirect_arg_t,
        equality_op_wrapper,
        indirect_arg_t,
        OffsetT,
        char,
        char,
        cub::detail::reduce_by_key::policy_selector,
        KernelSourceT,
        cub::detail::CudaDriverLauncherFactory>(
        d_temp_storage,
        *temp_storage_bytes,
        indirect_arg_t{d_keys_in},
        indirect_arg_t{d_unique_out},
        indirect_arg_t{d_values_in},
        indirect_arg_t{d_aggregates_out},
        indirect_arg_t{d_num_runs_out},
        equality_op_wrapper{},
        indirect_arg_t{op},
        static_cast<OffsetT>(num_items),
        stream,
        policy_selector,
        kernel_source,
        launcher_factory);
    };

    const int accum_size = static_cast<int>(build.aggregate_type.size);
    cudaError_t exec_status;
    if (reduce_by_key_is_single_word(reduce_by_key::run_index_size, accum_size))
    {
      switch (reduce_by_key_txn_word_size(reduce_by_key::run_index_size, accum_size))
      {
        case 4:
          exec_status = launch(reduce_by_key::reduce_by_key_single_word_kernel_source<4>{build});
          break;
        case 8:
          exec_status = launch(reduce_by_key::reduce_by_key_single_word_kernel_source<8>{build});
          break;
        default:
          exec_status = launch(reduce_by_key::reduce_by_key_single_word_kernel_source<16>{build});
          break;
      }
    }
    else
    {
      exec_status = launch(reduce_by_key::reduce_by_key_multi_word_kernel_source{build});
    }
    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_reduce_by_key(): %s\n", exc.what());
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

CUresult cccl_device_reduce_by_key(
  cccl_device_reduce_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  if (build.runtime_policy == nullptr || build.init_kernel == nullptr || build.sweep_kernel == nullptr
      || build.library == nullptr || temp_storage_bytes == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return cccl_device_reduce_by_key_launch(
    build,
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_values_in,
    d_unique_out,
    d_aggregates_out,
    d_num_runs_out,
    num_items,
    op,
    stream);
}

CUresult cccl_device_reduce_by_key_build_ex(
  cccl_device_reduce_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult build_status = cccl_device_reduce_by_key_compile(
    build_ptr,
    d_keys_in,
    d_values_in,
    d_unique_out,
    d_aggregates_out,
    d_num_runs_out,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    config);
  if (build_status != CUDA_SUCCESS)
  {
    return build_status;
  }

  CUresult load_status = cccl_device_reduce_by_key_load(build_ptr);
  if (load_status != CUDA_SUCCESS)
  {
    cccl_device_reduce_by_key_cleanup(build_ptr);
  }
  return load_status;
}

CUresult cccl_device_reduce_by_key_build(
  cccl_device_reduce_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_reduce_by_key_build_ex(
    build_ptr,
    d_keys_in,
    d_values_in,
    d_unique_out,
    d_aggregates_out,
    d_num_runs_out,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_reduce_by_key_cleanup(cccl_device_reduce_by_key_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> payload(reinterpret_cast<char*>(build_ptr->payload));
  std::free(build_ptr->runtime_policy);
  std::unique_ptr<char[]> init_name(build_ptr->init_kernel_lowered_name);
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
  printf("\nEXCEPTION in cccl_device_reduce_by_key_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_by_key_link_ltoir(
  cccl_device_reduce_by_key_build_result_t* build_ptr,
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
  printf("\nEXCEPTION in cccl_device_reduce_by_key_link_ltoir(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_by_key_serialize(
  const cccl_device_reduce_by_key_build_result_t* build_ptr, void** out_buf, size_t* out_size)
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

  using namespace cccl::serialization;
  buffer_writer w;
  write_header(w, CCCL_SERIALIZATION_ALGO_REDUCE_BY_KEY, build_ptr->payload_kind, build_ptr->cc);
  w.write_pod<uint64_t>(build_ptr->description_bytes_per_tile);
  w.write_pod<uint64_t>(build_ptr->payload_bytes_per_tile);
  // The accumulator's size picks the tile state layout at launch time, so the types have to survive a
  // round trip even though only their sizes are read back.
  for (const cccl_type_info& t : {build_ptr->key_type, build_ptr->value_type, build_ptr->aggregate_type})
  {
    w.write_pod<uint64_t>(t.size);
    w.write_pod<uint64_t>(t.alignment);
    w.write_pod<uint64_t>(static_cast<uint64_t>(t.type));
  }
  w.write_blob(build_ptr->payload, build_ptr->payload_size);
  w.write_blob(build_ptr->runtime_policy, build_ptr->runtime_policy_size);
  w.write_cstring(build_ptr->init_kernel_lowered_name);
  w.write_cstring(build_ptr->sweep_kernel_lowered_name);
  w.release(out_buf, out_size);
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_reduce_by_key_serialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_by_key_deserialize(
  cccl_device_reduce_by_key_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  using namespace cccl::serialization;
  buffer_reader r{buf, size};
  const auto h = read_and_validate_header(r, CCCL_SERIALIZATION_ALGO_REDUCE_BY_KEY);

  const auto desc_bytes = r.read_pod<uint64_t>();
  const auto pay_bytes  = r.read_pod<uint64_t>();

  cccl_type_info types[3] = {};
  for (cccl_type_info& t : types)
  {
    t.size      = r.read_pod<uint64_t>();
    t.alignment = r.read_pod<uint64_t>();
    t.type      = static_cast<cccl_type_enum>(r.read_pod<uint64_t>());
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
    throw std::runtime_error("serialization blob: empty payload");
  }

  std::unique_ptr<cub::detail::reduce_by_key::policy_selector, decltype(&std::free)> policy(
    static_cast<cub::detail::reduce_by_key::policy_selector*>(
      std::malloc(sizeof(cub::detail::reduce_by_key::policy_selector))),
    std::free);
  if (!policy)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  r.read_into(policy.get(), sizeof(cub::detail::reduce_by_key::policy_selector));

  std::unique_ptr<char[]> n_init{r.read_cstring_dup()};
  std::unique_ptr<char[]> n_sweep{r.read_cstring_dup()};

  cccl_device_reduce_by_key_build_result_t result{};
  result.cc                          = static_cast<int>(h.cc);
  result.payload_kind                = static_cast<cccl_payload_kind_t>(h.payload_kind);
  result.description_bytes_per_tile  = desc_bytes;
  result.payload_bytes_per_tile      = pay_bytes;
  result.key_type                    = types[0];
  result.value_type                  = types[1];
  result.aggregate_type              = types[2];
  result.payload                     = payload_owner.release();
  result.payload_size                = payload_size;
  result.runtime_policy              = policy.release();
  result.runtime_policy_size         = sizeof(cub::detail::reduce_by_key::policy_selector);
  result.init_kernel_lowered_name    = n_init.release();
  result.sweep_kernel_lowered_name   = n_sweep.release();
  *build_ptr                         = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_reduce_by_key_deserialize(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}
