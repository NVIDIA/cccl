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
#include <cub/device/dispatch/dispatch_reduce.cuh> // cub::DispatchSegmentedReduce
#include <cub/thread/thread_load.cuh> // cub::LoadModifier

#include <exception> // std::exception
#include <format> // std::format
#include <string> // std::string
#include <string_view> // std::string_view
#include <type_traits> // std::is_same_v

#include "kernels/iterators.h"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/segmented_reduce.h>
#include <cccl/c/types.h> // cccl_type_info
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <stdio.h> // printf

struct op_wrapper;
struct device_segmented_reduce_policy;
using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

// check we can map OffsetT to ::cuda::std::uint64_t
static_assert(std::is_unsigned_v<OffsetT>);
static_assert(sizeof(OffsetT) == sizeof(::cuda::std::uint64_t));

struct input_iterator_t;
struct output_iterator_t;
struct start_offset_iterator_t;
struct end_offset_iterator_t;

namespace segmented_reduce
{

struct segmented_reduce_runtime_tuning_policy
{
  int block_size;
  int items_per_thread;
  int vector_load_length;

  segmented_reduce_runtime_tuning_policy SegmentedReduce() const
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

struct segmented_reduce_tuning_t
{
  int cc;
  int block_size;
  int items_per_thread;
  int vector_load_length;
};

segmented_reduce_runtime_tuning_policy get_policy(int cc, cccl_type_info accumulator_type)
{
  // TODO: we should update this once we figure out a way to reuse
  // tuning logic from C++. Alternately, we should implement
  // something better than a hardcoded default:
  constexpr segmented_reduce_tuning_t chain[] = {{16, 256, 16, 4}, {35, 256, 20, 4}};

  auto [_, block_size, items_per_thread, vector_load_length] = find_tuning(cc, chain);

  auto four_bytes_per_thread = items_per_thread * 4 / accumulator_type.size;
  items_per_thread           = _CUDA_VSTD::min<decltype(items_per_thread)>(four_bytes_per_thread, items_per_thread * 2);
  items_per_thread           = _CUDA_VSTD::min(1, items_per_thread);

  auto work_per_sm    = cub::detail::max_smem_per_block / (accumulator_type.size * items_per_thread);
  auto max_block_size = cuda::round_up(work_per_sm, 32);
  block_size          = _CUDA_VSTD::min<decltype(block_size)>(block_size, max_block_size);

  return {block_size, items_per_thread, vector_load_length};
}

static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_value_t init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init.type;
}

template <typename Type>
std::string get_iterator_name()
{
  std::string iterator_t;
  check(nvrtcGetTypeName<Type>(&iterator_t));
  return iterator_t;
}

std::string get_input_iterator_name()
{
  return get_iterator_name<input_iterator_t>();
}

std::string get_output_iterator_name()
{
  return get_iterator_name<output_iterator_t>();
}

std::string get_start_offset_iterator_name()
{
  return get_iterator_name<start_offset_iterator_t>();
}

std::string get_end_offset_iterator_name()
{
  return get_iterator_name<end_offset_iterator_t>();
}

std::string get_device_segmented_reduce_kernel_name(
  cccl_op_t op,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  cccl_value_t init)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_segmented_reduce_policy>(&chained_policy_t));

  const std::string input_iterator_t =
    input_it.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(input_it.value_type.type, true) //
      : get_input_iterator_name();

  const std::string output_iterator_t =
    output_it.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(output_it.value_type.type, true) //
      : get_output_iterator_name();

  const std::string start_offset_iterator_t =
    start_offset_it.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(start_offset_it.value_type.type, true) //
      : get_start_offset_iterator_name();

  const std::string end_offset_iterator_t =
    end_offset_it.type == cccl_iterator_kind_t::CCCL_POINTER //
      ? cccl_type_enum_to_name(end_offset_it.value_type.type, true) //
      : get_end_offset_iterator_name();

  const std::string accum_t = cccl_type_enum_to_name(get_accumulator_type(op, input_it, init).type);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string reduction_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&reduction_op_t));

  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  /*
  template <typename ChainedPolicyT,       // 0
            typename InputIteratorT,       // 1
            typename OutputIteratorT,      // 2
            typename BeginOffsetIteratorT, // 3
            typename EndOffsetIteratorT,   // 4
            typename OffsetT,              // 5
            typename ReductionOpT,         // 6
            typename InitT,                // 7
            typename AccumT>               // 8
   DeviceSegmentedReduceKernel(...);
  */
  return std::format(
    "cub::detail::reduce::DeviceSegmentedReduceKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}>",
    chained_policy_t, // 0
    input_iterator_t, // 1
    output_iterator_t, // 2
    start_offset_iterator_t, // 3
    end_offset_iterator_t, // 4
    offset_t, // 5
    reduction_op_t, // 6
    init_t, // 7
    accum_t); // 8
}

template <auto* GetPolicy>
struct dynamic_reduce_policy_t
{
  using MaxPolicy = dynamic_reduce_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<segmented_reduce_runtime_tuning_policy>(GetPolicy(device_ptx_version, accumulator_type));
  }

  cccl_type_info accumulator_type;
};

struct segmented_reduce_kernel_source
{
  cccl_device_segmented_reduce_build_result_t& build;

  std::size_t AccumSize() const
  {
    return build.accumulator_size;
  }
  CUkernel SegmentedReduceKernel() const
  {
    return build.segmented_reduce_kernel;
  }
};
} // namespace segmented_reduce

CUresult cccl_device_segmented_reduce_build(
  cccl_device_segmented_reduce_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "device_segmented_reduce";

    const int cc                       = cc_major * 10 + cc_minor;
    const cccl_type_info accum_t       = segmented_reduce::get_accumulator_type(op, input_it, init);
    const auto policy                  = segmented_reduce::get_policy(cc, accum_t);
    const auto accum_cpp               = cccl_type_enum_to_name(accum_t.type);
    const auto input_it_value_t        = cccl_type_enum_to_name(input_it.value_type.type);
    const auto start_offset_it_value_t = cccl_type_enum_to_name(start_offset_it.value_type.type);
    const auto end_offset_it_value_t   = cccl_type_enum_to_name(end_offset_it.value_type.type);
    // OffsetT is checked to match have 64-bit size
    const auto offset_t = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

    const std::string input_iterator_src =
      make_kernel_input_iterator(offset_t, "input_iterator_t", input_it_value_t, input_it);
    const std::string output_iterator_src =
      make_kernel_output_iterator(offset_t, "output_iterator_t", accum_cpp, output_it);

    const std::string start_offset_iterator_src =
      make_kernel_input_iterator(offset_t, "start_offset_iterator_t", start_offset_it_value_t, start_offset_it);
    const std::string end_offset_iterator_src =
      make_kernel_input_iterator(offset_t, "end_offset_iterator_t", end_offset_it_value_t, end_offset_it);

    const std::string op_src = make_kernel_user_binary_operator(accum_cpp, accum_cpp, accum_cpp, op);

    // agent_policy_t is to specify parameters like policy_hub does in dispatch_reduce.cuh
    constexpr std::string_view program_preamble_template = R"XXX(
#include <cub/block/block_reduce.cuh>
#include <cub/device/dispatch/kernels/segmented_reduce.cuh>
struct __align__({1}) storage_t {{
   char data[{0}];
}};
{4}
{5}
{8}
{9}
struct agent_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {2};
  static constexpr int BLOCK_THREADS = {3};
  static constexpr int VECTOR_LOAD_LENGTH = {7};
  static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM = cub::BLOCK_REDUCE_WARP_REDUCTIONS;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
}};
struct device_segmented_reduce_policy {{
  struct ActivePolicy {{
    using ReducePolicy = agent_policy_t;
    using SegmentedReducePolicy = agent_policy_t;
  }};
}};
{6}
)XXX";

    std::string src = std::format(
      program_preamble_template,
      input_it.value_type.size, // 0
      input_it.value_type.alignment, // 1
      policy.items_per_thread, // 2
      policy.block_size, // 3
      input_iterator_src, // 4
      output_iterator_src, // 5
      op_src, // 6
      policy.vector_load_length, // 7
      start_offset_iterator_src, // 8
      end_offset_iterator_src // 9
    );

    std::string segmented_reduce_kernel_name = segmented_reduce::get_device_segmented_reduce_kernel_name(
      op, input_it, output_it, start_offset_it, end_offset_it, init);
    std::string segmented_reduce_kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    constexpr size_t num_args  = 8;
    const char* args[num_args] = {
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    nvrtc_ltoir_list_appender appender{ltoir_list};

    // add definition of binary operation op
    appender.append({op.ltoir, op.ltoir_size});
    // add iterator definitions
    appender.add_iterator_definition(input_it);
    appender.add_iterator_definition(output_it);
    appender.add_iterator_definition(start_offset_it);
    appender.add_iterator_definition(end_offset_it);

    nvrtc_link_result result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{src.c_str(), name})
        .add_expression({segmented_reduce_kernel_name})
        .compile_program({args, num_args})
        .get_name({segmented_reduce_kernel_name, segmented_reduce_kernel_lowered_name})
        .cleanup_program()
        .add_link_list(ltoir_list)
        .finalize_program(num_lto_args, lopts);

    // populate build struct members
    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(
      &build_ptr->segmented_reduce_kernel, build_ptr->library, segmented_reduce_kernel_lowered_name.c_str()));

    build_ptr->cc               = cc;
    build_ptr->cubin            = (void*) result.data.release();
    build_ptr->cubin_size       = result.size;
    build_ptr->accumulator_size = accum_t.size;
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_segmented_reduce_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

CUresult cccl_device_segmented_reduce(
  cccl_device_segmented_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_segments,
  cccl_iterator_t start_offset,
  cccl_iterator_t end_offset,
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

    auto exec_status = cub::DispatchSegmentedReduce<
      indirect_arg_t, // InputIteratorT
      indirect_arg_t, // OutputIteratorT
      indirect_arg_t, // BeginSegmentIteratorT
      indirect_arg_t, // EndSegmentIteratorT
      OffsetT, // OffsetT
      indirect_arg_t, // ReductionOpT
      indirect_arg_t, // InitT
      void, // AccumT
      segmented_reduce::dynamic_reduce_policy_t<&segmented_reduce::get_policy>, // PolicHub
      segmented_reduce::segmented_reduce_kernel_source, // KernelSource
      cub::detail::CudaDriverLauncherFactory>:: // KernelLaunchFactory
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        start_offset,
        end_offset,
        op,
        init,
        stream,
        /* kernel_source */ {build},
        /* launcher_factory &*/ cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        /* policy */ {segmented_reduce::get_accumulator_type(op, d_in, init)});

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_reduce(): %s\n", exc.what());
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

CUresult cccl_device_segmented_reduce_cleanup(cccl_device_segmented_reduce_build_result_t* build_ptr)
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    // allocation behind cubin is owned by unique_ptr with delete[] deleter now
    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
    check(cuLibraryUnload(build_ptr->library));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_reduce_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
};
