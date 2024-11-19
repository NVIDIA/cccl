//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/launcher/cuda_driver.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/variant>

#include <format>
#include <iostream>
#include <memory>

#include "kernels/iterators.h"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/reduce.h>
#include <nvrtc/command_list.h>

struct op_wrapper;
struct device_reduce_policy;
using TransformOpT = ::cuda::std::__identity;
using OffsetT      = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

struct nothing_t
{};

struct reduce_runtime_tuning_policy
{
  int block_size;
  int items_per_thread;
  int vector_load_length;

  reduce_runtime_tuning_policy SingleTile() const
  {
    return *this;
  }
  reduce_runtime_tuning_policy Reduce() const
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

struct reduce_tuning_t
{
  int cc;
  int block_size;
  int items_per_thread;
  int vector_load_length;
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

reduce_runtime_tuning_policy get_policy(int cc, cccl_type_info accumulator_type)
{
  reduce_tuning_t chain[] = {{60, 256, 16, 4}, {35, 256, 20, 4}};

  auto [_, block_size, items_per_thread, vector_load_length] = find_tuning(cc, chain);

  // Implement part of MemBoundScaling
  items_per_thread = CUB_MAX(1, CUB_MIN(items_per_thread * 4 / accumulator_type.size, items_per_thread * 2));
  block_size       = CUB_MIN(block_size, (((1024 * 48) / (accumulator_type.size * items_per_thread)) + 31) / 32 * 32);

  return {block_size, items_per_thread, vector_load_length};
}

static cccl_type_info get_accumulator_type(cccl_op_t /*op*/, cccl_iterator_t /*input_it*/, cccl_value_t init)
{
  // TODO Should be decltype(op(init, *input_it)) but haven't implemented type arithmetic yet
  //      so switching back to the old accumulator type logic for now
  return init.type;
}

struct input_iterator_state_t;
struct output_iterator_t;

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

std::string get_single_tile_kernel_name(
  cccl_iterator_t input_it, cccl_iterator_t output_it, cccl_op_t op, cccl_value_t init, bool is_second_kernel)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_reduce_policy>(&chained_policy_t));

  const cccl_type_info accum_t  = get_accumulator_type(op, input_it, init);
  const std::string accum_cpp_t = cccl_type_enum_to_name(accum_t.type);
  const std::string input_iterator_t =
    is_second_kernel ? cccl_type_enum_to_name(accum_t.type, true)
    : input_it.type == cccl_iterator_kind_t::pointer //
      ? cccl_type_enum_to_name(input_it.value_type.type, true) //
      : get_input_iterator_name();
  const std::string output_iterator_t =
    output_it.type == cccl_iterator_kind_t::pointer //
      ? cccl_type_enum_to_name(output_it.value_type.type, true) //
      : get_output_iterator_name();
  const std::string init_t = cccl_type_enum_to_name(init.type.type);

  std::string offset_t;
  if (is_second_kernel)
  {
    // Second kernel is always invoked with an int offset.
    // See the definition of the local variable `reduce_grid_size`
    // in DispatchReduce::InvokePasses.
    check(nvrtcGetTypeName<int>(&offset_t));
  }
  else
  {
    check(nvrtcGetTypeName<OffsetT>(&offset_t));
  }

  std::string reduction_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&reduction_op_t));

  return std::format(
    "cub::DeviceReduceSingleTileKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}>",
    chained_policy_t,
    input_iterator_t,
    output_iterator_t,
    offset_t,
    reduction_op_t,
    init_t,
    accum_cpp_t);
}

std::string get_device_reduce_kernel_name(cccl_op_t op, cccl_iterator_t input_it, cccl_value_t init)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_reduce_policy>(&chained_policy_t));

  const std::string input_iterator_t =
    input_it.type == cccl_iterator_kind_t::pointer //
      ? cccl_type_enum_to_name(input_it.value_type.type, true) //
      : get_input_iterator_name();

  const std::string accum_t = cccl_type_enum_to_name(get_accumulator_type(op, input_it, init).type);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string reduction_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&reduction_op_t));

  std::string transform_op_t;
  check(nvrtcGetTypeName<cuda::std::__identity>(&transform_op_t));

  return std::format(
    "cub::DeviceReduceKernel<{0}, {1}, {2}, {3}, {4}, {5}>",
    chained_policy_t,
    input_iterator_t,
    offset_t,
    reduction_op_t,
    accum_t,
    transform_op_t);
}

extern "C" CCCL_C_API CUresult cccl_device_reduce_build(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path) noexcept
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "test";

    const int cc                              = cc_major * 10 + cc_minor;
    const cccl_type_info accum_t              = get_accumulator_type(op, input_it, init);
    const reduce_runtime_tuning_policy policy = get_policy(cc, accum_t);
    const auto accum_cpp                      = cccl_type_enum_to_string(accum_t.type);
    const auto input_it_value_t               = cccl_type_enum_to_string(input_it.value_type.type);
    const auto offset_t                       = cccl_type_enum_to_string(cccl_type_enum::UINT64);

    const std::string input_iterator_src  = make_kernel_input_iterator(offset_t, input_it_value_t, input_it);
    const std::string output_iterator_src = make_kernel_output_iterator(offset_t, accum_cpp, output_it);

    const std::string op_src = make_kernel_user_binary_operator(accum_cpp, op);

    const std::string src = std::format(
      "#include <cub/block/block_reduce.cuh>\n"
      "#include <cub/device/dispatch/kernels/reduce.cuh>\n"
      "struct __align__({1}) storage_t {{\n"
      "  char data[{0}];\n"
      "}};\n"
      "{4}\n"
      "{5}\n"
      "struct agent_policy_t {{\n"
      "  static constexpr int ITEMS_PER_THREAD = {2};\n"
      "  static constexpr int BLOCK_THREADS = {3};\n"
      "  static constexpr int VECTOR_LOAD_LENGTH = {7};\n"
      "  static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM = cub::BLOCK_REDUCE_WARP_REDUCTIONS;\n"
      "  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;\n"
      "}};\n"
      "struct device_reduce_policy {{\n"
      "  struct ActivePolicy {{\n"
      "    using ReducePolicy = agent_policy_t;\n"
      "    using SingleTilePolicy = agent_policy_t;\n"
      "  }};\n"
      "}};\n"
      "{6};\n",
      input_it.value_type.size, // 0
      input_it.value_type.alignment, // 1
      policy.items_per_thread, // 2
      policy.block_size, // 3
      input_iterator_src, // 4
      output_iterator_src, // 5
      op_src, // 6
      policy.vector_load_length); // 7

    std::string single_tile_kernel_name        = get_single_tile_kernel_name(input_it, output_it, op, init, false);
    std::string single_tile_second_kernel_name = get_single_tile_kernel_name(input_it, output_it, op, init, true);
    std::string reduction_kernel_name          = get_device_reduce_kernel_name(op, input_it, init);
    std::string single_tile_kernel_lowered_name;
    std::string single_tile_second_kernel_lowered_name;
    std::string reduction_kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    constexpr size_t num_args  = 7;
    const char* args[num_args] = {arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    auto ltoir_list_append = [&ltoir_list](nvrtc_ltoir lto) {
      if (lto.ltsz)
      {
        ltoir_list.push_back(std::move(lto));
      }
    };
    ltoir_list_append({op.ltoir, op.ltoir_size});
    if (cccl_iterator_kind_t::iterator == input_it.type)
    {
      ltoir_list_append({input_it.advance.ltoir, input_it.advance.ltoir_size});
      ltoir_list_append({input_it.dereference.ltoir, input_it.dereference.ltoir_size});
    }
    if (cccl_iterator_kind_t::iterator == output_it.type)
    {
      ltoir_list_append({output_it.advance.ltoir, output_it.advance.ltoir_size});
      ltoir_list_append({output_it.dereference.ltoir, output_it.dereference.ltoir_size});
    }

    nvrtc_cubin result =
      make_nvrtc_command_list()
        .add_program(nvrtc_translation_unit{src.c_str(), name})
        .add_expression({single_tile_kernel_name})
        .add_expression({single_tile_second_kernel_name})
        .add_expression({reduction_kernel_name})
        .compile_program({args, num_args})
        .get_name({single_tile_kernel_name, single_tile_kernel_lowered_name})
        .get_name({single_tile_second_kernel_name, single_tile_second_kernel_lowered_name})
        .get_name({reduction_kernel_name, reduction_kernel_lowered_name})
        .cleanup_program()
        .add_link_list(ltoir_list)
        .finalize_program(num_lto_args, lopts);

    cuLibraryLoadData(&build->library, result.cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build->single_tile_kernel, build->library, single_tile_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build->single_tile_second_kernel, build->library, single_tile_second_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build->reduction_kernel, build->library, reduction_kernel_lowered_name.c_str()));

    build->cc               = cc;
    build->cubin            = (void*) result.cubin.release();
    build->cubin_size       = result.size;
    build->accumulator_size = accum_t.size;
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

template <auto* GetPolicy>
struct dynamic_reduce_policy_t
{
  using MaxPolicy = dynamic_reduce_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<reduce_runtime_tuning_policy>(GetPolicy(device_ptx_version, accumulator_type));
  }

  cccl_type_info accumulator_type;
};

struct reduce_kernel_source
{
  cccl_device_reduce_build_result_t& build;

  std::size_t AccumSize() const
  {
    return build.accumulator_size;
  }
  CUkernel SingleTileKernel() const
  {
    return build.single_tile_kernel;
  }
  CUkernel SingleTileSecondKernel() const
  {
    return build.single_tile_second_kernel;
  }
  CUkernel ReductionKernel() const
  {
    return build.reduction_kernel;
  }
};

extern "C" CCCL_C_API CUresult cccl_device_reduce(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  unsigned long long num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream) noexcept
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    cub::DispatchReduce<indirect_arg_t,
                        indirect_arg_t,
                        ::cuda::std::size_t,
                        indirect_arg_t,
                        indirect_arg_t,
                        void,
                        dynamic_reduce_policy_t<&get_policy>,
                        ::cuda::std::__identity,
                        reduce_kernel_source,
                        cub::CudaDriverLauncherFactory>::
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        op,
        init,
        stream,
        {},
        {build},
        cub::CudaDriverLauncherFactory{cu_device, build.cc},
        {get_accumulator_type(op, d_in, init)});
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

extern "C" CCCL_C_API CUresult cccl_device_reduce_cleanup(cccl_device_reduce_build_result_t* bld_ptr)
{
  try
  {
    if (bld_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(bld_ptr->cubin));
    check(cuLibraryUnload(bld_ptr->library));
  }
  catch (...)
  {
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
