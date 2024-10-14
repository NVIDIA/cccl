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
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <format>
#include <iostream>
#include <memory>

#include "util/context.h"
#include "util/errors.h"
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

struct runtime_tuning_policy
{
  int block_size;
  int items_per_thread;
  int vector_load_length;
};

struct reduce_tuning_t
{
  int cc;
  int block_size;
  int items_per_thread;
  int vector_load_length;
};

template <int N>
static reduce_tuning_t find_tuning(int cc, const reduce_tuning_t (&tunings)[N])
{
  for (const reduce_tuning_t& tuning : tunings)
  {
    if (cc >= tuning.cc)
    {
      return tuning;
    }
  }

  return tunings[N - 1];
}

static runtime_tuning_policy get_policy(int cc, cccl_type_info accumulator_type, cccl_type_info /*input_type*/)
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

static cudaError_t InvokeSingleTile(
  void* d_temp_storage,
  std::size_t& temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  unsigned long long num_items,
  cccl_op_t op,
  cccl_value_t init,
  int cc,
  CUkernel single_tile_kernel,
  CUstream stream)
{
  const runtime_tuning_policy policy = get_policy(cc, d_in.value_type, d_in.value_type);

  cudaError error = cudaSuccess;
  do
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      break;
    }

    nothing_t nothing{};
    TransformOpT transform_op{};
    void* op_state = op.type == cccl_op_kind_t::stateless ? &nothing : op.state;
    void* in_ptr   = d_in.type == cccl_iterator_kind_t::pointer ? &d_in.state : d_in.state;
    void* out_ptr  = d_out.type == cccl_iterator_kind_t::pointer ? &d_out.state : d_out.state;
    void* args[]   = {in_ptr, out_ptr, &num_items, op_state, init.state, &transform_op};

    check(cuLaunchKernel((CUfunction) single_tile_kernel, 1, 1, 1, policy.block_size, 1, 1, 0, stream, args, 0));

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      break;
    }
  } while (0);

  return error;
}

static cudaError_t InvokePasses(
  void* d_temp_storage,
  std::size_t& temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  unsigned long long num_items,
  cccl_op_t op,
  cccl_value_t init,
  int cc,
  CUkernel reduce_kernel,
  CUkernel single_tile_kernel,
  CUdevice device,
  CUstream stream)
{
  const cccl_type_info accum_t       = get_accumulator_type(op, d_in, init);
  const runtime_tuning_policy policy = get_policy(cc, accum_t, d_in.value_type);

  cudaError error = cudaSuccess;
  do
  {
    void* in_ptr  = d_in.type == cccl_iterator_kind_t::pointer ? &d_in.state : d_in.state;
    void* out_ptr = d_out.type == cccl_iterator_kind_t::pointer ? &d_out.state : d_out.state;

    // Get SM count
    int sm_count;
    check(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

    // Init regular kernel configuration
    const auto tile_size = policy.block_size * policy.items_per_thread;

    // Older drivers have issues handling CUkernel in the occupancy queries, get the CUfunction instead.
    // Assumes that the current device is properly set, it needs to be set for the occupancy queries anyway
    CUfunction reduce_kernel_fn;
    check(cuKernelGetFunction(&reduce_kernel_fn, reduce_kernel));

    int sm_occupancy = 1;
    check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&sm_occupancy, reduce_kernel_fn, policy.block_size, 0));

    int reduce_device_occupancy = sm_occupancy * sm_count;

    // Even-share work distribution
    int max_blocks = reduce_device_occupancy * CUB_SUBSCRIPTION_FACTOR(0);
    cub::GridEvenShare<OffsetT> even_share;
    even_share.DispatchInit(num_items, max_blocks, tile_size);

    // Temporary storage allocation requirements
    void* allocations[1]       = {};
    size_t allocation_sizes[1] = {
      max_blocks * static_cast<std::size_t>(d_in.value_type.size) // bytes needed for privatized block reductions
    };

    // Alias the temporary allocations from the single storage blob (or
    // compute the necessary size of the blob)
    error = CubDebug(cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
    if (cudaSuccess != error)
    {
      break;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      return cudaSuccess;
    }

    // Get grid size for device_reduce_sweep_kernel
    OffsetT reduce_grid_size = even_share.grid_size;

    // Invoke DeviceReduceKernel
    // reduce_kernel<<<reduce_grid_size, ActivePolicyT::ReducePolicy::BLOCK_THREADS>>>(
    //    d_in, d_block_reductions, num_items, even_share, ReductionOpT{}, TransformOpT{});

    nothing_t nothing{};
    void* op_state = op.type == cccl_op_kind_t::stateless ? &nothing : op.state;

    TransformOpT transform_op{};
    void* reduce_args[] = {in_ptr, &allocations[0], &num_items, &even_share, op_state, &transform_op};

    check(cuLaunchKernel(
      (CUfunction) reduce_kernel, reduce_grid_size, 1, 1, policy.block_size, 1, 1, 0, stream, reduce_args, 0));

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      break;
    }

    // single_tile_kernel<<<1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS>>>(
    //     d_block_reductions, d_out, reduce_grid_size, ReductionOpT{}, 0, TransformOpT{});

    void* single_tile_kernel_args[] = {&allocations[0], out_ptr, &reduce_grid_size, op_state, init.state, &transform_op};

    check(cuLaunchKernel(
      (CUfunction) single_tile_kernel, 1, 1, 1, policy.block_size, 1, 1, 0, stream, single_tile_kernel_args, 0));

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      break;
    }
  } while (0);

  return error;
}

static cudaError_t Invoke(
  void* d_temp_storage,
  std::size_t& temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  OffsetT num_items,
  cccl_op_t op,
  cccl_value_t init,
  int cc,
  CUkernel single_tile_kernel,
  CUkernel single_tile_second_kernel,
  CUkernel reduce_kernel,
  CUdevice device,
  CUstream stream)
{
  const cccl_type_info accum_t = get_accumulator_type(op, d_in, init);
  runtime_tuning_policy policy = get_policy(cc, accum_t, d_in.value_type);

  // Force kernel code-generation in all compiler passes
  if (num_items <= static_cast<OffsetT>(policy.block_size * policy.items_per_thread))
  {
    // Small, single tile size
    return InvokeSingleTile(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, cc, single_tile_kernel, stream);
  }
  else
  {
    // Multi-tile pass
    return InvokePasses(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      op,
      init,
      cc,
      reduce_kernel,
      single_tile_second_kernel,
      device,
      stream);
  }
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
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

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

    const int cc                       = cc_major * 10 + cc_minor;
    const cccl_type_info accum_t       = get_accumulator_type(op, input_it, init);
    const std::string accum_cpp        = cccl_type_enum_to_string(accum_t.type);
    const runtime_tuning_policy policy = get_policy(cc, accum_t, input_it.value_type);
    const std::string input_it_value_t = cccl_type_enum_to_string(input_it.value_type.type);
    const std::string offset_t         = cccl_type_enum_to_string(cccl_type_enum::UINT64);

    const std::string input_iterator_src =
      input_it.type == cccl_iterator_kind_t::pointer
        ? std::string{}
        : std::format(
            "extern \"C\" __device__ {3} {4}(const void *self_ptr);\n"
            "extern \"C\" __device__ void {5}(void *self_ptr, {0} offset);\n"
            "struct __align__({2}) input_iterator_state_t {{\n"
            "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
            "  using value_type = {3};\n"
            "  using difference_type = {0};\n"
            "  using pointer = {3}*;\n"
            "  using reference = {3}&;\n"
            "  __device__ value_type operator*() const {{ return {4}(this); }}\n"
            "  __device__ input_iterator_state_t& operator+=(difference_type diff) {{\n"
            "      {5}(this, diff);\n"
            "      return *this;\n"
            "  }}\n"
            "  __device__ value_type operator[](difference_type diff) const {{\n"
            "      return *(*this + diff);\n"
            "  }}\n"
            "  __device__ input_iterator_state_t operator+(difference_type diff) const {{\n"
            "      input_iterator_state_t result = *this;\n"
            "      result += diff;\n"
            "      return result;\n"
            "  }}\n"
            "  char data[{1}];\n"
            "}};\n",
            offset_t, // 0
            input_it.size, // 1
            input_it.alignment, // 2
            input_it_value_t, // 3
            input_it.dereference.name, // 4
            input_it.advance.name); // 5

    const std::string output_iterator_src =
      output_it.type == cccl_iterator_kind_t::pointer
        ? std::string{}
        : std::format(
            "extern \"C\" __device__ void {2}(const void *self_ptr, {1} x);\n"
            "extern \"C\" __device__ void {3}(void *self_ptr, {0} offset);\n"
            "struct __align__({5}) output_iterator_state_t{{\n"
            "  char data[{4}];\n"
            "}};\n"
            "struct output_iterator_proxy_t {{\n"
            "  __device__ output_iterator_proxy_t operator=({1} x) {{\n"
            "    {2}(&state, x);\n"
            "    return *this;\n"
            "  }}\n"
            "  output_iterator_state_t state;\n"
            "}};\n"
            "struct output_iterator_t {{\n"
            "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
            "  using difference_type   = {0};\n"
            "  using value_type        = void;\n"
            "  using pointer           = output_iterator_proxy_t*;\n"
            "  using reference         = output_iterator_proxy_t;\n"
            "  __device__ output_iterator_proxy_t operator*() const {{ return {{state}}; }}\n"
            "  __device__ output_iterator_t& operator+=(difference_type diff) {{\n"
            "      {3}(&state, diff);\n"
            "      return *this;\n"
            "  }}\n"
            "  __device__ output_iterator_proxy_t operator[](difference_type diff) const {{\n"
            "    output_iterator_t result = *this;\n"
            "    result += diff;\n"
            "    return {{ result.state }};\n"
            "  }}\n"
            "  __device__ output_iterator_t operator+(difference_type diff) const {{\n"
            "    output_iterator_t result = *this;\n"
            "    result += diff;\n"
            "    return result;\n"
            "  }}\n"
            "  output_iterator_state_t state;\n"
            "}};",
            offset_t, // 0
            accum_cpp, // 1
            output_it.dereference.name, // 2
            output_it.advance.name, // 3
            output_it.size, // 4
            output_it.alignment); // 5

    const std::string op_src =
      op.type == cccl_op_kind_t::stateless
        ? std::format(
            "extern \"C\" __device__ {0} {1}({0} lhs, {0} rhs);\n"
            "struct op_wrapper {{\n"
            "  __device__ {0} operator()({0} lhs, {0} rhs) const {{\n"
            "    return {1}(lhs, rhs);\n"
            "  }}\n"
            "}};\n",
            accum_cpp,
            op.name)
        : std::format(
            "struct __align__({2}) op_state {{\n"
            "  char data[{3}];\n"
            "}};"
            "extern \"C\" __device__ {0} {1}(op_state *state, {0} lhs, {0} rhs);\n"
            "struct op_wrapper {{\n"
            "  op_state state;\n"
            "  __device__ {0} operator()({0} lhs, {0} rhs) {{\n"
            "    return {1}(&state, lhs, rhs);\n"
            "  }}\n"
            "}};\n",
            accum_cpp,
            op.name,
            op.alignment,
            op.size);

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

    auto cl =
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
        .add_link({op.ltoir, op.ltoir_size});

    nvrtc_cubin result{};

    if (cccl_iterator_kind_t::iterator == input_it.type && cccl_iterator_kind_t::iterator == output_it.type)
    {
      result = cl.add_link({input_it.advance.ltoir, input_it.advance.ltoir_size})
                 .add_link({input_it.dereference.ltoir, input_it.dereference.ltoir_size})
                 .add_link({output_it.advance.ltoir, output_it.advance.ltoir_size})
                 .add_link({output_it.dereference.ltoir, output_it.dereference.ltoir_size})
                 .finalize_program(num_lto_args, lopts);
    }
    else if (cccl_iterator_kind_t::iterator == input_it.type)
    {
      result = cl.add_link({input_it.advance.ltoir, input_it.advance.ltoir_size})
                 .add_link({input_it.dereference.ltoir, input_it.dereference.ltoir_size})
                 .finalize_program(num_lto_args, lopts);
    }
    else if (cccl_iterator_kind_t::iterator == output_it.type)
    {
      result = cl.add_link({output_it.advance.ltoir, output_it.advance.ltoir_size})
                 .add_link({output_it.dereference.ltoir, output_it.dereference.ltoir_size})
                 .finalize_program(num_lto_args, lopts);
    }
    else
    {
      result = cl.finalize_program(num_lto_args, lopts);
    }

    cuLibraryLoadData(&build->library, result.cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build->single_tile_kernel, build->library, single_tile_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build->single_tile_second_kernel, build->library, single_tile_second_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build->reduction_kernel, build->library, reduction_kernel_lowered_name.c_str()));

    build->cc         = cc;
    build->cubin      = (void*) result.cubin.release();
    build->cubin_size = result.size;
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

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

    Invoke(
      d_temp_storage,
      *temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      op,
      init,
      build.cc,
      build.single_tile_kernel,
      build.single_tile_second_kernel,
      build.reduction_kernel,
      cu_device,
      stream);
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
