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

#include "util/errors.h"
#include "util/small_storage.h"
#include <cccl/c/for.h>
#include <cccl/c/types.h>
#include <nvJitLink.h>
#include <nvrtc.h>

struct op_wrapper;
struct device_reduce_policy;

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

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

struct for_each_kernel_params
{
  void* data;
};

inline cudaError_t
Invoke(cccl_iterator_t d_in, size_t num_items, cccl_op_t op, int cc, CUfunction static_kernel, CUstream stream)
{
  cudaError error = cudaSuccess;

  if (num_items == 0)
  {
    return error;
  }

  small_aligned_storage<for_each_kernel_params> op_params =
    (op.type == cccl_op_kind_t::stateful)
      ? small_aligned_storage<for_each_kernel_params>({d_in.state}, op.state, op.alignment, op.size)
      : small_aligned_storage<for_each_kernel_params>({d_in.state});

  void* args[] = {&num_items, op_params.get()};

  int thread_count = 256;
  int block_count  = (num_items + 511) / 512;
  check(cuLaunchKernel(static_kernel, block_count, 1, 1, thread_count, 1, 1, 0, stream, args, 0));

  // Check for failure to launch
  error = CubDebug(cudaPeekAtLastError());

  return error;
}

inline std::string get_device_for_kernel_name(std::string diff_type)
{
  return std::format("cub::detail::for_each::static_kernel<device_for_policy, {0}, op_iter_wrapper>", diff_type);
}

inline std::string
format_device_for_kernel(cccl_op_t op, cccl_iterator_t d_data, std::string diff_type, std::string for_kernel)
{
  // TODO: Maybe break out the iter wrapper
  return std::format(
    R"XXX(
#include <cuda/std/iterator>
#include <cub/agent/agent_for.cuh>
#include <cub/device/dispatch/kernels/for_each.cuh>

#if {5} // enable stateful op dispatch
extern "C" __device__ void {4}(void* state, void* data);
struct __align__({7}) storage_t {{
  char state[{6}];
}};
#else
extern "C" __device__ void {4}(void* data);
struct storage_t {{}};
#endif

struct op_iter_wrapper
{{
  using iterator_category        = cuda::std::random_access_iterator_tag;
  using value_type               = {0};
  using difference_type          = {2};
  using pointer                  = value_type*;
  using reference                = value_type&;
  static constexpr difference_type size = {3};
  value_type* data;
  storage_t state;

  __device__ void operator()(difference_type idx)
  {{
#if {5} // enable stateful op dispatch
    {4}(&state, data + (idx*size));
#else
    {4}(data + (idx*size));
#endif
  }}
}};

using policy_dim_t = cub::detail::for_each::policy_t<256, 2>;

struct device_for_policy
{{
  struct ActivePolicy
  {{
    using for_policy_t = policy_dim_t;
  }};
}};

// Instantiate device kernel
template
__global__ void {1}({2} num_items, op_iter_wrapper op);
)XXX",
    "uint8_t", // 0 - value type
    for_kernel, // 1 - Kernel name
    diff_type, // 2 - difference type
    d_data.value_type.size, // 3 - value_type size
    op.name, // 4 - Operator name
    op.type == cccl_op_kind_t::stateful ? 1 : 0, // 5 - Enable use of state
    op.size, // 6 - Operator state size
    op.alignment // 7 - Operator alignment
  );
}

extern "C" CCCL_C_API CUresult cccl_device_for_build(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
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
    nvrtcProgram prog{};
    const char* name = "test";

    const int cc                     = cc_major * 10 + cc_minor;
    const std::string d_data_value_t = cccl_type_enum_to_string(d_data.value_type.type);
    const std::string offset_t       = cccl_type_enum_to_string(cccl_type_enum::UINT64);

    const std::string for_kernel_name   = get_device_for_kernel_name("size_t");
    const std::string device_for_kernel = format_device_for_kernel(op, d_data, "size_t", for_kernel_name);

    check(nvrtcCreateProgram(&prog, device_for_kernel.c_str(), name, 0, nullptr, nullptr));

    check(nvrtcAddNameExpression(prog, for_kernel_name.c_str()));

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    constexpr int num_args = 7;
    const char* args[] = {arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-G"};

    std::size_t log_size{};
    nvrtcResult compile_result = nvrtcCompileProgram(prog, num_args, args);

    check(nvrtcGetProgramLogSize(prog, &log_size));

    std::unique_ptr<char[]> log{new char[log_size]};
    check(nvrtcGetProgramLog(prog, log.get()));

    if (log_size > 1)
    {
      std::cerr << log.get() << std::endl;
    }

    std::string for_kernel_lowered_name;
    {
      const char* for_kernel_lowered_name_temp;
      check(nvrtcGetLoweredName(prog, for_kernel_name.c_str(), &for_kernel_lowered_name_temp));
      for_kernel_lowered_name = for_kernel_lowered_name_temp;
    }

    check(compile_result);

    std::size_t ltoir_size{};
    check(nvrtcGetLTOIRSize(prog, &ltoir_size));
    std::unique_ptr<char[]> ltoir{new char[ltoir_size]};
    check(nvrtcGetLTOIR(prog, ltoir.get()));
    check(nvrtcDestroyProgram(&prog));

    nvJitLinkHandle handle;
    const char* lopts[] = {"-lto", arch.c_str()};
    check(nvJitLinkCreate(&handle, 2, lopts));

    check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoir.get(), ltoir_size, name));
    check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, op.ltoir, op.ltoir_size, name));

    check(nvJitLinkComplete(handle));

    std::size_t cubin_size{};
    check(nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
    std::unique_ptr<char[]> cubin{new char[cubin_size]};
    check(nvJitLinkGetLinkedCubin(handle, cubin.get()));
    check(nvJitLinkDestroy(&handle));

    cuLibraryLoadData(&build->library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build->static_kernel, build->library, for_kernel_lowered_name.c_str()));

    build->cc         = cc;
    build->cubin      = cubin.release();
    build->cubin_size = cubin_size;
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

extern "C" CCCL_C_API CUresult cccl_device_for(
  cccl_device_for_build_result_t build,
  cccl_iterator_t d_data,
  int64_t num_items,
  cccl_op_t op,
  CUstream stream) noexcept
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    Invoke(d_data, num_items, op, build.cc, (CUfunction) build.static_kernel, stream);
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

extern "C" CCCL_C_API CUresult cccl_device_for_cleanup(cccl_device_for_build_result_t* bld_ptr)
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
