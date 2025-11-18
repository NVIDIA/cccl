/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#  include <thrust/system/cuda/detail/core/util.h>

#  include <cuda/std/cassert>

#  include <nv/target>

/**
 * @def THRUST_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION
 * If defined, the default suppression of kernel visibility attribute warning is disabled.
 */
#  if !defined(THRUST_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_CLANG("-Wattributes")
#    if !_CCCL_CUDA_COMPILER(NVHPC)
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)
#    endif // !_CCCL_CUDA_COMPILER(NVHPC)
#  endif // !THRUST_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION

THRUST_NAMESPACE_BEGIN

namespace cuda_cub::core::detail
{
#  ifndef THRUST_DETAIL_KERNEL_ATTRIBUTES
#    define THRUST_DETAIL_KERNEL_ATTRIBUTES CCCL_DETAIL_KERNEL_ATTRIBUTES
#  endif

#  if _CCCL_DEVICE_COMPILATION()
template <class Agent, class... Args>
THRUST_DETAIL_KERNEL_ATTRIBUTES void __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS) _kernel_agent(Args... args)
{
  extern __shared__ char shmem[];
  Agent::entry(args..., shmem);
}

template <class Agent, class... Args>
THRUST_DETAIL_KERNEL_ATTRIBUTES void __launch_bounds__(Agent::ptx_plan::BLOCK_THREADS)
  _kernel_agent_vshmem(char* vshmem, Args... args)
{
  extern __shared__ char shmem[];
  vshmem = vshmem == nullptr ? shmem : vshmem + blockIdx.x * temp_storage_size<typename Agent::ptx_plan>::value;
  Agent::entry(args..., vshmem);
}

#  else // ^^^ _CCCL_DEVICE_COMPILATION() ^^^ / vvv !_CCCL_DEVICE_COMPILATION() vvv
template <class, class... Args>
THRUST_DETAIL_KERNEL_ATTRIBUTES void _kernel_agent(Args... args)
{}

template <class, class... Args>
THRUST_DETAIL_KERNEL_ATTRIBUTES void _kernel_agent_vshmem(char*, Args... args)
{}
#  endif // ^^^ !_CCCL_DEVICE_COMPILATION() ^^^

template <class Agent>
struct AgentLauncher : Agent
{
  AgentPlan plan;
  size_t count;
  cudaStream_t stream;
  char const* name;
  unsigned int grid;
  char* vshmem;
  bool has_shmem;
  size_t shmem_size;

  static constexpr int MAX_SHMEM_PER_BLOCK = 48 * 1024;

  using has_enough_shmem_t = typename has_enough_shmem<Agent, MAX_SHMEM_PER_BLOCK>::type;
  using shm1               = has_enough_shmem<Agent, MAX_SHMEM_PER_BLOCK>;

  template <class Size>
  THRUST_RUNTIME_FUNCTION AgentLauncher(AgentPlan plan_, Size count_, cudaStream_t stream_, char const* name_)
      : plan(plan_)
      , count((size_t) count_)
      , stream(stream_)
      , name(name_)
      , grid(static_cast<unsigned int>((count + plan.items_per_tile - 1) / plan.items_per_tile))
      , vshmem(nullptr)
      , has_shmem((size_t) get_max_shared_memory_per_block() >= (size_t) plan.shared_memory_size)
      , shmem_size(has_shmem ? plan.shared_memory_size : 0)
  {
    assert(count > 0);
  }

  template <class Size>
  THRUST_RUNTIME_FUNCTION
  AgentLauncher(AgentPlan plan_, Size count_, cudaStream_t stream_, char* vshmem, char const* name_)
      : plan(plan_)
      , count((size_t) count_)
      , stream(stream_)
      , name(name_)
      , grid(static_cast<unsigned int>((count + plan.items_per_tile - 1) / plan.items_per_tile))
      , vshmem(vshmem)
      , has_shmem((size_t) get_max_shared_memory_per_block() >= (size_t) plan.shared_memory_size)
      , shmem_size(has_shmem ? plan.shared_memory_size : 0)
  {
    assert(count > 0);
  }

  THRUST_RUNTIME_FUNCTION AgentLauncher(AgentPlan plan_, cudaStream_t stream_, char const* name_)
      : plan(plan_)
      , count(0)
      , stream(stream_)
      , name(name_)
      , grid(plan.grid_size)
      , vshmem(nullptr)
      , has_shmem((size_t) get_max_shared_memory_per_block() >= (size_t) plan.shared_memory_size)
      , shmem_size(has_shmem ? plan.shared_memory_size : 0)
  {
    assert(plan.grid_size > 0);
  }

  THRUST_RUNTIME_FUNCTION AgentLauncher(AgentPlan plan_, cudaStream_t stream_, char* vshmem, char const* name_)
      : plan(plan_)
      , count(0)
      , stream(stream_)
      , name(name_)
      , grid(plan.grid_size)
      , vshmem(vshmem)
      , has_shmem((size_t) get_max_shared_memory_per_block() >= (size_t) plan.shared_memory_size)
      , shmem_size(has_shmem ? plan.shared_memory_size : 0)
  {
    assert(plan.grid_size > 0);
  }

  THRUST_RUNTIME_FUNCTION typename get_plan<Agent>::type static get_plan(cudaStream_t, void* /* d_ptr */ = 0)
  {
    return get_agent_plan<Agent>(get_ptx_version());
  }

  THRUST_RUNTIME_FUNCTION typename detail::get_plan<Agent>::type static get_plan()
  {
    return get_agent_plan<Agent>(lowest_supported_sm_arch::ver);
  }

  THRUST_RUNTIME_FUNCTION void sync() const
  {
    CubDebug(cub::detail::DebugSyncStream(stream));
  }

  template <class K>
  static cuda_optional<int> THRUST_RUNTIME_FUNCTION max_blocks_per_sm_impl(K k, int block_threads)
  {
    int occ;
    cudaError_t status = cub::MaxSmOccupancy(occ, k, block_threads);
    return cuda_optional<int>(status == cudaSuccess ? occ : -1, status);
  }

  template <class K>
  cuda_optional<int> THRUST_RUNTIME_FUNCTION max_sm_occupancy(K k) const
  {
    return max_blocks_per_sm_impl(k, plan.block_threads);
  }

  template <class K>
  THRUST_RUNTIME_FUNCTION void print_info([[maybe_unused]] K k) const
  {
#  if THRUST_DEBUG_SYNC_FLAG
    cuda_optional<int> occ = max_sm_occupancy(k);
    const int ptx_version  = get_ptx_version();
    if (count > 0)
    {
      _CubLog(
        "Invoking %s<<<%u, %d, %d, %lld>>>(), %llu items total, %d items per thread, %d SM occupancy, %d vshmem size, "
        "%d ptx_version \n",
        name,
        grid,
        plan.block_threads,
        (has_shmem ? (int) plan.shared_memory_size : 0),
        (long long) stream,
        (long long) count,
        plan.items_per_thread,
        (int) occ,
        (!has_shmem ? (int) plan.shared_memory_size : 0),
        (int) ptx_version);
    }
    else
    {
      _CubLog(
        "Invoking %s<<<%u, %d, %d, %lld>>>(), %d items per thread, %d SM occupancy, %d vshmem size, %d ptx_version\n",
        name,
        grid,
        plan.block_threads,
        (has_shmem ? (int) plan.shared_memory_size : 0),
        (long long) stream,
        plan.items_per_thread,
        (int) occ,
        (!has_shmem ? (int) plan.shared_memory_size : 0),
        (int) ptx_version);
    }
#  endif
  }

  template <class... Args>
  static cuda_optional<int> THRUST_RUNTIME_FUNCTION get_max_blocks_per_sm(AgentPlan plan)
  {
    return max_blocks_per_sm_impl(_kernel_agent<Agent, Args...>, plan.block_threads);
  }

  // If we are guaranteed to have enough shared memory
  // don't compile other kernel which accepts pointer
  // and save on compilations
  template <class... Args>
  void THRUST_RUNTIME_FUNCTION launch_impl(thrust::detail::true_type, Args... args) const
  {
    assert(has_shmem && vshmem == nullptr);
    print_info(_kernel_agent<Agent, Args...>);
    cuda_cub::detail::triple_chevron(grid, plan.block_threads, shmem_size, stream)
      .doit(_kernel_agent<Agent, Args...>, args...);
  }

  // If there is a risk of not having enough shared memory
  // we compile generic kernel instead.
  // This kernel is likely to be somewhat slower, but it can accommodate
  // both shared and virtualized shared memories.
  // Alternative option is to compile two kernels, one using shared and one
  // using virtualized shared memory. While this can be slightly faster if we
  // do actually have enough shared memory, the compilation time will double.
  //
  template <class... Args>
  void THRUST_RUNTIME_FUNCTION launch_impl(thrust::detail::false_type, Args... args) const
  {
    assert((has_shmem && vshmem == nullptr) || (!has_shmem && vshmem != nullptr && shmem_size == 0));
    print_info(_kernel_agent_vshmem<Agent, Args...>);
    cuda_cub::detail::triple_chevron(grid, plan.block_threads, shmem_size, stream)
      .doit(_kernel_agent_vshmem<Agent, Args...>, vshmem, args...);
  }

  template <class... Args>
  void THRUST_RUNTIME_FUNCTION launch(Args... args) const
  {
    launch_impl(has_enough_shmem_t(), args...);
    sync();
  }
};
} // namespace cuda_cub::core::detail

THRUST_NAMESPACE_END

#endif // _CCCL_CUDA_COMPILATION()
