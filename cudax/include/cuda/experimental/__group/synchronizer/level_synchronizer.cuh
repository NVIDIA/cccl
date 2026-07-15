//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_LEVEL_SYNCHRONIZER_CUH
#define _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_LEVEL_SYNCHRONIZER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__limits/numeric_limits.h>

#include <cuda/experimental/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
class level_synchronizer
{
public:
  template <class _Level>
  struct __synchronizer_instance;

  _CCCL_HIDE_FROM_ABI explicit level_synchronizer() = default;

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  make_instance(const _Unit&, const _ParentGroup&, const _MappingResult&) const noexcept
  {
    return __synchronizer_instance<typename _ParentGroup::level_type>{};
  }
};

template <bool _Aligned>
_CCCL_DEVICE_API void __block_sync() noexcept
{
  if constexpr (_Aligned)
  {
    ::__syncthreads();
  }
  else
  {
    ::__barrier_sync(0);
  }
}

template <>
struct level_synchronizer::__synchronizer_instance<thread_level>
{
  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _MappingResult&, const level_synchronizer&, const _Hierarchy&) const noexcept
  {}

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void
  do_sync_aligned(const _MappingResult&, const level_synchronizer&, const _Hierarchy&) const noexcept
  {}
};

template <>
struct level_synchronizer::__synchronizer_instance<warp_level>
{
  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _MappingResult&, const level_synchronizer&, const _Hierarchy&) const noexcept
  {
    ::__syncwarp();
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void
  do_sync_aligned(const _MappingResult&, const level_synchronizer&, const _Hierarchy&) const noexcept
  {
    ::__syncwarp();
  }
};

template <>
struct level_synchronizer::__synchronizer_instance<block_level>
{
  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _MappingResult&, const level_synchronizer&, const _Hierarchy&) const noexcept
  {
    ::cuda::experimental::__block_sync<false>();
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void
  do_sync_aligned(const _MappingResult&, const level_synchronizer&, const _Hierarchy&) const noexcept
  {
    ::cuda::experimental::__block_sync<true>();
  }
};

template <>
struct level_synchronizer::__synchronizer_instance<cluster_level>
{
  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _MappingResult&, const level_synchronizer&, const _Hierarchy&) const noexcept
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      ({
                        ::__cluster_barrier_arrive();
                        ::__cluster_barrier_wait();
                      }),
                      ({ ::cuda::experimental::__block_sync<false>(); }))
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void
  do_sync_aligned(const _MappingResult&, const level_synchronizer&, const _Hierarchy&) const noexcept
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      ({
                        asm volatile("barrier.cluster.arrive.aligned;");
                        asm volatile("barrier.cluster.wait.aligned;");
                      }),
                      ({ ::cuda::experimental::__block_sync<true>(); }))
  }
};

// Synchronizing whole grid requires driver support and the kernel must be launched using the cooperative launch API.
// This part is extracted from grid synchronization implementation in cooperative groups.
template <>
struct level_synchronizer::__synchronizer_instance<grid_level>
{
  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void
  do_sync(const _MappingResult&, const level_synchronizer&, const _Hierarchy& __hier) const noexcept
  {
    __sync_impl<false>(__hier);
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void
  do_sync_aligned(const _MappingResult&, const level_synchronizer&, const _Hierarchy& __hier) const noexcept
  {
    __sync_impl<true>(__hier);
  }

  template <bool _Aligned, class _Hierarchy>
  _CCCL_DEVICE_API void __sync_impl(const _Hierarchy& __hier) const noexcept
  {
    struct __grid_workspace
    {
      unsigned __size_;
      unsigned __barrier_;
    };

    __grid_workspace* __grid_workspace_ptr;
    asm("mov.b64 %0, {%%envreg2, %%envreg1};" : "=l"(__grid_workspace_ptr));
    _CCCL_ASSERT(__grid_workspace_ptr != nullptr,
                 "Synchronizing grid requires the kernel to be launched using the cooperative launch.");

    const auto __barrier_ptr = &__grid_workspace_ptr->__barrier_;

    // Synchronize the block before synchronizing with the other blocks.
    ::cuda::experimental::__block_sync<_Aligned>();

    // Synchronize with other blocks using the thread 0 in block.
    const auto __thread_idx = gpu_thread.index(block, __hier);
    if ((__thread_idx.x | __thread_idx.y | __thread_idx.z) == 0)
    {
      const auto __expected = block.count_as<unsigned>(grid, __hier);
      unsigned __nblocks    = 1;

      const auto __block_idx = block.index(grid, __hier);
      if ((__block_idx.x | __block_idx.y | __block_idx.z) == 0)
      {
        __nblocks = unsigned{::cuda::std::numeric_limits<int>::min()} - (__expected - 1);
      }

      unsigned __old_barrier_value;
#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
      __old_barrier_value =
        __nv_atomic_fetch_add(__barrier_ptr, __nblocks, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
      asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;"
                   : "=r"(__old_barrier_value)
                   : "l"(__barrier_ptr), "r"(__nblocks)
                   : "memory");
#  endif // ^^^ !_CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^
      unsigned __curr_barrier_value;
      do
      {
#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
        __nv_atomic_load(__barrier_ptr, &__curr_barrier_value, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
        asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(__curr_barrier_value) : "l"(__barrier_ptr) : "memory");
#  endif // ^^^ !_CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^
      } while (static_cast<int>(__old_barrier_value) < 0 == static_cast<int>(__curr_barrier_value) < 0);
    }

    // Wait for the thread 0 to finish the inter block synchronization.
    ::cuda::experimental::__block_sync<_Aligned>();
  }
};
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_LEVEL_SYNCHRONIZER_CUH
