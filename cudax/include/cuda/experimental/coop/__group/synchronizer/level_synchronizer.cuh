//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_LEVEL_SYNCHRONIZER_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_LEVEL_SYNCHRONIZER_CUH

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/synchronizer/arrival_token.cuh>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
class level_synchronizer
{
public:
  template <class _Level>
  struct __synchronizer_instance;

  _CCCL_HIDE_FROM_ABI explicit level_synchronizer() = default;

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto make_instance(const _Unit&, const _ParentGroup&, const _MappingResult&) const
  {
    return __synchronizer_instance<typename _ParentGroup::bound_level>{};
  }
};

template <bool _Aligned>
_CCCL_DEVICE_API void __block_sync()
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

template <class _Level>
struct level_synchronizer::__synchronizer_instance
{
  using arrival_token _CCCL_NODEBUG = __dummy_arrival_token<__synchronizer_instance>;

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _Hierarchy& __hier) const
  {
    do_wait(__hier, do_arrive(__hier));
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _Hierarchy& __hier) const
  {
    do_wait_aligned(__hier, do_arrive_aligned(__hier));
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive(const _Hierarchy&) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
                   {
                     ::__cluster_barrier_arrive();
                   }
                 }))
    return arrival_token{};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive_aligned(const _Hierarchy&) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
                   {
                     asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
                   }
                 }))
    return arrival_token{};
  }

  template <bool _Aligned>
  _CCCL_DEVICE_API void __wait_impl([[maybe_unused]] arrival_token __tok) const
  {
    _CCCL_ASSERT(__tok.__is_valid(), "invalid arrival token passed to wait");

    if constexpr (::cuda::std::is_same_v<_Level, warp_level>)
    {
      ::__syncwarp();
    }
    else if constexpr (::cuda::std::is_same_v<_Level, block_level> || ::cuda::std::is_same_v<_Level, cluster_level>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                     if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
                     {
                       if constexpr (_Aligned)
                       {
                         asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
                       }
                       else
                       {
                         ::__cluster_barrier_wait();
                       }
                       return;
                     }
                   }))
      ::cuda::experimental::coop::__block_sync<_Aligned>();
    }
    else
    {
      static_assert(::cuda::std::is_same_v<_Level, thread_level>, "Unsupported level");
    }
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait(const _Hierarchy&, arrival_token&& __tok) const
  {
    __wait_impl<false>(::cuda::std::move(__tok));
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait_aligned(const _Hierarchy&, arrival_token&& __tok) const
  {
    __wait_impl<true>(::cuda::std::move(__tok));
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&)
  {}
};

struct __grid_workspace
{
  ::cuda::std::uint32_t __size_;
  ::cuda::std::uint32_t __barrier_;
};

[[nodiscard]] _CCCL_CONST _CCCL_DEVICE_API inline __grid_workspace* __get_grid_barrier_ptr()
{
  __grid_workspace* __ret;
  asm("mov.b64 %0, {%%envreg2, %%envreg1};" : "=l"(__ret));
  return __ret;
}

// Synchronizing whole grid requires driver support and the kernel must be launched using the cooperative launch API.
// This part is extracted from grid synchronization implementation in cooperative groups.
template <>
struct level_synchronizer::__synchronizer_instance<grid_level>
{
  using arrival_token _CCCL_NODEBUG = __int_arrival_token<::cuda::std::uint32_t, __synchronizer_instance<grid_level>>;

  static constexpr ::cuda::std::uint32_t __phase_bit = 0x8000'0000u;

  __grid_workspace* __grid_workspace_ptr_ = ::cuda::experimental::coop::__get_grid_barrier_ptr();

  _CCCL_DEVICE_API ::cuda::std::uint32_t* __get_grid_barrier_ptr() const
  {
    _CCCL_ASSERT(__grid_workspace_ptr_ != nullptr,
                 "Synchronizing grid requires the kernel to be launched using the cooperative launch.");
    return &__grid_workspace_ptr_->__barrier_;
  }

  _CCCL_DEVICE_API static void
  __wait_for_phase_change(const ::cuda::std::uint32_t* __barrier_ptr, ::cuda::std::uint32_t __old_barrier_value)
  {
    while (true)
    {
      ::cuda::std::uint32_t __curr_barrier_value;
#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
      __nv_atomic_load(__barrier_ptr, &__curr_barrier_value, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
      asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(__curr_barrier_value) : "l"(__barrier_ptr) : "memory");
#  endif // ^^^ !_CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^
      if (((__old_barrier_value ^ __curr_barrier_value) & __phase_bit) != 0)
      {
        return;
      }
    }
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _Hierarchy& __hier) const
  {
    __sync_impl<false>(__hier);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _Hierarchy& __hier) const
  {
    __sync_impl<true>(__hier);
  }

  template <bool _Aligned, class _Hierarchy>
  _CCCL_DEVICE_API void __sync_impl(const _Hierarchy& __hier) const
  {
    const auto __barrier_ptr = __get_grid_barrier_ptr();

    // Synchronize the block before synchronizing with the other blocks.
    ::cuda::experimental::coop::__block_sync<_Aligned>();

    // Synchronize with other blocks using the thread 0 in block.
    const auto __thread_idx = gpu_thread.index(block, __hier);
    if ((__thread_idx.x | __thread_idx.y | __thread_idx.z) == 0)
    {
      const auto __expected           = block.count_as<::cuda::std::uint32_t>(grid, __hier);
      ::cuda::std::uint32_t __nblocks = 1;

      const auto __block_idx = block.index(grid, __hier);
      if ((__block_idx.x | __block_idx.y | __block_idx.z) == 0)
      {
        __nblocks = __phase_bit - (__expected - 1);
      }

      ::cuda::std::uint32_t __old_barrier_value;
#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
      __old_barrier_value =
        __nv_atomic_fetch_add(__barrier_ptr, __nblocks, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
      asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;"
                   : "=r"(__old_barrier_value)
                   : "l"(__barrier_ptr), "r"(__nblocks)
                   : "memory");
#  endif // ^^^ !_CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^
      __wait_for_phase_change(__barrier_ptr, __old_barrier_value);
    }

    // Wait for the thread 0 to finish the inter block synchronization.
    ::cuda::experimental::coop::__block_sync<_Aligned>();
  }

  template <bool _Aligned, class _Hierarchy>
  _CCCL_DEVICE_API arrival_token __arrive_impl(const _Hierarchy& __hier) const
  {
    const auto __barrier_ptr = __get_grid_barrier_ptr();

    // Synchronize the block before synchronizing with the other blocks.
    ::cuda::experimental::coop::__block_sync<_Aligned>();

    ::cuda::std::uint32_t __old_barrier_value = 0;

    // Synchronize with other blocks using the thread 0 in block.
    const auto __thread_idx = gpu_thread.index(block, __hier);
    if ((__thread_idx.x | __thread_idx.y | __thread_idx.z) == 0)
    {
      const auto __expected           = block.count_as<::cuda::std::uint32_t>(grid, __hier);
      ::cuda::std::uint32_t __nblocks = 1;

      const auto __block_idx = block.index(grid, __hier);
      if ((__block_idx.x | __block_idx.y | __block_idx.z) == 0)
      {
        __nblocks = __phase_bit - (__expected - 1);
      }

#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
      __old_barrier_value =
        __nv_atomic_fetch_add(__barrier_ptr, __nblocks, __NV_ATOMIC_RELEASE, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
      asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;"
                   : "=r"(__old_barrier_value)
                   : "l"(__barrier_ptr), "r"(__nblocks)
                   : "memory");
#  endif // ^^^ !_CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^
    }
    return arrival_token{__old_barrier_value};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive(const _Hierarchy& __hier) const
  {
    return __arrive_impl<false>(__hier);
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive_aligned(const _Hierarchy& __hier) const
  {
    return __arrive_impl<true>(__hier);
  }

  template <bool _Aligned, class _Hierarchy>
  _CCCL_DEVICE_API void __wait_impl(const _Hierarchy& __hier, arrival_token __tok) const
  {
    const auto __barrier_ptr = __get_grid_barrier_ptr();

    // Synchronize with other blocks using the thread 0 in block.
    const auto __thread_idx = gpu_thread.index(block, __hier);
    if ((__thread_idx.x | __thread_idx.y | __thread_idx.z) == 0)
    {
      __wait_for_phase_change(__barrier_ptr, __tok.__value());
    }

    // Wait for the thread 0 to finish the inter block synchronization.
    ::cuda::experimental::coop::__block_sync<_Aligned>();
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait(const _Hierarchy& __hier, arrival_token&& __tok) const
  {
    __wait_impl<false>(__hier, ::cuda::std::move(__tok));
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait_aligned(const _Hierarchy& __hier, arrival_token&& __tok) const
  {
    __wait_impl<true>(__hier, ::cuda::std::move(__tok));
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&)
  {}
};
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_LEVEL_SYNCHRONIZER_CUH
