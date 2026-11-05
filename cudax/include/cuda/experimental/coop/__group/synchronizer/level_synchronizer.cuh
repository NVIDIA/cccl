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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/synchronizer/arrival_token.cuh>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

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
    return __synchronizer_instance<typename _ParentGroup::level_type>{};
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
    if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90, ({ ::__cluster_barrier_arrive(); }))
    }
    return arrival_token{};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API arrival_token do_arrive_aligned(const _Hierarchy&) const
  {
    if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90, ({ asm volatile("barrier.cluster.arrive.aligned;" ::: "memory"); }))
    }
    return arrival_token{};
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait(const _Hierarchy&, [[maybe_unused]] arrival_token&& __tok) const
  {
    _CCCL_ASSERT(__tok.__valid(), "invalid arrival token passed to do_wait function");

    if constexpr (::cuda::std::is_same_v<_Level, thread_level>)
    {
    }
    else if constexpr (::cuda::std::is_same_v<_Level, warp_level>)
    {
      ::__syncwarp();
    }
    else if constexpr (::cuda::std::is_same_v<_Level, block_level>)
    {
      ::cuda::experimental::coop::__block_sync</*is-aligned*/ false>();
    }
    else if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
    {
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, ({ ::__cluster_barrier_wait(); }), ({
                          ::cuda::experimental::coop::__block_sync</*is-aligned*/ false>();
                        }))
    }
    else
    {
      static_assert(::cuda::std::__always_false_v<_Level>, "Unsupported level");
    }
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait_aligned(const _Hierarchy&, [[maybe_unused]] arrival_token&& __tok) const
  {
    _CCCL_ASSERT(__tok.__valid(), "invalid arrival token passed to do_wait_aligned function");

    if constexpr (::cuda::std::is_same_v<_Level, thread_level>)
    {
    }
    else if constexpr (::cuda::std::is_same_v<_Level, warp_level>)
    {
      ::__syncwarp();
    }
    else if constexpr (::cuda::std::is_same_v<_Level, block_level>)
    {
      ::cuda::experimental::coop::__block_sync</*is-aligned*/ true>();
    }
    else if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
    {
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                        ({
                          asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
                        }),
                        ({ ::cuda::experimental::coop::__block_sync</*is-aligned*/ true>(); }))
    }
    else
    {
      static_assert(::cuda::std::__always_false_v<_Level>, "Unsupported level");
    }
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&)
  {}

  [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance view() const
  {
    return *this;
  }
};

struct __grid_workspace
{
  ::cuda::std::uint32_t __size_;
  ::cuda::std::uint32_t __barrier_;
};

[[nodiscard]] _CCCL_CONST _CCCL_DEVICE inline __grid_workspace* __get_grid_barrier_ptr()
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

  __grid_workspace* __grid_workspace_ptr_ = ::cuda::experimental::coop::__get_grid_barrier_ptr();

  _CCCL_DEVICE_API ::cuda::std::uint32_t* __get_grid_barrier_ptr() const
  {
    _CCCL_ASSERT(__grid_workspace_ptr_ != nullptr,
                 "Synchronizing grid requires the kernel to be launched using the cooperative launch.");
    return &__grid_workspace_ptr_->__barrier_;
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
        __nblocks = ::cuda::std::uint32_t{::cuda::std::numeric_limits<int>::min()} - (__expected - 1);
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
      ::cuda::std::uint32_t __curr_barrier_value;
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
        __nblocks = ::cuda::std::uint32_t{::cuda::std::numeric_limits<int>::min()} - (__expected - 1);
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
  _CCCL_DEVICE_API void __wait_impl(const _Hierarchy& __hier, ::cuda::std::uint32_t __old_barrier_value) const
  {
    const auto __barrier_ptr = __get_grid_barrier_ptr();

    // Synchronize with other blocks using the thread 0 in block.
    const auto __thread_idx = gpu_thread.index(block, __hier);
    if ((__thread_idx.x | __thread_idx.y | __thread_idx.z) == 0)
    {
      ::cuda::std::uint32_t __curr_barrier_value;
      do
      {
#  if _CCCL_HAS_NV_ATOMIC_BUILTINS()
        __nv_atomic_load(__barrier_ptr, &__curr_barrier_value, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_DEVICE);
#  else // ^^^ _CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^ / vvv !_CCCL_HAS_NV_ATOMIC_BUILTINS() vvv
        asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(__curr_barrier_value) : "l"(__barrier_ptr) : "memory");
#  endif // ^^^ !_CCCL_HAS_NV_ATOMIC_BUILTINS() ^^^
      } while ((static_cast<int>(__old_barrier_value) < 0) == (static_cast<int>(__curr_barrier_value) < 0));
    }

    // Wait for the thread 0 to finish the inter block synchronization.
    ::cuda::experimental::coop::__block_sync<_Aligned>();
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait(const _Hierarchy& __hier, arrival_token&& __tok) const
  {
    __wait_impl<false>(__hier, __tok.__value());
  }

  template <class _Hierarchy>
  _CCCL_DEVICE_API void do_wait_aligned(const _Hierarchy& __hier, arrival_token&& __tok) const
  {
    __wait_impl<true>(__hier, __tok.__value());
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&)
  {}

  [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance view() const
  {
    return *this;
  }
};
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_LEVEL_SYNCHRONIZER_CUH
