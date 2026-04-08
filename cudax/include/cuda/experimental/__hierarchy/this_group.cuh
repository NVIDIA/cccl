//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___THIS_GROUP_CUH
#define _CUDA_EXPERIMENTAL___THIS_GROUP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__hierarchy/fwd.cuh>
#include <cuda/experimental/__hierarchy/implicit_hierarchy.cuh>

#if _CCCL_HAS_COOPERATIVE_GROUPS()
#  include <cooperative_groups.h>
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <class _HierarchyLike>
using __hierarchy_type_of =
  ::cuda::std::remove_cvref_t<decltype(::cuda::__unpack_hierarchy_if_needed(::cuda::std::declval<_HierarchyLike>()))>;

#if _CCCL_CUDA_COMPILATION()
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

template <bool _Aligned>
_CCCL_DEVICE_API void __cluster_sync() noexcept
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                    ({
                      if constexpr (_Aligned)
                      {
                        asm volatile("barrier.cluster.arrive.aligned;");
                        asm volatile("barrier.cluster.wait.aligned;");
                      }
                      else
                      {
                        ::__cluster_barrier_arrive();
                        ::__cluster_barrier_wait();
                      }
                    }),
                    ({ ::cuda::experimental::__block_sync<_Aligned>(); }))
}
#endif // _CCCL_CUDA_COMPILATION()

// todo: use __hier_ in queries
template <class _Level, class _Hierarchy>
class __this_group_base
{
  static_assert(__is_hierarchy_level_v<_Level>);
  static_assert(__is_hierarchy_v<_Hierarchy>);

protected:
  _Hierarchy __hier_;

public:
  _CCCL_DEVICE_API explicit __this_group_base() noexcept
      : __hier_{::cuda::experimental::__implicit_hierarchy()}
  {}

  _CCCL_TEMPLATE(class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Hierarchy, __hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API __this_group_base(const _HierarchyLike& __hier_like) noexcept
      : __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {}

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND(
    !::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp count_as(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.template count_as<_Tp>(__in_level);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND(!::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.count(__in_level);
  }

#if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND(
    !::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API _Tp rank_as(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.template rank_as<_Tp>(__in_level);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND(!::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API auto rank(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.rank(__in_level);
  }
#endif // _CCCL_CUDA_COMPILATION()
};

template <class _Hierarchy>
class this_thread : __this_group_base<thread_level, _Hierarchy>
{
  using __base_type = __this_group_base<thread_level, _Hierarchy>;

public:
  using unit_type      = thread_level;
  using level_type     = thread_level;
  using hierarchy_type = _Hierarchy;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;
#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
  template <class _Parent>
  _CCCL_DEVICE_API this_thread(const ::cooperative_groups::thread_block_tile<1, _Parent>&) noexcept
  {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

  _CCCL_DEVICE_API void sync() noexcept {}

  _CCCL_DEVICE_API void sync_aligned() noexcept {}

  [[nodiscard]] _CCCL_DEVICE_API const _Hierarchy& hierarchy() const noexcept
  {
    return __base_type::__hier_;
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_HOST_DEVICE this_thread() -> this_thread<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE this_thread(const _Hierarchy&) -> this_thread<__hierarchy_type_of<_Hierarchy>>;

#if _CCCL_HAS_COOPERATIVE_GROUPS()
_CCCL_HOST_DEVICE this_thread(const ::cooperative_groups::thread_block_tile<1, void>&)
  -> this_thread<__implicit_hierarchy_t>;
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

template <class _Hierarchy>
class this_warp : __this_group_base<warp_level, _Hierarchy>
{
  using __base_type = __this_group_base<warp_level, _Hierarchy>;

public:
  using unit_type      = warp_level;
  using level_type     = warp_level;
  using hierarchy_type = _Hierarchy;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;
#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
  template <class _Parent>
  _CCCL_DEVICE_API this_warp(const ::cooperative_groups::thread_block_tile<32, _Parent>&) noexcept
  {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::__syncwarp();
  }

  _CCCL_DEVICE_API void sync_aligned() noexcept
  {
    ::__syncwarp();
  }

  [[nodiscard]] _CCCL_DEVICE_API const _Hierarchy& hierarchy() const noexcept
  {
    return __base_type::__hier_;
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_HOST_DEVICE this_warp() -> this_warp<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE this_warp(const _Hierarchy&) -> this_warp<__hierarchy_type_of<_Hierarchy>>;

#if _CCCL_HAS_COOPERATIVE_GROUPS()
template <class _Parent>
_CCCL_HOST_DEVICE this_warp(const ::cooperative_groups::thread_block_tile<32, _Parent>&)
  -> this_warp<__implicit_hierarchy_t>;
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

template <class _Hierarchy>
class this_block : __this_group_base<block_level, _Hierarchy>
{
  using __base_type = __this_group_base<block_level, _Hierarchy>;

public:
  using unit_type      = block_level;
  using level_type     = block_level;
  using hierarchy_type = _Hierarchy;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;

#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
  _CCCL_DEVICE_API this_block(const ::cooperative_groups::thread_block&) noexcept {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::cuda::experimental::__block_sync<false>();
  }

  _CCCL_DEVICE_API void sync_aligned() noexcept
  {
    ::cuda::experimental::__block_sync<true>();
  }

  [[nodiscard]] _CCCL_DEVICE_API const _Hierarchy& hierarchy() const noexcept
  {
    return __base_type::__hier_;
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_HOST_DEVICE this_block() -> this_block<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE this_block(const _Hierarchy&) -> this_block<__hierarchy_type_of<_Hierarchy>>;

#if _CCCL_HAS_COOPERATIVE_GROUPS()
_CCCL_HOST_DEVICE this_block(const ::cooperative_groups::thread_block&) -> this_block<__implicit_hierarchy_t>;
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

template <class _Hierarchy>
class this_cluster : __this_group_base<cluster_level, _Hierarchy>
{
  using __base_type = __this_group_base<cluster_level, _Hierarchy>;

public:
  using unit_type      = cluster_level;
  using level_type     = cluster_level;
  using hierarchy_type = _Hierarchy;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;

#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

#  if _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)
  _CCCL_DEVICE_API this_cluster(const ::cooperative_groups::cluster_group&) noexcept {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)

  _CCCL_DEVICE_API void sync() noexcept
  {
    if constexpr (_Hierarchy::has_level(cluster))
    {
      ::cuda::experimental::__cluster_sync<false>();
    }
    else
    {
      ::cuda::experimental::__block_sync<false>();
    }
  }

  _CCCL_DEVICE_API void sync_aligned() noexcept
  {
    if constexpr (_Hierarchy::has_level(cluster))
    {
      ::cuda::experimental::__cluster_sync<true>();
    }
    else
    {
      ::cuda::experimental::__block_sync<true>();
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API const _Hierarchy& hierarchy() const noexcept
  {
    return __base_type::__hier_;
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_HOST_DEVICE this_cluster() -> this_cluster<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE this_cluster(const _Hierarchy&) -> this_cluster<__hierarchy_type_of<_Hierarchy>>;

#if _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)
_CCCL_HOST_DEVICE this_cluster(const ::cooperative_groups::cluster_group&) -> this_cluster<__implicit_hierarchy_t>;
#endif // _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)

// Synchronizing whole grid requires driver support and the kernel must be launched using the cooperative launch API.
// This part is extracted from grid synchronization implementation in cooperative groups.
#if _CCCL_CUDA_COMPILATION()
[[nodiscard]] _CCCL_DEVICE_API inline unsigned* __get_grid_barrier_ptr() noexcept
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

  return &__grid_workspace_ptr->__barrier_;
}
#endif // _CCCL_CUDA_COMPILATION()

template <class _Hierarchy>
class this_grid : __this_group_base<grid_level, _Hierarchy>
{
  using __base_type = __this_group_base<grid_level, _Hierarchy>;

#if _CCCL_CUDA_COMPILATION()
  template <bool _Aligned>
  _CCCL_DEVICE_API void __sync_impl() noexcept
  {
    const auto __barrier_ptr = ::cuda::experimental::__get_grid_barrier_ptr();

    // Synchronize the block before synchronizing with the other blocks.
    ::cuda::experimental::__block_sync<_Aligned>();

    // Synchronize with other blocks using the thread 0 in block.
    const auto __thread_idx = gpu_thread.index(block, hierarchy());
    if ((__thread_idx.x | __thread_idx.y | __thread_idx.z) == 0)
    {
      const auto __expected = block.count_as<unsigned>(grid, hierarchy());
      unsigned __nblocks    = 1;

      const auto __block_idx = block.index(grid, hierarchy());
      if ((__block_idx.x | __block_idx.y | __block_idx.z) == 0)
      {
        __nblocks = unsigned{::cuda::std::numeric_limits<int>::min()} - (__expected - 1);
      }

      unsigned __old_barrier_value;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_70,
        ({
          asm volatile("atom.add.release.gpu.u32 %0, [%1], %2;"
                       : "=r"(__old_barrier_value)
                       : "l"(__barrier_ptr), "r"(__nblocks)
                       : "memory");
          unsigned __curr_barrier_value;
          do
          {
            asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(__curr_barrier_value) : "l"(__barrier_ptr) : "memory");
          } while (static_cast<int>(__old_barrier_value) < 0 == static_cast<int>(__curr_barrier_value) < 0);
        }),
        ({
          ::__threadfence();
          __old_barrier_value = ::atomicAdd(__barrier_ptr, __nblocks);
          while (static_cast<int>(__old_barrier_value) < 0 == static_cast<int>(*__barrier_ptr) < 0)
          {
          }
          ::__threadfence();
        }))
    }

    // Wait for the thread 0 to finish the inter block synchronization.
    ::cuda::experimental::__block_sync<_Aligned>();
  }
#endif // _CCCL_CUDA_COMPILATION()

public:
  using unit_type      = grid_level;
  using level_type     = grid_level;
  using hierarchy_type = _Hierarchy;

  using __base_type::__base_type;

#if _CCCL_CUDA_COMPILATION()
#  if _CCCL_HAS_COOPERATIVE_GROUPS()
  _CCCL_DEVICE_API this_grid(const ::cooperative_groups::grid_group&) noexcept {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

  _CCCL_DEVICE_API void sync() noexcept
  {
    __sync_impl<false>();
  }

  _CCCL_DEVICE_API void sync_aligned() noexcept
  {
    __sync_impl<true>();
  }

  [[nodiscard]] _CCCL_DEVICE_API const _Hierarchy& hierarchy() const noexcept
  {
    return __base_type::__hier_;
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_HOST_DEVICE this_grid() -> this_grid<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE this_grid(const _Hierarchy&) -> this_grid<__hierarchy_type_of<_Hierarchy>>;

#if _CCCL_HAS_COOPERATIVE_GROUPS()
_CCCL_HOST_DEVICE this_grid(const ::cooperative_groups::grid_group&) -> this_grid<__implicit_hierarchy_t>;
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___THIS_GROUP_CUH
