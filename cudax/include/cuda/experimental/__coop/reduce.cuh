//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_REDUCE_CUH
#define _CUDA_EXPERIMENTAL___COOP_REDUCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/array>
#include <cuda/std/optional>

#include <cuda/experimental/group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): We share the temporary storage in shared/global memory for all reduce invocations. This is a temporary
// state before we make it a parameter.

namespace cuda::experimental::coop
{
template <class _Hierarchy, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(this_thread<_Hierarchy>, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  return ::cub::ThreadReduce(__thread_data, __red_fn);
}

template <class _Hierarchy, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(this_warp<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _WarpReduce = ::cub::WarpReduce<_Tp>;
  __shared__ typename _WarpReduce::TempStorage __scratch;

  const auto __result = _WarpReduce{__scratch}.Reduce(__thread_data, __red_fn);
  return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
}

template <class _Hierarchy, class _Tp, cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(this_block<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _BlockExts = decltype(gpu_thread.extents(block, __group.hierarchy()));
  static_assert(_BlockExts::rank_dynamic() == 0,
                "cuda::coop::reduce requires the block level to have all static extents.");

  using _BlockReduce =
    ::cub::BlockReduce<_Tp,
                       static_cast<int>(_BlockExts::static_extent(0)),
                       ::cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                       static_cast<int>(_BlockExts::static_extent(1)),
                       static_cast<int>(_BlockExts::static_extent(2))>;
  __shared__ typename _BlockReduce::TempStorage __scratch;

  const auto __result = _BlockReduce{__scratch}.Reduce(__thread_data, __red_fn);
  return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
}

template <class _Hierarchy, class _Tp, cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(this_cluster<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _ClusterExts = decltype(block.extents(cluster, __group.hierarchy()));
  static_assert(_ClusterExts::rank_dynamic() == 0,
                "cuda::coop::reduce requires the cluster level to have all static extents.");

  constexpr auto __nblocks_in_cluster =
    _ClusterExts::static_extent(0) * _ClusterExts::static_extent(1) * _ClusterExts::static_extent(2);
  if constexpr (__nblocks_in_cluster == 1)
  {
    return ::cuda::experimental::coop::__reduce_impl(this_block{__group.hierarchy()}, __thread_data, __red_fn);
  }
  else
  {
    using _BlockExts = decltype(gpu_thread.extents(block, __group.hierarchy()));
    static_assert(_BlockExts::rank_dynamic() == 0,
                  "cuda::coop::reduce requires the block level to have all static extents.");

    using _BlockReduce =
      ::cub::BlockReduce<_Tp,
                         static_cast<int>(_BlockExts::static_extent(0)),
                         ::cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                         static_cast<int>(_BlockExts::static_extent(1)),
                         static_cast<int>(_BlockExts::static_extent(2))>;

    using _RootWarpReduce = ::cub::WarpReduce<_Tp>;
    struct _RootScratch
    {
      _Tp __partials_[__nblocks_in_cluster];
      typename _RootWarpReduce::TempStorage __warp_reduce_;
    };

    union _Scratch
    {
      typename _BlockReduce::TempStorage __block_;
      _RootScratch __root_;
    };
    __shared__ _Scratch __scratch;

    const auto __partial = _BlockReduce{__scratch.__block_}.Reduce(__thread_data, __red_fn);
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   const auto __partials_root =
                     static_cast<_Tp*>(::__cluster_map_shared_rank(__scratch.__root_.__partials_, 0));

                   __group.sync_aligned();
                   if (gpu_thread.is_root_rank(this_block{__group.hierarchy()}))
                   {
                     __partials_root[block.rank(__group)] = __partial;
                   }
                   __group.sync_aligned();

                   if (warp.is_root_rank(__group))
                   {
                     this_warp __warp{__group.hierarchy()};
                     const auto __value  = (gpu_thread.rank(__warp) < __nblocks_in_cluster)
                                           ? __scratch.__root_.__partials_[gpu_thread.rank(__warp)]
                                           : ::cuda::identity_element<_RedFn, _Tp>();
                     const auto __result = _RootWarpReduce{__scratch.__root_.__warp_reduce_}.Reduce(__value, __red_fn);
                     if (gpu_thread.is_root_rank(__group))
                     {
                       return ::cuda::std::optional{__result};
                     }
                   }
                 }))
    return ::cuda::std::nullopt;
  }
}

template <class _Tp, ::cuda::std::size_t _Np>
_CCCL_DEVICE ::cuda::std::array<_Tp, _Np> __reduce_grid_partials;

template <class _Hierarchy, class _Tp, cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(this_grid<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _GridExts = decltype(cluster.extents(grid, __group.hierarchy()));
  static_assert(_GridExts::rank_dynamic() == 0,
                "cuda::coop::reduce requires the grid level to have all static extents.");

  constexpr auto __nclusters_in_grid =
    _GridExts::static_extent(0) * _GridExts::static_extent(1) * _GridExts::static_extent(2);

  this_cluster __cluster{__group.hierarchy()};
  const auto __partial = ::cuda::experimental::coop::__reduce_impl(__cluster, __thread_data, __red_fn);

  if (gpu_thread.is_root_rank(__cluster))
  {
    __reduce_grid_partials<_Tp, __nclusters_in_grid>[cluster.rank(__group)] = __partial.value();
  }
  __group.sync_aligned();

  if (block.is_root_rank(__group))
  {
    this_block __block{__group.hierarchy()};

    constexpr auto __npartials_per_thread = ::cuda::ceil_div(__nclusters_in_grid, gpu_thread.static_count(__block));
    _Tp __thread_partials[__npartials_per_thread];
    const auto __offset = gpu_thread.rank(__block) * __npartials_per_thread;

    // todo(dabayer): This is not the most efficient way to load values, it doesn't take into account element size and
    // reads N consecutive elements by 1 thread.
    for (unsigned __i = 0; __i < __npartials_per_thread; ++__i)
    {
      __thread_partials[__i] =
        (__offset + __i < __nclusters_in_grid)
          ? __reduce_grid_partials<_Tp, __nclusters_in_grid>[__offset + __i]
          : ::cuda::identity_element<_RedFn, _Tp>();
    }
    return ::cuda::experimental::coop::__reduce_impl(__block, __thread_partials, __red_fn);
  }
  return ::cuda::std::nullopt;
}

_CCCL_TEMPLATE(class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn)
_CCCL_REQUIRES(::cuda::std::is_same_v<warp_level, typename _Group::unit_type>
                 _CCCL_AND ::cuda::std::is_same_v<block_level, typename _Group::level_type>)
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(_Group __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  constexpr auto __nwarps_in_group = warp.static_count(__group);
  static_assert(__nwarps_in_group != ::cuda::std::dynamic_extent,
                "cuda::coop::reduce requires the group to have statically known size");

  using _WarpReduce = ::cub::WarpReduce<_Tp>;
  union _Scratch
  {
    typename _WarpReduce::TempStorage __warp_reduce_[__nwarps_in_group];
    _Tp __partials_[__nwarps_in_group];
  };
  __shared__ _Scratch __scratch;

  const auto __partial = _WarpReduce{__scratch.__warp_reduce_[warp.rank(__group)]}.Reduce(__thread_data, __red_fn);
  __group.sync_aligned();

  this_warp __warp{__group.hierarchy()};
  if (gpu_thread.is_root_rank(__warp))
  {
    __scratch.__partials_[warp.rank(__group)] = __partial;
  }
  __group.sync_aligned();

  if (warp.is_root_rank(__group))
  {
    const auto __value  = (gpu_thread.rank(__warp) < __nwarps_in_group)
                          ? __scratch.__partials_[gpu_thread.rank(__warp)]
                          : ::cuda::identity_element<_RedFn, _Tp>();
    const auto __result = _WarpReduce{__scratch.__warp_reduce_[0]}.Reduce(__value, __red_fn);
    if (gpu_thread.is_root_rank(__warp))
    {
      return ::cuda::std::optional{__result};
    }
  }
  return ::cuda::std::nullopt;
}

template <class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
reduce(_Group __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  static_assert(gpu_thread.static_count(__group) != ::cuda::std::dynamic_extent,
                "cuda::coop::reduce requires the group to have statically known size");

  if (!gpu_thread.is_part_of(__group))
  {
    return ::cuda::std::nullopt;
  }

  return ::cuda::experimental::coop::__reduce_impl(__group, __thread_data, __red_fn);
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_CUH
