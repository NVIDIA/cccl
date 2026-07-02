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
#include <cuda/__cmath/pow2.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/array>
#include <cuda/std/optional>

#include <cuda/experimental/__coop/shuffle_down.cuh>
#include <cuda/experimental/__utility/broadcasted.cuh>
#include <cuda/experimental/group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): We share the temporary storage in shared/global memory for all reduce invocations. This is a temporary
// state before we make it a parameter.

namespace cuda::experimental::coop
{
template <bool _Dummy = false>
_CCCL_DEVICE_API auto __reduce_impl(...)
{
  static_assert(_Dummy, "cudax::coop::reduce is not supported for the group");
}

template <bool _Broadcasted, class _Hierarchy, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API auto __reduce_impl(
  ::cuda::std::bool_constant<_Broadcasted>, this_thread<_Hierarchy>, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  const auto __result = ::cub::ThreadReduce(__thread_data, __red_fn);
  if constexpr (_Broadcasted)
  {
    return __result;
  }
  else
  {
    return ::cuda::std::optional{__result};
  }
}

template <bool _Broadcasted, class _Hierarchy, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API auto __reduce_impl(
  ::cuda::std::bool_constant<_Broadcasted>, this_warp<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _BlockExts = decltype(gpu_thread.extents(block, __group.hierarchy()));
  constexpr auto __nwarps_in_block =
    ::cuda::ceil_div(_BlockExts::static_extent(0) * _BlockExts::static_extent(1) * _BlockExts::static_extent(2), 32);

  using _WarpReduce = ::cub::WarpReduce<_Tp>;

  union _Scratch
  {
    typename _WarpReduce::TempStorage __warp_reduce_[__nwarps_in_block];
  };
  __shared__ _Scratch __scratch;

  const auto __warp_rank_in_block = __group.rank(block);
  const auto __result = _WarpReduce{__scratch.__warp_reduce_[__warp_rank_in_block]}.Reduce(__thread_data, __red_fn);
  if constexpr (_Broadcasted)
  {
    return ::cuda::device::warp_shuffle_idx(__result, 0).data;
  }
  else
  {
    return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
  }
}

template <bool _Broadcasted, class _Hierarchy, class _Tp, cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API auto __reduce_impl(
  ::cuda::std::bool_constant<_Broadcasted>, this_block<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
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

  union _Scratch
  {
    typename _BlockReduce::TempStorage __block_reduce_;
    _Tp __bcast_;
  };
  __shared__ _Scratch __scratch;

  const auto __result = _BlockReduce{__scratch.__block_reduce_}.Reduce(__thread_data, __red_fn);
  if constexpr (_Broadcasted)
  {
    if (gpu_thread.is_root_rank(__group))
    {
      __scratch.__bcast_ = __result;
    }
    __group.sync_aligned();
    return __scratch.__bcast_;
  }
  else
  {
    return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
  }
}

template <bool _Broadcasted, class _Hierarchy, class _Tp, cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API auto __reduce_impl(
  ::cuda::std::bool_constant<_Broadcasted>, this_cluster<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _ClusterExts = decltype(block.extents(cluster, __group.hierarchy()));
  static_assert(_ClusterExts::rank_dynamic() == 0,
                "cuda::coop::reduce requires the cluster level to have all static extents.");

  constexpr auto __nblocks_in_cluster =
    _ClusterExts::static_extent(0) * _ClusterExts::static_extent(1) * _ClusterExts::static_extent(2);
  if constexpr (__nblocks_in_cluster == 1)
  {
    return ::cuda::experimental::coop::__reduce_impl(
      ::cuda::std::bool_constant<_Broadcasted>{}, this_block{__group.hierarchy()}, __thread_data, __red_fn);
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
      _Tp __bcast_;
    };

    union _Scratch
    {
      typename _BlockReduce::TempStorage __block_;
      _RootScratch __root_;
    };
    __shared__ _Scratch __scratch;

    const auto __partial = _BlockReduce{__scratch.__block_}.Reduce(__thread_data, __red_fn);
    _Tp __result{};
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   const auto __root_scratch = static_cast<_Scratch*>(::__cluster_map_shared_rank(&__scratch, 0));
                   auto& __partials_root     = __root_scratch->__root_.__partials_;

                   __group.sync_aligned();
                   if (gpu_thread.is_root_rank(this_block{__group.hierarchy()}))
                   {
                     __partials_root[block.rank(__group)] = __partial;
                   }
                   __group.sync_aligned();

                   if (warp.is_root_rank(__group))
                   {
                     this_warp __warp{__group.hierarchy()};
                     const auto __value = (gpu_thread.rank(__warp) < __nblocks_in_cluster)
                                          ? __scratch.__root_.__partials_[gpu_thread.rank(__warp)]
                                          : ::cuda::identity_element<_RedFn, _Tp>();
                     __result           = _RootWarpReduce{__scratch.__root_.__warp_reduce_}.Reduce(__value, __red_fn);
                   }

                   if constexpr (_Broadcasted)
                   {
                     if (gpu_thread.is_root_rank(__group))
                     {
                       __scratch.__root_.__bcast_ = __result;
                     }
                     __group.sync_aligned();
                     __result = __root_scratch->__root_.__bcast_;

                     // Wait until all threads are done reading the result.
                     __group.sync_aligned();
                   }
                 }))

    if constexpr (_Broadcasted)
    {
      return __result;
    }
    else
    {
      return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
    }
  }
}

template <class _Tp, ::cuda::std::size_t _Np>
_CCCL_DEVICE ::cuda::std::array<_Tp, _Np> __reduce_grid_partials;

template <bool _Broadcasted, class _Hierarchy, class _Tp, cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API auto __reduce_impl(
  ::cuda::std::bool_constant<_Broadcasted>, this_grid<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _GridExts = decltype(cluster.extents(grid, __group.hierarchy()));
  static_assert(_GridExts::rank_dynamic() == 0,
                "cuda::coop::reduce requires the grid level to have all static extents.");

  constexpr auto __nclusters_in_grid =
    _GridExts::static_extent(0) * _GridExts::static_extent(1) * _GridExts::static_extent(2);

  this_cluster __cluster{__group.hierarchy()};
  const auto __partial =
    ::cuda::experimental::coop::__reduce_impl(::cuda::std::false_type{}, __cluster, __thread_data, __red_fn);

  if (gpu_thread.is_root_rank(__cluster))
  {
    __reduce_grid_partials<_Tp, __nclusters_in_grid>[cluster.rank(__group)] = __partial.value();
  }
  __group.sync_aligned();

  ::cuda::std::optional<_Tp> __result;
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
    __result =
      ::cuda::experimental::coop::__reduce_impl(::cuda::std::false_type{}, __block, __thread_partials, __red_fn);
  }

  if constexpr (_Broadcasted)
  {
    if (gpu_thread.is_root_rank(__group))
    {
      __reduce_grid_partials<_Tp, __nclusters_in_grid>[0] = *__result;
    }
    __group.sync_aligned();
    const auto __result2 = __reduce_grid_partials<_Tp, __nclusters_in_grid>[0];

    // Wait until all threads are done reading the result.
    __group.sync_aligned();
    return __result2;
  }
  else
  {
    return __result;
  }
}

_CCCL_TEMPLATE(bool _Broadcasted, class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn)
_CCCL_REQUIRES(::cuda::std::is_same_v<thread_level, typename _Group::unit_type>
                 _CCCL_AND ::cuda::std::is_same_v<warp_level, typename _Group::level_type>)
[[nodiscard]] _CCCL_DEVICE_API auto
__reduce_impl(::cuda::std::bool_constant<_Broadcasted>, _Group __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _MappingResult         = typename _Group::__mapping_result_type;
  const auto& __mapping_result = __group.__mapping_result();

  const auto __lane_mask = __mapping_result.lane_mask();
  const auto __lane      = ::cuda::ptx::get_sreg_laneid();
  auto __result          = ::cub::ThreadReduce(__thread_data, __red_fn);

  _CCCL_PRAGMA_UNROLL_FULL()
  for (unsigned __stride = 1; __stride < ::cuda::next_power_of_two(__mapping_result.unit_count()); __stride *= 2)
  {
    const auto __other = ::cuda::experimental::coop::shuffle_down(__group, __result, __stride);
    if (__other.has_value())
    {
      __result = __red_fn(__result, *__other);
    }
  }

  if constexpr (_Broadcasted)
  {
    return ::cuda::device::warp_shuffle_idx(__result, ::cuda::std::countr_zero(__lane_mask.value()), __lane_mask.value())
      .data;
  }
  else
  {
    return (__mapping_result.unit_rank() == 0) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
  }
}

_CCCL_TEMPLATE(bool _Broadcasted, class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn)
_CCCL_REQUIRES(::cuda::std::is_same_v<warp_level, typename _Group::unit_type>
                 _CCCL_AND ::cuda::std::is_same_v<block_level, typename _Group::level_type>)
[[nodiscard]] _CCCL_DEVICE_API auto
__reduce_impl(::cuda::std::bool_constant<_Broadcasted>, _Group __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  constexpr auto __nwarps_in_group = warp.static_count(__group);
  static_assert(__nwarps_in_group != ::cuda::std::dynamic_extent,
                "cuda::coop::reduce requires the group to have statically known size");

  using _WarpReduce = ::cub::WarpReduce<_Tp>;
  struct _AdditionalScratch
  {
    _Tp __partials_[__nwarps_in_group];
    _Tp __bcast_;
  };

  union _Scratch
  {
    typename _WarpReduce::TempStorage __warp_reduce_[__nwarps_in_group];
    _AdditionalScratch __additional_;
  };
  __shared__ _Scratch __scratch;

  const auto __partial = _WarpReduce{__scratch.__warp_reduce_[warp.rank(__group)]}.Reduce(__thread_data, __red_fn);
  __group.sync_aligned();

  this_warp __warp{__group.hierarchy()};
  if (gpu_thread.is_root_rank(__warp))
  {
    __scratch.__additional_.__partials_[warp.rank(__group)] = __partial;
  }
  __group.sync_aligned();

  _Tp __result;
  if (warp.is_root_rank(__group))
  {
    const auto __value = (gpu_thread.rank(__warp) < __nwarps_in_group)
                         ? __scratch.__additional_.__partials_[gpu_thread.rank(__warp)]
                         : ::cuda::identity_element<_RedFn, _Tp>();
    __result           = _WarpReduce{__scratch.__warp_reduce_[0]}.Reduce(__value, __red_fn);
  }

  if constexpr (_Broadcasted)
  {
    if (gpu_thread.is_root_rank(__group))
    {
      __scratch.__additional_.__bcast_ = __result;
    }
    __group.sync_aligned();
    return __scratch.__additional_.__bcast_;
  }
  else
  {
    return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
  }
}

template <class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
reduce(_Group __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  static_assert(gpu_thread.static_count(__group) != ::cuda::std::dynamic_extent,
                "cuda::coop::reduce requires the group to have statically known size");

  _CCCL_ASSERT(gpu_thread.is_part_of(__group), "Only threads that are part of the group can call cudax::coop::reduce");

  return ::cuda::experimental::coop::__reduce_impl(::cuda::std::false_type{}, __group, __thread_data, __red_fn);
}

template <class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API _Tp reduce(broadcasted_t, _Group __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  static_assert(gpu_thread.static_count(__group) != ::cuda::std::dynamic_extent,
                "cuda::coop::reduce requires the group to have statically known size");

  _CCCL_ASSERT(gpu_thread.is_part_of(__group), "Only threads that are part of the group can call cudax::coop::reduce");

  return ::cuda::experimental::coop::__reduce_impl(::cuda::std::true_type{}, __group, __thread_data, __red_fn);
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_CUH
