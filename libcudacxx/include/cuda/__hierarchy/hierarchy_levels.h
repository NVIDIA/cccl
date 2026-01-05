//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_HIERARCHY_LEVELS_H
#define _CUDA___HIERARCHY_HIERARCHY_LEVELS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/dimensions.h>
#  include <cuda/std/__type_traits/type_list.h>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace hierarchy
{
template <class _Unit, class _Level>
_CCCL_DEVICE auto rank(const _Unit& = _Unit(), const _Level& = _Level());

template <class _Unit, class _Level>
_CCCL_DEVICE auto count(const _Unit& = _Unit(), const _Level& = _Level());

template <class _Unit, class _Level>
_CCCL_DEVICE auto index(const _Unit& = _Unit(), const _Level& = _Level());

template <class _Unit, class _Level>
_CCCL_DEVICE auto extents(const _Unit& = _Unit(), const _Level& = _Level());
} // namespace hierarchy

namespace __detail
{
template <class _Level>
struct __dimensions_query
{
  template <class _Unit>
  /* [[nodiscard]] */ _CCCL_DEVICE static auto rank(const _Unit& = _Unit())
  {
    return hierarchy::rank<_Unit, _Level>();
  }

  template <class _Unit>
  /* [[nodiscard]] */ _CCCL_DEVICE static auto count(const _Unit& = _Unit())
  {
    return hierarchy::count<_Unit, _Level>();
  }

  template <class _Unit>
  /* [[nodiscard]] */ _CCCL_DEVICE static auto index(const _Unit& = _Unit())
  {
    return hierarchy::index<_Unit, _Level>();
  }

  template <class _Unit>
  /* [[nodiscard]] */ _CCCL_DEVICE static auto extents(const _Unit& = _Unit())
  {
    return hierarchy::extents<_Unit, _Level>();
  }
};
} // namespace __detail

// Struct to represent levels allowed below or above a certain level,
//  used for hierarchy sorting, validation and for hierarchy traversal
template <typename... _Levels>
struct allowed_levels
{
  using default_unit = ::cuda::std::__type_index_c<0, _Levels..., void>;
};

namespace __detail
{
template <typename LevelType>
using __default_unit_below = typename LevelType::allowed_below::default_unit;

template <class _QueryLevel, class _AllowedLevels>
inline constexpr bool __is_level_allowed = false;

template <class _QueryLevel, class... _Levels>
inline constexpr bool __is_level_allowed<_QueryLevel, allowed_levels<_Levels...>> =
  ::cuda::std::disjunction_v<::cuda::std::is_same<_QueryLevel, _Levels>...>;

template <class _L1, class _L2>
inline constexpr bool __can_rhs_stack_on_lhs =
  __is_level_allowed<_L1, typename _L2::allowed_below> || __is_level_allowed<_L2, typename _L1::allowed_above>;

template <class _Unit, class _Level>
inline constexpr bool __legal_unit_for_level =
  __can_rhs_stack_on_lhs<_Unit, _Level> || __legal_unit_for_level<_Unit, __default_unit_below<_Level>>;

template <class _Unit>
inline constexpr bool __legal_unit_for_level<_Unit, void> = false;
} // namespace __detail

template <typename _Level>
constexpr bool is_core_cuda_hierarchy_level =
  ::cuda::std::is_same_v<_Level, grid_level> || ::cuda::std::is_same_v<_Level, cluster_level>
  || ::cuda::std::is_same_v<_Level, block_level> || ::cuda::std::is_same_v<_Level, thread_level>;

namespace __detail
{
template <typename _Unit, typename _Level>
struct __dims_helper;

#  if _CCCL_CUDA_COMPILATION()
template <typename _Level>
struct __dims_helper<_Level, _Level>
{
  [[nodiscard]] _CCCL_DEVICE static ::dim3 extents()
  {
    return ::dim3(1, 1, 1);
  }

  [[nodiscard]] _CCCL_DEVICE static ::dim3 index()
  {
    return ::dim3(0, 0, 0);
  }
};

template <>
struct __dims_helper<thread_level, block_level>
{
  [[nodiscard]] _CCCL_DEVICE static ::dim3 extents()
  {
    return ::blockDim;
  }

  [[nodiscard]] _CCCL_DEVICE static ::dim3 index()
  {
    return ::threadIdx;
  }
};

template <>
struct __dims_helper<block_level, cluster_level>
{
  [[nodiscard]] _CCCL_DEVICE static ::dim3 extents()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterDim();), (return ::dim3(1, 1, 1);));
  }

  [[nodiscard]] _CCCL_DEVICE static ::dim3 index()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterRelativeBlockIdx();), (return ::dim3(0, 0, 0);));
  }
};

template <>
struct __dims_helper<block_level, grid_level>
{
  [[nodiscard]] _CCCL_DEVICE static ::dim3 extents()
  {
    return ::gridDim;
  }

  [[nodiscard]] _CCCL_DEVICE static ::dim3 index()
  {
    return ::blockIdx;
  }
};

template <>
struct __dims_helper<cluster_level, grid_level>
{
  [[nodiscard]] _CCCL_DEVICE static ::dim3 extents()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterGridDimInClusters();), (return gridDim;));
  }

  [[nodiscard]] _CCCL_DEVICE static ::dim3 index()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterIdx();), (return ::dim3(0, 0, 0);));
  }
};
#  endif // _CCCL_CUDA_COMPILATION()

// Seems like a compiler bug, where NODISCARD is marked as ignored due to void
// return type, while its not possible to ever have void return type here
template <typename _Unit, typename _Level>
/* [[nodiscard]] */ _CCCL_DEVICE auto __extents_impl()
{
  if constexpr (::cuda::std::is_same_v<_Unit, _Level> || __can_rhs_stack_on_lhs<_Unit, _Level>)
  {
    return ::cuda::__detail::__dim3_to_dims(__dims_helper<_Unit, _Level>::extents());
  }
  else
  {
    using _SplitLevel = __detail::__default_unit_below<_Level>;
    return ::cuda::__detail::__dims_product<typename _Level::product_type>(
      __extents_impl<_SplitLevel, _Level>(), __extents_impl<_Unit, _SplitLevel>());
  }
}

template <typename _Unit, typename _Level>
/* [[nodiscard]] */ _CCCL_DEVICE auto __index_impl()
{
  if constexpr (::cuda::std::is_same_v<_Unit, _Level> || __detail::__can_rhs_stack_on_lhs<_Unit, _Level>)
  {
    return ::cuda::__detail::__dim3_to_dims(__dims_helper<_Unit, _Level>::index());
  }
  else
  {
    using _SplitLevel = __detail::__default_unit_below<_Level>;
    return ::cuda::__detail::__dims_sum<typename _Level::product_type>(
      ::cuda::__detail::__dims_product<typename _Level::product_type>(
        __index_impl<_SplitLevel, _Level>(), __extents_impl<_Unit, _SplitLevel>()),
      __index_impl<_Unit, _SplitLevel>());
  }
}
} // namespace __detail

namespace hierarchy
{
/**
 * @brief Counts the number of entities in a CUDA hierarchy level
 *
 * Returns how many instances of Unit are in Level.
 * Unit and Level need to be core CUDA hierarchy levels, for example grid_level
 * or block_level. This function is also available as a level type member
 * function, in that case it only takes a unit argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda;
 *
 * __global__ void kernel()
 * {
 *     // Can be called with the instances of level types
 *     int num_threads_in_block = hierarchy::count(thread, block);
 *     int num_blocks_in_grid = grid.count(block);
 *
 *     // Or using the level types as template arguments
 *     int num_threads_in_grid = hierarchy::count<thread_level, grid_level>();
 * }
 * @endcode
 * @par
 *
 * @tparam Unit
 *  Specifies what should be counted
 *
 * @tparam Level
 *  Specifies at what CUDA hierarchy level the count should happen
 */
template <typename _Unit, typename _Level>
_CCCL_DEVICE auto count(const _Unit&, const _Level&)
{
  static_assert(__detail::__legal_unit_for_level<_Unit, _Level>);
  auto __dims = __detail::__extents_impl<_Unit, _Level>();
  return __dims.extent(0) * __dims.extent(1) * __dims.extent(2);
}

/**
 * @brief Ranks an entity the calling thread belongs to in a CUDA hierarchy
 * level
 *
 * Returns a unique numeric rank within Level of the Unit that the calling
 * thread belongs to. Returned rank is always in range 0 to count - 1. Unit and
 * Level need to be core CUDA hierarchy levels, for example grid_level or
 * block_level. This function is also available as a level type member function,
 * in that case it only takes a unit argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda;
 *
 * __global__ void kernel()
 * {
 *     // Can be called with the instances of level types
 *     int thread_rank_in_block = hierarchy::rank(thread, block);
 *     int block_rank_in_grid = grid.rank(block);
 *
 *     // Or using the level types as template arguments
 *     int thread_rank_in_grid = hierarchy::rank<thread_level, grid_level>();
 * }
 * @endcode
 * @par
 *
 * @tparam Unit
 *  Specifies the entity that the rank is requested for
 *
 * @tparam Level
 *  Specifies at what CUDA hierarchy level the rank is requested
 */
template <typename _Unit, typename _Level>
_CCCL_DEVICE auto rank(const _Unit&, const _Level&)
{
  static_assert(__detail::__legal_unit_for_level<_Unit, _Level>);
  if constexpr (__detail::__can_rhs_stack_on_lhs<_Unit, _Level>)
  {
    return ::cuda::__detail::__index_to_linear<typename _Level::product_type>(
      __detail::__index_impl<_Unit, _Level>(), __detail::__extents_impl<_Unit, _Level>());
  }
  else
  {
    /* Its interesting that there is a need for else here, but using the above
       in all cases would result in a different numbering scheme, where adjacent
       ranks in lower level would not be adjacent in this level */
    using _SplitLevel = __detail::__default_unit_below<_Level>;
    return rank<_SplitLevel, _Level>() * count<_Unit, _SplitLevel>() + rank<_Unit, _SplitLevel>();
  }
}

/**
 * @brief Returns extents of multi-dimensional index space of a CUDA hierarchy
 * level
 *
 * Returned extents are in line with intrinsic CUDA dimensions vectors like
 * blockDim and gridDim, extentded to more unit/level combinations. Returns
 * hierarchy_query_result, which can be used like cuda::std::extents or dim3.
 * Unit and Level need to be a core CUDA hierarchy levels, for example
 * grid_level or block_level. This function is also available as a level type
 * member function, in that case it only takes a unit argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 * #include <cassert>
 *
 * using namespace cuda;
 *
 * __global__ void kernel()
 * {
 *     // Can be called with the instances of level types
 *     auto block_dims = hierarchy::extents(thread, block);
 *     assert(block_dims == blockDim);
 *     auto grid_dims = grid.extents(block);
 *     assert(grid_dims == gridDim);
 *
 *     // Or using the level types as template arguments
 *     auto grid_dims_in_threads = hierarchy::extents<thread_level,
 * grid_level>();
 * }
 * @endcode
 * @par
 *
 * @tparam Unit
 *  Specifies the unit of the index space
 *
 * @tparam Level
 *  Specifies at what CUDA hierarchy level the extents are requested
 */
template <typename _Unit, typename _Level>
_CCCL_DEVICE auto extents(const _Unit&, const _Level&)
{
  static_assert(__detail::__legal_unit_for_level<_Unit, _Level>);
  return ::cuda::__detail::__convert_to_query_result(__detail::__extents_impl<_Unit, _Level>());
}

/**
 * @brief Returns a 3-dimensional index of an entity the calling thread belongs
 * to in a CUDA hierarchy level
 *
 * Returned index is in line with intrinsic CUDA indexing like threadIdx and
 * blockIdx, extentded to more unit/level combinations. Returns a
 * hierarchy_query_result object, which can be used like cuda::std::extents or
 * dim3. Unit and Level need to be a core CUDA hierarchy levels, for example
 * grid_level or block_level. This function is also available as a level type
 * member function, in that case it only takes a unit argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 * #include <cassert>
 *
 * using namespace cuda;
 *
 * __global__ void kernel()
 * {
 *     // Can be called with the instances of level types
 *     auto thread_index_in_block = hierarchy::index(thread, block);
 *     assert(thread_index_in_block == threadIdx);
 *     auto block_index_in_grid = grid.index(block);
 *     assert(block_index_in_grid == blockIdx);
 *
 *     // Or using the level types as template arguments
 *     auto thread_index_in_grid = hierarchy::index<thread_level, grid_level>();
 * }
 * @endcode
 * @par
 *
 * @tparam Unit
 *  Specifies the entity that the index is requested for
 *
 * @tparam Level
 *  Specifies at what hierarchy level the index is requested
 */
template <typename _Unit, typename _Level>
_CCCL_DEVICE auto index(const _Unit&, const _Level&)
{
  static_assert(__detail::__legal_unit_for_level<_Unit, _Level>);
  return ::cuda::__detail::__convert_to_query_result(__detail::__index_impl<_Unit, _Level>());
}
} // namespace hierarchy
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_HIERARCHY_LEVELS_H
