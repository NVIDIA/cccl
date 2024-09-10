//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__HIERARCHY_HIERARCHY_LEVELS
#define _CUDAX__HIERARCHY_HIERARCHY_LEVELS

#include <cuda/experimental/__hierarchy/dimensions.cuh>

#include <nv/target>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

namespace hierarchy
{
template <typename Unit, typename Level>
_CCCL_DEVICE auto rank(const Unit& = Unit(), const Level& = Level());

template <typename Unit, typename Level>
_CCCL_DEVICE auto count(const Unit& = Unit(), const Level& = Level());

template <typename Unit, typename Level>
_CCCL_DEVICE auto index(const Unit& = Unit(), const Level& = Level());

template <typename Unit, typename Level>
_CCCL_DEVICE auto extents(const Unit& = Unit(), const Level& = Level());
} // namespace hierarchy

namespace detail
{
template <typename Level>
struct dimensions_query;

template <typename Level>
struct dimensions_query
{
  template <typename Unit>
  /* _CCCL_NODISCARD */ _CCCL_DEVICE static auto rank(const Unit& = Unit())
  {
    return hierarchy::rank<Unit, Level>();
  }

  template <typename Unit>
  /* _CCCL_NODISCARD */ _CCCL_DEVICE static auto count(const Unit& = Unit())
  {
    return hierarchy::count<Unit, Level>();
  }

  template <typename Unit>
  /* _CCCL_NODISCARD */ _CCCL_DEVICE static auto index(const Unit& = Unit())
  {
    return hierarchy::index<Unit, Level>();
  }

  template <typename Unit>
  /* _CCCL_NODISCARD */ _CCCL_DEVICE static auto extents(const Unit& = Unit())
  {
    return hierarchy::extents<Unit, Level>();
  }
};

template <typename L1, typename... Levels>
struct get_first_level_type
{
  using type = L1;
};
} // namespace detail

// Struct to represent levels allowed below or above a certain level,
//  used for hierarchy sorting, validation and for hierarchy traversal
template <typename... Levels>
struct allowed_levels
{
  using default_unit = typename detail::get_first_level_type<Levels...>::type;
};

template <>
struct allowed_levels<>
{
  using default_unit = void;
};

namespace detail
{
template <typename QueryLevel, typename AllowedLevels>
_LIBCUDACXX_INLINE_VAR constexpr bool is_level_allowed = false;

template <typename QueryLevel, typename... Levels>
_LIBCUDACXX_INLINE_VAR constexpr bool is_level_allowed<QueryLevel, allowed_levels<Levels...>> =
  ::cuda::std::disjunction_v<::cuda::std::is_same<QueryLevel, Levels>...>;

template <typename L1, typename L2>
_LIBCUDACXX_INLINE_VAR constexpr bool can_stack_on_top =
  is_level_allowed<L1, typename L2::allowed_below> || is_level_allowed<L2, typename L1::allowed_above>;

template <typename Unit, typename Level>
_LIBCUDACXX_INLINE_VAR constexpr bool legal_unit_for_level =
  can_stack_on_top<Unit, Level> || legal_unit_for_level<Unit, typename Level::allowed_below::default_unit>;

template <typename Unit>
_LIBCUDACXX_INLINE_VAR constexpr bool legal_unit_for_level<Unit, void> = false;
} // namespace detail

// Base type for all hierarchy levels
struct hierarchy_level
{};

struct grid_level;
struct cluster_level;
struct block_level;
struct thread_level;

/*
  Types to represent CUDA threads hierarchy levels
  All metadata about the hierarchy level goes here including certain forward progress information
  or what adjecent levels are valid in the hierarchy for validation.
*/

/**
 * @brief Type representing the grid level in CUDA thread hierarchy
 *
 * This type can be used in hierarchy queries to refer to the
 * grid level or to get that level from the hierarchy.
 * There is a constexpr variable of this type available for convenience
 * named grid.
 */
struct grid_level
    : public hierarchy_level
    , public detail::dimensions_query<grid_level>
{
  using product_type  = unsigned long long;
  using allowed_above = allowed_levels<>;
  using allowed_below = allowed_levels<block_level, cluster_level>;
};
_CCCL_GLOBAL_CONSTANT grid_level grid;

/**
 * @brief Type representing the cluster level in CUDA thread hierarchy
 *
 * This type can be used in hierarchy queries to refer to the
 * cluster level or to get that level from the hierarchy.
 * There is a constexpr variable of this type available for convenience
 * named cluster.
 */
struct cluster_level
    : public hierarchy_level
    , public detail::dimensions_query<cluster_level>
{
  using product_type  = unsigned int;
  using allowed_above = allowed_levels<grid_level>;
  using allowed_below = allowed_levels<block_level>;
};
_CCCL_GLOBAL_CONSTANT cluster_level cluster;

/**
 * @brief Type representing the block level in CUDA thread hierarchy
 *
 * This type can be used in hierarchy queries to refer to the
 * block level or to get that level from the hierarchy.
 * There is a constexpr variable of this type available for convenience
 * named block.
 */
struct block_level
    : public hierarchy_level
    , public detail::dimensions_query<block_level>
{
  using product_type  = unsigned int;
  using allowed_above = allowed_levels<grid_level, cluster_level>;
  using allowed_below = allowed_levels<thread_level>;
};
_CCCL_GLOBAL_CONSTANT block_level block;

/**
 * @brief Type representing the thread level in CUDA thread hierarchy
 *
 * This type can be used in hierarchy queries to specify threads as a
 * unit of the query.
 * There is a constexpr variable of this type available for convenience
 * named thread.
 */
struct thread_level
    : public hierarchy_level
    , public detail::dimensions_query<thread_level>
{
  using product_type  = unsigned int;
  using allowed_above = allowed_levels<block_level>;
  using allowed_below = allowed_levels<>;
};
_CCCL_GLOBAL_CONSTANT thread_level thread;

template <typename Level>
constexpr bool is_core_cuda_hierarchy_level =
  ::cuda::std::is_same_v<Level, grid_level> || ::cuda::std::is_same_v<Level, cluster_level>
  || ::cuda::std::is_same_v<Level, block_level> || ::cuda::std::is_same_v<Level, thread_level>;

namespace detail
{

template <typename Unit, typename Level>
struct dims_helper;

template <typename Level>
struct dims_helper<Level, Level>
{
  _CCCL_NODISCARD _CCCL_DEVICE static dim3 extents()
  {
    return dim3(1, 1, 1);
  }

  _CCCL_NODISCARD _CCCL_DEVICE static dim3 index()
  {
    return dim3(0, 0, 0);
  }
};

template <>
struct dims_helper<thread_level, block_level>
{
  _CCCL_NODISCARD _CCCL_DEVICE static dim3 extents()
  {
    return blockDim;
  }

  _CCCL_NODISCARD _CCCL_DEVICE static dim3 index()
  {
    return threadIdx;
  }
};

template <>
struct dims_helper<block_level, cluster_level>
{
  _CCCL_NODISCARD _CCCL_DEVICE static dim3 extents()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterDim();), (return dim3(1, 1, 1);));
  }

  _CCCL_NODISCARD _CCCL_DEVICE static dim3 index()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterRelativeBlockIdx();), (return dim3(0, 0, 0);));
  }
};

template <>
struct dims_helper<block_level, grid_level>
{
  _CCCL_NODISCARD _CCCL_DEVICE static dim3 extents()
  {
    return gridDim;
  }

  _CCCL_NODISCARD _CCCL_DEVICE static dim3 index()
  {
    return blockIdx;
  }
};

template <>
struct dims_helper<cluster_level, grid_level>
{
  _CCCL_NODISCARD _CCCL_DEVICE static dim3 extents()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterGridDimInClusters();), (return gridDim;));
  }

  _CCCL_NODISCARD _CCCL_DEVICE static dim3 index()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterIdx();), (return dim3(0, 0, 0);));
  }
};

// Seems like a compiler bug, where NODISCARD is marked as ignored due to void return type,
// while its not possible to ever have void return type here
template <typename Unit, typename Level>
/* _CCCL_NODISCARD */ _CCCL_DEVICE auto extents_impl()
{
  if constexpr (::cuda::std::is_same_v<Unit, Level> || can_stack_on_top<Unit, Level>)
  {
    return dim3_to_dims(dims_helper<Unit, Level>::extents());
  }
  else
  {
    using SplitLevel = typename Level::allowed_below::default_unit;
    return dims_product<typename Level::product_type>(
      extents_impl<SplitLevel, Level>(), extents_impl<Unit, SplitLevel>());
  }
  _CCCL_UNREACHABLE();
}

template <typename Unit, typename Level>
/* _CCCL_NODISCARD */ _CCCL_DEVICE auto index_impl()
{
  if constexpr (::cuda::std::is_same_v<Unit, Level> || detail::can_stack_on_top<Unit, Level>)
  {
    return dim3_to_dims(dims_helper<Unit, Level>::index());
  }
  else
  {
    using SplitLevel = typename Level::allowed_below::default_unit;
    return dims_sum<typename Level::product_type>(
      dims_product<typename Level::product_type>(index_impl<SplitLevel, Level>(), extents_impl<Unit, SplitLevel>()),
      index_impl<Unit, SplitLevel>());
  }
  _CCCL_UNREACHABLE();
}
} // namespace detail

namespace hierarchy
{

/**
 * @brief Counts the number of entities in a CUDA hierarchy level
 *
 * Returns how many instances of Unit are in Level.
 * Unit and Level need to be core CUDA hierarchy levels, for example grid_level or block_level.
 * This function is also available as a level type member function, in that case it only takes
 * a unit argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda::experimental;
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
template <typename Unit, typename Level>
_CCCL_DEVICE auto count(const Unit&, const Level&)
{
  static_assert(detail::legal_unit_for_level<Unit, Level>);
  auto d = detail::extents_impl<Unit, Level>();
  return d.extent(0) * d.extent(1) * d.extent(2);
}

/**
 * @brief Ranks an entity the calling thread belongs to in a CUDA hierarchy level
 *
 * Returns a unique numeric rank within Level of the Unit that the calling thread belongs to.
 * Returned rank is always in range 0 to count - 1.
 * Unit and Level need to be core CUDA hierarchy levels, for example grid_level or block_level.
 * This function is also available as a level type member function, in that case it only takes
 * a unit argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda::experimental;
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
template <typename Unit, typename Level>
_CCCL_DEVICE auto rank(const Unit&, const Level&)
{
  static_assert(detail::legal_unit_for_level<Unit, Level>);
  if constexpr (detail::can_stack_on_top<Unit, Level>)
  {
    return detail::index_to_linear<typename Level::product_type>(
      detail::index_impl<Unit, Level>(), detail::extents_impl<Unit, Level>());
  }
  else
  {
    /* Its interesting that there is a need for else here, but using the above in all cases would result in
        a different numbering scheme, where adjacent ranks in lower level would not be adjacent in this level */
    using SplitLevel = typename Level::allowed_below::default_unit;
    return rank<SplitLevel, Level>() * count<Unit, SplitLevel>() + rank<Unit, SplitLevel>();
  }
}

/**
 * @brief Returns extents of multi-dimensional index space of a CUDA hierarchy level
 *
 * Returned extents are in line with intrinsic CUDA dimensions vectors like blockDim and gridDim,
 * extentded to more unit/level combinations. Returns hierarchy_query_result, which can be used
 * like cuda::std::extents or dim3.
 * Unit and Level need to be a core CUDA hierarchy levels, for example grid_level or block_level.
 * This function is also available as a level type member function, in that case it only takes
 * a unit argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 * #include <cassert>
 *
 * using namespace cuda::experimental;
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
 *     auto grid_dims_in_threads = hierarchy::extents<thread_level, grid_level>();
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
template <typename Unit, typename Level>
_CCCL_DEVICE auto extents(const Unit&, const Level&)
{
  static_assert(detail::legal_unit_for_level<Unit, Level>);
  return hierarchy_query_result(detail::extents_impl<Unit, Level>());
}

/**
 * @brief Returns a 3-dimensional index of an entity the calling thread belongs to in a CUDA hierarchy level
 *
 * Returned index is in line with intrinsic CUDA indexing like threadIdx and blockIdx,
 * extentded to more unit/level combinations. Returns a hierarchy_query_result object, which can be used
 * like cuda::std::extents or dim3.
 * Unit and Level need to be a core CUDA hierarchy levels, for example grid_level or block_level.
 * This function is also available as a level type member function, in that case it only takes
 * a unit argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 * #include <cassert>
 *
 * using namespace cuda::experimental;
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
template <typename Unit, typename Level>
_CCCL_DEVICE auto index(const Unit&, const Level&)
{
  static_assert(detail::legal_unit_for_level<Unit, Level>);
  return hierarchy_query_result(detail::index_impl<Unit, Level>());
}
} // namespace hierarchy
} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017
#endif // _CUDAX__HIERARCHY_HIERARCHY_LEVELS
