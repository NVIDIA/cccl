//===----------------------------------------------------------------------===//
//
// Part of CUDA Next in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_NEXT_DETAIL_HIERARCHY_LEVELS
#define _CUDA_NEXT_DETAIL_HIERARCHY_LEVELS

#include "dimensions.cuh"
#include <nv/target>

namespace cuda::experimental
{

// Base type for all hierarchy levels
struct hierarchy_level
{};

namespace hierarchy
{
template <typename Unit, typename Level>
auto __device__ rank(const Unit& = Unit(), const Level& = Level());

template <typename Unit, typename Level>
auto __device__ count(const Unit& = Unit(), const Level& = Level());

template <typename Unit, typename Level>
auto __device__ index(const Unit& = Unit(), const Level& = Level());

template <typename Unit, typename Level>
auto __device__ dims(const Unit& = Unit(), const Level& = Level());
} // namespace hierarchy

namespace detail
{
template <typename Level>
struct dimensions_query;

template <typename Level>
struct dimensions_query
{
  template <typename Unit>
  static auto __device__ rank(const Unit& = Unit())
  {
    return hierarchy::rank<Unit, Level>();
  }

  template <typename Unit>
  static auto __device__ count(const Unit& = Unit())
  {
    return hierarchy::count<Unit, Level>();
  }

  template <typename Unit>
  static auto __device__ index(const Unit& = Unit())
  {
    return hierarchy::index<Unit, Level>();
  }

  template <typename Unit>
  static auto __device__ dims(const Unit& = Unit())
  {
    return hierarchy::dims<Unit, Level>();
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
constexpr bool is_level_allowed = false;
template <typename QueryLevel, typename... Levels>
constexpr bool is_level_allowed<QueryLevel, allowed_levels<Levels...>> =
  ::cuda::std::disjunction_v<::cuda::std::is_same<QueryLevel, Levels>...>;

template <typename L1, typename L2>
constexpr bool can_stack_on_top =
  is_level_allowed<L1, typename L2::allowed_below> || is_level_allowed<L2, typename L1::allowed_above>;

template <typename Unit, typename Level>
constexpr bool legal_unit_for_level =
  can_stack_on_top<Unit, Level> || legal_unit_for_level<Unit, typename Level::allowed_below::default_unit>;

template <typename Unit>
constexpr bool legal_unit_for_level<Unit, void> = false;
} // namespace detail

struct grid_level;
struct cluster_level;
struct block_level;
struct thread_level;

// Types to represent CUDA threads hierarchy levels
struct grid_level
    : public hierarchy_level
    , public detail::dimensions_query<grid_level>
{
  /* All metadata about the hierarchy level goes here including certain forward progress information
       or what adjecent levels are valid in the hierarchy for validation. */
  using product_type  = unsigned long long;
  using allowed_above = allowed_levels<>;
  using allowed_below = allowed_levels<block_level, cluster_level>;
};
_LIBCUDACXX_CPO_ACCESSIBILITY grid_level grid;

struct cluster_level
    : public hierarchy_level
    , public detail::dimensions_query<cluster_level>
{
  using product_type  = unsigned int;
  using allowed_above = allowed_levels<grid_level>;
  using allowed_below = allowed_levels<block_level>;
};
_LIBCUDACXX_CPO_ACCESSIBILITY cluster_level cluster;

struct block_level
    : public hierarchy_level
    , public detail::dimensions_query<block_level>
{
  using product_type  = unsigned int;
  using allowed_above = allowed_levels<grid_level, cluster_level>;
  using allowed_below = allowed_levels<thread_level>;
};
_LIBCUDACXX_CPO_ACCESSIBILITY block_level block;

struct thread_level
    : public hierarchy_level
    , public detail::dimensions_query<thread_level>
{
  using product_type  = unsigned int;
  using allowed_above = allowed_levels<block_level>;
  using allowed_below = allowed_levels<>;
};
_LIBCUDACXX_CPO_ACCESSIBILITY thread_level thread;

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
  static dim3 __device__ dims()
  {
    return dim3(1, 1, 1);
  }

  static dim3 __device__ index()
  {
    return dim3(0, 0, 0);
  }
};

template <>
struct dims_helper<thread_level, block_level>
{
  static dim3 __device__ dims()
  {
    return blockDim;
  }

  static dim3 __device__ index()
  {
    return threadIdx;
  }
};

template <>
struct dims_helper<block_level, cluster_level>
{
  static dim3 __device__ dims()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterDim();), (return dim3(1, 1, 1);));
  }

  static dim3 __device__ index()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterRelativeBlockIdx();), (return dim3(0, 0, 0);));
  }
};

template <>
struct dims_helper<block_level, grid_level>
{
  static dim3 __device__ dims()
  {
    return gridDim;
  }

  static dim3 __device__ index()
  {
    return blockIdx;
  }
};

template <>
struct dims_helper<cluster_level, grid_level>
{
  static dim3 __device__ dims()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterGridDimInClusters();), (return gridDim;));
  }

  static dim3 __device__ index()
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __clusterIdx();), (return dim3(0, 0, 0);));
  }
};

template <typename Unit, typename Level>
auto __device__ dims_impl()
{
  if constexpr (::cuda::std::is_same_v<Unit, Level> || can_stack_on_top<Unit, Level>)
  {
    return dim3_to_query_result(dims_helper<Unit, Level>::dims());
  }
  else
  {
    using SplitLevel = typename Level::allowed_below::default_unit;
    return dims_product<typename Level::product_type>(dims_impl<SplitLevel, Level>(), dims_impl<Unit, SplitLevel>());
  }
}

template <typename Unit, typename Level>
auto __device__ index_impl()
{
  if constexpr (::cuda::std::is_same_v<Unit, Level> || detail::can_stack_on_top<Unit, Level>)
  {
    return dim3_to_query_result(dims_helper<Unit, Level>::index());
  }
  else
  {
    using SplitLevel = typename Level::allowed_below::default_unit;
    return dims_sum<typename Level::product_type>(
      dims_product<typename Level::product_type>(index_impl<SplitLevel, Level>(), dims_impl<Unit, SplitLevel>()),
      index_impl<Unit, SplitLevel>());
  }
}
} // namespace detail

namespace hierarchy
{
template <typename Unit, typename Level>
auto __device__ count(const Unit&, const Level&)
{
  static_assert(detail::legal_unit_for_level<Unit, Level>);
  auto d = detail::dims_impl<Unit, Level>();
  return d.extent(0) * d.extent(1) * d.extent(2);
}

template <typename Unit, typename Level>
auto __device__ rank(const Unit&, const Level&)
{
  static_assert(detail::legal_unit_for_level<Unit, Level>);
  if constexpr (detail::can_stack_on_top<Unit, Level>)
  {
    return detail::index_to_linear<typename Level::product_type>(
      detail::index_impl<Unit, Level>(), detail::dims_impl<Unit, Level>());
  }
  else
  {
    /* Its interesting that there is a need for else here, but using the above in all cases would result in
        a different numbering scheme, where adjacent ranks in lower level would not be adjacent in this level */
    using SplitLevel = typename Level::allowed_below::default_unit;
    return rank<SplitLevel, Level>() * count<Unit, SplitLevel>() + rank<Unit, SplitLevel>();
  }
}

template <typename Unit, typename Level>
auto __device__ dims(const Unit&, const Level&)
{
  static_assert(detail::legal_unit_for_level<Unit, Level>);
  return hierarchy_query_result(detail::dims_impl<Unit, Level>());
}

template <typename Unit, typename Level>
auto __device__ index(const Unit&, const Level&)
{
  static_assert(detail::legal_unit_for_level<Unit, Level>);
  return hierarchy_query_result(detail::index_impl<Unit, Level>());
}
} // namespace hierarchy
} // namespace cuda::experimental
#endif
