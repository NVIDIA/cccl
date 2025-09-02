//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__HIERARCHY_HIERARCHY_DIMENSIONS_CUH
#define _CUDAX__HIERARCHY_HIERARCHY_DIMENSIONS_CUH

#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/span>
#include <cuda/std/tuple>

#include <cuda/experimental/__hierarchy/level_dimensions.cuh>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

/* TODO right now operator stacking can end up with a wrong unit, we could use below type, but we would need an explicit
 thread_level inserter
struct unknown_unit : public hierarchy_level
{
  using product_type  = unsigned int;
  using allowed_above = allowed_levels<>;
  using allowed_below = allowed_levels<>;
};
*/

namespace __detail
{
template <typename _Level>
[[nodiscard]] _CCCL_API constexpr auto __as_level(_Level __l) noexcept -> _Level
{
  return __l;
}

template <typename _LevelFn>
[[nodiscard]] _CCCL_API constexpr auto __as_level(_LevelFn* __fn) noexcept -> decltype(__fn())
{
  return {};
}
} // namespace __detail

template <class _Level>
using __level_type_of = typename _Level::level_type;

template <typename BottomUnit = thread_level, typename... Levels>
struct hierarchy_dimensions;

namespace __detail
{
// Function to sometimes convince the compiler something is a constexpr and not really accessing runtime storage
// Mostly a work around for what was addressed in P2280 (c++23) by leveraging the argumentless constructor of extents
template <typename T, size_t... Extents>
[[nodiscard]] _CCCL_API constexpr auto fool_compiler(const dimensions<T, Extents...>& ex)
{
  if constexpr (dimensions<T, Extents...>::rank_dynamic() == 0)
  {
    return dimensions<T, Extents...>();
  }
  else
  {
    return ex;
  }
  _CCCL_UNREACHABLE();
}

template <typename QueryLevel, typename Hierarchy>
struct has_level_helper;

template <typename QueryLevel, typename Unit, typename... Levels>
struct has_level_helper<QueryLevel, hierarchy_dimensions<Unit, Levels...>>
    : public ::cuda::std::__fold_or<::cuda::std::is_same_v<QueryLevel, __level_type_of<Levels>>...>
{};

// Is this needed?
template <typename QueryLevel, typename... Levels>
struct has_level_helper<QueryLevel, hierarchy_dimensions<Levels...>>
    : public ::cuda::std::__fold_or<::cuda::std::is_same_v<QueryLevel, __level_type_of<Levels>>...>
{};

template <typename QueryLevel, typename Hierarchy>
struct has_unit
{};

template <typename QueryLevel, typename Unit, typename... Levels>
struct has_unit<QueryLevel, hierarchy_dimensions<Unit, Levels...>> : ::cuda::std::is_same<QueryLevel, Unit>
{};

template <typename QueryLevel>
struct get_level_helper
{
  template <typename TopLevel, typename... Levels>
  [[nodiscard]] _CCCL_API constexpr auto& operator()(const TopLevel& top, const Levels&... levels)
  {
    if constexpr (::cuda::std::is_same_v<QueryLevel, __level_type_of<TopLevel>>)
    {
      return top;
    }
    else
    {
      return (*this)(levels...);
    }
    _CCCL_UNREACHABLE();
  }
};
} // namespace __detail

template <typename QueryLevel, typename Hierarchy>
inline constexpr bool has_level = __detail::has_level_helper<QueryLevel, ::cuda::std::remove_cvref_t<Hierarchy>>::value;

template <typename QueryLevel, typename Hierarchy>
inline constexpr bool has_level_or_unit =
  __detail::has_level_helper<QueryLevel, ::cuda::std::remove_cvref_t<Hierarchy>>::value
  || __detail::has_unit<QueryLevel, ::cuda::std::remove_cvref_t<Hierarchy>>::value;

namespace __detail
{
template <typename... Levels>
struct can_stack_checker
{
  template <typename... LevelsShifted>
  using can_stack = ::cuda::std::__fold_and<__detail::can_rhs_stack_on_lhs<LevelsShifted, Levels>...>;
};

template <typename LUnit, typename L1, typename... Levels>
inline constexpr bool __can_stack =
  can_stack_checker<__level_type_of<L1>,
                    __level_type_of<Levels>...>::template can_stack<__level_type_of<Levels>..., LUnit>::value;

template <size_t... _Id>
_CCCL_API constexpr auto __reverse_indices(::cuda::std::index_sequence<_Id...>) noexcept
{
  return ::cuda::std::index_sequence<(sizeof...(_Id) - 1 - _Id)...>();
}

template <typename LUnit, bool Reversed = false>
struct __make_hierarchy
{
  template <class Levels, size_t... _Ids>
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto
  __apply_reverse(const Levels& ls, ::cuda::std::index_sequence<_Ids...>) noexcept
  {
    return __make_hierarchy<LUnit, true>()(::cuda::std::get<_Ids>(ls)...);
  }

  template <typename... Levels>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const Levels&... ls) const noexcept
  {
    using UnitOrDefault = ::cuda::std::conditional_t<
      ::cuda::std::is_same_v<void, LUnit>,
      __default_unit_below<::cuda::std::__type_index_c<sizeof...(Levels) - 1, __level_type_of<Levels>...>>,
      LUnit>;
    if constexpr (__can_stack<UnitOrDefault, Levels...>)
    {
      return hierarchy_dimensions(UnitOrDefault{}, ls...);
    }
    else if constexpr (!Reversed)
    {
      return __apply_reverse(::cuda::std::tie(ls...), __reverse_indices(::cuda::std::index_sequence_for<Levels...>()));
    }
    else
    {
      static_assert(__can_stack<UnitOrDefault, Levels...>,
                    "Provided levels can't create a valid hierarchy when stacked in the provided order or reversed");
    }
    _CCCL_UNREACHABLE();
  }
};

template <typename LUnit>
[[nodiscard]] _CCCL_API constexpr auto get_levels_range_end() noexcept
{
  return ::cuda::std::make_tuple();
}

// Find LUnit in Levels... and discard the rest
// maybe_unused needed for MSVC
template <typename LUnit, typename LDims, typename... Levels>
[[nodiscard]] _CCCL_API constexpr auto
get_levels_range_end(const LDims& l, [[maybe_unused]] const Levels&... levels) noexcept
{
  if constexpr (::cuda::std::is_same_v<LUnit, __level_type_of<LDims>>)
  {
    return ::cuda::std::make_tuple();
  }
  else
  {
    return ::cuda::std::tuple_cat(::cuda::std::tie(l), get_levels_range_end<LUnit>(levels...));
  }
}

// Find the LTop in Levels... and discard the preceding ones
template <typename LTop, typename LUnit, typename LTopDims, typename... Levels>
[[nodiscard]] _CCCL_API constexpr auto get_levels_range_start(const LTopDims& ltop, const Levels&... levels) noexcept
{
  if constexpr (::cuda::std::is_same_v<LTop, __level_type_of<LTopDims>>)
  {
    return get_levels_range_end<LUnit>(ltop, levels...);
  }
  else
  {
    return get_levels_range_start<LTop, LUnit>(levels...);
  }
}

// Creates a new hierarchy from Levels... cutting out levels between LTop and LUnit
template <typename LTop, typename LUnit, typename... Levels>
[[nodiscard]] _CCCL_API constexpr auto get_levels_range(const Levels&... levels) noexcept
{
  return get_levels_range_start<LTop, LUnit>(levels...);
}

template <typename T, size_t... Extents, size_t... Ids>
[[nodiscard]] _CCCL_API constexpr auto
dims_to_count_helper(const dimensions<T, Extents...>& ex, ::cuda::std::index_sequence<Ids...>)
{
  return (ex.extent(Ids) * ...);
}

template <typename T, size_t... Extents>
[[nodiscard]] _CCCL_API constexpr auto dims_to_count(const dimensions<T, Extents...>& dims) noexcept
{
  return dims_to_count_helper(dims, ::cuda::std::make_index_sequence<sizeof...(Extents)>{});
}

template <typename... Levels>
[[nodiscard]] _CCCL_API constexpr auto get_level_counts_helper(const Levels&... ls)
{
  return ::cuda::std::make_tuple(dims_to_count(ls.dims)...);
}

template <typename Unit, typename Level, typename Dims>
[[nodiscard]] _CCCL_API constexpr auto replace_with_intrinsics_or_constexpr(const Dims& dims)
{
  if constexpr (is_core_cuda_hierarchy_level<Level> && is_core_cuda_hierarchy_level<Unit> && Dims::rank_dynamic() != 0)
  {
    // We replace hierarchy access with CUDA intrinsic to enable compiler optimizations, its ok for the prototype,
    // but might lead to unexpected results and should be eventually addressed at the API level
    // TODO with device side launch we should have a way to disable it for the device-side created hierarchy
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (dim3 intr_dims = dims_helper<Unit, Level>::extents();
                       return fool_compiler(Dims(intr_dims.x, intr_dims.y, intr_dims.z));),
                      (return fool_compiler(dims);));
  }
  else
  {
    return fool_compiler(dims);
  }
}

template <typename BottomUnit>
struct hierarchy_extents_helper
{
  template <typename LTopDims, typename... Levels>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const LTopDims& ltop, const Levels&... levels) noexcept
  {
    using TopLevel = __level_type_of<LTopDims>;
    if constexpr (sizeof...(Levels) == 0)
    {
      return replace_with_intrinsics_or_constexpr<BottomUnit, TopLevel>(ltop.dims);
    }
    else
    {
      using Unit = ::cuda::std::__type_index_c<0, __level_type_of<Levels>...>;
      return dims_product<typename TopLevel::product_type>(
        replace_with_intrinsics_or_constexpr<Unit, TopLevel>(ltop.dims), (*this)(levels...));
    }
  }
};

template <typename T, size_t... Extents>
[[nodiscard]] _CCCL_DEVICE constexpr auto static_index_hint(const dimensions<T, Extents...>& dims, dim3 index)
{
  using hinted_index_t = dimensions<T, (Extents == 1 ? 0 : ::cuda::std::dynamic_extent)...>;
  return hinted_index_t(index.x, index.y, index.z);
}

template <typename BottomUnit>
struct index_helper
{
  template <typename LTopDims, typename... Levels>
  [[nodiscard]] _CCCL_DEVICE constexpr auto operator()(const LTopDims& ltop, const Levels&... levels) noexcept
  {
    using TopLevel = __level_type_of<LTopDims>;
    if constexpr (sizeof...(Levels) == 0)
    {
      return static_index_hint(ltop.dims, dims_helper<BottomUnit, TopLevel>::index());
    }
    else
    {
      using Unit        = ::cuda::std::__type_index_c<0, __level_type_of<Levels>...>;
      auto hinted_index = static_index_hint(ltop.dims, dims_helper<Unit, TopLevel>::index());
      return dims_sum<typename TopLevel::product_type>(
        dims_product<typename TopLevel::product_type>(hinted_index, hierarchy_extents_helper<BottomUnit>()(levels...)),
        index_helper<BottomUnit>()(levels...));
    }
  }
};

template <typename BottomUnit>
struct rank_helper
{
  template <typename LTopDims, typename... Levels>
  [[nodiscard]] _CCCL_DEVICE constexpr auto operator()(const LTopDims& ltop, const Levels&... levels) noexcept
  {
    using TopLevel = __level_type_of<LTopDims>;
    if constexpr (sizeof...(Levels) == 0)
    {
      auto hinted_index = static_index_hint(ltop.dims, dims_helper<BottomUnit, TopLevel>::index());
      return __detail::index_to_linear<typename TopLevel::product_type>(hinted_index, ltop.dims);
    }
    else
    {
      using Unit        = ::cuda::std::__type_index_c<0, __level_type_of<Levels>...>;
      auto hinted_index = static_index_hint(ltop.dims, dims_helper<Unit, TopLevel>::index());
      auto level_rank   = __detail::index_to_linear<typename TopLevel::product_type>(hinted_index, ltop.dims);
      return level_rank * dims_to_count(hierarchy_extents_helper<BottomUnit>()(levels...))
           + rank_helper<BottomUnit>()(levels...);
    }
  }
};
} // namespace __detail

// Artificial empty hierarchy to make it possible for the config type to be empty,
// seems easier than checking everywhere in hierarchy APIs if its not empty.
// Any usage of an empty hierarchy other than combine should lead to an error anyway
struct __empty_hierarchy
{
  template <typename _Other>
  [[nodiscard]] _Other combine(const _Other& __other) const
  {
    return __other;
  }
};

/**
 * @brief Type representing a hierarchy of CUDA threads
 *
 * This type combines a number of level_dimensions objects to represent dimensions of a (possibly partial)
 * hierarchy of CUDA threads. It supports accessing individual levels or queries combining dimensions
 * of multiple levels.
 * This type should not be created directly and make_hierarchy function should be used instead.
 * For every level, the unit for its dimensions is implied by the next level in the hierarchy, except
 * for the last type, for which its the BottomUnit template argument.
 * In case the BottomUnit type is thread_level, the hierarchy is considered complete and there
 * exist an alias template for it named hierarchy_dimensions, that only takes the Levels... template argument.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * auto hierarchy = make_hierarchy(grid_dims(256), block_dims<8, 8, 8>());
 * assert(hierarchy.level(grid).dims.x == 256);
 * static_assert(hierarchy.count(thread, block) == 8 * 8 * 8);
 * @endcode
 * @par
 *
 * @tparam BottomUnit
 *   Type indicating what is the unit of the last level in the hierarchy
 *
 * @tparam Levels
 *   Template parameter pack with the types of levels in the hierarchy, must be
 *   level_dimensions instances or types derived from it
 */
template <typename BottomUnit, typename... Levels>
struct hierarchy_dimensions
{
  static_assert(::cuda::std::is_base_of_v<hierarchy_level, BottomUnit> || ::cuda::std::is_same_v<BottomUnit, void>);
  ::cuda::std::tuple<Levels...> levels;

  _CCCL_API constexpr hierarchy_dimensions(const Levels&... ls) noexcept
      : levels(ls...)
  {}
  _CCCL_API constexpr hierarchy_dimensions(const BottomUnit&, const Levels&... ls) noexcept
      : levels(ls...)
  {}

  _CCCL_API constexpr hierarchy_dimensions(const ::cuda::std::tuple<Levels...>& ls) noexcept
      : levels(ls)
  {}

  _CCCL_API constexpr hierarchy_dimensions(const BottomUnit&, const ::cuda::std::tuple<Levels...>& ls) noexcept
      : levels(ls)
  {}

#  if !defined(_CCCL_NO_THREE_WAY_COMPARISON) && !_CCCL_COMPILER(MSVC, <, 19, 39) && !_CCCL_COMPILER(GCC, <, 12)
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr bool operator==(const hierarchy_dimensions&) const noexcept = default;
#  else // ^^^ !_CCCL_NO_THREE_WAY_COMPARISON ^^^ / vvv _CCCL_NO_THREE_WAY_COMPARISON vvv
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const hierarchy_dimensions& left, const hierarchy_dimensions& right) noexcept
  {
    return left.levels == right.levels;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const hierarchy_dimensions& left, const hierarchy_dimensions& right) noexcept
  {
    return left.levels != right.levels;
  }
#  endif // _CCCL_NO_THREE_WAY_COMPARISON

private:
  // This being static is a bit of a hack to make extents_type working without incomplete class member access
  template <typename Unit, typename Level>
  [[nodiscard]] _CCCL_API static constexpr auto levels_range_static(const ::cuda::std::tuple<Levels...>& levels) noexcept
  {
    static_assert(has_level<Level, hierarchy_dimensions<BottomUnit, Levels...>>);
    static_assert(has_level_or_unit<Unit, hierarchy_dimensions<BottomUnit, Levels...>>);
    static_assert(__detail::legal_unit_for_level<Unit, Level>);
    return ::cuda::std::apply(__detail::get_levels_range<Level, Unit, Levels...>, levels);
  }

  // TODO is this useful enough to expose?
  template <typename Unit, typename Level>
  [[nodiscard]] _CCCL_API constexpr auto levels_range() const noexcept
  {
    return levels_range_static<Unit, Level>(levels);
  }

  template <typename Unit>
  struct fragment_helper
  {
    template <typename... Selected>
    [[nodiscard]] _CCCL_API constexpr auto operator()(const Selected&... levels) const noexcept
    {
      return hierarchy_dimensions<Unit, Selected...>(levels...);
    }
  };

public:
  template <typename, typename...>
  friend struct hierarchy_dimensions;

  template <typename Unit, typename Level>
  using extents_type = decltype(::cuda::std::apply(
    ::cuda::std::declval<__detail::hierarchy_extents_helper<Unit>>(),
    hierarchy_dimensions::levels_range_static<Unit, Level>(::cuda::std::declval<::cuda::std::tuple<Levels...>>())));

  /**
   * @brief Get a fragment of this hierarchy
   *
   * This member function can be used to get a fragment of the hierarchy its called on.
   * It returns a hierarchy_dimensions that includes levels starting with the
   * level specified in Level and ending with a level before Unit. Toegether with
   * hierarchy_add_level function it can be used to create a new hierarchy that is a modification
   * of an existing hierarchy.
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
   * auto fragment = hierarchy.fragment(block, grid);
   * auto new_hierarchy = hierarchy_add_level(fragment, block_dims<128>());
   * static_assert(new_hierarchy.count(thread, block) == 128);
   * @endcode
   * @par
   *
   * @tparam Unit
   *   Type indicating what should be the unit of the resulting fragment
   *
   * @tparam Level
   *   Type indicating what should be the top most level of the resulting fragment
   */
  template <typename Unit, typename Level>
  _CCCL_API constexpr auto fragment(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    auto selected = levels_range<Unit, Level>();
    // TODO fragment can't do constexpr queries because we use references here, can we create copies of the levels in
    // some cases and move to the constructor?
    return ::cuda::std::apply(fragment_helper<Unit>(), selected);
  }

  /**
   * @brief Returns extents of multi-dimensional index space of a specified range of levels in this hierarchy.
   *
   * Each dimension in the returned extents is a product of the corresponding dimension in extents
   * of each level in the range between Level and Unit.
   * The returned hierarchy_query_result type can be used like cuda::std::extents or dim3.
   * Unit and Level need to be levels present in this hierarchy.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   * #include <cassert>
   *
   * using namespace cuda::experimental;
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
   * static_assert(hierarchy.extents(thread, cluster).extent(0) == 4 * 8);
   * static_assert(hierarchy.extents(thread, cluster).extent(1) == 8);
   * static_assert(hierarchy.extents(thread, cluster).extent(2) == 8);
   *
   * // Using default arguments:
   * assert(hierarchy.extents().extent(0) == 256 * 4 * 8);
   * assert(hierarchy.extents(cluster).extent(0) == 256);
   * @endcode
   * @par
   *
   * @tparam Unit
   *  Specifies the unit of the requested extents
   *
   * @tparam Level
   *  Specifies at what CUDA hierarchy level the extents are requested
   */
  template <typename Unit = BottomUnit, typename Level = __level_type_of<::cuda::std::__type_index_c<0, Levels...>>>
  _CCCL_API constexpr auto extents(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    auto selected = levels_range<Unit, Level>();
    return __detail::convert_to_query_result(::cuda::std::apply(__detail::hierarchy_extents_helper<Unit>{}, selected));
  }

  // template <typename Unit, typename Level>
  // using extents_type = ::cuda::std::invoke_result_t<
  //   decltype(&hierarchy_dimensions<BottomUnit, Levels...>::template extents<Unit, Level>),
  //   hierarchy_dimensions<BottomUnit, Levels...>,
  //   Unit(),
  //   Level()>;

  /**
   * @brief Returns a count of specified entities at a level in this hierarchy.
   *
   * This function return a product of all dimensions of each level in the range between Level and Unit.
   * Unit and Level need to be levels present in this hierarchy.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   * #include <cassert>
   *
   * using namespace cuda::experimental;
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
   * static_assert(hierarchy.count(thread, cluster) == 4 * 8 * 8 * 8);
   *
   * // Using default arguments:
   * assert(hierarchy.count() == 256 * 4 * 8 * 8 * 8);
   * assert(hierarchy.count(cluster) == 256);
   * @endcode
   * @par
   *
   * @tparam Unit
   *  Specifies what should be counted
   *
   * @tparam Level
   *  Specifies at what level the count should happen
   */
  template <typename Unit = BottomUnit, typename Level = __level_type_of<::cuda::std::__type_index_c<0, Levels...>>>
  _CCCL_API constexpr auto count(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    return __detail::dims_to_count(extents<Unit, Level>());
  }

  // TODO static extents?

  /**
   * @brief Returns a compile time count of specified entities at a level in this hierarchy type.
   *
   * This function return a product of all dimensions of each level in the range between Level and Unit,
   * if all of those dimensions are specified statically. If at least one of them is a dynamic value,
   * this function returns cuda::std::dynamic_extent instead.
   * Unit and Level need to be levels present in this hierarchy.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   * #include <cassert>
   *
   * using namespace cuda::experimental;
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
   * static_assert(hierarchy.static_count(thread, cluster) == 4 * 8 * 8 * 8);
   *
   * // Using default arguments:
   * assert(hierarchy.static_count() == cuda::std::dynamic_extent);
   * @endcode
   * @par
   *
   * @tparam Unit
   *  Specifies what should be counted
   *
   * @tparam Level
   *  Specifies at what level the count should happen
   */
  template <typename Unit = BottomUnit, typename Level = __level_type_of<::cuda::std::__type_index_c<0, Levels...>>>
  _CCCL_API constexpr static auto static_count(const Unit& = Unit(), const Level& = Level()) noexcept
  {
    if constexpr (extents_type<Unit, Level>::rank_dynamic() == 0)
    {
      return __detail::dims_to_count(extents_type<Unit, Level>());
    }
    else
    {
      return ::cuda::std::dynamic_extent;
    }
  }

  /**
   * @brief Returns a 3-dimensional index of an entity the calling thread belongs to in a hierarchy level
   *
   * Returned index is in line with intrinsic CUDA indexing like threadIdx and blockIdx,
   * extentded to more unit/level combinations. Returns a hierarchy_query_result object, which can be used
   * like cuda::std::extents or dim3. This query will use any statically available
   * information in the hierarchy to simplify rank calculation compared to the rank function operating
   * only on level types (for example if extent of a certain dimnsion is 1, then index will be statically 0).
   * Unit and Level need to be present in the hierarchy. Available only in device code.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   * #include <cassert>
   *
   * using namespace cuda::experimental;
   *
   * template <typename Dimensions>
   * __global__ void kernel(Dimensions dims)
   * {
   *     // Can be called with the instances of level types
   *     auto thread_index_in_block = dims.index(thread, block);
   *     assert(thread_index_in_block == threadIdx);
   *     // With default arguments:
   *     auto block_index_in_grid = dims.index(block);
   *     assert(block_index_in_grid == blockIdx);
   *
   *     // Or using the level types as template arguments
   *     int thread_index_in_grid = dims.template index<thread_level, grid_level>();
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
  template <typename Unit = BottomUnit, typename Level = __level_type_of<::cuda::std::__type_index_c<0, Levels...>>>
  _CCCL_DEVICE constexpr auto index(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    auto selected = levels_range<Unit, Level>();
    return __detail::convert_to_query_result(::cuda::std::apply(__detail::index_helper<Unit>{}, selected));
  }

  /**
   * @brief Ranks an entity the calling thread belongs to in a hierarchy level
   *
   * Returns a unique numeric rank within Level of the Unit that the calling thread belongs to.
   * Returned rank is always in in range 0 to count - 1. This query will use any statically available
   * information in the hierarchy to simplify rank calculation compared to the rank function operating
   * only on level types.
   * Unit and Level need to be present in the hierarchy. Available only in device code.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   *
   * using namespace cuda::experimental;
   *
   * template <typename Dimensions>
   * __global__ void kernel(Dimensions dims)
   * {
   *     // Can be called with the instances of level types
   *     int thread_rank_in_block = dims.rank(thread, block);
   *     // With default arguments:
   *     int block_rank_in_grid = dims.rank(block);
   *
   *     // Or using the level types as template arguments
   *     int thread_rank_in_grid = dimensions.template rank<thread_level, grid_level>();
   * }
   * @endcode
   * @par
   *
   * @tparam Unit
   *  Specifies the entity that the rank is requested for
   *
   * @tparam Level
   *  Specifies at what level the rank is requested
   */
  template <typename Unit = BottomUnit, typename Level = __level_type_of<::cuda::std::__type_index_c<0, Levels...>>>
  _CCCL_DEVICE constexpr auto rank(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    auto selected = levels_range<Unit, Level>();
    return ::cuda::std::apply(__detail::rank_helper<Unit>{}, selected);
  }

  /**
   * @brief Returns level description associated with a specified hierarchy level in this hierarchy.
   *
   * This function returns a copy of the object associated with the specified level, that was passed
   * into the hierarchy on its creation.
   * Level need to be levels present in this hierarchy.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   *
   * using namespace cuda::experimental;
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
   * static_assert(decltype(hierarchy.level(cluster).dims)::static_extent(0) == 4);
   * @endcode
   * @par
   *
   * @tparam Level
   *  Specifies the requested level
   */
  template <typename Level>
  _CCCL_API constexpr auto level(const Level&) const noexcept
  {
    static_assert(has_level<Level, hierarchy_dimensions<BottomUnit, Levels...>>);

    return ::cuda::std::apply(__detail::get_level_helper<Level>{}, levels);
  }

  //! @brief Returns a new hierarchy with combined levels of this and the other supplied hierarchy
  //!
  //! This function combines this hierarchy with the supplied hierarchy, the resulting hierarchy
  //! holds levels present in both hierarchies. In case of overlap of levels this hierarchy
  //! is prioritized, so the result always holds all levels from this hierarchy and non-overlapping
  //! levels from the other hierarchy.
  //!
  //! @param other The other hierarchy to be combined with this hierarchy
  //!
  //! @return Hierarchy holding the combined levels from both hierarchies
  template <typename OtherUnit, typename... OtherLevels>
  constexpr auto combine(const hierarchy_dimensions<OtherUnit, OtherLevels...>& other) const
  {
    using this_top_level     = __level_type_of<::cuda::std::__type_index_c<0, Levels...>>;
    using this_bottom_level  = __level_type_of<::cuda::std::__type_index_c<sizeof...(Levels) - 1, Levels...>>;
    using other_top_level    = __level_type_of<::cuda::std::__type_index_c<0, OtherLevels...>>;
    using other_bottom_level = __level_type_of<::cuda::std::__type_index_c<sizeof...(OtherLevels) - 1, OtherLevels...>>;
    if constexpr (__detail::can_rhs_stack_on_lhs<other_top_level, this_bottom_level>)
    {
      // Easily stackable case, example this is (grid), other is (cluster, block)
      return ::cuda::std::apply(fragment_helper<OtherUnit>(), ::cuda::std::tuple_cat(levels, other.levels));
    }
    else if constexpr (has_level<this_bottom_level, hierarchy_dimensions<OtherUnit, OtherLevels...>>
                       && (!has_level<this_top_level, hierarchy_dimensions<OtherUnit, OtherLevels...>>
                           || ::cuda::std::is_same_v<this_top_level, other_top_level>) )
    {
      // Overlap with this on the top, e.g. this is (grid, cluster), other is (cluster, block), can fully overlap
      // Do we have some CCCL tuple utils that can select all but the first?
      auto to_add_with_one_too_many = other.template levels_range<OtherUnit, this_bottom_level>();
      auto to_add                   = ::cuda::std::apply(
        [](auto&&, auto&&... rest) {
          return ::cuda::std::make_tuple(rest...);
        },
        to_add_with_one_too_many);
      return ::cuda::std::apply(fragment_helper<OtherUnit>(), ::cuda::std::tuple_cat(levels, to_add));
    }
    else
    {
      if constexpr (__detail::can_rhs_stack_on_lhs<this_top_level, other_bottom_level>)
      {
        // Easily stackable case again, just reversed
        return ::cuda::std::apply(fragment_helper<BottomUnit>(), ::cuda::std::tuple_cat(other.levels, levels));
      }
      else
      {
        // Overlap with this on the bottom, e.g. this is (cluster, block), other is (grid, cluster), can fully overlap
        static_assert(has_level<other_bottom_level, hierarchy_dimensions<BottomUnit, Levels...>>
                        && (!has_level<this_bottom_level, hierarchy_dimensions<OtherUnit, OtherLevels...>>
                            || ::cuda::std::is_same_v<this_bottom_level, other_bottom_level>),
                      "Can't combine the hierarchies");

        auto to_add = other.template levels_range<this_top_level, other_top_level>();
        return ::cuda::std::apply(fragment_helper<BottomUnit>(), ::cuda::std::tuple_cat(to_add, levels));
      }
    }
  }

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  constexpr hierarchy_dimensions combine([[maybe_unused]] __empty_hierarchy __empty) const
  {
    return *this;
  }
#  endif // _CCCL_DOXYGEN_INVOKED
};

/**
 * @brief Returns a tuple of dim3 compatible objects that can be used to launch a kernel
 *
 * This function returns a tuple of hierarchy_query_result objects that contain dimensions from
 * the supplied hierarchy, that can be used to launch that hierarchy. It is meant to
 * allow for easy usage of hierarchy dimensions with the <<<>>> launch syntax or
 * cudaLaunchKernelEx in case of a cluster launch.
 * Contained hierarchy_query_result objects are results of extents() member function on
 * the hierarchy passed in.
 * The returned tuple has three elements if cluster_level is present in the hierarchy
 * (extents(block, grid), extents(cluster, block), extents(thread, block)).
 * Otherwise it contains only two elements, without the middle one related to the cluster.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda::experimental;
 *
 * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
 * auto [grid_dimensions, cluster_dimensions, block_dimensions] = get_launch_dimensions(hierarchy);
 * assert(grid_dimensions.x == 256);
 * assert(cluster_dimensions.x == 4);
 * assert(block_dimensions.x == 8);
 * assert(block_dimensions.y == 8);
 * assert(block_dimensions.z == 8);
 * @endcode
 * @par
 *
 * @param hierarchy
 *  Hierarchy that the launch dimensions are requested for
 */
template <typename... Levels>
constexpr auto _CCCL_HOST get_launch_dimensions(const hierarchy_dimensions<Levels...>& hierarchy)
{
  if constexpr (has_level<cluster_level, hierarchy_dimensions<Levels...>>)
  {
    return ::cuda::std::make_tuple(
      hierarchy.extents(block, grid), hierarchy.extents(block, cluster), hierarchy.extents(thread, block));
  }
  else
  {
    return ::cuda::std::make_tuple(hierarchy.extents(block, grid), hierarchy.extents(thread, block));
  }
}

// TODO consider having LUnit optional argument for template argument deduction
/**
 * @brief Creates a hierarchy from passed in levels.
 *
 * This function takes any number of level_dimensions or derived objects
 * and creates a hierarchy out of them. Levels need to be in ascending
 * or descending order and the lowest level needs to be valid for thread_level unit.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda::experimental;
 *
 * auto hierarchy1 = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
 * auto hierarchy2 = make_hierarchy(block_dims<8, 8, 8>(), cluster_dims<4>(), grid_dims(256));
 * static_assert(cuda::std::is_same_v<decltype(hierarchy1), decltype(hierarchy2)>);
 * @endcode
 * @par
 */
template <typename LUnit = void, typename L1, typename... Levels>
constexpr auto make_hierarchy(L1 l1, Levels... ls) noexcept
{
  return __detail::__make_hierarchy<LUnit>()(__detail::__as_level(l1), __detail::__as_level(ls)...);
}

/**
 * @brief Add a level to a hierarchy
 *
 * This function returns a new hierarchy, that is a copy of the supplied hierarchy
 * with the supplied level added to it. This function will examine the supplied
 * level and add it either at the top or at the bottom of the hierarchy, depending
 * on what levels above and below it are valid for it.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda::experimental;
 *
 * auto partial1 = make_hierarchy<block_level>(grid_dims(256), cluster_dims<4>());
 * auto hierarchy1 = hierarchy_add_level(partial1, block_dims<8, 8, 8>());
 * auto partial2 = make_hierarchy<thread_level>(block_dims<8, 8, 8>(), cluster_dims<4>());
 * auto hierarchy2 = hierarchy_add_level(partial2, grid_dims(256));
 * static_assert(cuda::std::is_same_v<decltype(hierarchy1), decltype(hierarchy2)>);
 * @endcode
 * @par
 */
template <typename NewLevel, typename Unit, typename... Levels>
constexpr auto hierarchy_add_level(const hierarchy_dimensions<Unit, Levels...>& hierarchy, NewLevel lnew)
{
  auto new_level     = __detail::__as_level(lnew);
  using AddedLevel   = decltype(new_level);
  using top_level    = __level_type_of<::cuda::std::__type_index_c<0, Levels...>>;
  using bottom_level = __level_type_of<::cuda::std::__type_index_c<sizeof...(Levels) - 1, Levels...>>;

  if constexpr (__detail::can_rhs_stack_on_lhs<top_level, __level_type_of<AddedLevel>>)
  {
    return hierarchy_dimensions<Unit, AddedLevel, Levels...>(
      ::cuda::std::tuple_cat(::cuda::std::make_tuple(new_level), hierarchy.levels));
  }
  else
  {
    static_assert(__detail::can_rhs_stack_on_lhs<__level_type_of<AddedLevel>, bottom_level>,
                  "Not supported order of levels in hierarchy");
    using NewUnit = __detail::__default_unit_below<__level_type_of<AddedLevel>>;
    return hierarchy_dimensions<NewUnit, Levels..., AddedLevel>(
      ::cuda::std::tuple_cat(hierarchy.levels, ::cuda::std::make_tuple(new_level)));
  }
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__HIERARCHY_HIERARCHY_DIMENSIONS_CUH
