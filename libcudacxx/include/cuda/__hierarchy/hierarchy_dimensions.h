//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_HIERARCHY_DIMENSIONS_H
#define _CUDA___HIERARCHY_HIERARCHY_DIMENSIONS_H

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
#  include <cuda/__hierarchy/level_dimensions.h>
#  include <cuda/__hierarchy/traits.h>
#  include <cuda/std/__type_traits/fold.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/remove_cvref.h>
#  include <cuda/std/__type_traits/type_list.h>
#  include <cuda/std/__utility/declval.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/array>
#  include <cuda/std/span>
#  include <cuda/std/tuple>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
template <typename _Level>
[[nodiscard]] _CCCL_API constexpr auto __as_level(_Level __lvl) noexcept -> _Level
{
  return __lvl;
}

template <typename _LevelFn>
[[nodiscard]] _CCCL_API constexpr auto __as_level(_LevelFn* __fn) noexcept -> decltype(__fn())
{
  return {};
}
} // namespace __detail

namespace __detail
{
// Function to sometimes convince the compiler something is a constexpr and not
// really accessing runtime storage Mostly a work around for what was addressed
// in P2280 (c++23) by leveraging the argumentless constructor of extents
template <class _Tp, size_t... _Extents>
[[nodiscard]] _CCCL_API constexpr auto __fool_compiler(const dimensions<_Tp, _Extents...>& __ex)
{
  if constexpr (dimensions<_Tp, _Extents...>::rank_dynamic() == 0)
  {
    return dimensions<_Tp, _Extents...>();
  }
  else
  {
    return __ex;
  }
}

template <class _QueryLevel>
struct __get_level_helper
{
  template <class _TopLevel, class... _Levels>
  [[nodiscard]] _CCCL_API constexpr auto& operator()(const _TopLevel& __top, const _Levels&... __levels)
  {
    if constexpr (::cuda::std::is_same_v<_QueryLevel, __level_type_of<_TopLevel>>)
    {
      return __top;
    }
    else
    {
      return (*this)(__levels...);
    }
  }
};
} // namespace __detail

namespace __detail
{
template <class... _Levels>
struct __can_stack_checker
{
  template <class... _LevelsShifted>
  using __can_stack = ::cuda::std::__fold_and<__detail::__can_rhs_stack_on_lhs<_LevelsShifted, _Levels>...>;
};

template <class _LUnit, class _L1, class... _Levels>
inline constexpr bool __can_stack =
  __can_stack_checker<__level_type_of<_L1>,
                      __level_type_of<_Levels>...>::template __can_stack<__level_type_of<_Levels>..., _LUnit>::value;

template <size_t... _Id>
_CCCL_API constexpr auto __reverse_indices(::cuda::std::index_sequence<_Id...>) noexcept
{
  return ::cuda::std::index_sequence<(sizeof...(_Id) - 1 - _Id)...>();
}

template <class _LUnit, bool _Reversed = false>
struct __make_hierarchy
{
  template <class _Levels, size_t... _Ids>
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto
  __apply_reverse(const _Levels& __ls, ::cuda::std::index_sequence<_Ids...>) noexcept
  {
    return __make_hierarchy<_LUnit, true>()(::cuda::std::get<_Ids>(__ls)...);
  }

  template <class... _Levels2>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Levels2&... __ls) const noexcept
  {
    using _UnitOrDefault = ::cuda::std::conditional_t<
      ::cuda::std::is_same_v<void, _LUnit>,
      __default_unit_below<::cuda::std::__type_index_c<sizeof...(_Levels2) - 1, __level_type_of<_Levels2>...>>,
      _LUnit>;
    if constexpr (__can_stack<_UnitOrDefault, _Levels2...>)
    {
      return hierarchy_dimensions(_UnitOrDefault{}, __ls...);
    }
    else if constexpr (!_Reversed)
    {
      return __apply_reverse(::cuda::std::tie(__ls...),
                             __reverse_indices(::cuda::std::index_sequence_for<_Levels2...>()));
    }
    else
    {
      static_assert(__can_stack<_UnitOrDefault, _Levels2...>,
                    "Provided levels can't create a valid hierarchy when "
                    "stacked in the provided order or reversed");
      _CCCL_UNREACHABLE();
    }
  }
};

template <class _LUnit>
[[nodiscard]] _CCCL_API constexpr auto __get_levels_range_end() noexcept
{
  return ::cuda::std::make_tuple();
}

// Find LUnit in Levels... and discard the rest
// maybe_unused needed for MSVC
template <class _LUnit, class _LDims, class... _Levels>
[[nodiscard]] _CCCL_API constexpr auto
__get_levels_range_end(const _LDims& __lvl, [[maybe_unused]] const _Levels&... __levels) noexcept
{
  if constexpr (::cuda::std::is_same_v<_LUnit, __level_type_of<_LDims>>)
  {
    return ::cuda::std::make_tuple();
  }
  else
  {
    return ::cuda::std::tuple_cat(::cuda::std::tie(__lvl), __get_levels_range_end<_LUnit>(__levels...));
  }
}

// Find the LTop in Levels... and discard the preceding ones
template <class _LTop, class _LUnit, class _LTopDims, class... _Levels>
[[nodiscard]] _CCCL_API constexpr auto
__get_levels_range_start(const _LTopDims& __ltop, const _Levels&... __levels) noexcept
{
  if constexpr (::cuda::std::is_same_v<_LTop, __level_type_of<_LTopDims>>)
  {
    return __get_levels_range_end<_LUnit>(__ltop, __levels...);
  }
  else
  {
    return __get_levels_range_start<_LTop, _LUnit>(__levels...);
  }
}

// Creates a new hierarchy from Levels... cutting out levels between LTop and
// LUnit
template <class _LTop, class _LUnit, class... _Levels>
[[nodiscard]] _CCCL_API constexpr auto __get_levels_range(const _Levels&... __levels) noexcept
{
  return __get_levels_range_start<_LTop, _LUnit>(__levels...);
}

template <class _Tp, size_t... _Extents, size_t... _Ids>
[[nodiscard]] _CCCL_API constexpr auto
__dims_to_count_helper(const dimensions<_Tp, _Extents...>& __ex, ::cuda::std::index_sequence<_Ids...>)
{
  return (__ex.extent(_Ids) * ...);
}

template <class _Tp, size_t... _Extents>
[[nodiscard]] _CCCL_API constexpr auto __dims_to_count(const dimensions<_Tp, _Extents...>& __dims) noexcept
{
  return __dims_to_count_helper(__dims, ::cuda::std::make_index_sequence<sizeof...(_Extents)>{});
}

template <class... _Levels>
[[nodiscard]] _CCCL_API constexpr auto __get_level_counts_helper(const _Levels&... __ls)
{
  return ::cuda::std::make_tuple(__dims_to_count(__ls.dims)...);
}

template <class _Unit, class _Level, class _Dims>
[[nodiscard]] _CCCL_API constexpr auto __replace_with_intrinsics_or_constexpr(const _Dims& __dims)
{
  if constexpr (is_core_cuda_hierarchy_level<_Level> && is_core_cuda_hierarchy_level<_Unit>
                && _Dims::rank_dynamic() != 0)
  {
    // We replace hierarchy access with CUDA intrinsic to enable compiler
    // optimizations, its ok for the prototype, but might lead to unexpected
    // results and should be eventually addressed at the API level
    // TODO with device side launch we should have a way to disable it for the
    // device-side created hierarchy
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (::dim3 __intr_dims = ::cuda::__detail::__dims_helper<_Unit, _Level>::extents();
                       return __fool_compiler(_Dims(__intr_dims.x, __intr_dims.y, __intr_dims.z));),
                      (return __fool_compiler(__dims);));
  }
  else
  {
    return __fool_compiler(__dims);
  }
}

template <class _BottomUnit>
struct __hierarchy_extents_helper
{
  template <class _LTopDims, class... _Levels>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _LTopDims& __ltop, const _Levels&... __levels) noexcept
  {
    using _TopLevel = __level_type_of<_LTopDims>;
    if constexpr (sizeof...(_Levels) == 0)
    {
      return __replace_with_intrinsics_or_constexpr<_BottomUnit, _TopLevel>(__ltop.dims);
    }
    else
    {
      using _Unit = ::cuda::std::__type_index_c<0, __level_type_of<_Levels>...>;
      return ::cuda::__detail::__dims_product<typename _TopLevel::product_type>(
        __replace_with_intrinsics_or_constexpr<_Unit, _TopLevel>(__ltop.dims), (*this)(__levels...));
    }
  }
};

template <class _Tp, size_t... _Extents>
[[nodiscard]] _CCCL_DEVICE constexpr auto __static_index_hint(const dimensions<_Tp, _Extents...>&, ::dim3 __index)
{
  using _HintedIndexT = dimensions<_Tp, (_Extents == 1 ? 0 : ::cuda::std::dynamic_extent)...>;
  return _HintedIndexT(__index.x, __index.y, __index.z);
}

template <class _BottomUnit>
struct __index_helper
{
  template <class _LTopDims, class... _Levels>
  [[nodiscard]] _CCCL_DEVICE constexpr auto operator()(const _LTopDims& __ltop, const _Levels&... __levels) noexcept
  {
    using _TopLevel = __level_type_of<_LTopDims>;
    if constexpr (sizeof...(_Levels) == 0)
    {
      return __static_index_hint(__ltop.dims, ::cuda::__detail::__dims_helper<_BottomUnit, _TopLevel>::index());
    }
    else
    {
      using _Unit = ::cuda::std::__type_index_c<0, __level_type_of<_Levels>...>;
      const auto __hinted_index =
        __static_index_hint(__ltop.dims, ::cuda::__detail::__dims_helper<_Unit, _TopLevel>::index());
      return ::cuda::__detail::__dims_sum<typename _TopLevel::product_type>(
        ::cuda::__detail::__dims_product<typename _TopLevel::product_type>(
          __hinted_index, __hierarchy_extents_helper<_BottomUnit>()(__levels...)),
        __index_helper<_BottomUnit>()(__levels...));
    }
  }
};

template <class _BottomUnit>
struct __rank_helper
{
  template <class _LTopDims, class... _Levels>
  [[nodiscard]] _CCCL_DEVICE constexpr auto operator()(const _LTopDims& __ltop, const _Levels&... __levels) noexcept
  {
    using _TopLevel = __level_type_of<_LTopDims>;
    if constexpr (sizeof...(_Levels) == 0)
    {
      const auto __hinted_index =
        __static_index_hint(__ltop.dims, ::cuda::__detail::__dims_helper<_BottomUnit, _TopLevel>::index());
      return ::cuda::__detail::__index_to_linear<typename _TopLevel::product_type>(__hinted_index, __ltop.dims);
    }
    else
    {
      using _Unit = ::cuda::std::__type_index_c<0, __level_type_of<_Levels>...>;
      const auto __hinted_index =
        __static_index_hint(__ltop.dims, ::cuda::__detail::__dims_helper<_Unit, _TopLevel>::index());
      auto __level_rank =
        ::cuda::__detail::__index_to_linear<typename _TopLevel::product_type>(__hinted_index, __ltop.dims);
      return __level_rank * __dims_to_count(__hierarchy_extents_helper<_BottomUnit>()(__levels...))
           + __rank_helper<_BottomUnit>()(__levels...);
    }
  }
};
} // namespace __detail

// Artificial empty hierarchy to make it possible for the config type to be
// empty, seems easier than checking everywhere in hierarchy APIs if its not
// empty. Any usage of an empty hierarchy other than combine should lead to an
// error anyway
struct __empty_hierarchy
{
  template <class _Other>
  [[nodiscard]] _Other combine(const _Other& __other) const
  {
    return __other;
  }
};

/**
 * @brief Type representing a hierarchy of CUDA threads
 *
 * This type combines a number of level_dimensions objects to represent
 * dimensions of a (possibly partial) hierarchy of CUDA threads. It supports
 * accessing individual levels or queries combining dimensions of multiple
 * levels. This type should not be created directly and make_hierarchy function
 * should be used instead. For every level, the unit for its dimensions is
 * implied by the next level in the hierarchy, except for the last type, for
 * which its the BottomUnit template argument. In case the BottomUnit type is
 * thread_level, the hierarchy is considered complete and there exist an alias
 * template for it named hierarchy_dimensions, that only takes the Levels...
 * template argument.
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
template <class _BottomUnit, class... _Levels>
struct hierarchy_dimensions
{
  static_assert(__is_hierarchy_level_v<_BottomUnit> || ::cuda::std::is_same_v<_BottomUnit, void>);
  ::cuda::std::tuple<_Levels...> levels;

  _CCCL_API constexpr hierarchy_dimensions(const _Levels&... __ls) noexcept
      : levels(__ls...)
  {}
  _CCCL_API constexpr hierarchy_dimensions(const _BottomUnit&, const _Levels&... __ls) noexcept
      : levels(__ls...)
  {}

  _CCCL_API constexpr hierarchy_dimensions(const ::cuda::std::tuple<_Levels...>& __ls) noexcept
      : levels(__ls)
  {}

  _CCCL_API constexpr hierarchy_dimensions(const _BottomUnit&, const ::cuda::std::tuple<_Levels...>& __ls) noexcept
      : levels(__ls)
  {}

#  if !defined(_CCCL_NO_THREE_WAY_COMPARISON) && !_CCCL_COMPILER(MSVC, <, 19, 39) && !_CCCL_COMPILER(GCC, <, 12)
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr bool operator==(const hierarchy_dimensions&) const noexcept = default;
#  else // ^^^ !_CCCL_NO_THREE_WAY_COMPARISON ^^^ / vvv
        // _CCCL_NO_THREE_WAY_COMPARISON vvv
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const hierarchy_dimensions& __left, const hierarchy_dimensions& __right) noexcept
  {
    return __left.levels == __right.levels;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const hierarchy_dimensions& __left, const hierarchy_dimensions& __right) noexcept
  {
    return __left.levels != __right.levels;
  }
#  endif // _CCCL_NO_THREE_WAY_COMPARISON

private:
  // This being static is a bit of a hack to make extents_type working without
  // incomplete class member access
  template <class _Unit, class _Level>
  [[nodiscard]] _CCCL_API static constexpr auto
  levels_range_static(const ::cuda::std::tuple<_Levels...>& __levels) noexcept
  {
    static_assert(has_level_v<_Level, hierarchy_dimensions<_BottomUnit, _Levels...>>);
    static_assert(has_unit_or_level_v<_Unit, hierarchy_dimensions<_BottomUnit, _Levels...>>);
    static_assert(__detail::__legal_unit_for_level<_Unit, _Level>);
    auto __fn = __detail::__get_levels_range<_Level, _Unit, _Levels...>;
    return ::cuda::std::apply(__fn, __levels);
  }

  // TODO is this useful enough to expose?
  template <class _Unit, class _Level>
  [[nodiscard]] _CCCL_API constexpr auto levels_range() const noexcept
  {
    return levels_range_static<_Unit, _Level>(levels);
  }

  template <class _Unit>
  struct fragment_helper
  {
    template <class... _Selected>
    [[nodiscard]] _CCCL_API constexpr auto operator()(const _Selected&... __levels) const noexcept
    {
      return hierarchy_dimensions<_Unit, _Selected...>(__levels...);
    }
  };

public:
  template <class, class...>
  friend struct hierarchy_dimensions;

  template <class _Unit, class _Level>
  using extents_type = decltype(::cuda::std::apply(
    ::cuda::std::declval<__detail::__hierarchy_extents_helper<_Unit>>(),
    hierarchy_dimensions::levels_range_static<_Unit, _Level>(::cuda::std::declval<::cuda::std::tuple<_Levels...>>())));

  /**
   * @brief Get a fragment of this hierarchy
   *
   * This member function can be used to get a fragment of the hierarchy its
   * called on. It returns a hierarchy_dimensions that includes levels starting
   * with the level specified in Level and ending with a level before Unit.
   * Toegether with hierarchy_add_level function it can be used to create a new
   * hierarchy that is a modification of an existing hierarchy.
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(),
   * block_dims<8, 8, 8>()); auto fragment = hierarchy.fragment(block, grid);
   * auto new_hierarchy = hierarchy_add_level(fragment, block_dims<128>());
   * static_assert(new_hierarchy.count(thread, block) == 128);
   * @endcode
   * @par
   *
   * @tparam Unit
   *   Type indicating what should be the unit of the resulting fragment
   *
   * @tparam Level
   *   Type indicating what should be the top most level of the resulting
   * fragment
   */
  template <typename _Unit, typename _Level>
  _CCCL_API constexpr auto fragment(const _Unit& = _Unit(), const _Level& = _Level()) const noexcept
  {
    auto selected = levels_range<_Unit, _Level>();
    // TODO fragment can't do constexpr queries because we use references here,
    // can we create copies of the levels in some cases and move to the
    // constructor?
    return ::cuda::std::apply(fragment_helper<_Unit>(), selected);
  }

  /**
   * @brief Returns extents of multi-dimensional index space of a specified
   * range of levels in this hierarchy.
   *
   * Each dimension in the returned extents is a product of the corresponding
   * dimension in extents of each level in the range between Level and Unit. The
   * returned hierarchy_query_result type can be used like cuda::std::extents or
   * dim3. Unit and Level need to be levels present in this hierarchy.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   * #include <cassert>
   *
   * using namespace cuda;
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(),
   * block_dims<8, 8, 8>()); static_assert(hierarchy.extents(thread,
   * cluster).extent(0) == 4 * 8); static_assert(hierarchy.extents(thread,
   * cluster).extent(1) == 8); static_assert(hierarchy.extents(thread,
   * cluster).extent(2) == 8);
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
  template <typename _Unit = _BottomUnit, typename _Level = __level_type_of<::cuda::std::__type_index_c<0, _Levels...>>>
  _CCCL_API constexpr auto extents(const _Unit& = _Unit(), const _Level& = _Level()) const noexcept
  {
    auto selected = levels_range<_Unit, _Level>();
    return ::cuda::__detail::__convert_to_query_result(
      ::cuda::std::apply(__detail::__hierarchy_extents_helper<_Unit>{}, selected));
  }

  //!
  //! @brief Get static extents of a multi-dimensional index space of a
  //! specified range of levels in this hierarchy.
  //!
  //! Each dimension in the returned static extents is a product of the
  //! corresponding dimension in static extents of each level in the range
  //! between Level and Unit. Unit and Level need to be levels present in this
  //! hierarchy.
  //!
  //! @return Returns a cuda::std::array object of type size_t.
  //!
  //! @par Snippet
  //! @code
  //! #include <cudax/hierarchy_dimensions.cuh>
  //! #include <cassert>
  //!
  //! using namespace cuda;
  //!
  //! auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(),
  //! block_dims<8, 8, 8>()); static_assert(hierarchy.static_extents(thread,
  //! cluster)[0] == 4 * 8); static_assert(hierarchy.static_extents(thread,
  //! cluster)[1] == 8); static_assert(hierarchy.static_extents(thread,
  //! cluster)[2] == 8); static_assert(hierarchy.static_extents(thread, grid)[0]
  //! == cuda::std::dynamic_extent);
  //!
  //! // Using default arguments:
  //! assert(hierarchy.static_extents()[0] == cuda::std::dynamic_extent);
  //! assert(hierarchy.static_extents(cluster)[0] == cuda::std::dynamic_extent);
  //! @endcode
  //! @par
  //!
  //! @tparam Unit
  //!  Specifies the unit of the requested extents
  //!
  //! @tparam Level
  //!  Specifies at what CUDA hierarchy level the extents are requested
  template <class _Unit = _BottomUnit, class _Level = __level_type_of<::cuda::std::__type_index_c<0, _Levels...>>>
  [[nodiscard]] _CCCL_API static constexpr auto static_extents(const _Unit& = _Unit(), const _Level& = _Level()) noexcept
  {
    using Exts = extents_type<_Unit, _Level>;
    ::cuda::std::array<::cuda::std::size_t, Exts::rank()> __ret{0};
    for (::cuda::std::size_t i = 0; i < Exts::rank(); ++i)
    {
      __ret[i] = Exts::static_extent(i);
    }
    return __ret;
  }

  /**
   * @brief Returns a count of specified entities at a level in this hierarchy.
   *
   * This function return a product of all dimensions of each level in the range
   * between Level and Unit. Unit and Level need to be levels present in this
   * hierarchy.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   * #include <cassert>
   *
   * using namespace cuda;
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(),
   * block_dims<8, 8, 8>()); static_assert(hierarchy.count(thread, cluster) == 4
   * * 8 * 8 * 8);
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
  template <typename _Unit = _BottomUnit, typename _Level = __level_type_of<::cuda::std::__type_index_c<0, _Levels...>>>
  _CCCL_API constexpr auto count(const _Unit& = _Unit(), const _Level& = _Level()) const noexcept
  {
    return ::cuda::__detail::__dims_to_count(extents<_Unit, _Level>());
  }

  /**
   * @brief Returns a compile time count of specified entities at a level in
   * this hierarchy type.
   *
   * This function return a product of all dimensions of each level in the range
   * between Level and Unit, if all of those dimensions are specified
   * statically. If at least one of them is a dynamic value, this function
   * returns cuda::std::dynamic_extent instead. Unit and Level need to be levels
   * present in this hierarchy.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   * #include <cassert>
   *
   * using namespace cuda;
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(),
   * block_dims<8, 8, 8>()); static_assert(hierarchy.static_count(thread,
   * cluster) == 4 * 8 * 8 * 8);
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
  template <typename _Unit = _BottomUnit, typename _Level = __level_type_of<::cuda::std::__type_index_c<0, _Levels...>>>
  _CCCL_API constexpr static auto static_count(const _Unit& = _Unit(), const _Level& = _Level()) noexcept
  {
    if constexpr (extents_type<_Unit, _Level>::rank_dynamic() == 0)
    {
      return __detail::__dims_to_count(extents_type<_Unit, _Level>());
    }
    else
    {
      return ::cuda::std::dynamic_extent;
    }
  }

  /**
   * @brief Returns a 3-dimensional index of an entity the calling thread
   * belongs to in a hierarchy level
   *
   * Returned index is in line with intrinsic CUDA indexing like threadIdx and
   * blockIdx, extentded to more unit/level combinations. Returns a
   * hierarchy_query_result object, which can be used like cuda::std::extents or
   * dim3. This query will use any statically available information in the
   * hierarchy to simplify rank calculation compared to the rank function
   * operating only on level types (for example if extent of a certain dimnsion
   * is 1, then index will be statically 0). Unit and Level need to be present
   * in the hierarchy. Available only in device code.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   * #include <cassert>
   *
   * using namespace cuda;
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
   *     int thread_index_in_grid = dims.template index<thread_level,
   * grid_level>();
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
  template <typename _Unit = _BottomUnit, typename _Level = __level_type_of<::cuda::std::__type_index_c<0, _Levels...>>>
  _CCCL_DEVICE constexpr auto index(const _Unit& = _Unit(), const _Level& = _Level()) const noexcept
  {
    auto selected = levels_range<_Unit, _Level>();
    return ::cuda::__detail::__convert_to_query_result(::cuda::std::apply(__detail::__index_helper<_Unit>{}, selected));
  }

  /**
   * @brief Ranks an entity the calling thread belongs to in a hierarchy level
   *
   * Returns a unique numeric rank within Level of the Unit that the calling
   * thread belongs to. Returned rank is always in in range 0 to count - 1. This
   * query will use any statically available information in the hierarchy to
   * simplify rank calculation compared to the rank function operating only on
   * level types. Unit and Level need to be present in the hierarchy. Available
   * only in device code.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   *
   * using namespace cuda;
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
   *     int thread_rank_in_grid = dimensions.template rank<thread_level,
   * grid_level>();
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
  template <typename _Unit = _BottomUnit, typename _Level = __level_type_of<::cuda::std::__type_index_c<0, _Levels...>>>
  _CCCL_DEVICE constexpr auto rank(const _Unit& = _Unit(), const _Level& = _Level()) const noexcept
  {
    auto selected = levels_range<_Unit, _Level>();
    return ::cuda::std::apply(__detail::__rank_helper<_Unit>{}, selected);
  }

  /**
   * @brief Returns level description associated with a specified hierarchy
   * level in this hierarchy.
   *
   * This function returns a copy of the object associated with the specified
   * level, that was passed into the hierarchy on its creation. Level need to be
   * levels present in this hierarchy.
   *
   * @par Snippet
   * @code
   * #include <cudax/hierarchy_dimensions.cuh>
   *
   * using namespace cuda;
   *
   * auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(),
   * block_dims<8, 8, 8>());
   * static_assert(decltype(hierarchy.level(cluster).dims)::static_extent(0) ==
   * 4);
   * @endcode
   * @par
   *
   * @tparam Level
   *  Specifies the requested level
   */
  template <typename _Level>
  _CCCL_API constexpr auto level(const _Level&) const noexcept
  {
    static_assert(has_level_v<_Level, hierarchy_dimensions<_BottomUnit, _Levels...>>);

    return ::cuda::std::apply(__detail::__get_level_helper<_Level>{}, levels);
  }

  //! @brief Returns a new hierarchy with combined levels of this and the other
  //! supplied hierarchy
  //!
  //! This function combines this hierarchy with the supplied hierarchy, the
  //! resulting hierarchy holds levels present in both hierarchies. In case of
  //! overlap of levels this hierarchy is prioritized, so the result always
  //! holds all levels from this hierarchy and non-overlapping levels from the
  //! other hierarchy.
  //!
  //! @param other The other hierarchy to be combined with this hierarchy
  //!
  //! @return Hierarchy holding the combined levels from both hierarchies
  template <class _OtherUnit, class... _OtherLevels>
  constexpr auto combine(const hierarchy_dimensions<_OtherUnit, _OtherLevels...>& __other) const
  {
    using __this_top_level    = __level_type_of<::cuda::std::__type_index_c<0, _Levels...>>;
    using __this_bottom_level = __level_type_of<::cuda::std::__type_index_c<sizeof...(_Levels) - 1, _Levels...>>;
    using __other_top_level   = __level_type_of<::cuda::std::__type_index_c<0, _OtherLevels...>>;
    using __other_bottom_level =
      __level_type_of<::cuda::std::__type_index_c<sizeof...(_OtherLevels) - 1, _OtherLevels...>>;
    if constexpr (__detail::__can_rhs_stack_on_lhs<__other_top_level, __this_bottom_level>)
    {
      // Easily stackable case, example this is (grid), other is (cluster,
      // block)
      return ::cuda::std::apply(fragment_helper<_OtherUnit>(), ::cuda::std::tuple_cat(levels, __other.levels));
    }
    else if constexpr (has_level_v<__this_bottom_level, hierarchy_dimensions<_OtherUnit, _OtherLevels...>>
                       && (!has_level_v<__this_top_level, hierarchy_dimensions<_OtherUnit, _OtherLevels...>>
                           || ::cuda::std::is_same_v<__this_top_level, __other_top_level>) )
    {
      // Overlap with this on the top, e.g. this is (grid, cluster), other is
      // (cluster, block), can fully overlap Do we have some CCCL tuple utils
      // that can select all but the first?
      auto __to_add_with_one_too_many = __other.template levels_range<_OtherUnit, __this_bottom_level>();
      auto __to_add                   = ::cuda::std::apply(
        [](auto&&, auto&&... __rest) {
          return ::cuda::std::make_tuple(__rest...);
        },
        __to_add_with_one_too_many);
      return ::cuda::std::apply(fragment_helper<_OtherUnit>(), ::cuda::std::tuple_cat(levels, __to_add));
    }
    else
    {
      if constexpr (__detail::__can_rhs_stack_on_lhs<__this_top_level, __other_bottom_level>)
      {
        // Easily stackable case again, just reversed
        return ::cuda::std::apply(fragment_helper<_BottomUnit>(), ::cuda::std::tuple_cat(__other.levels, levels));
      }
      else
      {
        // Overlap with this on the bottom, e.g. this is (cluster, block), other
        // is (grid, cluster), can fully overlap
        static_assert(has_level_v<__other_bottom_level, hierarchy_dimensions<_BottomUnit, _Levels...>>
                        && (!has_level_v<__this_bottom_level, hierarchy_dimensions<_OtherUnit, _OtherLevels...>>
                            || ::cuda::std::is_same_v<__this_bottom_level, __other_bottom_level>),
                      "Can't combine the hierarchies");

        auto __to_add = __other.template levels_range<__this_top_level, __other_top_level>();
        return ::cuda::std::apply(fragment_helper<_BottomUnit>(), ::cuda::std::tuple_cat(__to_add, levels));
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

// TODO consider having LUnit optional argument for template argument deduction
/**
 * @brief Creates a hierarchy from passed in levels.
 *
 * This function takes any number of level_dimensions or derived objects
 * and creates a hierarchy out of them. Levels need to be in ascending
 * or descending order and the lowest level needs to be valid for thread_level
 * unit.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda;
 *
 * auto hierarchy1 = make_hierarchy(grid_dims(256), cluster_dims<4>(),
 * block_dims<8, 8, 8>()); auto hierarchy2 = make_hierarchy(block_dims<8, 8,
 * 8>(), cluster_dims<4>(), grid_dims(256));
 * static_assert(cuda::std::is_same_v<decltype(hierarchy1),
 * decltype(hierarchy2)>);
 * @endcode
 * @par
 */
template <class _LUnit = void, class _L1, class... _Levels>
constexpr auto make_hierarchy(_L1 __l1, _Levels... __ls) noexcept
{
  return __detail::__make_hierarchy<_LUnit>()(__detail::__as_level(__l1), __detail::__as_level(__ls)...);
}

/**
 * @brief Add a level to a hierarchy
 *
 * This function returns a new hierarchy, that is a copy of the supplied
 * hierarchy with the supplied level added to it. This function will examine the
 * supplied level and add it either at the top or at the bottom of the
 * hierarchy, depending on what levels above and below it are valid for it.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * using namespace cuda;
 *
 * auto partial1 = make_hierarchy<block_level>(grid_dims(256),
 * cluster_dims<4>()); auto hierarchy1 = hierarchy_add_level(partial1,
 * block_dims<8, 8, 8>()); auto partial2 =
 * make_hierarchy<thread_level>(block_dims<8, 8, 8>(), cluster_dims<4>()); auto
 * hierarchy2 = hierarchy_add_level(partial2, grid_dims(256));
 * static_assert(cuda::std::is_same_v<decltype(hierarchy1),
 * decltype(hierarchy2)>);
 * @endcode
 * @par
 */
template <class _NewLevel, class _Unit, class... _Levels>
constexpr auto hierarchy_add_level(const hierarchy_dimensions<_Unit, _Levels...>& hierarchy, _NewLevel __lnew)
{
  auto __new_level     = __detail::__as_level(__lnew);
  using __added_level  = decltype(__new_level);
  using __top_level    = __level_type_of<::cuda::std::__type_index_c<0, _Levels...>>;
  using __bottom_level = __level_type_of<::cuda::std::__type_index_c<sizeof...(_Levels) - 1, _Levels...>>;

  if constexpr (__detail::__can_rhs_stack_on_lhs<__top_level, __level_type_of<__added_level>>)
  {
    return hierarchy_dimensions<_Unit, __added_level, _Levels...>(
      ::cuda::std::tuple_cat(::cuda::std::make_tuple(__new_level), hierarchy.levels));
  }
  else
  {
    static_assert(__detail::__can_rhs_stack_on_lhs<__level_type_of<__added_level>, __bottom_level>,
                  "Not supported order of levels in hierarchy");
    using __new_unit = __detail::__default_unit_below<__level_type_of<__added_level>>;
    return hierarchy_dimensions<__new_unit, _Levels..., __added_level>(
      ::cuda::std::tuple_cat(hierarchy.levels, ::cuda::std::make_tuple(__new_level)));
  }
}
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_HIERARCHY_DIMENSIONS_H
