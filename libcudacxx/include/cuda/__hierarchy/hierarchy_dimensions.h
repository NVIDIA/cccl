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
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/type_list.h>
#  include <cuda/std/__utility/integer_sequence.h>
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
template <class... _Levels>
struct __can_stack_checker
{
  template <class... _LevelsShifted>
  static constexpr bool __can_stack = (__detail::__can_rhs_stack_on_lhs<_LevelsShifted, _Levels> && ...);
};

template <class _LUnit, class _L1, class... _Levels>
inline constexpr bool __can_stack =
  __can_stack_checker<__level_type_of<_L1>,
                      __level_type_of<_Levels>...>::template __can_stack<__level_type_of<_Levels>..., _LUnit>;

template <::cuda::std::size_t... _Id>
_CCCL_API constexpr auto __reverse_indices(::cuda::std::index_sequence<_Id...>) noexcept
{
  return ::cuda::std::index_sequence<(sizeof...(_Id) - 1 - _Id)...>();
}

template <class _LUnit, bool _Reversed = false>
struct __make_hierarchy
{
  template <class _Levels, ::cuda::std::size_t... _Ids>
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
      return hierarchy(_UnitOrDefault{}, __ls...);
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
 * This type combines a number of hierarchy_level_desc objects to represent
 * dimensions of a (possibly partial) hierarchy of CUDA threads. It supports
 * accessing individual levels or queries combining dimensions of multiple
 * levels. This type should not be created directly and make_hierarchy function
 * should be used instead. For every level, the unit for its dimensions is
 * implied by the next level in the hierarchy, except for the last type, for
 * which its the BottomUnit template argument. In case the BottomUnit type is
 * thread_level, the hierarchy is considered complete and there exist an alias
 * template for it named hierarchy, that only takes the Levels...
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
 *   hierarchy_level_desc instances or types derived from it
 */
template <class _BottomUnit, class... _LevelDescs>
class hierarchy
{
  static_assert(__is_hierarchy_level_v<_BottomUnit>);
  static_assert(__detail::__can_stack<_BottomUnit, typename _LevelDescs::level_type...>);

  template <class, class...>
  friend class hierarchy;

  ::cuda::std::tuple<_LevelDescs...> __descs_;

  // This being static is a bit of a hack to make extents_type working without
  // incomplete class member access
  template <class _Unit, class _Level>
  [[nodiscard]] _CCCL_API static constexpr auto
  __levels_range_static(const ::cuda::std::tuple<_LevelDescs...>& __levels) noexcept
  {
    static_assert(hierarchy::has_level<_Level>());
    static_assert(__has_bottom_unit_or_level_v<_Unit, hierarchy<_BottomUnit, _LevelDescs...>>);
    static_assert(__detail::__legal_unit_for_level<_Unit, _Level>);
    auto __fn = __detail::__get_levels_range<_Level, _Unit, _LevelDescs...>;
    return ::cuda::std::apply(__fn, __levels);
  }

  // TODO is this useful enough to expose?
  template <class _Unit, class _Level>
  [[nodiscard]] _CCCL_API constexpr auto __levels_range() const noexcept
  {
    return __levels_range_static<_Unit, _Level>(__descs_);
  }

  template <class _Unit>
  struct __fragment_helper
  {
    template <class... _Selected>
    [[nodiscard]] _CCCL_API constexpr auto operator()(const _Selected&... __levels) const noexcept
    {
      return hierarchy<_Unit, _Selected...>(__levels...);
    }
  };

public:
  template <class _Level>
  static constexpr auto __level_idx =
    ::cuda::std::__find_exactly_one_t<_Level, typename _LevelDescs::level_type...>::value;

  using bottom_unit_type = _BottomUnit;
  using top_level_type   = ::cuda::std::__type_index_c<0, typename _LevelDescs::level_type...>;

  template <class _Level>
  using level_desc_type = ::cuda::std::__type_index_c<__level_idx<_Level>, _LevelDescs...>;

  template <class _Level>
  [[nodiscard]] _CCCL_API static constexpr bool has_level(const _Level& = _Level{}) noexcept
  {
    return (::cuda::std::is_same_v<_Level, typename _LevelDescs::level_type> || ...);
  }

  _CCCL_API constexpr hierarchy(const _LevelDescs&... __lds) noexcept
      : __descs_(__lds...)
  {}

  _CCCL_TEMPLATE(class _BottomUnit2 = _BottomUnit)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<void, _BottomUnit2>) )
  _CCCL_API constexpr hierarchy(const _BottomUnit2&, const _LevelDescs&... __lds) noexcept
      : __descs_(__lds...)
  {}

  _CCCL_API constexpr hierarchy(const ::cuda::std::tuple<_LevelDescs...>& __lds) noexcept
      : __descs_(__lds)
  {}

  _CCCL_TEMPLATE(class _BottomUnit2 = _BottomUnit)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<void, _BottomUnit2>) )
  _CCCL_API constexpr hierarchy(const _BottomUnit2&, const ::cuda::std::tuple<_LevelDescs...>& __lds) noexcept
      : __descs_(__lds)
  {}

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const hierarchy& __lhs, const hierarchy& __rhs) noexcept
  {
    return __lhs.__descs_ == __rhs.__descs_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const hierarchy& __lhs, const hierarchy& __rhs) noexcept
  {
    return __lhs.__descs_ != __rhs.__descs_;
  }

  /**
   * @brief Get a fragment of this hierarchy
   *
   * This member function can be used to get a fragment of the hierarchy its
   * called on. It returns a hierarchy that includes levels starting
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
    auto __selected = __levels_range<_Unit, _Level>();
    // TODO fragment can't do constexpr queries because we use references here,
    // can we create copies of the levels in some cases and move to the
    // constructor?
    return ::cuda::std::apply(__fragment_helper<_Unit>(), __selected);
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
  [[nodiscard]] _CCCL_API constexpr const level_desc_type<_Level>& level(const _Level&) const noexcept
  {
    static_assert(hierarchy::has_level<_Level>());
    return ::cuda::std::get<__level_idx<_Level>>(__descs_);
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
  constexpr auto combine(const hierarchy<_OtherUnit, _OtherLevels...>& __other) const
  {
    using _BottomLevel    = __level_type_of<::cuda::std::__type_index_c<sizeof...(_LevelDescs) - 1, _LevelDescs...>>;
    using _OtherHierarchy = hierarchy<_OtherUnit, _OtherLevels...>;
    using _OtherTopLevel  = typename _OtherHierarchy::top_level_type;
    using _OtherBottomLevel =
      __level_type_of<::cuda::std::__type_index_c<sizeof...(_OtherLevels) - 1, _OtherLevels...>>;
    if constexpr (__detail::__can_rhs_stack_on_lhs<_OtherTopLevel, _BottomLevel>)
    {
      // Easily stackable case, example this is (grid), other is (cluster,
      // block)
      return ::cuda::std::apply(__fragment_helper<_OtherUnit>(), ::cuda::std::tuple_cat(__descs_, __other.__descs_));
    }
    else if constexpr (_OtherHierarchy::template has_level<_BottomLevel>()
                       && (!_OtherHierarchy::template has_level<top_level_type>()
                           || ::cuda::std::is_same_v<top_level_type, _OtherTopLevel>) )
    {
      // Overlap with this on the top, e.g. this is (grid, cluster), other is
      // (cluster, block), can fully overlap Do we have some CCCL tuple utils
      // that can select all but the first?
      auto __to_add_with_one_too_many = __other.template __levels_range<_OtherUnit, _BottomLevel>();
      auto __to_add                   = ::cuda::std::apply(
        [](auto&&, auto&&... __rest) {
          return ::cuda::std::make_tuple(__rest...);
        },
        __to_add_with_one_too_many);
      return ::cuda::std::apply(__fragment_helper<_OtherUnit>(), ::cuda::std::tuple_cat(__descs_, __to_add));
    }
    else
    {
      if constexpr (__detail::__can_rhs_stack_on_lhs<top_level_type, _OtherBottomLevel>)
      {
        // Easily stackable case again, just reversed
        return ::cuda::std::apply(__fragment_helper<_BottomUnit>(), ::cuda::std::tuple_cat(__other.__descs_, __descs_));
      }
      else
      {
        // Overlap with this on the bottom, e.g. this is (cluster, block), other
        // is (grid, cluster), can fully overlap
        static_assert(hierarchy::has_level<_OtherBottomLevel>()
                        && (!_OtherHierarchy::template has_level<_BottomLevel>()
                            || ::cuda::std::is_same_v<_BottomLevel, _OtherBottomLevel>),
                      "Can't combine the hierarchies");

        auto __to_add = __other.template __levels_range<top_level_type, _OtherTopLevel>();
        return ::cuda::std::apply(__fragment_helper<_BottomUnit>(), ::cuda::std::tuple_cat(__to_add, __descs_));
      }
    }
  }

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  constexpr hierarchy combine([[maybe_unused]] __empty_hierarchy __empty) const
  {
    return *this;
  }
#  endif // _CCCL_DOXYGEN_INVOKED

  template <class _NewLevel, class _Unit, class... _LevelDescs2>
  friend constexpr auto hierarchy_add_level(const hierarchy<_Unit, _LevelDescs2...>& hierarchy, _NewLevel __lnew);
};

_CCCL_TEMPLATE(class... _LevelDescs)
_CCCL_REQUIRES(::cuda::std::__fold_and_v<__is_hierarchy_level_desc_v<_LevelDescs>...>)
_CCCL_HOST_DEVICE hierarchy(const _LevelDescs&...)
  -> hierarchy<__detail::__default_unit_below<
                 ::cuda::std::__type_index_c<sizeof...(_LevelDescs) - 1, __level_type_of<_LevelDescs>...>>,
               _LevelDescs...>;

_CCCL_TEMPLATE(class _BottomUnit, class... _LevelDescs)
_CCCL_REQUIRES(
  __is_hierarchy_level_v<_BottomUnit> _CCCL_AND ::cuda::std::__fold_and_v<__is_hierarchy_level_desc_v<_LevelDescs>...>)
_CCCL_HOST_DEVICE hierarchy(const _BottomUnit&, const _LevelDescs&...) -> hierarchy<_BottomUnit, _LevelDescs...>;

_CCCL_TEMPLATE(class... _LevelDescs)
_CCCL_REQUIRES(::cuda::std::__fold_and_v<__is_hierarchy_level_desc_v<_LevelDescs>...>)
_CCCL_HOST_DEVICE hierarchy(const ::cuda::std::tuple<_LevelDescs...>&)
  -> hierarchy<__detail::__default_unit_below<
                 ::cuda::std::__type_index_c<sizeof...(_LevelDescs) - 1, __level_type_of<_LevelDescs>...>>,
               _LevelDescs...>;

_CCCL_TEMPLATE(class _BottomUnit, class... _LevelDescs)
_CCCL_REQUIRES(
  __is_hierarchy_level_v<_BottomUnit> _CCCL_AND ::cuda::std::__fold_and_v<__is_hierarchy_level_desc_v<_LevelDescs>...>)
_CCCL_HOST_DEVICE hierarchy(const _BottomUnit&, const ::cuda::std::tuple<_LevelDescs...>&)
  -> hierarchy<_BottomUnit, _LevelDescs...>;

// TODO consider having LUnit optional argument for template argument deduction
/**
 * @brief Creates a hierarchy from passed in levels.
 *
 * This function takes any number of hierarchy_level_desc or derived objects
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
template <class _LUnit = void, class _L1, class... _LevelDescs>
constexpr auto make_hierarchy(_L1 __l1, _LevelDescs... __ls) noexcept
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
template <class _NewLevel, class _Unit, class... _LevelDescs>
constexpr auto hierarchy_add_level(const hierarchy<_Unit, _LevelDescs...>& __hierarchy, _NewLevel __lnew)
{
  auto __new_level     = __detail::__as_level(__lnew);
  using __added_level  = decltype(__new_level);
  using __top_level    = __level_type_of<::cuda::std::__type_index_c<0, _LevelDescs...>>;
  using __bottom_level = __level_type_of<::cuda::std::__type_index_c<sizeof...(_LevelDescs) - 1, _LevelDescs...>>;

  if constexpr (__detail::__can_rhs_stack_on_lhs<__top_level, __level_type_of<__added_level>>)
  {
    return hierarchy<_Unit, __added_level, _LevelDescs...>(
      ::cuda::std::tuple_cat(::cuda::std::make_tuple(__new_level), __hierarchy.__descs_));
  }
  else
  {
    static_assert(__detail::__can_rhs_stack_on_lhs<__level_type_of<__added_level>, __bottom_level>,
                  "Not supported order of levels in hierarchy");
    using __new_unit = __detail::__default_unit_below<__level_type_of<__added_level>>;
    return hierarchy<__new_unit, _LevelDescs..., __added_level>(
      ::cuda::std::tuple_cat(__hierarchy.__descs_, ::cuda::std::make_tuple(__new_level)));
  }
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_HIERARCHY_DIMENSIONS_H
