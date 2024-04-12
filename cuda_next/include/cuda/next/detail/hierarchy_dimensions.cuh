//===----------------------------------------------------------------------===//
//
// Part of CUDA Next in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_NEXT_DETAIL_HIERARCHY_DIMENSIONS
#define _CUDA_NEXT_DETAIL_HIERARCHY_DIMENSIONS

#include <cuda/std/tuple>

#include "level_dimensions.cuh"
#include <nv/target>

namespace cuda_next
{

// TODO right now operator stacking can end up with a wrong unit, we could use below type, but we would need an explicit
// thread_level inserter
struct unknown_unit : public hierarchy_level
{
  using product_type  = unsigned int;
  using allowed_above = allowed_levels<>;
  using allowed_below = allowed_levels<>;
};

template <typename BottomUnit, typename... Levels>
struct hierarchy_dimensions_fragment;

// If lowest unit in the hierarchy is thread, it can be considered a full hierarchy and not only a fragment
template <typename... Levels>
using hierarchy_dimensions = hierarchy_dimensions_fragment<thread_level, Levels...>;

namespace detail
{
// Function to sometimes convince the compiler something is a constexpr and not really accessing runtime storage
// Mostly a work around for what was addressed in P2280 (c++23) by leveraging a default constructor of extents
template <typename T, size_t... Extents>
_CCCL_HOST_DEVICE constexpr auto fool_compiler(const dimensions<T, Extents...>& ex)
{
  if constexpr (dimensions<T, Extents...>::rank_dynamic() == 0)
  {
    return dimensions<T, Extents...>();
  }
  else
  {
    return ex;
  }
}

template <typename QueryLevel, typename Hierarchy>
struct has_level_helper;

template <typename QueryLevel, typename Unit, typename... Levels>
struct has_level_helper<QueryLevel, hierarchy_dimensions_fragment<Unit, Levels...>>
    : public ::cuda::std::disjunction<::cuda::std::is_same<QueryLevel, typename Levels::level_type>...>
{};

// Is this needed?
template <typename QueryLevel, typename... Levels>
struct has_level_helper<QueryLevel, hierarchy_dimensions<Levels...>>
    : public ::cuda::std::disjunction<::cuda::std::is_same<QueryLevel, typename Levels::level_type>...>
{};

template <typename QueryLevel, typename Hierarchy>
struct has_unit
{};

template <typename QueryLevel, typename Unit, typename... Levels>
struct has_unit<QueryLevel, hierarchy_dimensions_fragment<Unit, Levels...>> : ::cuda::std::is_same<QueryLevel, Unit>
{};

template <unsigned int Id, typename... Levels>
using level_at_index = typename ::cuda::std::tuple_element<Id, ::cuda::std::tuple<Levels...>>::type;

template <typename QueryLevel>
struct get_level_helper
{
  template <typename TopLevel, typename... Levels>
  _CCCL_HOST_DEVICE constexpr auto& operator()(const TopLevel& top, const Levels&... levels)
  {
    if constexpr (::cuda::std::is_same_v<QueryLevel, typename TopLevel::level_type>)
    {
      return top;
    }
    else
    {
      return (*this)(levels...);
    }
  }
};
} // namespace detail

template <typename QueryLevel, typename Hierarchy>
bool constexpr has_level =
  detail::has_level_helper<QueryLevel, ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<Hierarchy>>>::value;

template <typename QueryLevel, typename Hierarchy>
bool constexpr has_level_or_unit =
  detail::has_level_helper<QueryLevel, ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<Hierarchy>>>::value
  || detail::has_unit<QueryLevel, ::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<Hierarchy>>>::value;

// We can consider removing the operator, but its convinient for in-line construction
// TODO accept forwarding references
template <typename LUnit, typename L1, typename... Levels>
_CCCL_HOST_DEVICE constexpr auto
operator&(const hierarchy_dimensions_fragment<LUnit, Levels...>& ls, const L1& l1) noexcept
{
  using top_level    = typename detail::level_at_index<0, Levels...>::level_type;
  using bottom_level = typename detail::level_at_index<sizeof...(Levels) - 1, Levels...>::level_type;

  if constexpr (detail::can_stack_on_top<top_level, typename L1::level_type>)
  {
    return hierarchy_dimensions_fragment<LUnit, L1, Levels...>(
      ::cuda::std::tuple_cat(::cuda::std::make_tuple(l1), ls.levels));
  }
  else
  {
    static_assert(detail::can_stack_on_top<typename L1::level_type, bottom_level>,
                  "Not supported order of levels in hierarchy");
    using NewUnit = typename L1::level_type::allowed_below::default_unit;
    return hierarchy_dimensions_fragment<NewUnit, Levels..., L1>(
      ::cuda::std::tuple_cat(ls.levels, ::cuda::std::make_tuple(l1)));
  }
}

template <typename L1, typename LUnit, typename... Levels>
_CCCL_HOST_DEVICE constexpr auto
operator&(const L1& l1, const hierarchy_dimensions_fragment<LUnit, Levels...>& ls) noexcept
{
  return ls & l1;
}

template <typename L1, typename Dims1, typename L2, typename Dims2>
_CCCL_HOST_DEVICE constexpr auto
operator&(const level_dimensions<L1, Dims1>& l1, const level_dimensions<L2, Dims2>& l2) noexcept
{
  return hierarchy_dimensions<level_dimensions<L1, Dims1>>(l1) & l2;
}

namespace detail
{
template <typename... Levels>
struct can_stack_checker
{
  template <typename... LevelsShifted>
  static constexpr bool can_stack = (detail::can_stack_on_top<LevelsShifted, Levels> && ...);
};

template <typename LUnit, ::cuda::std::size_t... Ids, typename... Levels>
auto constexpr hierarchy_dimensions_fragment_reversed(::cuda::std::index_sequence<Ids...>, const Levels&&... ls);

template <bool Reversed, typename LUnit, typename L1, typename... Levels>
auto constexpr make_hierarchy_fragment_reversable(L1&& l1, Levels&&... ls) noexcept
{
  using checker = can_stack_checker<typename ::cuda::std::remove_reference_t<L1>::level_type,
                                    typename ::cuda::std::remove_reference_t<Levels>::level_type...>;
  constexpr bool can_stack =
    checker::template can_stack<typename ::cuda::std::remove_reference_t<Levels>::level_type..., LUnit>;
  static_assert(can_stack || !Reversed,
                "Provided levels can't create a valid hierarchy when stacked in the provided order or reversed");
  if constexpr (can_stack)
  {
    return hierarchy_dimensions_fragment(LUnit{}, ::cuda::std::forward<L1>(l1), ::cuda::std::forward<Levels>(ls)...);
  }
  else
  {
    return hierarchy_dimensions_fragment_reversed<LUnit>(
      ::cuda::std::index_sequence_for<L1, Levels...>(),
      ::cuda::std::forward<L1>(l1),
      ::cuda::std::forward<Levels>(ls)...);
  }
}

template <typename LUnit, ::cuda::std::size_t... Ids, typename... Levels>
auto constexpr hierarchy_dimensions_fragment_reversed(::cuda::std::index_sequence<Ids...>, Levels&&... ls)
{
  auto tied = ::cuda::std::forward_as_tuple(::cuda::std::forward<Levels>(ls)...);
  return make_hierarchy_fragment_reversable<true, LUnit>(
    ::cuda::std::get<sizeof...(Levels) - 1 - Ids>(::cuda::std::move(tied))...);
}

} // namespace detail

// TODO consider having LUnit optional argument for template argument deduction
template <typename LUnit, typename L1, typename... Levels>
auto constexpr make_hierarchy_fragment(L1&& l1, Levels&&... ls) noexcept
{
  return detail::make_hierarchy_fragment_reversable<false, thread_level>(
    ::cuda::std::forward<L1>(l1), ::cuda::std::forward<Levels>(ls)...);
}

template <typename L1, typename... Levels>
auto constexpr make_hierarchy(L1&& l1, Levels&&... ls) noexcept
{
  return detail::make_hierarchy_fragment_reversable<false, thread_level>(
    ::cuda::std::forward<L1>(l1), ::cuda::std::forward<Levels>(ls)...);
}

// Should we enforce unit match?
template <typename NewLevel, typename Unit, typename... Levels>
auto constexpr hierarchy_add_level(const hierarchy_dimensions_fragment<Unit, Levels...>& hierarchy, NewLevel&& level)
{
  return hierarchy & ::cuda::std::forward<NewLevel>(level);
}

namespace detail
{

template <typename LUnit>
_CCCL_HOST_DEVICE constexpr auto get_levels_range_end() noexcept
{
  return ::cuda::std::make_tuple();
}

// Find LUnit in Levels... and discard the rest
template <typename LUnit, typename LDims, typename... Levels>
_CCCL_HOST_DEVICE constexpr auto get_levels_range_end(const LDims& l, const Levels&... levels) noexcept
{
  if constexpr (::cuda::std::is_same_v<LUnit, typename LDims::level_type>)
  {
    return ::cuda::std::make_tuple();
  }
  else
  {
    return ::cuda::std::tuple_cat(::cuda::std::tie(l), get_levels_range_end<LUnit>(levels...));
  }
}

// Find the LTop in Levels... and discard the preceeding ones
template <typename LTop, typename LUnit, typename LTopDims, typename... Levels>
_CCCL_HOST_DEVICE constexpr auto get_levels_range_start(const LTopDims& ltop, const Levels&... levels) noexcept
{
  if constexpr (::cuda::std::is_same_v<LTop, typename LTopDims::level_type>)
  {
    return get_levels_range_end<LUnit>(ltop, levels...);
  }
  else
  {
    return get_levels_range_start<LTop, LUnit>(levels...);
  }
}

// Creates a new hierachy from Levels... cutting out levels between LTop and LUnit
template <typename LTop, typename LUnit, typename... Levels>
_CCCL_HOST_DEVICE constexpr auto get_levels_range(const Levels&... levels) noexcept
{
  return get_levels_range_start<LTop, LUnit>(levels...);
}

template <typename T, size_t... Extents, size_t... Ids>
_CCCL_HOST_DEVICE constexpr auto
dims_to_count_helper(const dimensions<T, Extents...> ex, ::cuda::std::integer_sequence<size_t, Ids...>)
{
  return (ex.extent(Ids) * ...);
}

template <typename T, size_t... Extents>
_CCCL_HOST_DEVICE constexpr auto dims_to_count(const dimensions<T, Extents...>& dims) noexcept
{
  return dims_to_count_helper(dims, ::cuda::std::make_integer_sequence<size_t, sizeof...(Extents)>{});
}

template <typename... Levels>
_CCCL_HOST_DEVICE constexpr auto get_level_counts_helper(const Levels&... ls)
{
  return ::cuda::std::make_tuple(dims_to_count(ls.dims)...);
}

template <typename Unit, typename Level, typename Dims>
_CCCL_HOST_DEVICE constexpr auto replace_with_intrinsics_or_constexpr(const Dims& dims)
{
  if constexpr (is_core_cuda_hierarchy_level<Level> && is_core_cuda_hierarchy_level<Unit> && Dims::rank_dynamic() != 0)
  {
    // We replace hierarchy access with CUDA intrinsic to enable compiler optimizations, its ok for the prototype,
    // but might lead to unexpected results and should be eventually addressed at the API level
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (dim3 intr_dims = dims_helper<Unit, Level>::dims();
                       return fool_compiler(Dims(intr_dims.x, intr_dims.y, intr_dims.z));),
                      (return fool_compiler(dims);));
  }
  else
  {
    return fool_compiler(dims);
  }
}

template <typename BottomUnit>
struct hierarchy_flatten_helper
{
  template <typename LTopDims, typename... Levels>
  _CCCL_HOST_DEVICE constexpr auto operator()(const LTopDims& ltop, const Levels&... levels) noexcept
  {
    using TopLevel = typename LTopDims::level_type;
    if constexpr (sizeof...(Levels) == 0)
    {
      return replace_with_intrinsics_or_constexpr<BottomUnit, TopLevel>(ltop.dims);
    }
    else
    {
      using Unit = typename detail::get_first_level_type<typename Levels::level_type...>::type;
      return dims_product<typename TopLevel::product_type>(
        replace_with_intrinsics_or_constexpr<Unit, TopLevel>(ltop.dims), (*this)(levels...));
    }
  }
};

template <typename T, size_t... Extents>
constexpr __device__ auto static_index_hint(const dimensions<T, Extents...>& dims, dim3 index)
{
  using hinted_index_t = dimensions<T, (Extents == 1 ? 0 : ::cuda::std::dynamic_extent)...>;
  return hinted_index_t(index.x, index.y, index.z);
}

template <typename BottomUnit>
struct index_helper
{
  template <typename LTopDims, typename... Levels>
  constexpr __device__ auto operator()(const LTopDims& ltop, const Levels&... levels) noexcept
  {
    using TopLevel = typename LTopDims::level_type;
    if constexpr (sizeof...(Levels) == 0)
    {
      return static_index_hint(ltop.dims, dims_helper<BottomUnit, TopLevel>::index());
    }
    else
    {
      using Unit        = typename detail::get_first_level_type<typename Levels::level_type...>::type;
      auto hinted_index = static_index_hint(ltop.dims, dims_helper<Unit, TopLevel>::index());
      return dims_sum<typename TopLevel::product_type>(
        dims_product<typename TopLevel::product_type>(hinted_index, hierarchy_flatten_helper<BottomUnit>()(levels...)),
        index_helper<BottomUnit>()(levels...));
    }
  }
};

template <typename BottomUnit>
struct rank_helper
{
  template <typename LTopDims, typename... Levels>
  constexpr __device__ auto operator()(const LTopDims& ltop, const Levels&... levels) noexcept
  {
    using TopLevel = typename LTopDims::level_type;
    if constexpr (sizeof...(Levels) == 0)
    {
      auto hinted_index = static_index_hint(ltop.dims, dims_helper<BottomUnit, TopLevel>::index());
      return detail::index_to_linear<typename TopLevel::product_type>(hinted_index, ltop.dims);
    }
    else
    {
      using Unit        = typename detail::get_first_level_type<typename Levels::level_type...>::type;
      auto hinted_index = static_index_hint(ltop.dims, dims_helper<Unit, TopLevel>::index());
      auto level_rank   = detail::index_to_linear<typename TopLevel::product_type>(hinted_index, ltop.dims);
      return level_rank * dims_to_count(hierarchy_flatten_helper<BottomUnit>()(levels...))
           + rank_helper<BottomUnit>()(levels...);
    }
  }
};
} // namespace detail

// Type to represent (possibly partial) hierarchy of CUDA threads. Each level is an instance of level_dimensions
// template or derived type.
//  Dimensions of a given level are expressed in using what is the next level in Levels. Unit for last level is
//  BottomUnit
template <typename BottomUnit, typename... Levels>
struct hierarchy_dimensions_fragment
{
  static_assert(::cuda::std::is_base_of_v<hierarchy_level, BottomUnit> || ::cuda::std::is_same_v<BottomUnit, void>);
  ::cuda::std::tuple<Levels...> levels;

  constexpr _CCCL_HOST_DEVICE hierarchy_dimensions_fragment(const Levels&... ls) noexcept
      : levels(ls...)
  {}
  constexpr _CCCL_HOST_DEVICE hierarchy_dimensions_fragment(Levels&&... ls) noexcept
      : levels(::cuda::std::forward<Levels>(ls)...)
  {}
  constexpr _CCCL_HOST_DEVICE hierarchy_dimensions_fragment(const BottomUnit& /* unit */, const Levels&... ls) noexcept
      : levels(ls...)
  {}
  constexpr _CCCL_HOST_DEVICE hierarchy_dimensions_fragment(const BottomUnit& /* unit */, Levels&&... ls) noexcept
      : levels(::cuda::std::forward<Levels>(ls)...)
  {}

  constexpr _CCCL_HOST_DEVICE hierarchy_dimensions_fragment(const ::cuda::std::tuple<Levels...>& ls) noexcept
      : levels(ls)
  {}
  constexpr _CCCL_HOST_DEVICE hierarchy_dimensions_fragment(::cuda::std::tuple<Levels...>&& ls) noexcept
      : levels(::cuda::std::forward<::cuda::std::tuple<Levels...>>(ls))
  {}

  constexpr _CCCL_HOST_DEVICE
  hierarchy_dimensions_fragment(const BottomUnit& unit, const ::cuda::std::tuple<Levels...>& ls) noexcept
      : levels(ls)
  {}
  constexpr _CCCL_HOST_DEVICE
  hierarchy_dimensions_fragment(const BottomUnit& unit, ::cuda::std::tuple<Levels...>&& ls) noexcept
      : levels(::cuda::std::forward<::cuda::std::tuple<Levels...>>(ls))
  {}

private:
  // TODO is this useful enough to expose?
  template <typename Unit, typename Level>
  _CCCL_HOST_DEVICE constexpr auto levels_range() const noexcept
  {
    static_assert(has_level<Level, hierarchy_dimensions_fragment<BottomUnit, Levels...>>);
    static_assert(has_level_or_unit<Unit, hierarchy_dimensions_fragment<BottomUnit, Levels...>>);
    static_assert(detail::legal_unit_for_level<Unit, Level>);
    return ::cuda::std::apply(detail::get_levels_range<Level, Unit, Levels...>, levels);
  }

public:
  template <typename Unit, typename Level>
  _CCCL_HOST_DEVICE constexpr auto fragment(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    auto selected = levels_range<Unit, Level>();
    // TODO fragment can't do constexpr queries because we use references here, can we create copies of the levels in
    // some cases and move to the constructor?
    return ::cuda::std::apply(
      [](const auto&... levels) {
        return hierarchy_dimensions_fragment<Unit, ::cuda::std::remove_reference_t<decltype(levels)>...>(levels...);
      },
      selected);
  }

  // TODO Its not the best name, rename to dims? Its not really better
  template <typename Unit  = BottomUnit,
            typename Level = typename detail::get_first_level_type<Levels...>::type::level_type>
  _CCCL_HOST_DEVICE constexpr auto flatten(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    auto selected = levels_range<Unit, Level>();
    return detail::convert_to_query_result(::cuda::std::apply(detail::hierarchy_flatten_helper<Unit>{}, selected));
  }

  template <typename Unit  = BottomUnit,
            typename Level = typename detail::get_first_level_type<Levels...>::type::level_type>
  _CCCL_HOST_DEVICE constexpr auto count(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    return detail::dims_to_count(flatten<Unit, Level>());
  }

  // TODO static flatten?

  template <typename Unit  = BottomUnit,
            typename Level = typename detail::get_first_level_type<Levels...>::type::level_type>
  _CCCL_HOST_DEVICE constexpr static auto static_count(const Unit& = Unit(), const Level& = Level()) noexcept
  {
    using flattened_type = decltype(::cuda::std::declval<hierarchy_dimensions_fragment<BottomUnit, Levels...>>()
                                      .template flatten<Unit, Level>());
    if constexpr (flattened_type::rank_dynamic() == 0)
    {
      return detail::dims_to_count(flattened_type());
    }
    else
    {
      return ::cuda::std::dynamic_extent;
    }
  }

  template <typename Unit  = BottomUnit,
            typename Level = typename detail::get_first_level_type<Levels...>::type::level_type>
  _CCCL_HOST_DEVICE constexpr auto index(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    auto selected = levels_range<Unit, Level>();
    return detail::convert_to_query_result(::cuda::std::apply(detail::index_helper<Unit>{}, selected));
  }

  template <typename Unit  = BottomUnit,
            typename Level = typename detail::get_first_level_type<Levels...>::type::level_type>
  _CCCL_HOST_DEVICE constexpr auto rank(const Unit& = Unit(), const Level& = Level()) const noexcept
  {
    auto selected = levels_range<Unit, Level>();
    return ::cuda::std::apply(detail::rank_helper<Unit>{}, selected);
  }

  template <typename Level>
  _CCCL_HOST_DEVICE constexpr auto level(const Level& /*level*/)
  {
    static_assert(has_level<Level, hierarchy_dimensions_fragment<BottomUnit, Levels...>>);

    return ::cuda::std::apply(detail::get_level_helper<Level>{}, levels);
  }
};

template <typename... Levels>
auto constexpr _CCCL_HOST get_launch_dimensions(const hierarchy_dimensions<Levels...>& h)
{
  if constexpr (has_level<cluster_level, hierarchy_dimensions<Levels...>>)
  {
    return ::cuda::std::make_tuple(h.flatten(block, grid), h.flatten(block, cluster), h.flatten(thread, block));
  }
  else
  {
    return ::cuda::std::make_tuple(h.flatten(block, grid), h.flatten(thread, block));
  }
}

} // namespace cuda_next
#endif
