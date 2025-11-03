// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_for.cuh>
#include <cub/detail/mdspan_utils.cuh> // is_sub_size_static
#include <cub/detail/type_traits.cuh> // implicit_prom_t

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_trivially_constructible.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstddef> // size_t

CUB_NAMESPACE_BEGIN

namespace detail::for_each
{
template <class Fn>
struct first_parameter
{
  using type = void;
};

template <class C, class R, class A>
struct first_parameter<R (C::*)(A)>
{
  using type = A;
};

template <class C, class R, class A>
struct first_parameter<R (C::*)(A) const>
{
  using type = A;
};

template <class Fn>
using first_parameter_t = typename first_parameter<decltype(&Fn::operator())>::type;

template <class Value, class Fn, class = void>
struct has_unique_value_overload : ::cuda::std::false_type
{};

// clang-format off
template <class Value, class Fn>
struct has_unique_value_overload<
  Value,
  Fn,
  ::cuda::std::enable_if_t<
              !::cuda::std::is_reference_v<first_parameter_t<Fn>> &&
              ::cuda::std::is_convertible_v<Value, first_parameter_t<Fn>
             >>>
    : ::cuda::std::true_type
{};

// For trivial types, foreach is not allowed to copy values, even if those are trivially copyable.
// This can be observable if the unary operator takes parameter by reference and modifies it or uses address.
// The trait below checks if the freedom to copy trivial types can be regained.
template <typename Value, typename Fn>
using can_regain_copy_freedom =
  ::cuda::std::integral_constant<
    bool,
    ::cuda::std::is_trivially_constructible_v<Value> &&
    ::cuda::std::is_trivially_copy_assignable_v<Value> &&
    ::cuda::std::is_trivially_move_assignable_v<Value> &&
    ::cuda::std::is_trivially_destructible_v<Value> &&
    has_unique_value_overload<Value, Fn>::value>;
// clang-format on

// This kernel is used when the block size is not known at compile time
template <class ChainedPolicyT, class OffsetT, class OpT>
CUB_DETAIL_KERNEL_ATTRIBUTES void dynamic_kernel(OffsetT num_items, OpT op)
{
  using active_policy_t = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using agent_t         = agent_block_striped_t<active_policy_t, OffsetT, OpT>;

  const auto block_threads  = static_cast<OffsetT>(blockDim.x);
  const auto items_per_tile = active_policy_t::items_per_thread * block_threads;
  const auto tile_base      = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining  = num_items - tile_base;
  const auto items_in_tile  = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);

  if (items_in_tile == items_per_tile)
  {
    agent_t{tile_base, op}.template consume_tile<true>(items_per_tile, block_threads);
  }
  else
  {
    agent_t{tile_base, op}.template consume_tile<false>(items_in_tile, block_threads);
  }
}

// This kernel is used when the block size is known at compile time
template <class ChainedPolicyT, class OffsetT, class OpT>
CUB_DETAIL_KERNEL_ATTRIBUTES //
__launch_bounds__(ChainedPolicyT::ActivePolicy::for_policy_t::block_threads) //
  void static_kernel(OffsetT num_items, OpT op)
{
  using active_policy_t = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using agent_t         = agent_block_striped_t<active_policy_t, OffsetT, OpT>;

  constexpr auto block_threads  = active_policy_t::block_threads;
  constexpr auto items_per_tile = active_policy_t::items_per_thread * block_threads;

  const auto tile_base     = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining = num_items - tile_base;
  const auto items_in_tile = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);

  if (items_in_tile == items_per_tile)
  {
    agent_t{tile_base, op}.template consume_tile<true>(items_per_tile, block_threads);
  }
  else
  {
    agent_t{tile_base, op}.template consume_tile<false>(items_in_tile, block_threads);
  }
}

/***********************************************************************************************************************
 * ForEachInExtents
 **********************************************************************************************************************/

// Retrieves the extent (dimension size) at a specific position in a multi-dimensional array
//
// This function efficiently returns the extent at the given position, optimizing for static extents by returning
// compile-time constants when possible. For dynamic extents, it returns the precomputed value to avoid runtime
// computation overhead.
template <int Position, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE_API auto extent_at(ExtentType extents, FastDivModType dynamic_extent)
{
  if constexpr (ExtentType::static_extent(Position) != ::cuda::std::dynamic_extent)
  {
    using extent_index_type   = typename ExtentType::index_type;
    using index_type          = implicit_prom_t<extent_index_type>;
    using unsigned_index_type = ::cuda::std::make_unsigned_t<index_type>;
    constexpr auto extent     = extents.static_extent(Position);
    return static_cast<unsigned_index_type>(extent);
  }
  else
  {
    return dynamic_extent;
  }
}

// Computes the product of extents in a specified range for multi-dimensional indexing.
// This function calculates the product of all extent dimensions from Start (inclusive) to End (exclusive).
//
// Performance characteristics:
//  - Static extents in range: Product computed at compile-time, zero runtime cost
//  - Dynamic extents present: Returns precomputed value, avoiding runtime multiplication
template <int Start, int End, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE_API auto get_extents_sub_size(ExtentType extents, FastDivModType extent_sub_size)
{
  if constexpr (cub::detail::are_extents_in_range_static<ExtentType>(Start, End))
  {
    using extent_index_type   = typename ExtentType::index_type;
    using index_type          = implicit_prom_t<extent_index_type>;
    using unsigned_index_type = ::cuda::std::make_unsigned_t<index_type>;
    auto sub_size             = cub::detail::size_range(extents, Start, End);
    return static_cast<unsigned_index_type>(sub_size);
  }
  else
  {
    return extent_sub_size;
  }
}

// Converts a linear index to a multi-dimensional coordinate at a specific position.
//
// This function performs the mathematical conversion from a linear (flat) index to the coordinate value at a specific
// position in a multi-dimensional array. It supports both row-major (layout_right) and column-major (layout_left)
// memory layouts, which affects the indexing calculation order.
//
// The mathematical formulation depends on the layout:
// - Right layout (row-major):   index_i = (index / product(extent[j] for j in [i+1, rank-1])) % extent[i]
// - Left layout (column-major): index_i = (index / product(extent[j] for j in [0, i])) % extent[i]
//
// This function leverages precomputed fast division and modulo operations to minimize runtime arithmetic overhead.
template <bool IsLayoutRight, int Position, typename IndexType, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE_API auto
coordinate_at(IndexType index, ExtentType extents, FastDivModType extent_sub_size, FastDivModType dynamic_extent)
{
  using cub::detail::for_each::extent_at;
  using cub::detail::for_each::get_extents_sub_size;
  using extent_index_type = typename ExtentType::index_type;
  constexpr auto start    = IsLayoutRight ? Position + 1 : 0;
  constexpr auto end      = IsLayoutRight ? ExtentType::rank() : Position;
  return static_cast<extent_index_type>((index / get_extents_sub_size<start, end>(extents, extent_sub_size))
                                        % extent_at<Position>(extents, dynamic_extent));
}

// Function object wrapper for applying operations with multi-dimensional coordinate conversion.
//
// The wrapped operation will be called with signature: `op(linear_index, coord_0, coord_1, ..., coord_n)`
// where the number of coordinate parameters matches the rank of the extents object.
//
// This wrapper is used internally by DeviceFor::ForEachInLayout/ForEachInExtents
template <typename OpT, typename ExtentsType, bool IsLayoutRight, typename FastDivModArrayT>
struct op_wrapper_extents_t
{
  OpT op; ///< The user-provided operation to be called with coordinates
  ExtentsType extents; ///< The multi-dimensional extents defining array dimensions
  FastDivModArrayT sub_sizes_div_array; ///< Precomputed fast division values for extent sub-products
  FastDivModArrayT extents_mod_array; ///< Precomputed fast modulo values for individual extents

  // Internal implementation that converts linear index to coordinates and calls the user operation
  template <typename IndexType, size_t... Positions>
  _CCCL_DEVICE_API void impl(IndexType i, ::cuda::std::index_sequence<Positions...>)
  {
    using cub::detail::for_each::coordinate_at;
    op(i,
       coordinate_at<IsLayoutRight, Positions>(
         i, extents, sub_sizes_div_array[Positions], extents_mod_array[Positions])...);
  }

  // Internal implementation that converts linear index to coordinates and calls the user operation
  template <typename IndexType, size_t... Positions>
  _CCCL_DEVICE_API void impl(IndexType i, ::cuda::std::index_sequence<Positions...>) const
  {
    using cub::detail::for_each::coordinate_at;
    op(i,
       coordinate_at<IsLayoutRight, Positions>(
         i, extents, sub_sizes_div_array[Positions], extents_mod_array[Positions])...);
  }

  // Function call operator that processes a linear index by converting it to multi-dimensional coordinates
  template <typename IndexType>
  _CCCL_DEVICE_API void operator()(IndexType i)
  {
    impl(i, ::cuda::std::make_index_sequence<ExtentsType::rank()>{});
  }

  // Function call operator that processes a linear index by converting it to multi-dimensional coordinates
  template <typename IndexType>
  _CCCL_DEVICE_API void operator()(IndexType i) const
  {
    impl(i, ::cuda::std::make_index_sequence<ExtentsType::rank()>{});
  }
};
} // namespace detail::for_each

CUB_NAMESPACE_END
