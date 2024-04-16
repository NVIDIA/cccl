//===----------------------------------------------------------------------===//
//
// Part of CUDA Next in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_NEXT_DETAIL_LEVEL_DIMENSIONS
#define _CUDA_NEXT_DETAIL_LEVEL_DIMENSIONS

#include <cuda/std/type_traits>

#include "hierarchy_levels.cuh"

namespace cuda_next
{

namespace detail
{

/* Keeping it around in case issues like https://github.com/NVIDIA/cccl/issues/522
template <typename T, size_t... Extents>
struct extents_corrected : public ::cuda::std::extents<T, Extents...> {
    using ::cuda::std::extents<T, Extents...>::extents;

    template <typename ::cuda::std::extents<T, Extents...>::rank_type Id>
    _CCCL_HOST_DEVICE constexpr auto extent_corrected() const {
        if constexpr (::cuda::std::extents<T, Extents...>::static_extent(Id) != ::cuda::std::dynamic_extent) {
            return this->static_extent(Id);
        }
        else {
            return this->extent(Id);
        }
    }
};
*/

// TODO might want to remove the below alias
template <size_t... Extents>
using dims = dimensions<dimensions_index_type, Extents...>;

template <typename Dims>
struct dimensions_handler
{
  static constexpr bool is_type_supported = ::cuda::std::is_integral_v<Dims>;

  static _CCCL_HOST_DEVICE constexpr auto translate(const Dims& d) noexcept
  {
    return dims<::cuda::std::dynamic_extent, 1, 1>(static_cast<unsigned int>(d));
  }
};

template <>
struct dimensions_handler<dim3>
{
  static constexpr bool is_type_supported = true;

  static _CCCL_HOST_DEVICE constexpr auto translate(const dim3& d) noexcept
  {
    return dims<::cuda::std::dynamic_extent, ::cuda::std::dynamic_extent, ::cuda::std::dynamic_extent>(d.x, d.y, d.z);
  }
};

template <typename Dims, Dims Val>
struct dimensions_handler<::cuda::std::integral_constant<Dims, Val>>
{
  static constexpr bool is_type_supported = true;

  static _CCCL_HOST_DEVICE constexpr auto translate(const Dims& d) noexcept
  {
    return dims<size_t(d), 1, 1>();
  }
};
} // namespace detail

/* Single instance of level_dimensions holds just one level,
  unit is implied by the next level in kernel_dimensions.
  Last unit is implied to be thread_level. */
template <typename Level, typename Dimensions>
struct level_dimensions
{
  static_assert(::cuda::std::is_base_of_v<hierarchy_level, Level>);
  using level_type = Level;

  // Needs alignas to work around an issue with tuple
  alignas(16) const Dimensions dims; // Unit for dimensions is implicit

  constexpr _CCCL_HOST_DEVICE level_dimensions(const Dimensions& d)
      : dims(d)
  {}
  constexpr _CCCL_HOST_DEVICE level_dimensions(Dimensions&& d)
      : dims(d)
  {}
  constexpr _CCCL_HOST_DEVICE level_dimensions(){};
};

template <size_t X, size_t Y = 1, size_t Z = 1>
_CCCL_HOST_DEVICE constexpr auto grid_dims() noexcept
{
  return level_dimensions<grid_level, detail::dims<X, Y, Z>>();
}

template <typename T>
_CCCL_HOST_DEVICE constexpr auto grid_dims(T t) noexcept
{
  static_assert(detail::dimensions_handler<T>::is_type_supported);
  auto dims = detail::dimensions_handler<T>::translate(t);
  return level_dimensions<grid_level, decltype(dims)>(dims);
}

template <size_t X, size_t Y = 1, size_t Z = 1>
_CCCL_HOST_DEVICE constexpr auto cluster_dims() noexcept
{
  return level_dimensions<cluster_level, detail::dims<X, Y, Z>>();
}

template <typename T>
_CCCL_HOST_DEVICE constexpr auto cluster_dims(T t) noexcept
{
  static_assert(detail::dimensions_handler<T>::is_type_supported);
  auto dims = detail::dimensions_handler<T>::translate(t);
  return level_dimensions<cluster_level, decltype(dims)>(dims);
}

template <size_t X, size_t Y = 1, size_t Z = 1>
_CCCL_HOST_DEVICE constexpr auto block_dims() noexcept
{
  return level_dimensions<block_level, detail::dims<X, Y, Z>>();
}

template <typename T>
_CCCL_HOST_DEVICE constexpr auto block_dims(T t) noexcept
{
  static_assert(detail::dimensions_handler<T>::is_type_supported);
  auto dims = detail::dimensions_handler<T>::translate(t);
  return level_dimensions<block_level, decltype(dims)>(dims);
}

} // namespace cuda_next
#endif
