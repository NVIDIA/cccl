//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__HIERARCHY_LEVEL_DIMENSIONS_CUH
#define _CUDAX__HIERARCHY_LEVEL_DIMENSIONS_CUH

#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda/experimental/__hierarchy/hierarchy_levels.cuh>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

namespace __detail
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

template <typename Dims>
struct dimensions_handler
{
  static constexpr bool is_type_supported = ::cuda::std::is_integral_v<Dims>;

  [[nodiscard]] _CCCL_HOST_DEVICE static constexpr auto translate(const Dims& d) noexcept
  {
    return dimensions<dimensions_index_type, ::cuda::std::dynamic_extent, 1, 1>(static_cast<unsigned int>(d));
  }
};

template <>
struct dimensions_handler<dim3>
{
  static constexpr bool is_type_supported = true;

  [[nodiscard]] _CCCL_HOST_DEVICE static constexpr auto translate(const dim3& d) noexcept
  {
    return dimensions<dimensions_index_type,
                      ::cuda::std::dynamic_extent,
                      ::cuda::std::dynamic_extent,
                      ::cuda::std::dynamic_extent>(d.x, d.y, d.z);
  }
};

template <typename Dims, Dims Val>
struct dimensions_handler<::cuda::std::integral_constant<Dims, Val>>
{
  static constexpr bool is_type_supported = true;

  [[nodiscard]] _CCCL_HOST_DEVICE static constexpr auto translate(const Dims& d) noexcept
  {
    return dimensions<dimensions_index_type, size_t(d), 1, 1>();
  }
};
} // namespace __detail

/**
 * @brief Type representing dimensions of a level in a thread hierarchy.
 *
 * This type combines a level type like grid_level or block_level with
 * a cuda::std::extents object to describe dimensions of a level in a thread hierarchy.
 * This type is not intended to be created explicitly and *_dims functions
 * creating them should be used instead. They will translate the input
 * arguments to a correct cuda::std::extents to be stored inside
 * level_dimensions.
 * While this type can be used to access the stored dimensions,
 * the main usage is to pass a number of level_dimensions objects
 * to make_hierarchy function in order to create a hierarchy.
 * This type does not store what the unit is for the stored dimensions,
 * it is instead implied by the level below it in a hierarchy object.
 * In case there is a need to store more information about a specific level,
 * for example some library-specific information, this type can be derived
 * from and the resulting type can be used to build the hierarchy.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * auto hierarchy = make_hierarchy(grid_dims(256), block_dims<8, 8, 8>());
 * assert(hierarchy.level(grid).dims.x == 256);
 * @endcode
 * @par
 *
 * @tparam Level
 *   Type indicating which hierarchy level this is
 *
 * @tparam Dimensions
 *   Type holding the dimensions of this level
 */
template <typename Level, typename Dimensions>
struct level_dimensions
{
  static_assert(::cuda::std::is_base_of_v<hierarchy_level, Level>);
  using level_type = Level;

  // Needs alignas to work around an issue with tuple
  alignas(16) Dimensions dims; // Unit for dimensions is implicit

  _CCCL_HOST_DEVICE constexpr level_dimensions(const Dimensions& d)
      : dims(d)
  {}
  _CCCL_HOST_DEVICE constexpr level_dimensions(Dimensions&& d)
      : dims(d)
  {}
  _CCCL_HOST_DEVICE constexpr level_dimensions()
      : dims(){};

#  if !defined(_CCCL_NO_THREE_WAY_COMPARISON) && !_CCCL_COMPILER(MSVC, <, 19, 39) && !_CCCL_COMPILER(GCC, <, 12)
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr bool operator==(const level_dimensions&) const noexcept = default;
#  else // ^^^ !_CCCL_NO_THREE_WAY_COMPARISON ^^^ / vvv _CCCL_NO_THREE_WAY_COMPARISON vvv
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const level_dimensions& left, const level_dimensions& right) noexcept
  {
    return left.dims == right.dims;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const level_dimensions& left, const level_dimensions& right) noexcept
  {
    return left.dims != right.dims;
  }
#  endif // _CCCL_NO_THREE_WAY_COMPARISON
};

/**
 * @brief Creates an instance of level_dimensions describing grid_level
 *
 * This function creates a statically sized level from up to three template arguments.
 */
template <size_t X, size_t Y = 1, size_t Z = 1>
_CCCL_HOST_DEVICE constexpr auto grid_dims() noexcept
{
  return level_dimensions<grid_level, dimensions<dimensions_index_type, X, Y, Z>>();
}

/**
 * @brief Creates an instance of level_dimensions describing grid_level
 *
 * This function creates the level from an integral or dim3 argument.
 */
template <typename T>
_CCCL_HOST_DEVICE constexpr auto grid_dims(T t) noexcept
{
  static_assert(__detail::dimensions_handler<T>::is_type_supported);
  auto dims = __detail::dimensions_handler<T>::translate(t);
  return level_dimensions<grid_level, decltype(dims)>(dims);
}

/**
 * @brief Creates an instance of level_dimensions describing cluster_level
 *
 * This function creates a statically sized level from up to three template arguments.
 */
template <size_t X, size_t Y = 1, size_t Z = 1>
_CCCL_HOST_DEVICE constexpr auto cluster_dims() noexcept
{
  return level_dimensions<cluster_level, dimensions<dimensions_index_type, X, Y, Z>>();
}

/**
 * @brief Creates an instance of level_dimensions describing cluster_level
 *
 * This function creates the level from an integral or dim3 argument.
 */
template <typename T>
_CCCL_HOST_DEVICE constexpr auto cluster_dims(T t) noexcept
{
  static_assert(__detail::dimensions_handler<T>::is_type_supported);
  auto dims = __detail::dimensions_handler<T>::translate(t);
  return level_dimensions<cluster_level, decltype(dims)>(dims);
}

/**
 * @brief Creates an instance of level_dimensions describing block_level
 *
 * This function creates a statically sized level from up to three template arguments.
 */
template <size_t X, size_t Y = 1, size_t Z = 1>
_CCCL_HOST_DEVICE constexpr auto block_dims() noexcept
{
  return level_dimensions<block_level, dimensions<dimensions_index_type, X, Y, Z>>();
}

/**
 * @brief Creates an instance of level_dimensions describing block_level
 *
 * This function creates the level from an integral or dim3 argument.
 */
template <typename T>
_CCCL_HOST_DEVICE constexpr auto block_dims(T t) noexcept
{
  static_assert(__detail::dimensions_handler<T>::is_type_supported);
  auto dims = __detail::dimensions_handler<T>::translate(t);
  return level_dimensions<block_level, decltype(dims)>(dims);
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__HIERARCHY_LEVEL_DIMENSIONS_CUH
