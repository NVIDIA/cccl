//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO_EXTENT_CUH
#define _CUDAX__CUCO_EXTENT_CUH

#include <cuda/__cccl_config>
#include <cuda/std/cstddef>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cstddef>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
static constexpr _CUDA_VSTD::size_t dynamic_extent = static_cast<std::size_t>(-1);

/**
 * @brief Static extent class.
 *
 * @tparam SizeType Size type
 * @tparam N Extent
 */
template <typename SizeType, _CUDA_VSTD::size_t N = dynamic_extent>
struct extent
{
  using value_type = SizeType; ///< Extent value type

  constexpr extent() = default;

  /// Constructs from `SizeType`
  _CCCL_HOST_DEVICE constexpr extent(SizeType) noexcept {}

  /**
   * @brief Conversion to value_type.
   *
   * @return Extent size
   */
  _CCCL_HOST_DEVICE constexpr operator value_type() const noexcept
  {
    return N;
  }
};

/**
 * @brief Dynamic extent class.
 *
 * @tparam SizeType Size type
 */
template <typename SizeType>
struct extent<SizeType, dynamic_extent>
{
  using value_type = SizeType; ///< Extent value type

  /**
   * @brief Constructs extent from a given `size`.
   *
   * @param size The extent size
   */
  _CCCL_HOST_DEVICE constexpr extent(SizeType size) noexcept
      : value_{size}
  {}

  /**
   * @brief Conversion to value_type.
   *
   * @return Extent size
   */
  _CCCL_HOST_DEVICE constexpr operator value_type() const noexcept
  {
    return value_;
  }

private:
  value_type value_; ///< Extent value
};

} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_EXTENT_CUH
