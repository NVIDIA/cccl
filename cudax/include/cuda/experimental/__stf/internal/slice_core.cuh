//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Definition of `slice` and related artifacts
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/mdspan>

#include <cuda/experimental/__stf/utility/dimensions.cuh>
#include <cuda/experimental/__stf/utility/each.cuh>
#include <cuda/experimental/__stf/utility/to_tuple.cuh>

namespace cuda::experimental::stf
{
using ::cuda::std::mdspan;
}

namespace cuda::experimental::stf
{

/**
 * @brief Abstraction for the shape of a structure such as a multidimensional view, i.e. everything but the data
 * itself. A type to be used with cudastf must specialize this template.
 *
 * @tparam T The data corresponding to this shape.
 *
 * Any specialization must be default constructible, copyable, and assignable. The required methods are constructor from
 * `const T&`. All other definitions are optional and should provide full information about the structure ("shape") of
 * an object of type `T`, without actually allocating any data for it.
 *
 * @class shape_of
 */
template <typename T>
class shape_of;

#if defined(CUDASTF_BOUNDSCHECK) && defined(NDEBUG)
#  error "CUDASTF_BOUNDSCHECK requires that NDEBUG is not defined."
#endif

/**
 * @brief A layout stride that can be used with `mdspan`.
 *
 * In debug mode (i.e., `NDEBUG` is not defined) all uses of `operator()()` are bounds-checked by means of `assert`.
 */
// struct layout_stride : ::cuda::std::layout_stride
// {
//   template <class Extents>
//   struct mapping : ::cuda::std::layout_stride::mapping<Extents>
//   {
//     constexpr mapping() = default;
//
//     template <typename... A>
//     constexpr _CCCL_HOST_DEVICE mapping(A&&... a)
//         : ::cuda::std::layout_stride::mapping<Extents>(::std::forward<A>(a)...)
//     {}
//
//     template <typename... is_t>
//     constexpr _CCCL_HOST_DEVICE auto operator()(is_t&&... is) const
//     {
// #ifdef CUDASTF_BOUNDSCHECK
//       each_in_pack(
//         [&](auto r, const auto& i) {
//           _CCCL_ASSERT(i < this->extents().extent(r), "Index out of bounds.");
//         },
//         is...);
// #endif
//       return ::cuda::std::layout_stride::mapping<Extents>::operator()(::std::forward<is_t>(is)...);
//     }
//   };
// };

using layout_stride = ::cuda::std::layout_stride;

/**
 * @brief Slice based on `mdspan`.
 *
 * @tparam T
 * @tparam dimensions
 */
template <typename T, size_t dimensions = 1>
using slice = mdspan<T, ::cuda::std::dextents<size_t, dimensions>, layout_stride>;

/** @brief Specialization of the `shape` template for `mdspan`
 *
 * @extends shape_of
 */
template <typename T, typename... P>
class shape_of<mdspan<T, P...>>
{
public:
  using described_type = mdspan<T, P...>;
  using coords_t       = array_tuple<size_t, described_type::rank()>;

  /**
   * @brief Dimensionality of the slice.
   *
   */
  _CCCL_HOST_DEVICE static constexpr size_t rank()
  {
    return described_type::rank();
  }

#ifndef __CUDACC_RTC__
  // Functions to create the begin and end iterators
  _CCCL_HOST_DEVICE auto begin()
  {
    ::std::array<size_t, rank()> sizes;
    unroll<rank()>([&](auto i) {
      sizes[i] = extents.extent(i);
    });
    return box<rank()>(sizes).begin();
  }

  _CCCL_HOST_DEVICE auto end()
  {
    ::std::array<size_t, rank()> sizes;
    unroll<rank()>([&](auto i) {
      sizes[i] = extents.extent(i);
    });
    return box<rank()>(sizes).end();
  }
#endif // __CUDACC_RTC__

  /**
   * @brief The default constructor builds a shape with size 0 in all dimensions.
   *
   * All `shape_of` specializations must define this constructor.
   */
  shape_of() = default;

  /**
   * @name Copies a shape.
   *
   * All `shape_of` specializations must define this constructor.
   */
  shape_of(const shape_of&) = default;

  /**
   * @brief Extracts the shape from a given slice.
   *
   * @param x object to get the shape from
   *
   * All `shape_of` specializations must define this constructor.
   */
  explicit _CCCL_HOST_DEVICE shape_of(const described_type& x)
      : extents(x.extents())
      , strides(x.mapping().strides())
  {}

#ifndef __CUDACC_RTC__
  /**
   * @brief Create a new `shape_of` object from a `coords_t` object.
   *
   * @tparam Sizes Types (all must convert to `size_t` implicitly)
   * @param size0 Size for the first dimension (
   * @param sizes Sizes of data for the other dimensions, one per dimension
   *
   * Initializes dimensions to `size0`, `sizes...`. This constructor is optional.
   */
  explicit shape_of(const coords_t& sizes)
      : extents(reserved::to_cuda_array(sizes))
  {
    size_t product_sizes = 1;
    unroll<rank()>([&](auto i) {
      if (i == 0)
      {
        strides[i] = ::std::get<0>(sizes) != 0;
      }
      else
      {
        product_sizes *= extent(i - 1);
        strides[i] = product_sizes;
      }
    });
  }

  /**
   * @brief Create a new `shape_of` object taking exactly `dimension` sizes.
   *
   * @tparam Sizes Types (all must convert to `size_t` implicitly)
   * @param size0 Size for the first dimension (
   * @param sizes Sizes of data for the other dimensions, one per dimension
   *
   * Initializes dimensions to `size0`, `sizes...`. This constructor is optional.
   */
  template <typename... Sizes>
  explicit shape_of(size_t size0, Sizes&&... sizes)
      : extents(size0, ::std::forward<Sizes>(sizes)...)
  {
    static_assert(sizeof...(sizes) + 1 == rank(), "Wrong number of arguments passed to shape_of.");

    strides[0]           = size0 != 0;
    size_t product_sizes = 1;
    for (size_t i = 1; i < rank(); ++i)
    {
      product_sizes *= extent(i - 1);
      strides[i] = product_sizes;
    }
  }

  ///@{ @name Constructors
  explicit shape_of(const ::std::array<size_t, rank()>& sizes)
      : extents(reserved::convert_to_cuda_array(sizes))
  {}

  explicit shape_of(const ::std::array<size_t, rank()>& sizes, const ::std::array<size_t, rank()>& _strides)
      : shape_of(sizes)
  {
    if constexpr (rank() > 1)
    {
      size_t n = 0;
      for (auto i = _strides.begin(); i != _strides.end(); ++i)
      {
        strides[n++] = *i;
      }
      for (auto i = strides.rbegin() + 1; i != strides.rend(); ++i)
      {
        *i *= i[1];
      }
    }
  }
  ///@}
#endif // __CUDACC_RTC__

  bool operator==(const shape_of& other) const
  {
    return extents == other.extents && strides == other.strides;
  }

  /**
   * @brief Returns the size of a slice in a given dimension (run-time version)
   *
   * @tparam dim The dimension to get the size for. Must be `< dimensions`
   * @return size_t The size
   *
   * This member function is optional.
   */
  constexpr _CCCL_HOST_DEVICE size_t extent(size_t dim) const
  {
    assert(dim < rank());
    return extents.extent(dim);
  }

  /**
   * @brief Get the stride for a specified dimension.
   *
   * @param dim The dimension for which to get the stride.
   * @return The stride for the specified dimension.
   */
  constexpr _CCCL_HOST_DEVICE size_t stride(size_t dim) const
  {
    assert(dim < rank());
    return strides[dim];
  }

  /**
   * @brief Total size of the slice in all dimensions (product of the sizes)
   *
   * @return size_t The total size
   *
   * This member function is optional.
   */
  _CCCL_HOST_DEVICE size_t size() const
  {
    size_t result = 1;
    for (size_t i = 0; i != rank(); ++i)
    {
      result *= extents.extent(i);
    }
    return result;
  }

#ifndef __CUDACC_RTC__
  /**
   * @brief Returns an array with sizes in all dimensions
   *
   * @return const std::array<size_t, dimensions>&
   */
  _CCCL_HOST_DEVICE ::cuda::std::array<size_t, rank()> get_sizes() const
  {
    ::cuda::std::array<size_t, rank()> result;
    for (size_t i = 0; i < rank(); ++i)
    {
      result[i] = extents.extent(i);
    }
    return result;
  }

  /**
   * @brief Returns the extents as a dim4 type
   *
   * @return const dim4
   */
  dim4 get_data_dims() const
  {
    static_assert(rank() < 5);

    dim4 dims(0);
    if constexpr (rank() >= 1)
    {
      dims.x = static_cast<int>(extents.extent(0));
    }
    if constexpr (rank() >= 2)
    {
      dims.y = static_cast<int>(extents.extent(1));
    }
    if constexpr (rank() >= 3)
    {
      dims.z = static_cast<int>(extents.extent(2));
    }
    if constexpr (rank() >= 4)
    {
      dims.t = static_cast<int>(extents.extent(3));
    }
    return dims;
  }

  /**
   * @brief Set contiguous strides based on the provided sizes.
   *
   * @note This function should not be called (calling it will engender a link-time error).
   *
   * @param sizes The sizes to set the contiguous strides.
   */
  void set_contiguous_strides(const ::std::array<size_t, rank()>& sizes);

  /**
   * @brief Create a new `described_type` object with the given base pointer.
   *
   * @param base Base pointer to create the described_type object.
   * @return described_type Newly created described_type object.
   */
  described_type create(T* base) const
  {
    return described_type(base, typename described_type::mapping_type(extents, strides));
  }
#endif // __CUDACC_RTC__

  // This transforms a tuple of (shape, 1D index) into a coordinate
  _CCCL_HOST_DEVICE coords_t index_to_coords(size_t index) const
  {
    ::cuda::std::array<size_t, shape_of::rank()> coordinates{};
    // for (int i = shape_of::rank() - 1; i >= 0; i--)
    for (auto i : each(0, shape_of::rank()))
    {
      coordinates[i] = index % extent(i);
      index /= extent(i);
    }

    return ::cuda::std::apply(
      [](const auto&... e) {
        return ::cuda::std::make_tuple(e...);
      },
      coordinates);
  }

private:
  typename described_type::extents_type extents{};
  ::cuda::std::array<typename described_type::index_type, described_type::rank()> strides{};
};

} // namespace cuda::experimental::stf
