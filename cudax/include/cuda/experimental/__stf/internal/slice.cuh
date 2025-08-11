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

#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>
#include <cuda/experimental/__stf/utility/memory.cuh>

#include <iostream>

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
struct layout_stride : ::cuda::std::layout_stride
{
  template <class Extents>
  struct mapping : ::cuda::std::layout_stride::mapping<Extents>
  {
    constexpr mapping() = default;

    template <typename... A>
    constexpr _CCCL_HOST_DEVICE mapping(A&&... a)
        : ::cuda::std::layout_stride::mapping<Extents>(::std::forward<A>(a)...)
    {}

    template <typename... is_t>
    constexpr _CCCL_HOST_DEVICE auto operator()(is_t&&... is) const
    {
#ifdef CUDASTF_BOUNDSCHECK
      each_in_pack(
        [&](auto r, const auto& i) {
          _CCCL_ASSERT(i < this->extents().extent(r), "Index out of bounds.");
        },
        is...);
#endif
      return ::cuda::std::layout_stride::mapping<Extents>::operator()(::std::forward<is_t>(is)...);
    }
  };
};

/**
 * @brief Slice based on `mdspan`.
 *
 * @tparam T
 * @tparam dimensions
 */
template <typename T, size_t dimensions = 1>
using slice = mdspan<T, ::cuda::std::dextents<size_t, dimensions>, layout_stride>;

/**
 * @brief Compute how many dimensions of a slice are actually contiguous.
 *
 * @tparam T Base type for the slice
 * @tparam dimensions
 * @param span
 * @return size_t
 */
template <typename S>
size_t contiguous_dims(const S& span)
{
  if (span.rank() < 2)
  {
    return span.rank();
  }

  // Repeat until size != stride
  size_t contiguous_dims = 1;
  size_t prod_strides    = 1;
  while (contiguous_dims < span.rank()
         && span.extent(contiguous_dims - 1) * prod_strides == span.stride(contiguous_dims))
  {
    prod_strides *= span.stride(contiguous_dims);
    contiguous_dims++;
  }

  assert(contiguous_dims <= span.rank());
  return contiguous_dims;
}

/**
 * @brief Returns a `slice` object starting at `data` with the given extents and strides.
 *
 * @tparam ElementType Element type of the slice
 * @tparam Extents Extents of the slice in each dimension packed in a tuple
 * @tparam Strides Strides of the slice in each dimension
 * @param data pointer to the beginning of data
 * @param extents Values for the extents, e.g. `std::tuple{1024, 512}`
 * @param strides Values for the strides; fastest-moving one is implicitly 1.
 * @return auto `slice<ElementType, sizeof...(Extents)>`
 */
template <typename ElementType, typename... Extents, typename... Strides>
_CCCL_HOST_DEVICE auto make_slice(ElementType* data, const ::std::tuple<Extents...>& extents, const Strides&... strides)
{
  static_assert(sizeof...(Extents) == sizeof...(Strides) + 1);
  using Result = slice<ElementType, sizeof...(Extents)>;
  auto sizes   = ::std::apply(
    [&](auto&&... e) {
      return ::cuda::std::array<size_t, sizeof...(Extents)>{size_t(e)...};
    },
    extents);
  ::cuda::std::array<size_t, Result::rank()> mdspan_strides{1, size_t(strides)...};
  return Result(data, typename Result::mapping_type(sizes, mdspan_strides));
}

/**
 * @brief Returns a contiguous mdspan for a dense multidimensional array.
 *
 * @tparam T Element type
 * @param data Pointer to first element of the data
 * @param extents Sizes
 * @return auto
 */
template <typename T, typename... Extents>
_CCCL_HOST_DEVICE auto make_slice(T* data, const Extents&... extents)
{
  using Result = slice<T, sizeof...(Extents)>;
  static_assert(sizeof...(Extents) == Result::rank());

  if constexpr (sizeof...(Extents) == 0)
  {
    return Result(data, typename Result::mapping_type());
  }
  else
  {
    ::cuda::std::array<size_t, Result::rank()> sizes{size_t(extents)...}, strides;
    for (size_t i = 1; i < strides.size(); ++i)
    {
      strides[i] = sizes[i - 1];
    }

    strides[0] = sizes.size() != 0 && sizes.begin()[0] != 0;
    for (size_t i = 2; i < strides.size(); ++i)
    {
      strides.begin()[i] *= strides.begin()[i - 1];
    }
    return Result(data, typename Result::mapping_type(sizes, strides));
  }
}

namespace reserved
{

template <typename View, typename... Whatevs>
auto make_mdview(Whatevs&&... whatevs)
{
  return make_slice(::std::forward<Whatevs>(whatevs)...);
}

} // namespace reserved

#ifdef UNITTESTED_FILE
#  ifdef STF_HAS_UNITTEST_WITH_ARGS
UNITTEST("slice usual suspects: default ctor, copy, move, assign", (slice<double, 2>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  UNITTEST("should work for several sizes", 1u, 13u)
  {
    View s;
    EXPECT(s.extent(0) == 0);
    EXPECT(s.extent(1) == 0);
    EXPECT(s.size() == 0);

    double data[200 * 100];
    s = reserved::make_mdview<View>(data, ::std::make_tuple(200, unittest_param), 200);
    EXPECT(s.extent(0) == 200);
    EXPECT(s.extent(1) == unittest_param);
    EXPECT(s.size() == 200 * unittest_param);
  };
};

UNITTEST("2D slice basics", (slice<double, 2>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  // Bidimensional array of 200 rows of 100 elements each
  double data[200 * 100];
  // Access the first 13 elements of each row in that array
  auto s = reserved::make_mdview<View>(data, ::std::make_tuple(200, 13), 200);
  EXPECT(s.extent(0) == 200);
  EXPECT(s.extent(1) == 13);
  EXPECT(s.size() == 200 * 13);
  EXPECT(&s(0, 0) == data);
  EXPECT(&s(1, 0) == data + 1);
  EXPECT(&s(2, 0) == data + 2);
  EXPECT(&s(2, 3) == data + 2 + 3 * 200);
};

UNITTEST("3D slice basics", (slice<double, 3>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  // 3-dimensional array of 200 by 100 by 27
  double data[14 * 27 * 300];
  // Access the first 13*14 elements of each row in that array
  auto s = reserved::make_mdview<View>(data, ::std::tuple{200, 13, 14}, 300, 27 * 300);
  EXPECT(s.extent(0) == 200);
  EXPECT(s.extent(1) == 13);
  EXPECT(s.extent(2) == 14);
  EXPECT(s.size() == 200 * 13 * 14);
  EXPECT(&s(0, 0, 0) == data);
  EXPECT(&s(1, 0, 0) == data + 1);
  EXPECT(&s(2, 3, 0) == data + 2 + 3 * 300);
  EXPECT(&s(3, 5, 7) == data + 3 + 5 * 300 + 7 * 27 * 300);
};

UNITTEST("2D tiles", (slice<int, 2>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  // 6x4 matrix, 4 tiles of size 3x2, make sure we touch all entries of the
  // original matrix once when iterating over the tiles
  int data[6 * 4] = {0};
  auto s00        = reserved::make_mdview<View>(data + 0, ::std::tuple{3, 2}, 6);
  auto s10        = reserved::make_mdview<View>(data + 3, ::std::tuple{3, 2}, 6);
  auto s01        = reserved::make_mdview<View>(data + 12, ::std::tuple{3, 2}, 6);
  auto s11        = reserved::make_mdview<View>(data + 15, ::std::tuple{3, 2}, 6);

  for (size_t j = 0; j < s00.extent(1); j++)
  {
    for (size_t i = 0; i < s00.extent(0); i++)
    {
      s00(i, j) = 42;
    }
  }

  for (size_t j = 0; j < s10.extent(1); j++)
  {
    for (size_t i = 0; i < s10.extent(0); i++)
    {
      s10(i, j) = 42;
    }
  }

  for (size_t j = 0; j < s01.extent(1); j++)
  {
    for (size_t i = 0; i < s01.extent(0); i++)
    {
      s01(i, j) = 42;
    }
  }

  for (size_t j = 0; j < s11.extent(1); j++)
  {
    for (size_t i = 0; i < s11.extent(0); i++)
    {
      s11(i, j) = 42;
    }
  }

  for (size_t i = 0; i < 4 * 6; i++)
  {
    EXPECT(data[i] == 42);
  }
};

UNITTEST("contiguous_dims 1D", (slice<int, 1>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  int a[10];
  auto s = reserved::make_mdview<View>(a, 10);
  EXPECT(contiguous_dims(s) == 1);
};

UNITTEST("contiguous_dims 2D", (slice<int, 2>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  // This is non-contiguous
  int a[3 * 2];
  auto s = reserved::make_mdview<View>(a, ::std::tuple{3, 2}, 6);
  EXPECT(contiguous_dims(s) == 1);

  // This is contiguous
  int b[6 * 2];
  auto s2 = reserved::make_mdview<View>(b, ::std::tuple{6, 2}, 6);
  EXPECT(contiguous_dims(s2) == 2);
};

UNITTEST("contiguous_dims 3D", (slice<int, 3>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  int a[3 * 2 * 10];
  auto s = reserved::make_mdview<View>(a, ::std::tuple{3, 2, 10}, 6, 12);
  EXPECT(contiguous_dims(s) == 1);

  auto s2 = reserved::make_mdview<View>(a, ::std::tuple{3, 2, 10}, 3, 3 * 2);
  EXPECT(contiguous_dims(s2) == 3);

  auto s3 = reserved::make_mdview<View>(a, ::std::tuple{3, 2, 10}, 3, 10);
  EXPECT(contiguous_dims(s3) == 2);
};

UNITTEST("implicit contiguous strides", (slice<int, 3>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  int a[3 * 2 * 10];
  auto s = reserved::make_mdview<View>(a, 3, 2, 10);
  EXPECT(s.stride(0) == 1);
  EXPECT(s.stride(1) == 3);
  EXPECT(s.stride(2) == 3 * 2);
  EXPECT(contiguous_dims(s) == 3);
};

#  endif // STF_HAS_UNITTEST_WITH_ARGS
#endif // UNITTESTED_FILE

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

  /**
   * @brief Returns an array with sizes in all dimensions
   *
   * @return const std::array<size_t, dimensions>&
   */
  _CCCL_HOST_DEVICE ::std::array<size_t, rank()> get_sizes() const
  {
    ::std::array<size_t, rank()> result;
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

  // This transforms a tuple of (shape, 1D index) into a coordinate
  _CCCL_HOST_DEVICE coords_t index_to_coords(size_t index) const
  {
    ::std::array<size_t, shape_of::rank()> coordinates{};
    // for (::std::ptrdiff_t i = _dimensions - 1; i >= 0; i--)
    for (auto i : each(0, shape_of::rank()))
    {
      coordinates[i] = index % extent(i);
      index /= extent(i);
    }

    return ::std::apply(
      [](const auto&... e) {
        return ::std::make_tuple(e...);
      },
      coordinates);
  }

private:
  typename described_type::extents_type extents{};
  ::cuda::std::array<typename described_type::index_type, described_type::rank()> strides{};
};

#ifdef UNITTESTED_FILE
#  ifdef STF_HAS_UNITTEST_WITH_ARGS
UNITTEST("shape_of for slice and mdspan", (slice<double, 3>()))
{
  using View = ::std::remove_reference_t<decltype(unittest_param)>;
  using s    = shape_of<View>;

  static_assert(s::rank() == 3);
  EXPECT(s::rank() == 3);

  s shape_obj1;
  assert(shape_obj1.get_sizes() == (::std::array<size_t, 3>{0, 0, 0}));
  EXPECT(shape_obj1.size() == 0);
  EXPECT(shape_obj1.extent(0) == 0);
  EXPECT(shape_obj1.extent(1) == 0);
  EXPECT(shape_obj1.extent(2) == 0);
  EXPECT(shape_obj1.stride(0) == 0);
  EXPECT(shape_obj1.stride(1) == 0);
  EXPECT(shape_obj1.stride(2) == 0);

  s shape_obj2{2, 3, 4};
  assert(shape_obj2.get_sizes() == (::std::array<size_t, 3>{2, 3, 4}));
  EXPECT(shape_obj2.size() == 2 * 3 * 4);
  EXPECT(shape_obj2.extent(0) == 2);
  EXPECT(shape_obj2.extent(1) == 3);
  EXPECT(shape_obj2.extent(2) == 4);
  EXPECT(shape_obj2.stride(0) == 1);
  EXPECT(shape_obj2.stride(1) == 2);
  EXPECT(shape_obj2.stride(2) == 2 * 3);

  s shape_obj3{{200, 13, 14}, {300, 27 * 300}};
  assert(shape_obj3.get_sizes() == (::std::array<size_t, 3>{200, 13, 14}));
  EXPECT(shape_obj3.size() == 200 * 13 * 14);
  EXPECT(shape_obj3.extent(0) == 200);
  EXPECT(shape_obj3.extent(1) == 13);
  EXPECT(shape_obj3.extent(2) == 14);

  for (auto i : shape_obj3)
  {
    EXPECT(i[0] < shape_obj3.extent(0));
    EXPECT(i[1] < shape_obj3.extent(1));
    EXPECT(i[2] < shape_obj3.extent(2));
  }
};

UNITTEST("3D slice should be similar to 3D mdspan", (slice<double, 3>()))
{
  // 3-dimensional array of 3 by 4 by 5
  using View = ::std::remove_reference_t<decltype(unittest_param)>;

  double data[3 * 4 * 5];
  for (size_t i = 0; i < sizeof(data) / sizeof(data[0]); ++i)
  {
    data[i] = 1.0 * i;
  }
  // Access each element of the array
  auto m = reserved::make_mdview<View>(data, ::std::tuple{3, 4, 5}, 3, 3 * 4);

  EXPECT(m.extent(0) == 3);
  EXPECT(m.extent(1) == 4);
  EXPECT(m.extent(2) == 5);

  for (size_t i = 0; i < m.extent(0); ++i)
  {
    for (size_t j = 0; j < m.extent(1); ++j)
    {
      for (size_t k = 0; k < m.extent(2); ++k)
      {
        EXPECT(m(i, j, k) == i + j * 3 + k * 3 * 4);
      }
    }
  }

  // An array without explicit strides should behave just the same as the one above
  m = reserved::make_mdview<View>(data, 3, 4, 5);

  EXPECT(m.extent(0) == 3);
  EXPECT(m.extent(1) == 4);
  EXPECT(m.extent(2) == 5);

  for (size_t i = 0; i < m.extent(0); ++i)
  {
    for (size_t j = 0; j < m.extent(1); ++j)
    {
      for (size_t k = 0; k < m.extent(2); ++k)
      {
        EXPECT(m(i, j, k) == i + j * 3 + k * 3 * 4);
      }
    }
  }

  // To access a subset of the array, use the same strides but different extents

  auto m2 = reserved::make_mdview<View>(data, ::std::tuple{3, 2, 1}, 3, 3 * 4);

  EXPECT(m2.extent(0) == 3);
  EXPECT(m2.extent(1) == 2);
  EXPECT(m2.extent(2) == 1);

  for (size_t i = 0; i < m2.extent(0); ++i)
  {
    for (size_t j = 0; j < m2.extent(1); ++j)
    {
      for (size_t k = 0; k < m2.extent(2); ++k)
      {
        // printf("m2(%zu, %zu, %zu) = %g\n", i, j, k, m2(i, j, k));
        m2(i, j, k) = -1;
      }
    }
  }

  // first dim is the fastest moving
  double witness[3 * 4 * 5] = {
    /*0, 0, 0*/ -1,
    /*1, 0, 0*/ -1,
    /*2, 0, 0*/ -1,
    /*0, 1, 0*/ -1,
    /*1, 1, 0*/ -1,
    /*2, 1, 0*/ -1,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59};
  EXPECT(::std::equal(data, data + 3 * 4 * 5, witness));
};

#  endif // STF_HAS_UNITTEST_WITH_ARGS
#endif // UNITTESTED_FILE

/**
 * @brief Pins a slice in host memory for efficient use with CUDA primitives
 *
 * @tparam T memory type
 * @tparam dimensions slice dimension
 * @param s slice to pin
 */
template <typename T, typename... P>
bool pin(mdspan<T, P...>& s)
{
  // We need the rank as a constexpr value
  constexpr auto rank = mdspan<T, P...>::extents_type::rank();

  if (address_is_pinned(s.data_handle()))
  {
    return false;
  }

  if constexpr (rank == 0)
  {
    cuda_safe_call(pin_memory(s.data_handle(), 1));
  }
  else if constexpr (rank == 1)
  {
    cuda_safe_call(pin_memory(s.data_handle(), s.extent(0)));
  }
  else if constexpr (rank == 2)
  {
    switch (contiguous_dims(s))
    {
      case 1:
        for (size_t index_1 = 0; index_1 < s.extent(1); index_1++)
        {
          cuda_safe_call(pin_memory(&s(0, index_1) + index_1 * s.stride(1), s.extent(0)));
        }
        break;
      case 2:
        // fprintf(stderr, "PIN 2D - contiguous\n");
        cuda_safe_call(pin_memory(s.data_handle(), s.extent(0) * s.extent(1)));
        break;
      default:
        assert(false);
        abort();
    }
  }
  else
  {
    static_assert(rank == 3, "Dimensionality not supported.");
    switch (contiguous_dims(s))
    {
      case 1:
        for (size_t index_2 = 0; index_2 < s.extent(2); index_2++)
        {
          for (size_t index_1 = 0; index_1 < s.extent(1); index_1++)
          {
            // fprintf(stderr, "ADDR %d,%d,0 = %p \n", index_2, index_1, &s(index_2, index_1, 0));
            cuda_safe_call(pin_memory(&s(0, index_1, index_2), s.extent(0)));
          }
        }
        break;
      case 2:
        for (size_t index_2 = 0; index_2 < s.extent(2); index_2++)
        {
          cuda_safe_call(pin_memory(&s(0, 0, index_2), s.extent(0) * s.extent(1)));
        }
        break;
      case 3:
        // fprintf(stderr, "PIN 3D - contiguous\n");
        cuda_safe_call(pin_memory(s.data_handle(), s.extent(0) * s.extent(1) * s.extent(2)));
        break;
      default:
        assert(false);
        abort();
    }
  }

  return true;
}

/**
 * @brief Unpin the memory associated with an mdspan object.
 *
 * @tparam T The type of elements in the mdspan.
 * @tparam P The properties of the mdspan.
 * @param s The mdspan object to unpin memory for.
 */
template <typename T, typename... P>
void unpin(mdspan<T, P...>& s)
{
  // We need the rank as a constexpr value
  constexpr auto rank = mdspan<T, P...>::extents_type::rank();

  if constexpr (rank == 0)
  {
    unpin_memory(s.data_handle());
  }
  else if constexpr (rank == 1)
  {
    unpin_memory(s.data_handle());
  }
  else if constexpr (rank == 2)
  {
    switch (contiguous_dims(s))
    {
      case 1:
        for (size_t index_1 = 0; index_1 < s.extent(1); index_1++)
        {
          unpin_memory(&s(0, index_1) + index_1 * s.extent(0));
        }
        break;
      case 2:
        // fprintf(stderr, "PIN 2D - contiguous\n");
        unpin_memory(s.data_handle());
        break;
      default:
        assert(false);
        abort();
    }
  }
  else
  {
    static_assert(rank == 3, "Dimensionality not supported.");
    switch (contiguous_dims(s))
    {
      case 1:
        for (size_t index_2 = 0; index_2 < s.extent(2); index_2++)
        {
          for (size_t index_1 = 0; index_1 < s.extent(1); index_1++)
          {
            // fprintf(stderr, "ADDR %d,%d,0 = %p \n", index_2, index_1, &s(index_2, index_1, 0));
            unpin_memory(&s(0, index_1, index_2));
          }
        }
        break;
      case 2:
        for (size_t index_2 = 0; index_2 < s.extent(2); index_2++)
        {
          unpin_memory(&s(0, 0, index_2));
        }
        break;
      case 3:
        // fprintf(stderr, "PIN 3D - contiguous\n");
        unpin_memory(s.data_handle());
        break;
      default:
        assert(false);
        abort();
    }
  }
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code

//! @brief Computes a hash value for the contents of an mdspan.
//!
//! @details
//! This function recursively hashes all elements of the provided mdspan using either
//! `std::hash<E>` or a custom hash function, if available. The hash is computed by
//! traversing the multidimensional array in a dimension-major order and combining the
//! hash values of each element. If neither a standard nor custom hash is available for
//! the element type, the function prints an error and aborts.
//!
//! The function supports both rank deduction and explicit index sequence specification.
//! When called without an explicit index sequence, it generates one corresponding to the
//! mdspan's rank.
//!
//! @tparam E Element type stored in the mdspan.
//! @tparam X Extents type describing the shape of the mdspan.
//! @tparam L Layout policy for the mdspan.
//! @tparam A Accessor policy for the mdspan.
//! @tparam i... Index sequence used for multidimensional traversal (automatically deduced).
//!
//! @param[in] s The mdspan whose contents will be hashed.
//!
//! @return The combined hash value of all elements in the mdspan.
//!
//! @note Requires that either `std::hash<E>` or a custom hash function for `E` is defined.
//!       If neither is available, the function will print an error and terminate the program.
//! @note If the mdspan is empty, the function returns 0.
template <typename E, typename X, typename L, typename A, size_t... i>
size_t data_hash([[maybe_unused]] mdspan<E, X, L, A> s, ::std::index_sequence<i...> = {})
{
  using Slice = mdspan<E, X, L, A>;
  if constexpr (!reserved::has_std_hash_v<E> && !reserved::has_cudastf_hash_v<E>)
  {
    fprintf(stderr, "Error: cannot compute data_hash on a mdspan<E, ...> if ::std::hash<E> is not defined.\n");
    abort();
    return 0;
  }
  else
  {
    if constexpr (sizeof...(i) != Slice::rank())
    {
      return data_hash(s, ::std::make_index_sequence<Slice::rank()>());
    }
    else
    {
      if (s.size() == 0)
      {
        return 0;
      }

      size_t h          = 0;
      auto content_hash = [&](auto... indices) -> void {
        for (;;)
        {
          cuda::experimental::stf::hash_combine(h, s(indices...));

          bool bump = true;
          each_in_pack(
            [&](auto current_dim, size_t& index) {
              if (!bump)
              {
                return;
              }
              ++index;
              if (index >= s.extent(current_dim))
              {
                index = 0;
              }
              else
              {
                bump = false;
              }
            },
            indices...);
          if (bump)
          {
            // Done with all dimensions
            break;
          }
        }
      };
      content_hash((i * 0)...);
      return h;
    }
  }
}
_CCCL_DIAG_POP

/**
 * Write the content of the mdspan into a file
 */
template <typename E, typename X, typename L, typename A, size_t... i>
void data_dump([[maybe_unused]] mdspan<E, X, L, A> s,
               ::std::ostream& file        = ::std::cerr,
               ::std::index_sequence<i...> = {})
{
  using Slice = mdspan<E, X, L, A>;
  if constexpr (reserved::has_ostream_operator<E>::value)
  {
    if constexpr (sizeof...(i) != Slice::rank())
    {
      return data_dump(s, file, ::std::make_index_sequence<Slice::rank()>());
    }
    else
    {
      if (s.size() == 0)
      {
        return;
      }

      auto print_element = [&](auto... indices) -> void {
        for (;;)
        {
          file << s(indices...) << ' ';
          bool bump = true;
          each_in_pack(
            [&](auto current_dim, size_t& index) {
              if (!bump)
              {
                return;
              }
              ++index;
              if (index >= s.extent(current_dim))
              {
                // Done printing last element on the current dimension, move to the next one
                if (current_dim < Slice::rank() - 1)
                {
                  file << "; ";
                }
                index = 0;
              }
              else
              {
                bump = false;
              }
            },
            indices...);
          if (bump)
          {
            // Done with all dimensions
            break;
          }
        }
      };

      print_element((i * 0)...);
      file << '\n';
    }
  }
  else
  {
    ::std::cerr << "Unsupported typed." << ::std::endl;
  }
}

#ifdef UNITTESTED_FILE
UNITTEST("data_dump")
{
  double data[4] = {1, 2, 3, 4};
  auto s         = make_slice(data, 4);
  ::std::cerr << type_name<decltype(s)> << '\n';
  data_dump(s, ::std::cerr);
  auto s1 = make_slice(data, 2, 2);
  ::std::cerr << type_name<decltype(s1)> << '\n';
  data_dump(s1, ::std::cerr);
};

UNITTEST("data_hash")
{
  // Select a random filename
  double data[4] = {1, 2, 3, 4};
  auto s         = make_slice(data, 4);

  double same_data[4] = {1, 2, 3, 4};
  auto same_s         = make_slice(same_data, 4);

  EXPECT(data_hash(s) == data_hash(same_s));

  double other_data[5] = {1, 2, 3, 4, 5};
  auto other_s         = make_slice(other_data, 5);

  EXPECT(data_hash(s) != data_hash(other_s));
};
#endif

// Note this is implementing it in the cudastf namespace, not std::hash
template <typename... P>
struct hash<mdspan<P...>>
{
  ::std::size_t operator()(mdspan<P...> const& s) const noexcept
  {
    static constexpr auto _dimensions = mdspan<P...>::rank();
    // Combine hashes from the ptr, sizes and strides
    size_t h = 0;
    hash_combine(h, s.data_handle());

    if constexpr (_dimensions > 0)
    {
      for (size_t i = 0; i < _dimensions; i++)
      {
        hash_combine(h, s.extent(i));
      }
    }

    if constexpr (_dimensions > 1)
    {
      for (size_t i = 1; i < _dimensions; i++)
      {
        hash_combine(h, s.stride(i));
      }
    }

    return h;
  }
};

#ifdef UNITTESTED_FILE

UNITTEST("slice hash")
{
  double A[5 * 2];
  auto s  = make_slice(A, ::std::tuple{5, 2}, 5);
  auto s2 = make_slice(A, ::std::tuple{4, 2}, 5);

  size_t h  = hash<slice<double, 2>>{}(s);
  size_t h2 = hash<slice<double, 2>>{}(s2);

  EXPECT(h != h2);
};

UNITTEST("slice hash 3D")
{
  double A[5 * 2 * 40];

  // contiguous
  auto s = make_slice(A, ::std::tuple{5, 2, 40}, 5, 5 * 2);

  // non-contiguous
  auto s2 = make_slice(A, ::std::tuple{4, 2, 40}, 5, 5 * 2);

  size_t h  = hash<slice<double, 3>>{}(s);
  size_t h2 = hash<slice<double, 3>>{}(s2);

  EXPECT(h != h2);
};

UNITTEST("shape_of<slice> basics")
{
  using namespace cuda::experimental::stf;
  auto s1 = shape_of<slice<double, 3>>(1, 2, 3);
  auto s2 = shape_of<slice<double, 3>>(::std::tuple(1, 2, 3));
  EXPECT(s1 == s2);
};

#endif // UNITTESTED_FILE

} // namespace cuda::experimental::stf
