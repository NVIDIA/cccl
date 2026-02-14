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
 * @brief Implementation of the pos4 class
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

#include <cuda/experimental/__stf/utility/cuda_attributes.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief pos4 class defining a position within a multidimensional object (default value 0 in each axis)
 */
class pos4
{
public:
  constexpr pos4() = default;

  /// Create a pos4 from its coordinates
  template <typename IntegralX, typename IntegralY = ssize_t, typename IntegralZ = ssize_t, typename IntegralT = ssize_t>
  _CCCL_HOST_DEVICE constexpr explicit pos4(IntegralX x, IntegralY y = 0, IntegralZ z = 0, IntegralT t = 0)
      : x(static_cast<ssize_t>(x))
      , y(static_cast<ssize_t>(y))
      , z(static_cast<ssize_t>(z))
      , t(static_cast<ssize_t>(t))
  {}

  /// Get the position along a specific axis
  _CCCL_HOST_DEVICE constexpr ssize_t get(size_t axis_id) const
  {
    switch (axis_id)
    {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        assert(axis_id == 3);
        return t;
    }
  }

  /// Get the position along a specific axis
  _CCCL_HOST_DEVICE constexpr ssize_t operator()(int axis_id) const
  {
    return get(axis_id);
  }

  /// Comparison of two pos4 in lexicographical order
  _CCCL_HOST_DEVICE constexpr bool operator<(const pos4& rhs) const
  {
    if (x != rhs.x)
    {
      return x < rhs.x;
    }
    if (y != rhs.y)
    {
      return y < rhs.y;
    }
    if (z != rhs.z)
    {
      return z < rhs.z;
    }
    return t < rhs.t;
  }

  /// Equality test between two pos4
  _CCCL_HOST_DEVICE constexpr bool operator==(const pos4& rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z && t == rhs.t;
  }

  /// Convert the pos4 to a string
  ::std::string to_string() const
  {
    return ::std::string("pos4(" + ::std::to_string(x) + "," + ::std::to_string(y) + "," + ::std::to_string(z) + ","
                         + ::std::to_string(t) + ")");
  }

  ssize_t x = 0;
  ssize_t y = 0;
  ssize_t z = 0;
  ssize_t t = 0;
};

/**
 * @brief dim4 class defining the size of a multidimensional object (default value 1 in each axis)
 */
class dim4
{
public:
  dim4() = default;

  /// Create a dim4 from its extents
  template <typename IntegralX, typename IntegralY = size_t, typename IntegralZ = size_t, typename IntegralT = size_t>
  _CCCL_HOST_DEVICE constexpr explicit dim4(IntegralX x, IntegralY y = 1, IntegralZ z = 1, IntegralT t = 1)
      : x(static_cast<size_t>(x))
      , y(static_cast<size_t>(y))
      , z(static_cast<size_t>(z))
      , t(static_cast<size_t>(t))
  {}

  /// Get the dimension along a specific axis
  _CCCL_HOST_DEVICE constexpr size_t get(size_t axis_id) const
  {
    switch (axis_id)
    {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        _CCCL_ASSERT(axis_id == 3, "Invalid axis");
        return t;
    }
  }

  /// Get the dimension along a specific axis
  _CCCL_HOST_DEVICE constexpr size_t operator()(int axis_id) const
  {
    return get(axis_id);
  }

  /// Get the total size (multiply all dimensions)
  _CCCL_HOST_DEVICE constexpr size_t size() const
  {
    const size_t result = x * y * z * t;
    return result;
  }

  /// Comparison of two dim4 in lexicographical order
  _CCCL_HOST_DEVICE constexpr bool operator<(const dim4& rhs) const
  {
    if (x != rhs.x)
    {
      return x < rhs.x;
    }
    if (y != rhs.y)
    {
      return y < rhs.y;
    }
    if (z != rhs.z)
    {
      return z < rhs.z;
    }
    return t < rhs.t;
  }

  /// Equality test between two dim4
  _CCCL_HOST_DEVICE constexpr bool operator==(const dim4& rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z && t == rhs.t;
  }

  /// Compute the dim4 class obtained by taking the minimum of two dim4 along each axis
  _CCCL_HOST_DEVICE static constexpr dim4 min(const dim4& a, const dim4& b)
  {
    return dim4(::std::min(a.x, b.x), ::std::min(a.y, b.y), ::std::min(a.z, b.z), ::std::min(a.t, b.t));
  }

  /// Get the 1D index of a coordinate defined by a pos4 class within a dim4 class
  _CCCL_HOST_DEVICE constexpr size_t get_index(const pos4& p) const
  {
    const size_t px = static_cast<size_t>(p.get(0));
    const size_t py = static_cast<size_t>(p.get(1));
    const size_t pz = static_cast<size_t>(p.get(2));
    const size_t pt = static_cast<size_t>(p.get(3));

    _CCCL_ASSERT(p.get(0) >= 0 && px <= x, "invalid position");
    _CCCL_ASSERT(p.get(1) >= 0 && py <= y, "invalid position");
    _CCCL_ASSERT(p.get(2) >= 0 && pz <= z, "invalid position");
    _CCCL_ASSERT(p.get(3) >= 0 && pt <= t, "invalid position");

    return px + x * (py + y * (pz + pt * z));
  }

  /// Get the maximum dimension that is not 1
  _CCCL_HOST_DEVICE constexpr size_t get_rank() const
  {
    if (t > 1)
    {
      return 3;
    }
    if (z > 1)
    {
      return 2;
    }
    if (y > 1)
    {
      return 1;
    }

    return 0;
  }

  /// Convert the dim4 to a string
  ::std::string to_string() const
  {
    return ::std::string("dim4(" + ::std::to_string(x) + "," + ::std::to_string(y) + "," + ::std::to_string(z) + ","
                         + ::std::to_string(t) + ")");
  }

  size_t x = 1;
  size_t y = 1;
  size_t z = 1;
  size_t t = 1;
};

/**
 * @brief An explicit shape is a shape or rank 'dimensions' where the bounds are explicit in each dimension.
 *
 * @tparam dimensions the rank of the shape
 */
template <size_t dimensions>
class box
{
public:
  ///@{ @name Constructors
  /// Construct an explicit shape from its lower and upper bounds (inclusive lower bounds, exclusive upper bounds)
  template <typename Int1, typename Int2>
  _CCCL_HOST_DEVICE box(const ::std::array<::std::pair<Int1, Int2>, dimensions>& s)
      : s(s)
  {}

  /// Construct an explicit shape from its upper bounds (exclusive upper bounds)
  template <typename Int>
  _CCCL_HOST_DEVICE box(const ::std::array<Int, dimensions>& sizes)
  {
    for (size_t ind : each(0, dimensions))
    {
      s[ind].first  = 0;
      s[ind].second = sizes[ind];
      if constexpr (::std::is_signed_v<Int>)
      {
        _CCCL_ASSERT(sizes[ind] >= 0, "Invalid shape.");
      }
    }
  }

  /// Construct an explicit shape from its upper bounds (exclusive upper bounds)
  template <typename... Int>
  _CCCL_HOST_DEVICE box(Int... args)
  {
    static_assert(sizeof...(Int) == dimensions, "Number of dimensions must match");
    each_in_pack(
      [&](auto i, const auto& e) {
        if constexpr (::std::is_arithmetic_v<::std::remove_reference_t<decltype(e)>>)
        {
          s[i].first  = 0;
          s[i].second = e;
        }
        else
        {
          // Assume a pair
          s[i].first  = e.first;
          s[i].second = e.second;
        }
      },
      args...);
  }

  /// Construct an explicit shape from its lower and upper bounds (inclusive lower bounds, exclusive upper bounds)
  template <typename... E>
  _CCCL_HOST_DEVICE box(::std::initializer_list<E>... args)
  {
    static_assert(sizeof...(E) == dimensions, "Number of dimensions must match");
    each_in_pack(
      [&](auto i, auto&& e) {
        _CCCL_ASSERT((e.size() == 1 || e.size() == 2), "Invalid arguments for box.");
        if (e.size() > 1)
        {
          s[i].first  = *e.begin();
          s[i].second = e.begin()[1];
        }
        else
        {
          s[i].first  = 0;
          s[i].second = *e.begin();
        }
      },
      args...);
  }

  // _CCCL_HOST_DEVICE box(const typename ::std::experimental::dextents<size_t, dimensions>& extents) {
  //     for (size_t i: each(0, dimensions)) {
  //         s[i].first = 0;
  //         s[i].second = extents[ind];
  //     }
  // }

  _CCCL_HOST_DEVICE void print()
  {
    printf("EXPLICIT SHAPE\n");
    for (size_t ind = 0; ind < dimensions; ind++)
    {
      assert(s[ind].first <= s[ind].second);
      printf("    %ld -> %ld\n", s[ind].first, s[ind].second);
    }
  }

  /// Get the number of elements along a dimension
  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_extent(size_t dim) const
  {
    return s[dim].second - s[dim].first;
  }

  /// Get the first coordinate (included) in a specific dimension
  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_begin(size_t dim) const
  {
    return s[dim].first;
  }

  /// Get the last coordinate (excluded) in a specific dimension
  _CCCL_HOST_DEVICE ::std::ptrdiff_t get_end(size_t dim) const
  {
    return s[dim].second;
  }

  /// Get the total number of elements in this explicit shape
  _CCCL_HOST_DEVICE ::std::ptrdiff_t size() const
  {
    if constexpr (dimensions == 1)
    {
      return s[0].second - s[0].first;
    }
    else
    {
      size_t res = 1;
      for (size_t d = 0; d < dimensions; d++)
      {
        res *= get_extent(d);
      }
      return res;
    }
  }

  /// get the dimensionnality of the explicit shape
  _CCCL_HOST_DEVICE constexpr size_t get_rank() const
  {
    return dimensions;
  }

  // Iterator class for box
  class iterator
  {
  private:
    box iterated; // A copy of the box being iterated
    ::std::array<::std::ptrdiff_t, dimensions> current; // Array to store the current position in each dimension

  public:
    _CCCL_HOST_DEVICE iterator(const box& b, bool at_end = false)
        : iterated(b)
    {
      if (at_end)
      {
        for (size_t i = 0; i < dimensions; ++i)
        {
          current[i] = iterated.get_end(i);
        }
      }
      else
      {
        for (size_t i = 0; i < dimensions; ++i)
        {
          current[i] = iterated.get_begin(i);
        }
      }
    }

    // Overload the dereference operator to get the current position
    _CCCL_HOST_DEVICE auto& operator*()
    {
      if constexpr (dimensions == 1UL)
      {
        return current[0];
      }
      else
      {
        return current;
      }
    }

    // Overload the pre-increment operator to move to the next position
    _CCCL_HOST_DEVICE iterator& operator++()
    {
      if constexpr (dimensions == 1UL)
      {
        current[0]++;
      }
      else
      {
        // Increment current with carry to next dimension
        for (size_t i : each(0, dimensions))
        {
          _CCCL_ASSERT(current[i] < iterated.get_end(i), "Attempt to increment past the end.");
          if (++current[i] < iterated.get_end(i))
          {
            // Found the new posish, now reset all lower dimensions to "zero"
            for (size_t j : each(0, i))
            {
              current[j] = iterated.get_begin(j);
            }
            break;
          }
        }
      }
      return *this;
    }

    // Overload the equality operator to check if two iterators are equal
    _CCCL_HOST_DEVICE bool operator==(const iterator& rhs) const
    { /*printf("EQUALITY TEST index %d %d shape equal ? %s\n", index,
           other.index, (&shape == &other.shape)?"yes":"no"); */
      _CCCL_ASSERT(iterated == rhs.iterated, "Cannot compare iterators in different boxes.");
      for (auto i : each(0, dimensions))
      {
        if (current[i] != rhs.current[i])
        {
          return false;
        }
      }
      return true;
    }

    // Overload the inequality operator to check if two iterators are not equal
    _CCCL_HOST_DEVICE bool operator!=(const iterator& other) const
    {
      return !(*this == other);
    }
  };

  // Functions to create the begin and end iterators
  _CCCL_HOST_DEVICE iterator begin()
  {
    return iterator(*this);
  }

  _CCCL_HOST_DEVICE iterator end()
  {
    return iterator(*this, true);
  }

  // Overload the equality operator to check if two shapes are equal
  _CCCL_HOST_DEVICE bool operator==(const box& rhs) const
  {
    for (size_t i : each(0, dimensions))
    {
      if (get_begin(i) != rhs.get_begin(i) || get_end(i) != rhs.get_end(i))
      {
        return false;
      }
    }
    return true;
  }

  _CCCL_HOST_DEVICE bool operator!=(const box& rhs) const
  {
    return !(*this == rhs);
  }

  using coords_t = array_tuple<size_t, dimensions>;

  // This transforms a tuple of (shape, 1D index) into a coordinate
  _CCCL_HOST_DEVICE coords_t index_to_coords(size_t index) const
  {
    // Help the compiler which may not detect that a device lambda is calling a device lambda
    CUDASTF_NO_DEVICE_STACK
    return make_tuple_indexwise<dimensions>([&](auto i) {
      // included
      const ::std::ptrdiff_t begin_i  = get_begin(i);
      const ::std::ptrdiff_t extent_i = get_extent(i);
      auto result                     = begin_i + (index % extent_i);
      index /= extent_i;
      return result;
    });
    CUDASTF_NO_DEVICE_STACK
  }

private:
  ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> s;
};

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
// Deduction guides
template <typename... Int>
box(Int...) -> box<sizeof...(Int)>;
template <typename... E>
box(::std::initializer_list<E>...) -> box<sizeof...(E)>;
template <typename E, size_t dimensions>
box(::std::array<E, dimensions>) -> box<dimensions>;
#endif // !_CCCL_DOXYGEN_INVOKED

#ifdef UNITTESTED_FILE
UNITTEST("box<3>")
{
  // Expect to iterate over Card({0, 1, 2}x{1, 2}x{10, 11, 12, 13}) = 3*2*4 = 24 items
  const size_t expected_cnt = 24;
  size_t cnt                = 0;
  auto shape                = box({0, 3}, {1, 3}, {10, 14});
  static_assert(::std::is_same_v<decltype(shape), box<3>>);
  for ([[maybe_unused]] const auto& pos : shape)
  {
    EXPECT(cnt < expected_cnt);
    cnt++;
  }

  EXPECT(cnt == expected_cnt);
};

UNITTEST("box<3> upper")
{
  // Expect to iterate over Card({0, 1, 2}x{0, 1}x{0, 1, 2, 3}) = 3*2*4 = 24 items
  const size_t expected_cnt = 24;
  size_t cnt                = 0;
  auto shape                = box(3, 2, 4);
  static_assert(::std::is_same_v<decltype(shape), box<3>>);
  for ([[maybe_unused]] const auto& pos : shape)
  {
    EXPECT(cnt < expected_cnt);
    cnt++;
  }

  EXPECT(cnt == expected_cnt);
};

UNITTEST("empty box<1>")
{
  auto shape = box({7, 7});
  static_assert(::std::is_same_v<decltype(shape), box<1>>);

  auto it_end   = shape.end();
  auto it_begin = shape.begin();
  if (it_end != it_begin)
  {
    fprintf(stderr, "Error: begin() != end()\n");
    abort();
  }

  // There should be no entry in this range
  for ([[maybe_unused]] const auto& pos : shape)
  {
    abort();
  }
};

UNITTEST("mix of integrals and pairs")
{
  const size_t expected_cnt = 12;
  size_t cnt                = 0;
  auto shape                = box(3, ::std::pair(1, 2), 4);
  static_assert(::std::is_same_v<decltype(shape), box<3>>);
  for ([[maybe_unused]] const auto& pos : shape)
  {
    EXPECT(cnt < expected_cnt);
    cnt++;
  }

  EXPECT(cnt == expected_cnt);
};

UNITTEST("pos4 large values")
{
  // Test that pos4 can handle values larger than int32 max (2^31-1 = 2,147,483,647)
  const ssize_t large_positive = 5000000000LL; // 5 billion
  const ssize_t large_negative = -3000000000LL; // -3 billion

  pos4 p(large_positive, large_negative, large_positive + 1000, large_negative - 1000);

  EXPECT(p.x == large_positive);
  EXPECT(p.y == large_negative);
  EXPECT(p.z == large_positive + 1000);
  EXPECT(p.t == large_negative - 1000);

  // Test accessors
  EXPECT(p.get(0) == large_positive);
  EXPECT(p.get(1) == large_negative);
  EXPECT(p.get(2) == large_positive + 1000);
  EXPECT(p.get(3) == large_negative - 1000);

  // Test operator()
  EXPECT(p(0) == large_positive);
  EXPECT(p(1) == large_negative);
};

UNITTEST("dim4 large values")
{
  // Test that dim4 can handle values larger than uint32 max (2^32-1 = 4,294,967,295)
  const size_t large_value = 6000000000ULL; // 6 billion

  dim4 d(large_value, large_value + 1000, large_value + 2000, large_value + 3000);

  EXPECT(d.x == large_value);
  EXPECT(d.y == large_value + 1000);
  EXPECT(d.z == large_value + 2000);
  EXPECT(d.t == large_value + 3000);

  // Test accessors
  EXPECT(d.get(0) == large_value);
  EXPECT(d.get(1) == large_value + 1000);
  EXPECT(d.get(2) == large_value + 2000);
  EXPECT(d.get(3) == large_value + 3000);

  // Test operator()
  EXPECT(d(0) == large_value);
  EXPECT(d(1) == large_value + 1000);
};

UNITTEST("dim4 very large total size")
{
  // Test dimensions that would exceed 2^32 when multiplied
  // 2000 * 2000 * 2000 * 64 = 1,024,000,000,000 = ~1T elements (2^40)
  dim4 d(2000, 2000, 2000, 64);

  const size_t expected_size = 2000ULL * 2000ULL * 2000ULL * 64ULL;
  EXPECT(d.size() == expected_size);
};

UNITTEST("pos4 dim4 interaction")
{
  // Test get_index with large coordinates
  const size_t large_dim = 100000; // 100K per dimension
  dim4 d(large_dim, large_dim, large_dim, large_dim);

  // Test position in the middle
  pos4 p(50000, 50000, 50000, 50000);
  size_t index = d.get_index(p);

  // Verify index calculation
  const size_t expected = 50000 + large_dim * (50000 + large_dim * (50000 + 50000 * large_dim));
  EXPECT(index == expected);

  // Test near the boundaries
  pos4 p_max(static_cast<ssize_t>(large_dim - 1),
             static_cast<ssize_t>(large_dim - 1),
             static_cast<ssize_t>(large_dim - 1),
             static_cast<ssize_t>(large_dim - 1));
  size_t max_index = d.get_index(p_max);
  EXPECT(max_index < d.size());
};

UNITTEST("dim4 comparison operators")
{
  dim4 d1(1000, 2000, 3000, 4000);
  dim4 d2(1000, 2000, 3000, 4000);
  dim4 d3(1000, 2000, 3000, 4001);

  // Test equality
  EXPECT(d1 == d2);
  EXPECT(!(d1 == d3));

  // Test lexicographical ordering
  EXPECT(d1 < d3);
  EXPECT(!(d3 < d1));
};

UNITTEST("pos4 comparison operators")
{
  pos4 p1(1000, -2000, 3000, -4000);
  pos4 p2(1000, -2000, 3000, -4000);
  pos4 p3(1000, -2000, 3000, -3999);

  // Test equality
  EXPECT(p1 == p2);
  EXPECT(!(p1 == p3));

  // Test lexicographical ordering
  EXPECT(p1 < p3);
  EXPECT(!(p3 < p1));
};

#endif // UNITTESTED_FILE

// So that we can create unordered_map of pos4 entries
template <>
struct hash<pos4>
{
  ::std::size_t operator()(pos4 const& s) const noexcept
  {
    return hash_all(s.x, s.y, s.z, s.t);
  }
};

// So that we can create maps of dim4 entries
template <>
struct hash<dim4>
{
  ::std::size_t operator()(const dim4& s) const noexcept
  {
    return hash_all(s.x, s.y, s.z, s.t);
  }
};
} // end namespace cuda::experimental::stf
