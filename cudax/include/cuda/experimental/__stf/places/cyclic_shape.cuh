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
 * @brief Define explicit shapes
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

#include <cuda/experimental/__stf/utility/dimensions.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief An cyclic shape is a shape or rank 'dimensions' where the bounds are
 * explicit in each dimension, and where we jump between elements with a
 * specific stride.
 *
 * @tparam dimensions the rank of the shape
 */
template <size_t dimensions = 1>
class cyclic_shape
{
public:
  ///@{ @name Constructors

  /// Construct and explicit shape from a list of lower and upper bounds
  _CCCL_HOST_DEVICE explicit cyclic_shape(const ::std::array<::std::tuple<size_t, size_t, size_t>, dimensions>& list)
  {
    size_t i = 0;
    for (auto& e : list)
    {
      begins[i]  = ::std::get<0>(e);
      ends[i]    = ::std::get<1>(e);
      strides[i] = ::std::get<2>(e);
      // printf("CYCLIC SHAPE : dim %ld : %ld %ld %ld\n", i, begin, end, stride);
      i++;
    }
  }

  /// Construct and explicit shape from a list of lower and upper bounds
  _CCCL_HOST_DEVICE cyclic_shape(const ::std::array<size_t, dimensions>& begins_,
                                 const ::std::array<size_t, dimensions>& ends_,
                                 const ::std::array<size_t, dimensions>& strides_)
      : begins(begins_)
      , ends(ends_)
      , strides(strides_)
  {}

  _CCCL_HOST_DEVICE void print() const
  {
    printf("CYCLIC SHAPE\n");
    for (size_t i = 0; i < dimensions; i++)
    {
      printf("\t%ld:%ld:%ld\n", begins[i], ends[i], strides[i]);
    }
  }

  ///@}

  /// Get the total number of elements in this explicit shape
  _CCCL_HOST_DEVICE ::std::ptrdiff_t size() const
  {
    size_t res = 1;
    for (size_t d = 0; d < dimensions; d++)
    {
      // TODO: should this be (ends[d] - begins[d] + strides[d] - 1) / strides[d];
      res *= (ends[d] - begins[d]) / strides[d];
    }

    return res;
  }

  // Overload the equality operator to check if two shapes are equal
  _CCCL_HOST_DEVICE bool operator==(const cyclic_shape& other) const
  {
    for (size_t i = 0; i < dimensions; ++i)
    {
      if (begins[i] != other.begins[i] || ends[i] != other.ends[i] || strides[i] != other.strides[i])
      {
        // printf("BEGIN[%ld] %ld != OTHER BEGIN[%ld] %ld\n", i, begins[i] , i , other.begins[i]);
        return false;
      }
    }
    return true;
  }

  /// get the dimensionnality of the explicit shape
  _CCCL_HOST_DEVICE size_t get_rank() const
  {
    return dimensions;
  }

  // Iterator class for cyclic_shape
  class iterator
  {
  private:
    cyclic_shape* shape;
    size_t index;
    ::std::array<size_t, dimensions> current; // Array to store the current position in each dimension

  public:
    _CCCL_HOST_DEVICE iterator(cyclic_shape& s, size_t idx = 0)
        : shape(&s)
        , index(idx)
        , current(s.begins)
    {}

    // Overload the dereference operator to get the current position
    _CCCL_HOST_DEVICE auto& operator*()
    {
      if constexpr (dimensions == 1)
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
      index++;

      for (auto dim : each(0, dimensions))
      {
        current[dim] += shape->strides[dim]; // Increment the current position by the stride
        // fprintf(stderr, "current[%ld] += %ld\n", dim, shape.get_stride(dim));

        // printf("TEST current[%ld] (%ld) > shape.ends[%ld] (%ld)\n", dim, current[dim], dim, shape.ends[dim]);
        if (current[dim] < shape->ends[dim])
        {
          break;
        }
        // Wrap around to the begin value if the current position exceeds the end value
        current[dim] = shape->begins[dim];
      }

      return *this;
    }

    // Overload the equality operator to check if two iterators are equal
    _CCCL_HOST_DEVICE bool operator==(const iterator& other) const
    { /*printf("EQUALITY TEST index %d %d shape equal ? %s\n", index,
         other.index, (&shape == &other.shape)?"yes":"no"); */
      // printf("check == : index %d %d => %s shape equal ? %s\n", index, other.index, (index ==
      // other.index)?"yes":"no", (shape == other.shape)?"yes":"no");
      return shape == other.shape && index == other.index;
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
    size_t total_positions = 1;
    for (auto i : each(0, dimensions))
    {
      auto begin  = begins[i]; // included
      auto end    = ends[i]; // excluded
      auto stride = strides[i];
      total_positions *= (end - begin + stride - 1) / stride;
    }
    // The first invalid position is index with "total_positions", even if there is no entry in the range
    return iterator(*this, total_positions);
  }

private:
  ::std::array<size_t, dimensions> begins;
  ::std::array<size_t, dimensions> ends;
  ::std::array<size_t, dimensions> strides;
};

/**
 * @brief Apply a round-robin distribution of elements
 */
class cyclic_partition
{
public:
  cyclic_partition() = default;

  template <size_t dimensions>
  _CCCL_HOST_DEVICE static auto apply(const box<dimensions>& in, pos4 place_position, dim4 grid_dims)
  {
    ::std::array<size_t, dimensions> begins;
    ::std::array<size_t, dimensions> ends;
    ::std::array<size_t, dimensions> strides;
    for (size_t d = 0; d < dimensions; d++)
    {
      begins[d]  = in.get_begin(d) + place_position.get(d);
      ends[d]    = in.get_end(d);
      strides[d] = grid_dims.get(d);
    }

    return cyclic_shape<dimensions>(begins, ends, strides);
  }

  template <typename mdspan_shape_t>
  _CCCL_HOST_DEVICE static auto apply(const mdspan_shape_t& in, pos4 place_position, dim4 grid_dims)
  {
    constexpr size_t dimensions = mdspan_shape_t::rank();

    ::std::array<::std::tuple<size_t, size_t, size_t>, dimensions> bounds;
    for (size_t d = 0; d < dimensions; d++)
    {
      // Can't assign the whole tuple because the assignment needs to run on device.
      ::std::get<0>(bounds[d]) = place_position.get(d);
      ::std::get<1>(bounds[d]) = in.extent(d);
      ::std::get<2>(bounds[d]) = grid_dims.get(d);
    }

    return cyclic_shape<dimensions>(bounds);
  }

  _CCCL_HOST_DEVICE static pos4 get_executor(pos4 /*unused*/, dim4 /*unused*/, dim4 /*unused*/)
  {
    abort();
    return pos4(0);
  }
};

#ifdef UNITTESTED_FILE
UNITTEST("cyclic_shape<3>")
{
  // Expect to iterate over Card({0, 2, 4, 6}x{1, 3}x{10, 15}) = 4*2*2 = 16 items
  const size_t expected_cnt = 16;
  size_t cnt                = 0;
  cyclic_shape<3> shape{{::std::make_tuple(0, 7, 2), ::std::make_tuple(1, 5, 2), ::std::make_tuple(10, 20, 5)}};
  for ([[maybe_unused]] const auto& pos : shape)
  {
    //// Use the position in each dimension
    // ::std::cout << "(";
    // for (const auto& p: pos) {
    //    ::std::cout << p << ", ";
    //}
    // ::std::cout << ")" << ::std::endl;
    EXPECT(cnt < expected_cnt);
    cnt++;
  }

  EXPECT(cnt == expected_cnt);
};

UNITTEST("empty cyclic_shape<1>")
{
  cyclic_shape<1> shape{{::std::make_tuple(7, 7, 1)}};

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

UNITTEST("apply cyclic ")
{
  box<3> e({{0, 7}, {1, 5}, {10, 20}});

  size_t dim0 = 2;
  size_t dim1 = 2;

  size_t cnt = 0;

  size_t expected_cnt = 7 * 4 * 10;

  for (size_t i0 = 0; i0 < dim0; i0++)
  {
    for (size_t i1 = 0; i1 < dim1; i1++)
    {
      auto c = cyclic_partition::apply(e, pos4(i0, i1), dim4(dim0, dim1));
      for ([[maybe_unused]] const auto& pos : c)
      {
        //// Use the position in each dimension
        // ::std::cout << " (";
        // for (const auto& p: pos) {
        //    ::std::cout << p << ", ";
        //}
        // ::std::cout << ")" << ::std::endl;

        // avoid infinite loops
        EXPECT(cnt < expected_cnt);

        cnt++;
      }
    }
  }

  // We must have gone over all elements exactly once
  EXPECT(cnt == expected_cnt);
};
#endif // UNITTESTED_FILE
} // namespace cuda::experimental::stf
