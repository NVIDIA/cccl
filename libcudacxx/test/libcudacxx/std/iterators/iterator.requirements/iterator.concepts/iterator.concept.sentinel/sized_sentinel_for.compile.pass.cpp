//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// [iterator.concept.sizedsentinel], concept sized_sentinel_for
//
// template<class S, class I>
//   inline constexpr bool disable_sized_sentinel_for = false;
//
// template<class S, class I>
//   concept sized_sentinel_for = see below;

#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

static_assert(cuda::std::sized_sentinel_for<random_access_iterator<int*>, random_access_iterator<int*>>, "");
static_assert(!cuda::std::sized_sentinel_for<bidirectional_iterator<int*>, bidirectional_iterator<int*>>, "");

struct int_sized_sentinel
{
#if TEST_STD_VER > 2017
  __host__ __device__ friend bool operator==(int_sized_sentinel, int*);
#else
  __host__ __device__ friend bool operator==(int_sized_sentinel, int*);
  __host__ __device__ friend bool operator==(int*, int_sized_sentinel);
  __host__ __device__ friend bool operator!=(int_sized_sentinel, int*);
  __host__ __device__ friend bool operator!=(int*, int_sized_sentinel);
#endif
  __host__ __device__ friend cuda::std::ptrdiff_t operator-(int_sized_sentinel, int*);
  __host__ __device__ friend cuda::std::ptrdiff_t operator-(int*, int_sized_sentinel);
};
static_assert(cuda::std::sized_sentinel_for<int_sized_sentinel, int*>, "");
// int_sized_sentinel is not an iterator.
static_assert(!cuda::std::sized_sentinel_for<int*, int_sized_sentinel>, "");

struct no_default_ctor
{
  no_default_ctor() = delete;
#if TEST_STD_VER > 2017
  __host__ __device__ bool operator==(cuda::std::input_or_output_iterator auto) const
  {
    return true;
  };
#else
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ bool operator==(const It&) const
  {
    return true;
  };
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ bool operator!=(const It&) const
  {
    return false;
  };
#endif
  __host__ __device__ friend cuda::std::ptrdiff_t operator-(no_default_ctor, int*);
  __host__ __device__ friend cuda::std::ptrdiff_t operator-(int*, no_default_ctor);
};
static_assert(!cuda::std::sized_sentinel_for<no_default_ctor, int*>, "");

struct not_copyable
{
  not_copyable()                    = default;
  not_copyable(not_copyable const&) = delete;
#if TEST_STD_VER > 2017
  __host__ __device__ bool operator==(cuda::std::input_or_output_iterator auto) const
  {
    return true;
  };
#else
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ bool operator==(const It&) const
  {
    return true;
  };
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ bool operator!=(const It&) const
  {
    return false;
  };
#endif
  __host__ __device__ friend cuda::std::ptrdiff_t operator-(not_copyable, int*);
  __host__ __device__ friend cuda::std::ptrdiff_t operator-(int*, not_copyable);
};
static_assert(!cuda::std::sized_sentinel_for<not_copyable, int*>, "");

struct double_sized_sentinel
{
  __host__ __device__ friend bool operator==(double_sized_sentinel, double*);
  __host__ __device__ friend int operator-(double_sized_sentinel, double*);
  __host__ __device__ friend int operator-(double*, double_sized_sentinel);
};

namespace cuda::std
{
template <>
inline constexpr bool disable_sized_sentinel_for<double_sized_sentinel, double*> = true;
} // namespace cuda::std

static_assert(!cuda::std::sized_sentinel_for<double_sized_sentinel, double*>, "");

struct only_one_sub_op
{
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int>>
  __host__ __device__ friend bool operator==(only_one_sub_op, It);
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int>>
  __host__ __device__ friend cuda::std::ptrdiff_t operator-(only_one_sub_op, It);
};
static_assert(!cuda::std::sized_sentinel_for<only_one_sub_op, int*>, "");

struct wrong_return_type
{
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int>>
  __host__ __device__ friend bool operator==(wrong_return_type, It);
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int>>
  __host__ __device__ friend cuda::std::ptrdiff_t operator-(wrong_return_type, It);
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int>>
  __host__ __device__ friend void operator-(It, wrong_return_type);
};
static_assert(!cuda::std::sized_sentinel_for<wrong_return_type, int*>, "");

// Standard types
static_assert(cuda::std::sized_sentinel_for<int*, int*>, "");
static_assert(cuda::std::sized_sentinel_for<const int*, int*>, "");
static_assert(cuda::std::sized_sentinel_for<const int*, const int*>, "");
static_assert(cuda::std::sized_sentinel_for<int*, const int*>, "");

int main(int, char**)
{
  return 0;
}
