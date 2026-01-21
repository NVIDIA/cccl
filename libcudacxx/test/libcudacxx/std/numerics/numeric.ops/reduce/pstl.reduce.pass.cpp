//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// XFAIL: true

// template<class ExecutionPolicy, class ForwardIterator>
//   typename iterator_traits<ForwardIterator>::value_type
//     reduce(ExecutionPolicy&& exec,
//            ForwardIterator first, ForwardIterator last);
// template<class ExecutionPolicy, class ForwardIterator, class T, class BinaryOperation>
//   T reduce(ExecutionPolicy&& exec,
//            ForwardIterator first, ForwardIterator last, T init,
//            BinaryOperation binary_op);

#include <cuda/std/__pstl/reduce.h>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/numeric>

#include "test_execution_policies.h"
#include "test_iterators.h"
#include "test_macros.h"

EXECUTION_POLICY_SFINAE_TEST(reduce);

static_assert(!sfinae_test_reduce<cuda::std::execution::parallel_policy, int*, int*>);
static_assert(sfinae_test_reduce<cuda::std::execution::parallel_policy, int*, int*>);

static_assert(!sfinae_test_reduce<cuda::std::execution::parallel_policy, int*, int*, int>);
static_assert(sfinae_test_reduce<cuda::std::execution::parallel_policy, int*, int*, int>);

static_assert(!sfinae_test_reduce<cuda::std::execution::parallel_policy, int*, int*, int, int (*)(int, int)>);
static_assert(sfinae_test_reduce<cuda::std::execution::parallel_policy, int*, int*, int, int (*)(int, int)>);

class MoveOnly
{
  int data_;

public:
  __host__ __device__ constexpr MoveOnly(int data = 1)
      : data_(data)
  {}

  MoveOnly(const MoveOnly&)            = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;

  __host__ __device__ constexpr MoveOnly(MoveOnly&& x)
      : data_(x.data_)
  {
    x.data_ = 0;
  }
  __host__ __device__ constexpr MoveOnly& operator=(MoveOnly&& x)
  {
    data_   = x.data_;
    x.data_ = 0;
    return *this;
  }

  __host__ __device__ constexpr int get() const
  {
    return data_;
  }

  __host__ __device__ friend constexpr bool operator==(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ == y.data_;
  }
  __host__ __device__ friend constexpr bool operator!=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ != y.data_;
  }
  __host__ __device__ friend constexpr bool operator<(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ < y.data_;
  }
  __host__ __device__ friend constexpr bool operator<=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ <= y.data_;
  }
  __host__ __device__ friend constexpr bool operator>(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ > y.data_;
  }
  __host__ __device__ friend constexpr bool operator>=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ >= y.data_;
  }

#if TEST_STD_VER > 2017 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  __host__ __device__ friend constexpr auto operator<=>(const MoveOnly&, const MoveOnly&) = default;
#endif // TEST_STD_VER > 2017 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  __host__ __device__ constexpr MoveOnly operator+(const MoveOnly& x) const
  {
    return MoveOnly(data_ + x.data_);
  }
  __host__ __device__ constexpr MoveOnly operator+(const int x) const
  {
    return MoveOnly(data_ + x);
  }
  __host__ __device__ constexpr MoveOnly operator*(const MoveOnly& x) const
  {
    return MoveOnly(data_ * x.data_);
  }

  __host__ __device__ constexpr operator int() const noexcept
  {
    return data_;
  }

  template <class T>
  void operator,(T const&) = delete;
};
static_assert(cuda::std::is_convertible_v<MoveOnly, MoveOnly>);

constexpr int max_size = 350;
int data[max_size];

template <class Iter, class ValueT>
struct Test
{
  template <class Policy>
  void operator()(Policy&& policy)
  {
    const cuda::std::pair<int, int> runs[] = {{0, 34}, {1, 36}, {2, 39}, {100, 5184}, {max_size, 61809}};
    for (const auto& pair : runs)
    {
      auto [size, expected] = pair;

      {
        decltype(auto) ret =
          cuda::std::reduce(policy, Iter(data), Iter(data + size), ValueT(34), [](ValueT i, ValueT j) -> ValueT {
            return i + j + ValueT{2};
          });
        static_assert(cuda::std::is_same_v<decltype(ret), ValueT>);
        assert(ret == ValueT{expected});
      }
      {
        decltype(auto) ret = cuda::std::reduce(policy, Iter(data), Iter(data + size), ValueT(34));
        static_assert(cuda::std::is_same_v<decltype(ret), ValueT>);
        assert(ret == ValueT{expected - 2 * size});
      }
      {
        decltype(auto) ret = cuda::std::reduce(policy, Iter(data), Iter(data + size));
        static_assert(cuda::std::is_same_v<decltype(ret), typename cuda::std::iterator_traits<Iter>::value_type>);
        assert(ret == expected - 2 * size - 34);
      }
    }
  }
};

__host__ void test()
{
  cuda::std::iota(data, data + max_size, 0);
  types::for_each(types::forward_iterator_list<int*>{}, types::apply_type_identity{[](auto v) {
                    using Iter = typename decltype(v)::type;
                    types::for_each(
                      types::type_list<int, MoveOnly>{},
                      TestIteratorWithPolicies<types::partial_instantiation<Test, Iter>::template apply>{});
                  }});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)

  return 0;
}
