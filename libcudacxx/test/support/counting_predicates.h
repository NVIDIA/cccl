//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_COUNTING_PREDICATES_H
#define TEST_SUPPORT_COUNTING_PREDICATES_H

#include <cuda/std/cstddef>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename Predicate, typename Arg>
struct unary_counting_predicate
{
public:
  typedef Arg argument_type;
  typedef bool result_type;

  __host__ __device__ TEST_CONSTEXPR_CXX14 unary_counting_predicate(Predicate p)
      : p_(p)
      , count_(0)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX14 bool operator()(const Arg& a)
  {
    ++count_;
    return p_(a);
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 size_t count() const
  {
    return count_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 void reset()
  {
    count_ = 0;
  }

private:
  Predicate p_;
  size_t count_;
};

template <typename Predicate, typename Arg1, typename Arg2 = Arg1>
struct binary_counting_predicate
{
public:
  typedef Arg1 first_argument_type;
  typedef Arg2 second_argument_type;
  typedef bool result_type;

  __host__ __device__ TEST_CONSTEXPR_CXX14 binary_counting_predicate(Predicate p)
      : p_(p)
      , count_(0)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX14 bool operator()(const Arg1& a1, const Arg2& a2)
  {
    ++count_;
    return p_(a1, a2);
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 size_t count() const
  {
    return count_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX14 void reset()
  {
    count_ = 0;
  }

private:
  Predicate p_;
  size_t count_;
};

template <class Predicate>
class counting_predicate
{
  Predicate pred_;
  int* count_ = nullptr;

public:
  constexpr counting_predicate() = default;
  __host__ __device__ constexpr counting_predicate(Predicate pred, int& count)
      : pred_(cuda::std::move(pred))
      , count_(&count)
  {}

  template <class... Args>
  __host__ __device__ TEST_CONSTEXPR_CXX14 auto
  operator()(Args&&... args) -> decltype(pred_(cuda::std::forward<Args>(args)...))
  {
    ++(*count_);
    return pred_(cuda::std::forward<Args>(args)...);
  }

  template <class... Args>
  __host__ __device__ TEST_CONSTEXPR_CXX14 auto
  operator()(Args&&... args) const -> decltype(pred_(cuda::std::forward<Args>(args)...))
  {
    ++(*count_);
    return pred_(cuda::std::forward<Args>(args)...);
  }
};

#if TEST_STD_VER > 2014

template <class Predicate>
counting_predicate(Predicate pred, int& count) -> counting_predicate<Predicate>;

#endif // TEST_STD_VER > 2014

#endif // TEST_SUPPORT_COUNTING_PREDICATES_H
