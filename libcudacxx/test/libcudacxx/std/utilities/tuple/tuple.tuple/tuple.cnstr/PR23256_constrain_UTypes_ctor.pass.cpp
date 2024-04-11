//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class ...UTypes>
//    EXPLICIT(...) tuple(UTypes&&...)

// Check that the UTypes... ctor is properly disabled before evaluating any
// SFINAE when the tuple-like copy/move ctor should *clearly* be selected
// instead. This happens 'sizeof...(UTypes) == 1' and the first element of
// 'UTypes...' is an instance of the tuple itself. See PR23256.

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct UnconstrainedCtor
{
  int value_;

  __host__ __device__ UnconstrainedCtor()
      : value_(0)
  {}

  // Blows up when instantiated for any type other than int. Because the ctor
  // is constexpr it is instantiated by 'is_constructible' and 'is_convertible'
  // for Clang based compilers. GCC does not instantiate the ctor body
  // but it does instantiate the noexcept specifier and it will blow up there.
  template <typename T>
  __host__ __device__ constexpr UnconstrainedCtor(T value) noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
    static_assert(cuda::std::is_same<int, T>::value, "");
  }
};

struct ExplicitUnconstrainedCtor
{
  int value_;

  __host__ __device__ ExplicitUnconstrainedCtor()
      : value_(0)
  {}

  template <typename T>
  __host__ __device__ constexpr explicit ExplicitUnconstrainedCtor(T value) noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
    static_assert(cuda::std::is_same<int, T>::value, "");
  }
};

int main(int, char**)
{
  typedef UnconstrainedCtor A;
  typedef ExplicitUnconstrainedCtor ExplicitA;
  {
    static_assert(cuda::std::is_copy_constructible<cuda::std::tuple<A>>::value, "");
    static_assert(cuda::std::is_move_constructible<cuda::std::tuple<A>>::value, "");
    static_assert(cuda::std::is_copy_constructible<cuda::std::tuple<ExplicitA>>::value, "");
    static_assert(cuda::std::is_move_constructible<cuda::std::tuple<ExplicitA>>::value, "");
  }
  // cuda::std::allocator not supported
  /*
  {
      static_assert(cuda::std::is_constructible<
          cuda::std::tuple<A>,
          cuda::std::allocator_arg_t, cuda::std::allocator<void>,
          cuda::std::tuple<A> const&
      >::value, "");
      static_assert(cuda::std::is_constructible<
          cuda::std::tuple<A>,
          cuda::std::allocator_arg_t, cuda::std::allocator<void>,
          cuda::std::tuple<A> &&
      >::value, "");
      static_assert(cuda::std::is_constructible<
          cuda::std::tuple<ExplicitA>,
          cuda::std::allocator_arg_t, cuda::std::allocator<void>,
          cuda::std::tuple<ExplicitA> const&
      >::value, "");
      static_assert(cuda::std::is_constructible<
          cuda::std::tuple<ExplicitA>,
          cuda::std::allocator_arg_t, cuda::std::allocator<void>,
          cuda::std::tuple<ExplicitA> &&
      >::value, "");
  }
  */
  {
    cuda::std::tuple<A&&> t(cuda::std::forward_as_tuple(A{}));
    ((void) t);
    cuda::std::tuple<ExplicitA&&> t2(cuda::std::forward_as_tuple(ExplicitA{}));
    ((void) t2);
  }

  return 0;
}
