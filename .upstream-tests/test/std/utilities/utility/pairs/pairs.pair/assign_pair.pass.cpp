//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include <cuda/std/utility>
// cuda/std/memory not supported
// #include <cuda/std/memory>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  CopyAssignable() = default;
  CopyAssignable(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
};
struct MoveAssignable {
  MoveAssignable() = default;
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};

struct CountAssign {
  STATIC_MEMBER_VAR(copied, int);
  STATIC_MEMBER_VAR(moved, int);

  __host__ __device__ static void reset() { copied() = moved() = 0; }
  CountAssign() = default;
  __host__ __device__ CountAssign& operator=(CountAssign const&) { ++copied(); return *this; }
  __host__ __device__ CountAssign& operator=(CountAssign&&) { ++moved(); return *this; }
};

struct Incomplete;

#define STATIC_EXTERN_DECL(name, type) \
  __device__ static type& name##_device(); \
  __host__   static type& name##_host();   \
  __host__ __device__ static type& name();

struct global {
    STATIC_EXTERN_DECL(inc_obj, Incomplete)
};

int main(int, char**)
{
    {
        typedef cuda::std::pair<CopyAssignable, short> P;
        const P p1(CopyAssignable(), 4);
        P p2;
        p2 = p1;
        assert(p2.second == 4);
    }
    {
        using P = cuda::std::pair<int&, int&&>;
        int x = 42;
        int y = 101;
        int x2 = -1;
        int y2 = 300;
        P p1(x, cuda::std::move(y));
        P p2(x2, cuda::std::move(y2));
        p1 = p2;
        assert(p1.first == x2);
        assert(p1.second == y2);
    }
    {
        using P = cuda::std::pair<int, NonAssignable>;
        static_assert(!cuda::std::is_copy_assignable<P>::value, "");
    }
    {
        CountAssign::reset();
        using P = cuda::std::pair<CountAssign, CopyAssignable>;
        static_assert(cuda::std::is_copy_assignable<P>::value, "");
        P p;
        P p2;
        p = p2;
        assert(CountAssign::copied() == 1);
        assert(CountAssign::moved() == 0);
    }
    {
        using P = cuda::std::pair<int, MoveAssignable>;
        static_assert(!cuda::std::is_copy_assignable<P>::value, "");
    }
    {
        using P = cuda::std::pair<int, Incomplete&>;
        static_assert(!cuda::std::is_copy_assignable<P>::value, "");
        P p(42, global::inc_obj());
        unused(p);
        assert(&p.second == &global::inc_obj());
    }

  return 0;
}

struct Incomplete {};

#define STATIC_EXTERN_IMPL(name, type) \
  __device__ type& name##_device() {              \
    __shared__ type v;                 \
    return v;                          \
  }                                    \
  __host__ type& name##_host()   {              \
    static type v;                     \
    return v;                          \
  }                                    \
  type& name() {                       \
    NV_DISPATCH_TARGET(                \
      NV_IS_DEVICE, (                  \
        return name##_device();        \
      ),                               \
      NV_IS_HOST, (                    \
        return name##_host();          \
      )                                \
    )                                  \
  }

STATIC_EXTERN_IMPL(global::inc_obj, Incomplete)
