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

// template <class Alloc> tuple(allocator_arg_t, Alloc const&)

// Libc++ has to deduce the 'allocator_arg_t' parameter for this constructor
// as 'AllocArgT'. Previously libc++ has tried to support tags derived from
// 'allocator_arg_t' by using 'is_base_of<AllocArgT, allocator_arg_t>'.
// However this breaks whenever a 2-tuple contains a reference to an incomplete
// type as its first parameter. See PR27684.

#include <cuda/std/tuple>
#include <cuda/std/cassert>

#include "test_macros.h"

struct IncompleteType;

#define STATIC_EXTERN_DECL(name, type) \
  __device__ static type& name##_device(); \
  __host__   static type& name##_host();   \
  __host__ __device__ static type& name();

struct global {
    STATIC_EXTERN_DECL(inc1, IncompleteType)
    STATIC_EXTERN_DECL(inc2, IncompleteType)
    __host__ __device__ static const IncompleteType& cinc1();
    __host__ __device__ static const IncompleteType& cinc2();
};

int main(int, char**) {
    using IT = IncompleteType;
    { // try calling tuple(Tp const&...)
        using Tup = cuda::std::tuple<const IT&, const IT&>;
        Tup t(global::cinc1(), global::cinc2());
        assert(&cuda::std::get<0>(t) == &global::inc1());
        assert(&cuda::std::get<1>(t) == &global::inc2());
    }
    { // try calling tuple(Up&&...)
        using Tup = cuda::std::tuple<const IT&, const IT&>;
        Tup t(global::inc1(), global::inc2());
        assert(&cuda::std::get<0>(t) == &global::inc1());
        assert(&cuda::std::get<1>(t) == &global::inc2());
    }

  return 0;
}

struct IncompleteType {};

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

STATIC_EXTERN_IMPL(global::inc1, IncompleteType)
STATIC_EXTERN_IMPL(global::inc2, IncompleteType)

__host__ __device__ const IncompleteType& global::cinc1() {
    return inc1();
}

__host__ __device__ const IncompleteType& global::cinc2() {
    return inc2();
}
