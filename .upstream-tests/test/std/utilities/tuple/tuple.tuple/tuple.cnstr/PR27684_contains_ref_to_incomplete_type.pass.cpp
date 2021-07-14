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
#ifdef __CUDA_ARCH__
__device__ extern IncompleteType inc1;
__device__ extern IncompleteType inc2;
__device__ IncompleteType const& cinc1 = inc1;
__device__ IncompleteType const& cinc2 = inc2;
#else
extern IncompleteType inc1;
extern IncompleteType inc2;
IncompleteType const& cinc1 = inc1;
IncompleteType const& cinc2 = inc2;
#endif

int main(int, char**) {
    using IT = IncompleteType;
    { // try calling tuple(Tp const&...)
        using Tup = cuda::std::tuple<const IT&, const IT&>;
        Tup t(cinc1, cinc2);
        assert(&cuda::std::get<0>(t) == &inc1);
        assert(&cuda::std::get<1>(t) == &inc2);
    }
    { // try calling tuple(Up&&...)
        using Tup = cuda::std::tuple<const IT&, const IT&>;
        Tup t(inc1, inc2);
        assert(&cuda::std::get<0>(t) == &inc1);
        assert(&cuda::std::get<1>(t) == &inc2);
    }

  return 0;
}

struct IncompleteType {};
#ifdef __CUDA_ARCH__
__device__ IncompleteType inc1;
__device__ IncompleteType inc2;
#else
IncompleteType inc1;
IncompleteType inc2;
#endif
