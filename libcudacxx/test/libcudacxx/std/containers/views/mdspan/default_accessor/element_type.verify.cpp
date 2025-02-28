//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <mdspan>

// template<class ElementType>
// class default_accessor;

// ElementType is required to be a complete object type that is neither an abstract class type nor an array type.

#include <cuda/std/mdspan>

#include "test_macros.h"

class AbstractClass
{
public:
  __host__ __device__ virtual void method() = 0;
};

__host__ __device__ void not_abstract_class()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}default_accessor: template argument may
  // not be an abstract class}}
  cuda::std::default_accessor<AbstractClass> acc;
  unused(acc);
}

__host__ __device__ void not_array_type()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}default_accessor: template argument may
  // not be an array type}}
  cuda::std::default_accessor<int[5]> acc;
  unused(acc);
}

int main(int, char**)
{
  return 0;
}
