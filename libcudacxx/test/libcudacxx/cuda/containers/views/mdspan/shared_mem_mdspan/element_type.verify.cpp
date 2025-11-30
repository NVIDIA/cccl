//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <mdspan>

// template<class ElementType, class Extents, class LayoutPolicy = layout_right, class AccessorPolicy =
// default_accessor> class mdspan;
//
// Mandates:
//   - ElementType is a complete object type that is neither an abstract class type nor an array type.
//   - is_same_v<ElementType, typename AccessorPolicy::element_type> is true.

#include <cuda/mdspan>

#include "test_macros.h"

class AbstractClass
{
public:
  __host__ __device__ virtual void method() = 0;
};

__host__ __device__ void not_abstract_class()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: ElementType template parameter
  // may not be an abstract class}}
  cuda::shared_memory_mdspan<AbstractClass, cuda::std::extents<int>> m;
  unused(m);
}

__host__ __device__ void not_array_type()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: ElementType template parameter
  // may not be an array type}}
  cuda::shared_memory_mdspan<int[5], cuda::std::extents<int>> m;
  unused(m);
}

__host__ __device__ void element_type_mismatch()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: ElementType template parameter
  // must match AccessorPolicy::element_type}}
  cuda::
    shared_memory_mdspan<int, cuda::std::extents<int>, cuda::std::layout_right, cuda::std::default_accessor<const int>>
      m;
  unused(m);
}

int main(int, char**)
{
  return 0;
}
