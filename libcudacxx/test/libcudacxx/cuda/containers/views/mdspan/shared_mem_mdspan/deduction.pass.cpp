//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/mdspan>

//  template<class CArray>
//    requires(is_array_v<CArray> && rank_v<CArray> == 1)
//    restrict_mdspan(CArray&)
//      -> restrict_mdspan<remove_all_extents_t<CArray>, extents<size_t, extent_v<CArray, 0>>>;
//
//  template<class Pointer>
//    requires(is_pointer_v<remove_reference_t<Pointer>>)
//    restrict_mdspan(Pointer&&)
//      -> restrict_mdspan<remove_pointer_t<remove_reference_t<Pointer>>, extents<size_t>>;
//
//  template<class ElementType, class... Integrals>
//    requires((is_convertible_v<Integrals, size_t> && ...) && sizeof...(Integrals) > 0)
//    explicit restrict_mdspan(ElementType*, Integrals...)
//      -> restrict_mdspan<ElementType, extents<size_t, maybe-static-ext<Integrals>...>>;
//
//  template<class ElementType, class OtherIndexType, size_t N>
//    restrict_mdspan(ElementType*, span<OtherIndexType, N>)
//      -> restrict_mdspan<ElementType, dextents<size_t, N>>;
//
//  template<class ElementType, class OtherIndexType, size_t N>
//    restrict_mdspan(ElementType*, const array<OtherIndexType, N>&)
//      -> restrict_mdspan<ElementType, dextents<size_t, N>>;
//
//  template<class ElementType, class IndexType, size_t... ExtentsPack>
//    restrict_mdspan(ElementType*, const extents<IndexType, ExtentsPack...>&)
//      -> restrict_mdspan<ElementType, extents<IndexType, ExtentsPack...>>;
//
//  template<class ElementType, class MappingType>
//    restrict_mdspan(ElementType*, const MappingType&)
//      -> restrict_mdspan<ElementType, typename MappingType::extents_type,
//                typename MappingType::layout_type>;
//
//  template<class MappingType, class AccessorType>
//    restrict_mdspan(const typename AccessorType::data_handle_type&, const MappingType&,
//           const AccessorType&)
//      -> restrict_mdspan<typename AccessorType::element_type, typename MappingType::extents_type,
//                typename MappingType::layout_type, AccessorType>;

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../CustomTestAccessors.h"
#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

template <class H, class M, class A>
__host__ __device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A& acc)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  // deduction from data_handle_type (including non-pointer), mapping and accessor
  static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle, map, acc)), MDS>);

  if constexpr (cuda::std::is_same_v<A, cuda::std::default_accessor<typename A::element_type>>)
  {
    // deduction from pointer and mapping
    // non-pointer data-handle-types have other accessor
    static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle, map)), MDS>);
    if constexpr (cuda::std::is_same_v<typename M::layout_type, cuda::std::layout_right>)
    {
      // deduction from pointer and extents
      static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle, map.extents())), MDS>);
    }
  }
}

template <class H, class L, class A>
__host__ __device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<int>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<char, D>(7)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

struct SizeTIntType
{
  size_t val;
  __host__ __device__ constexpr SizeTIntType(size_t val_)
      : val(val_)
  {}
  __host__ __device__ constexpr operator size_t() const noexcept
  {
    return size_t(val);
  }
};

template <class H>
_CCCL_CONCEPT can_deduce_layout = _CCCL_REQUIRES_EXPR((H))((cuda::shared_memory_mdspan(cuda::std::declval<H>(), 10)));

template <class H, class A, cuda::std::enable_if_t<can_deduce_layout<H>, int> = 0>
__host__ __device__ constexpr bool test_no_layout_deduction_guides(const H& handle, const A&)
{
  using T = typename A::element_type;
  // deduction from pointer alone
  static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle)),
                                     cuda::shared_memory_mdspan<T, cuda::std::extents<size_t>>>);
  // deduction from pointer and integral like
  static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle, 5, SizeTIntType(6))),
                                     cuda::shared_memory_mdspan<T, cuda::std::dextents<size_t, 2>>>);

  // P3029R1: deduction from `integral_constant`
  static_assert(
    cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle, cuda::std::integral_constant<size_t, 5>{})),
                         cuda::shared_memory_mdspan<T, cuda::std::extents<size_t, 5>>>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(
                           handle, cuda::std::integral_constant<size_t, 5>{}, cuda::std::dynamic_extent)),
                         cuda::shared_memory_mdspan<T, cuda::std::extents<size_t, 5, cuda::std::dynamic_extent>>>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle,
                                                             cuda::std::integral_constant<size_t, 5>{},
                                                             cuda::std::dynamic_extent,
                                                             cuda::std::integral_constant<size_t, 7>{})),
                         cuda::shared_memory_mdspan<T, cuda::std::extents<size_t, 5, cuda::std::dynamic_extent, 7>>>);

  cuda::std::array<char, 3> exts;
  // deduction from pointer and array
  static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle, exts)),
                                     cuda::shared_memory_mdspan<T, cuda::std::dextents<size_t, 3>>>);
  // deduction from pointer and span
  static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(handle, cuda::std::span(exts))),
                                     cuda::shared_memory_mdspan<T, cuda::std::dextents<size_t, 3>>>);
  return true;
}

template <class H, class A, cuda::std::enable_if_t<!can_deduce_layout<H>, int> = 0>
__host__ __device__ constexpr bool test_no_layout_deduction_guides(const H&, const A&)
{
  return false;
}

template <class H, class A>
__host__ __device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  mixin_extents(handle, cuda::std::layout_left(), acc);
  mixin_extents(handle, cuda::std::layout_right(), acc);
  mixin_extents(handle, layout_wrapping_integral<4>(), acc);

  // checking that there is no deduction happen for non-pointer handle type
  assert((test_no_layout_deduction_guides(handle, acc) == cuda::std::is_same_v<H, typename A::element_type*>) );
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible_v<T>, int> = 0>
__host__ __device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  mixin_layout(elements.data(), cuda::std::default_accessor<T>());
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible_v<T>, int> = 0>
__host__ __device__ TEST_CONSTEXPR_CXX20 void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  mixin_layout(elements.get_ptr(), cuda::std::default_accessor<T>());
}

__host__ __device__ constexpr bool test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();

  // deduction from array alone
  float a[12] = {};
  static_assert(cuda::std::is_same_v<decltype(cuda::shared_memory_mdspan(a)),
                                     cuda::shared_memory_mdspan<float, cuda::std::extents<size_t, 12>>>);
  unused(a);

  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_evil()
{
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();
  return true;
}

int main(int, char**)
{
  test();
  test_evil();

#if TEST_STD_VER >= 2020
  static_assert(test());
  static_assert(test_evil());
#endif // TEST_STD_VER >= 2020
  return 0;
}
