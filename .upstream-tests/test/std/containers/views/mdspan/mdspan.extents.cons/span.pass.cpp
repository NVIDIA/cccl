//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11, nvrtc && nvcc-12.0, nvrtc && nvcc-12.1

#include <cuda/std/mdspan>
#include <cuda/std/cassert>
#include "../mdspan.extents.util/extents_util.hpp"
#include "../my_int.hpp"

// TYPED_TEST(TestExtents, array_ctor)
template<class T>
__host__ __device__ void test_span_con()
{
    using TestFixture = TestExtents<T>;
    TestFixture t;

    auto s = cuda::std::span<const size_t, t.dyn_sizes.size()>( t.dyn_sizes );
    auto e = typename TestFixture::extents_type(s);
    assert(e == t.exts);
}

template< class T, class IndexType, size_t N, class = void >
struct is_span_cons_avail : cuda::std::false_type {};

template< class T, class IndexType, size_t N >
struct is_span_cons_avail< T
                         , IndexType
                         , N
                         , cuda::std::enable_if_t< cuda::std::is_same< decltype( T{ cuda::std::declval<cuda::std::span<IndexType, N>>() } )
                                                                     , T
                                                                     >::value
                                                 >
                         > : cuda::std::true_type {};

template< class T, class IndexType, size_t N >
constexpr bool is_span_cons_avail_v = is_span_cons_avail< T, IndexType, N >::value;

int main(int, char**)
{
    test_span_con< cuda::std::tuple_element_t< 0, extents_test_types > >();
    test_span_con< cuda::std::tuple_element_t< 1, extents_test_types > >();
    test_span_con< cuda::std::tuple_element_t< 2, extents_test_types > >();
    test_span_con< cuda::std::tuple_element_t< 3, extents_test_types > >();
    test_span_con< cuda::std::tuple_element_t< 4, extents_test_types > >();
    test_span_con< cuda::std::tuple_element_t< 5, extents_test_types > >();

    static_assert( is_span_cons_avail_v< cuda::std::dextents<int,2>, int   , 2 > == true , "" );

    static_assert( is_span_cons_avail_v< cuda::std::dextents<int,2>, my_int, 2 > == true , "" );

    // Constraint: rank consistency
    static_assert( is_span_cons_avail_v< cuda::std::dextents<int,1>, int   , 2 > == false, "" );

    // Constraint: convertibility
    static_assert( is_span_cons_avail_v< cuda::std::dextents<int,1>, my_int_non_convertible          , 1 > == false, "" );

    // Constraint: nonthrow-constructibility
    static_assert( is_span_cons_avail_v< cuda::std::dextents<int,1>, my_int_non_nothrow_constructible, 1 > == false, "" );

    return 0;
}
