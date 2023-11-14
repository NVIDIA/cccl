//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <cuda/std/iterator>
// template <class C> constexpr auto size(const C& c) -> decltype(c.size());         // C++17
// template <class T, size_t N> constexpr size_t size(const T (&array)[N]) noexcept; // C++17

#include <cuda/std/iterator>
#include <cuda/std/cassert>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#include <cuda/std/vector>
#endif
#include <cuda/std/array>
#if defined(_LIBCUDACXX_HAS_LIST)
#include <cuda/std/list>
#endif
#include <cuda/std/initializer_list>
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#include <cuda/std/string_view>
#endif

#include "test_macros.h"

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#if TEST_STD_VER > 14
#include <cuda/std/string_view>
#endif
#endif


template<typename C>
__host__ __device__
void test_container( C& c)
{
//  Can't say noexcept here because the container might not be
    assert ( cuda::std::size(c)   == c.size());
}

template<typename C>
__host__ __device__
void test_const_container( const C& c )
{
//  Can't say noexcept here because the container might not be
    assert ( cuda::std::size(c)   == c.size());
}

template<typename T>
__host__ __device__
void test_const_container( const cuda::std::initializer_list<T>& c)
{
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::size(c)); // our cuda::std::size is conditionally noexcept
    assert ( cuda::std::size(c)   == c.size());
}

template<typename T>
__host__ __device__
void test_container( cuda::std::initializer_list<T>& c )
{
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::size(c)); // our cuda::std::size is conditionally noexcept
    assert ( cuda::std::size(c)   == c.size());
}

template<typename T, size_t Sz>
__host__ __device__
void test_const_array( const T (&array)[Sz] )
{
    ASSERT_NOEXCEPT(cuda::std::size(array));
    assert ( cuda::std::size(array) == Sz );
}

STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL int arrA [] { 1, 2, 3 };

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_VECTOR)
    cuda::std::vector<int> v; v.push_back(1);
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    cuda::std::list<int>   l; l.push_back(2);
#endif
    cuda::std::array<int, 1> a; a[0] = 3;
    cuda::std::initializer_list<int> il = { 4 };
#if defined(_LIBCUDACXX_HAS_VECTOR)
    test_container ( v );
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    test_container ( l );
#endif
    test_container ( a );
    test_container ( il );

#if defined(_LIBCUDACXX_HAS_VECTOR)
    test_const_container ( v );
#endif
#if defined(_LIBCUDACXX_HAS_LIST)
    test_const_container ( l );
#endif
    test_const_container ( a );
    test_const_container ( il );

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#if TEST_STD_VER > 14
    cuda::std::string_view sv{"ABC"};
    test_container ( sv );
    test_const_container ( sv );
#endif
#endif

    test_const_array ( arrA );

  return 0;
}
