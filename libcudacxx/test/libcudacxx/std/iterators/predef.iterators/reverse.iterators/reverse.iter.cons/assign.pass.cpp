//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// template <class U>
// reverse_iterator& operator=(const reverse_iterator<U>& u); // constexpr since C++17

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It, class U>
TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 void test(U u) {
    const cuda::std::reverse_iterator<U> r2(u);
    cuda::std::reverse_iterator<It> r1;
    cuda::std::reverse_iterator<It>& rr = r1 = r2;
    assert(base(r1.base()) == base(u));
    assert(&rr == &r1);
}

struct Base { };
struct Derived : Base { };

struct ToIter {
    typedef cuda::std::bidirectional_iterator_tag iterator_category;
    typedef char *pointer;
    typedef char &reference;
    typedef char value_type;
    typedef value_type difference_type;

    TEST_HOST_DEVICE explicit TEST_CONSTEXPR_CXX14 ToIter() : m_value(0) {}
    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 ToIter(const ToIter &src) : m_value(src.m_value) {}
    // Intentionally not defined, must not be called.
    TEST_HOST_DEVICE ToIter(char *src);
    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 ToIter &operator=(char *src) {
        m_value = src;
        return *this;
    }
    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 ToIter &operator=(const ToIter &src) {
        m_value = src.m_value;
        return *this;
    }
    char *m_value;

    TEST_HOST_DEVICE reference operator*() const;
    TEST_HOST_DEVICE ToIter& operator++();
    TEST_HOST_DEVICE ToIter& operator--();
    TEST_HOST_DEVICE ToIter operator++(int);
    TEST_HOST_DEVICE ToIter operator--(int);
};

TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 bool tests() {
    Derived d{};
    test<bidirectional_iterator<Base*> >(bidirectional_iterator<Derived*>(&d));
    test<random_access_iterator<const Base*> >(random_access_iterator<Derived*>(&d));
    test<Base*>(&d);

    char c = '\0';
    char *fi = &c;
    const cuda::std::reverse_iterator<char *> rev_fi(fi);
    cuda::std::reverse_iterator<ToIter> rev_ti;
    rev_ti = rev_fi;
    assert(rev_ti.base().m_value == fi);

    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 2011
    static_assert(tests(), "");
#endif
    return 0;
}
