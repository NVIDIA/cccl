//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string<charT,traits,Allocator>& operator+=(const charT* s);

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(S s, const typename S::value_type* str, S expected)
{
    s += str;
    LIBCPP_ASSERT(s.__invariants());
    assert(s == expected);
}

int main(int, char**)
{
    {
    typedef std::string S;
    test(S(), "", S());
    test(S(), "12345", S("12345"));
    test(S(), "1234567890", S("1234567890"));
    test(S(), "12345678901234567890", S("12345678901234567890"));

    test(S("12345"), "", S("12345"));
    test(S("12345"), "12345", S("1234512345"));
    test(S("12345"), "1234567890", S("123451234567890"));
    test(S("12345"), "12345678901234567890", S("1234512345678901234567890"));

    test(S("1234567890"), "", S("1234567890"));
    test(S("1234567890"), "12345", S("123456789012345"));
    test(S("1234567890"), "1234567890", S("12345678901234567890"));
    test(S("1234567890"), "12345678901234567890", S("123456789012345678901234567890"));

    test(S("12345678901234567890"), "", S("12345678901234567890"));
    test(S("12345678901234567890"), "12345", S("1234567890123456789012345"));
    test(S("12345678901234567890"), "1234567890", S("123456789012345678901234567890"));
    test(S("12345678901234567890"), "12345678901234567890",
         S("1234567890123456789012345678901234567890"));
    }
#if TEST_STD_VER >= 2011
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S(), "", S());
    test(S(), "12345", S("12345"));
    test(S(), "1234567890", S("1234567890"));
    test(S(), "12345678901234567890", S("12345678901234567890"));

    test(S("12345"), "", S("12345"));
    test(S("12345"), "12345", S("1234512345"));
    test(S("12345"), "1234567890", S("123451234567890"));
    test(S("12345"), "12345678901234567890", S("1234512345678901234567890"));

    test(S("1234567890"), "", S("1234567890"));
    test(S("1234567890"), "12345", S("123456789012345"));
    test(S("1234567890"), "1234567890", S("12345678901234567890"));
    test(S("1234567890"), "12345678901234567890", S("123456789012345678901234567890"));

    test(S("12345678901234567890"), "", S("12345678901234567890"));
    test(S("12345678901234567890"), "12345", S("1234567890123456789012345"));
    test(S("12345678901234567890"), "1234567890", S("123456789012345678901234567890"));
    test(S("12345678901234567890"), "12345678901234567890",
         S("1234567890123456789012345678901234567890"));
    }
#endif

  return 0;
}
