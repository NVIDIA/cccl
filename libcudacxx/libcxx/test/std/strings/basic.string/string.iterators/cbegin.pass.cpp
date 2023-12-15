//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// const_iterator cbegin() const;

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(const S& s)
{
    typename S::const_iterator cb = s.cbegin();
    if (!s.empty())
    {
        assert(*cb == s[0]);
    }
    assert(cb == s.begin());
}

int main(int, char**)
{
    {
    typedef std::string S;
    test(S());
    test(S("123"));
    }
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S());
    test(S("123"));
    }

  return 0;
}
