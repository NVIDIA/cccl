//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// iterator, const_iterator

#include <cuda/std/array>
#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"

// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{
    {
    typedef cuda::std::array<int, 5> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }
    {
    typedef cuda::std::array<int, 0> C;
    C c;
    C::iterator i;
    i = c.begin();
    C::const_iterator j;
    j = c.cbegin();
    assert(i == j);
    }

#if TEST_STD_VER > 11
    { // N3644 testing
        {
        typedef cuda::std::array<int, 5> C;
        C::iterator ii1{}, ii2{};
        C::iterator ii4 = ii1;
        C::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );
        assert ( ii1 == cii );

        assert ( !(ii1 != ii2 ));
        assert ( !(ii1 != cii ));

        C c;
        assert ( c.begin()   == cuda::std::begin(c));
        assert ( c.cbegin()  == cuda::std::cbegin(c));
        assert ( c.rbegin()  == cuda::std::rbegin(c));
        assert ( c.crbegin() == cuda::std::crbegin(c));
        assert ( c.end()     == cuda::std::end(c));
        assert ( c.cend()    == cuda::std::cend(c));
        assert ( c.rend()    == cuda::std::rend(c));
        assert ( c.crend()   == cuda::std::crend(c));

        assert ( cuda::std::begin(c)   != cuda::std::end(c));
        assert ( cuda::std::rbegin(c)  != cuda::std::rend(c));
        assert ( cuda::std::cbegin(c)  != cuda::std::cend(c));
        assert ( cuda::std::crbegin(c) != cuda::std::crend(c));
        }
        {
        typedef cuda::std::array<int, 0> C;
        C::iterator ii1{}, ii2{};
        C::iterator ii4 = ii1;
        C::const_iterator cii{};
        assert ( ii1 == ii2 );
        assert ( ii1 == ii4 );

        assert (!(ii1 != ii2 ));

        assert ( (ii1 == cii ));
        assert ( (cii == ii1 ));
        assert (!(ii1 != cii ));
        assert (!(cii != ii1 ));
        assert (!(ii1 <  cii ));
        assert (!(cii <  ii1 ));
        assert ( (ii1 <= cii ));
        assert ( (cii <= ii1 ));
        assert (!(ii1 >  cii ));
        assert (!(cii >  ii1 ));
        assert ( (ii1 >= cii ));
        assert ( (cii >= ii1 ));
        assert (cii - ii1 == 0);
        assert (ii1 - cii == 0);

        C c;
        assert ( c.begin()   == cuda::std::begin(c));
        assert ( c.cbegin()  == cuda::std::cbegin(c));
        assert ( c.rbegin()  == cuda::std::rbegin(c));
        assert ( c.crbegin() == cuda::std::crbegin(c));
        assert ( c.end()     == cuda::std::end(c));
        assert ( c.cend()    == cuda::std::cend(c));
        assert ( c.rend()    == cuda::std::rend(c));
        assert ( c.crend()   == cuda::std::crend(c));

        assert ( cuda::std::begin(c)   == cuda::std::end(c));
        assert ( cuda::std::rbegin(c)  == cuda::std::rend(c));
        assert ( cuda::std::cbegin(c)  == cuda::std::cend(c));
        assert ( cuda::std::crbegin(c) == cuda::std::crend(c));
        }
    }
#endif
#if TEST_STD_VER > 14
    {
        typedef cuda::std::array<int, 5> C;
        constexpr C c{0,1,2,3,4};

        static_assert ( c.begin()   == cuda::std::begin(c), "");
        static_assert ( c.cbegin()  == cuda::std::cbegin(c), "");
        static_assert ( c.end()     == cuda::std::end(c), "");
        static_assert ( c.cend()    == cuda::std::cend(c), "");

        static_assert ( c.rbegin()  == cuda::std::rbegin(c), "");
        static_assert ( c.crbegin() == cuda::std::crbegin(c), "");
        static_assert ( c.rend()    == cuda::std::rend(c), "");
        static_assert ( c.crend()   == cuda::std::crend(c), "");

        static_assert ( cuda::std::begin(c)   != cuda::std::end(c), "");
        static_assert ( cuda::std::rbegin(c)  != cuda::std::rend(c), "");
        static_assert ( cuda::std::cbegin(c)  != cuda::std::cend(c), "");
        static_assert ( cuda::std::crbegin(c) != cuda::std::crend(c), "");

        static_assert ( *c.begin()  == 0, "");
        static_assert ( *c.rbegin()  == 4, "");

        static_assert ( *cuda::std::begin(c)   == 0, "" );
        static_assert ( *cuda::std::cbegin(c)  == 0, "" );
        static_assert ( *cuda::std::rbegin(c)  == 4, "" );
        static_assert ( *cuda::std::crbegin(c) == 4, "" );
    }
#endif

  return 0;
}
