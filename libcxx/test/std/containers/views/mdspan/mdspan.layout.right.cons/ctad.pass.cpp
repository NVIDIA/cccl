//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11

// No CTAD in C++14 or earlier
//UNSUPPORTED: c++14

#include <mdspan>
#include <cassert>


int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
    {
        typedef int    data_t ;
        typedef size_t index_t;

        std::layout_right::mapping m{std::extents{16, 32}};

        static_assert( m.is_exhaustive() == true, "" );

        assert( m.extents().rank()         == 2  );
        assert( m.extents().rank_dynamic() == 2  );
        assert( m.extents().extent(0)      == 16 );
        assert( m.extents().extent(1)      == 32 );
        assert( m.stride(0)                == 32 );
        assert( m.stride(1)                == 1  );
    }
#endif

    return 0;
}
