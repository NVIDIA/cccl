//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11

#include <mdspan>
#include <cassert>

constexpr auto dyn = std::dynamic_extent;

int main(int, char**)
{
    using index_t = int;

    {
        std::extents<index_t,16> e;
        std::array<index_t,1> a{1};
        std::layout_stride::mapping<std::extents<index_t,16>> m{e, a};

        static_assert( m.is_always_exhaustive() == false, "" );
        assert       ( m.is_exhaustive       () == true  );
    }

    {
        std::extents<index_t,16> e;
        std::array<index_t,1> a{2};
        std::layout_stride::mapping<std::extents<index_t,16>> m{e, a};

        static_assert( m.is_always_exhaustive() == false, "" );
        assert       ( m.is_exhaustive       () == false );
    }

    {
        std::extents<index_t,16,32> e;
        std::array<index_t,2> a{1,16};
        std::layout_stride::mapping<std::extents<index_t,dyn,dyn>> m{e, a};

        static_assert( m.is_always_exhaustive() == false, "" );
        assert       ( m.is_exhaustive       () == true  );
    }

    {
        std::extents<index_t,dyn,dyn> e{16, 32};
        std::array<index_t,2> a{1,128};
        std::layout_stride::mapping<std::extents<index_t,dyn,dyn>> m{e, a};

        static_assert( m.is_always_exhaustive() == false, "" );
        assert       ( m.is_exhaustive       () == false );
    }

    {
        std::extents<index_t,dyn,dyn,dyn> e{16, 32, 4};
        std::array<index_t,3> a{1,16*4,16};
        std::layout_stride::mapping<std::extents<index_t,dyn,dyn,dyn>> m{e, a};

        static_assert( m.is_always_exhaustive() == false, "" );
        assert       ( m.is_exhaustive       () == true  );
    }

    {
        std::extents<index_t,dyn,dyn,dyn> e{16, 32, 4};
        std::array<index_t,3> a{1,16*4+1,16};
        std::layout_stride::mapping<std::extents<index_t,dyn,dyn,dyn>> m{e, a};

        static_assert( m.is_always_exhaustive() == false, "" );
        assert       ( m.is_exhaustive       () == false );
    }

    return 0;
}
