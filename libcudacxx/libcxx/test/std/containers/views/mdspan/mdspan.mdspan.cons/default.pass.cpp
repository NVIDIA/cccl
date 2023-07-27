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
    {
        typedef int    data_t ;
        typedef size_t index_t;

        std::mdspan<data_t, std::dextents<index_t,1>> m;

        static_assert(m.is_exhaustive() == true, "");

        assert(m.data_handle()    == nullptr);
        assert(m.rank()           == 1      );
        assert(m.rank_dynamic()   == 1      );
        assert(m.extent(0)        == 0      );
        assert(m.static_extent(0) == dyn    );
        assert(m.stride(0)        == 1      );
        assert(m.size()           == 0      );
        assert(m.empty()          == true   );
    }

    return 0;
}
