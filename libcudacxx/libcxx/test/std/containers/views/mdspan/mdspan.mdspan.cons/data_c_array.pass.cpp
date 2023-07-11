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


int main(int, char**)
{
    {
        typedef int    data_t ;
        typedef size_t index_t;

        data_t data[1] = {42};
        std::mdspan<data_t, std::extents<index_t,1>> m(data);
        auto val = m(0);

        static_assert(m.is_exhaustive() == true, "");

        assert(m.data_handle()    == data );
        assert(m.rank()           == 1    );
        assert(m.rank_dynamic()   == 0    );
        assert(m.extent(0)        == 1    );
        assert(m.static_extent(0) == 1    );
        assert(m.stride(0)        == 1    );
        assert(val                == 42   );
        assert(m.size()           == 1    );
        assert(m.empty()          == false);
    }

    return 0;
}
