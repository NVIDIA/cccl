//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11, c++14

#include <mdspan>
#include <array>
#include <cassert>

int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
    {
        std::layout_stride::mapping m{std::dextents<size_t,2>{16, 32}, std::array{1, 128}};

        assert( m.is_exhaustive() == false);

        assert( m.extents().rank()         == 2   );
        assert( m.extents().rank_dynamic() == 2   );
        assert( m.extents().extent(0)      == 16  );
        assert( m.extents().extent(1)      == 32  );
        assert( m.stride(0)                == 1   );
        assert( m.stride(1)                == 128 );
        assert( m.strides()[0]             == 1   );
        assert( m.strides()[1]             == 128 );
    }
#endif

    return 0;
}
