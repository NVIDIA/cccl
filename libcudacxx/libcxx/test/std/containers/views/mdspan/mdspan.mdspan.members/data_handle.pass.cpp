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
    // C array
    {
        const int d[5] = {1,2,3,4,5};
#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
        std::mdspan m(d);
#else
        std::mdspan<const int, std::extents<size_t,5>> m(d);
#endif

        assert( m.data_handle() == d );
    }

    // std array
    {
        std::array<int,5> d = {1,2,3,4,5};
#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
        std::mdspan m(d.data());
#else
        std::mdspan<int, std::extents<size_t,5>> m(d.data());
#endif

        assert( m.data_handle() == d.data() );
    }

    // C pointer
    {
        std::array<int,5> d = {1,2,3,4,5};
        int* ptr = d.data();
#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
        std::mdspan m(ptr);
#else
        std::mdspan<int, std::extents<size_t,5>> m(ptr);
#endif

        assert( m.data_handle() == ptr );
    }

    // Copy constructor
    {
        std::array<int,5> d = {1,2,3,4,5};
#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
        std::mdspan m0(d.data());
        std::mdspan m (m0);
#else
        std::mdspan<int, std::extents<size_t,5>> m0(d.data());
        std::mdspan<int, std::extents<size_t,5>> m (m0);
#endif

        assert( m.data_handle() == m0.data_handle() );
    }

    return 0;
}
