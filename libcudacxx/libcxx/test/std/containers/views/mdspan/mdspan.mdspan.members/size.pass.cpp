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

template<size_t N, class Extents>
void test_mdspan_size(std::array<char,N>& storage, Extents&& e)
{
    using extents_type = std::remove_cv_t<std::remove_reference_t<Extents>>;
    std::mdspan<char, extents_type> m(storage.data(), std::forward<Extents>(e));

    static_assert(std::is_same<decltype(m.size()), std::size_t>::value,
                  "The return type of mdspan::size() must be size_t.");

    // m.size() must not overflow, as long as the product of extents
    // is representable as a value of type size_t.
    assert( m.size() == N );
}


int main(int, char**)
{
    // TEST(TestMdspan, MdspanSizeReturnTypeAndPrecondition)
    {
        std::array<char,12*11> storage;

        static_assert(std::numeric_limits<std::int8_t>::max() == 127, "max int8_t != 127");
        test_mdspan_size(storage, std::extents<std::int8_t, 12, 11>{}); // 12 * 11 == 132
    }

    {
        std::array<char,16*17> storage;

        static_assert(std::numeric_limits<std::uint8_t>::max() == 255, "max uint8_t != 255");
        test_mdspan_size(storage, std::extents<std::uint8_t, 16, 17>{}); // 16 * 17 == 272
    }

    return 0;
}
