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
#include "../mdspan.mdspan.util/mdspan_util.hpp"
#include "../my_accessor.hpp"

constexpr auto dyn = std::dynamic_extent;

template< class T, class DataHandleType, class MappingType, class = void >
struct is_mapping_cons_avail : std::false_type {};

template< class T, class DataHandleType, class MappingType >
struct is_mapping_cons_avail< T
                            , DataHandleType
                            , MappingType
                            , std::enable_if_t< std::is_same< decltype( T{ std::declval<DataHandleType>()
                                                                         , std::declval<MappingType>()
                                                                         }
                                                                      )
                                                            , T
                                                            >::value
                                              >
                            > : std::true_type {};

template< class T, class DataHandleType, class MappingType >
constexpr bool is_mapping_cons_avail_v = is_mapping_cons_avail< T, DataHandleType, MappingType >::value;

int main(int, char**)
{
    using    data_t = int;
    using   index_t = size_t;
    using     ext_t = std::extents<index_t,dyn,dyn>;
    using mapping_t = std::layout_left::mapping<ext_t>;

    // mapping
    {
        using  mdspan_t = std::mdspan<data_t, ext_t, std::layout_left>;

        static_assert( is_mapping_cons_avail_v< mdspan_t, int *, mapping_t > == true, "" );

        std::array<data_t, 1> d{42};
        mapping_t map{std::dextents<index_t,2>{64, 128}};
        mdspan_t  m{ d.data(), map };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // Constraint: is_default_constructible_v<accessor_type> is true
    {
        using  mdspan_t = std::mdspan<data_t, ext_t, std::layout_left, Foo::my_accessor<data_t>>;

        static_assert( is_mapping_cons_avail_v< mdspan_t, int *, mapping_t > == false, "" );
    }

    // mapping and accessor
    {

        std::array<data_t, 1> d{42};
        mapping_t map{std::dextents<index_t,2>{64, 128}};
        std::default_accessor<data_t> a;
        std::mdspan<data_t, ext_t, std::layout_left> m{ d.data(), map, a };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    return 0;
}
