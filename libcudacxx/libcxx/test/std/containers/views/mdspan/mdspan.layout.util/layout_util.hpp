//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <tuple>

template <class> struct TestLayoutCtors;
template <class Mapping, size_t... DynamicSizes>
struct TestLayoutCtors<std::tuple<
    Mapping,
    std::integer_sequence<size_t, DynamicSizes...>
>>
{
    using mapping_type = Mapping;
    using extents_type = typename mapping_type::extents_type;
    Mapping map = { extents_type{ DynamicSizes... } };
};

template<class T> void typed_test_default_ctor()
// TYPED_TEST( TestLayoutCtors, default_ctor )
{
    // Default constructor ensures extents() == Extents() is true.
    using TestFixture = TestLayoutCtors<T>;
    auto m  = typename TestFixture::mapping_type();
    assert( m .extents() == typename TestFixture::extents_type() );
    auto m2 = typename TestFixture::mapping_type{};
    assert( m2.extents() == typename TestFixture::extents_type{} );
    assert( m == m2 );
}

template <class> struct TestLayoutCompatCtors;
template <class Mapping, size_t... DynamicSizes, class Mapping2, size_t... DynamicSizes2>
struct TestLayoutCompatCtors<std::tuple<
  Mapping,
  std::integer_sequence<size_t, DynamicSizes...>,
  Mapping2,
  std::integer_sequence<size_t, DynamicSizes2...>
>> {
  using mapping_type1 = Mapping;
  using mapping_type2 = Mapping2;
  using extents_type1 = std::remove_reference_t<decltype(std::declval<mapping_type1>().extents())>;
  using extents_type2 = std::remove_reference_t<decltype(std::declval<mapping_type2>().extents())>;
  Mapping  map1 = { extents_type1{ DynamicSizes...  } };
  Mapping2 map2 = { extents_type2{ DynamicSizes2... } };
};

template<class T> void typed_test_compatible()
//TYPED_TEST(TestLayout{Left|Right}CompatCtors, compatible_construct_{1|2}) {
//TYPED_TEST(TestLayout{Left|Right}CompatCtors, compatible_assign_{1|2}) {
{
    using TestFixture = TestLayoutCompatCtors<T>;

    // Construct
    {
        TestFixture t;

        auto m1 = typename TestFixture::mapping_type1(t.map2);
        assert( m1.extents() == t.map2.extents() );

        auto m2 = typename TestFixture::mapping_type2(t.map1);
        assert( m2.extents() == t.map1.extents() );
    }

    // Assign
    {
        TestFixture t;

#if __MDSPAN_HAS_CXX_17
        if constexpr ( std::is_convertible<typename TestFixture::mapping_type2, typename TestFixture::mapping_type1>::value )
        {
            t.map1 = t.map2;
        }
        else
        {
            t.map1 = typename TestFixture::mapping_type1( t.map2 );
        }
#else
        t.map1 = typename TestFixture::mapping_type1( t.map2 );
#endif

        assert( t.map1.extents() == t.map2.extents() );
    }
}

template<class T> void typed_test_compare()
{
    using TestFixture = TestLayoutCompatCtors<T>;

    {
        TestFixture t;

        auto m1 = typename TestFixture::mapping_type1(t.map2);
        assert( m1 == t.map2 );

        auto m2 = typename TestFixture::mapping_type2(t.map1);
        assert( m2 == t.map1 );
    }
}

template <size_t... Ds>
using _sizes = std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using _exts  = std::extents<size_t,Ds...>;

template <class E1, class S1, class E2, class S2>
using test_left_type_pair = std::tuple<
  typename std::layout_left::template mapping<E1>, S1,
  typename std::layout_left::template mapping<E2>, S2
>;

template <class E1, class S1, class E2, class S2>
using test_right_type_pair = std::tuple<
    typename std::layout_right::template mapping<E1>, S1,
    typename std::layout_right::template mapping<E2>, S2
>;

template< class T1, class T2, class = void >
struct is_cons_avail : std::false_type {};

template< class T1, class T2 >
struct is_cons_avail< T1
                    , T2
                    , std::enable_if_t< std::is_same< decltype( T1{ std::declval<T2>() } )
                                                    , T1
                                                    >::value
                                      >
                    > : std::true_type {};

template< class T1, class T2 >
constexpr bool is_cons_avail_v = is_cons_avail< T1, T2 >::value;

template< class, class T, class... Indicies >
struct is_paren_op_avail : std::false_type {};

template< class T, class... Indicies >
struct is_paren_op_avail< std::enable_if_t< std::is_same< decltype(std::declval<T>()(std::declval<Indicies>()...))
                                                        , typename T::index_type
                                                        >::value
                                          >
                        , T
                        , Indicies...
                        > : std::true_type {};

template< class T, class... Indicies >
constexpr bool is_paren_op_avail_v = is_paren_op_avail< void, T, Indicies... >::value;

template< class T, class RankType, class = void >
struct is_stride_avail : std::false_type {};

template< class T, class RankType >
struct is_stride_avail< T
                      , RankType
                      , std::enable_if_t< std::is_same< decltype( std::declval<T>().stride( std::declval<RankType>() ) )
                                                      , typename T::index_type
                                                      >::value
                                        >
                      > : std::true_type {};

template< class T, class RankType >
constexpr bool is_stride_avail_v = is_stride_avail< T, RankType >::value;

// Workaround for variables that are only used in static_assert's
template< typename T >
constexpr bool unused( T && ) { return true; }
