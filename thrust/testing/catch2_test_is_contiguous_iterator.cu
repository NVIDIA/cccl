#include <thrust/detail/static_assert.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <array>
#include <deque>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "catch2_test_helper.h"

template <typename Iterator>
inline constexpr bool check_is_contiguous =
  thrust::is_contiguous_iterator_v<Iterator> && cuda::std::__can_to_address<Iterator>;

TEST_CASE("is_contiguous_iterator extra", "[iterators]")
{
  STATIC_REQUIRE(check_is_contiguous<std::string::iterator>);
  STATIC_REQUIRE(check_is_contiguous<std::wstring::iterator>);
  STATIC_REQUIRE(check_is_contiguous<std::string_view::iterator>);
  STATIC_REQUIRE(check_is_contiguous<std::wstring_view::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<std::vector<bool>::iterator>);
}

TEMPLATE_LIST_TEST_CASE("is_contiguous_iterator", "[iterators]", generic_list)
{
  using T = TestType;
  STATIC_REQUIRE(check_is_contiguous<T*>);
  STATIC_REQUIRE(check_is_contiguous<T const*>);
  STATIC_REQUIRE(check_is_contiguous<thrust::device_ptr<T>>);
  STATIC_REQUIRE(check_is_contiguous<typename std::vector<T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::vector<T>::reverse_iterator>);
  STATIC_REQUIRE(check_is_contiguous<typename std::array<T, 1>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::list<T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::deque<T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::set<T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::multiset<T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::map<T, T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::multimap<T, T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::unordered_set<T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::unordered_multiset<T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::unordered_map<T, T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<typename std::unordered_multimap<T, T>::iterator>);
  STATIC_REQUIRE(!check_is_contiguous<std::istream_iterator<T>>);
  STATIC_REQUIRE(!check_is_contiguous<std::ostream_iterator<T>>);
}

TEMPLATE_LIST_TEST_CASE("is_contiguous_iterator cvref", "[iterators]", generic_list)
{
  using T = TestType;
  STATIC_REQUIRE(check_is_contiguous<T* const>);
  STATIC_REQUIRE(check_is_contiguous<T* volatile>);
  STATIC_REQUIRE(check_is_contiguous<T*&>);
  STATIC_REQUIRE(check_is_contiguous<T* const&>);
  STATIC_REQUIRE(check_is_contiguous<T* volatile&>);

  STATIC_REQUIRE(!check_is_contiguous<std::vector<bool>::iterator const>);
  STATIC_REQUIRE(!check_is_contiguous<std::vector<bool>::iterator volatile>);
  STATIC_REQUIRE(!check_is_contiguous<std::vector<bool>::iterator&>);
  STATIC_REQUIRE(!check_is_contiguous<std::vector<bool>::iterator const&>);
  STATIC_REQUIRE(!check_is_contiguous<std::vector<bool>::iterator volatile&>);
}

TEMPLATE_LIST_TEST_CASE("is_contiguous_iterator vectors", "[iterators]", vector_list)
{
  STATIC_REQUIRE(check_is_contiguous<typename TestType::iterator>);
}

template <typename IteratorT, typename PointerT, bool CanUnwrap>
constexpr bool check_iterator_unwrapping()
{
  using unwrapped_t =
    ::cuda::std::remove_reference_t<decltype(thrust::try_unwrap_contiguous_iterator(cuda::std::declval<IteratorT>()))>;

  if constexpr (CanUnwrap)
  {
    static_assert(std::is_same_v<unwrapped_t, PointerT>);
    static_assert(cuda::std::__can_to_address<IteratorT>);
  }
  else
  {
    static_assert(std::is_same_v<unwrapped_t, IteratorT>);
  }
  return true;
}

TEMPLATE_LIST_TEST_CASE("try_unwrap_contiguous_iterator", "[iterators]", generic_list)
{
  using T = TestType;
  // Raw pointers should pass whether expecting pointers or passthrough.
  STATIC_REQUIRE(check_iterator_unwrapping<T*, T*, true>());
  STATIC_REQUIRE(check_iterator_unwrapping<T*, T*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<T const*, T const*, true>());
  STATIC_REQUIRE(check_iterator_unwrapping<T const*, T const*, false>());

  STATIC_REQUIRE(check_iterator_unwrapping<thrust::device_ptr<T>, T*, true>());
  STATIC_REQUIRE(check_iterator_unwrapping<thrust::device_ptr<T const>, T const*, true>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::vector<T>::iterator, T*, true>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::vector<T>::reverse_iterator, T*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::array<T, 1>::iterator, T*, true>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::array<T const, 1>::iterator, T const*, true>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::list<T>::iterator, T*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::deque<T>::iterator, T*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::set<T>::iterator, T*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::multiset<T>::iterator, T*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::map<T, T>::iterator, std::pair<T const, T>*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::multimap<T, T>::iterator, std::pair<T const, T>*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::unordered_set<T>::iterator, T*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<typename std::unordered_multiset<T>::iterator, T*, false>());
  STATIC_REQUIRE(
    check_iterator_unwrapping<typename std::unordered_map<T, T>::iterator, std::pair<T const, T>*, false>());
  STATIC_REQUIRE(
    check_iterator_unwrapping<typename std::unordered_multimap<T, T>::iterator, std::pair<T const, T>*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<std::istream_iterator<T>, T*, false>());
  STATIC_REQUIRE(check_iterator_unwrapping<std::ostream_iterator<T>, void, false>());
}
