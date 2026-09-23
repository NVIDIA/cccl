#include <thrust/detail/static_assert.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>

#include <array>
#include <deque>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unittest/unittest.h>

template <typename Iterator, bool Expected>
void check_is_contiguous()
{
  STATIC_REQUIRE(thrust::is_contiguous_iterator_v<Iterator> == Expected);
  if constexpr (Expected)
  {
    STATIC_REQUIRE(cuda::std::__can_to_address<Iterator> == Expected);
  }
}

TEST_CASE("is_contiguous_iterator extra", "[iterators]")
{
  check_is_contiguous<std::string::iterator, true>();
  check_is_contiguous<std::wstring::iterator, true>();
  check_is_contiguous<std::string_view::iterator, true>();
  check_is_contiguous<std::wstring_view::iterator, true>();
  check_is_contiguous<std::vector<bool>::iterator, false>();
}

TEMPLATE_LIST_TEST_CASE("is_contiguous_iterator", "[iterators]", generic_list)
{
  using T = TestType;
  check_is_contiguous<T*, true>();
  check_is_contiguous<T const*, true>();
  check_is_contiguous<thrust::device_ptr<T>, true>();
  check_is_contiguous<typename std::vector<T>::iterator, true>();
  check_is_contiguous<typename std::vector<T>::reverse_iterator, false>();
  check_is_contiguous<typename std::array<T, 1>::iterator, true>();
  check_is_contiguous<typename std::list<T>::iterator, false>();
  check_is_contiguous<typename std::deque<T>::iterator, false>();
  check_is_contiguous<typename std::set<T>::iterator, false>();
  check_is_contiguous<typename std::multiset<T>::iterator, false>();
  check_is_contiguous<typename std::map<T, T>::iterator, false>();
  check_is_contiguous<typename std::multimap<T, T>::iterator, false>();
  check_is_contiguous<typename std::unordered_set<T>::iterator, false>();
  check_is_contiguous<typename std::unordered_multiset<T>::iterator, false>();
  check_is_contiguous<typename std::unordered_map<T, T>::iterator, false>();
  check_is_contiguous<typename std::unordered_multimap<T, T>::iterator, false>();
  check_is_contiguous<std::istream_iterator<T>, false>();
  check_is_contiguous<std::ostream_iterator<T>, false>();
}

TEMPLATE_LIST_TEST_CASE("is_contiguous_iterator cvref", "[iterators]", generic_list)
{
  using T = TestType;
  check_is_contiguous<T* const, true>();
  check_is_contiguous<T* volatile, true>();
  check_is_contiguous<T*&, true>();
  check_is_contiguous<T* const&, true>();
  check_is_contiguous<T* volatile&, true>();

  check_is_contiguous<std::vector<bool>::iterator const, false>();
  check_is_contiguous<std::vector<bool>::iterator volatile, false>();
  check_is_contiguous<std::vector<bool>::iterator&, false>();
  check_is_contiguous<std::vector<bool>::iterator const&, false>();
  check_is_contiguous<std::vector<bool>::iterator volatile&, false>();
}

TEMPLATE_LIST_TEST_CASE("is_contiguous_iterator vectors", "[iterators]", vector_list)
{
  check_is_contiguous<typename TestType::iterator, true>();
}

template <typename IteratorT, typename PointerT, bool CanUnwrap>
void check_iterator_unwrapping()
{
  using unwrapped_t =
    ::cuda::std::remove_reference_t<decltype(thrust::try_unwrap_contiguous_iterator(cuda::std::declval<IteratorT>()))>;

  if constexpr (CanUnwrap)
  {
    STATIC_REQUIRE(std::is_same_v<unwrapped_t, PointerT>);
    STATIC_REQUIRE(cuda::std::__can_to_address<IteratorT>);
  }
  else
  {
    STATIC_REQUIRE(std::is_same_v<unwrapped_t, IteratorT>);
  }
}

TEMPLATE_LIST_TEST_CASE("try_unwrap_contiguous_iterator", "[iterators]", generic_list)
{
  using T = TestType;
  // Raw pointers should pass whether expecting pointers or passthrough.
  check_iterator_unwrapping<T*, T*, true>();
  check_iterator_unwrapping<T*, T*, false>();
  check_iterator_unwrapping<T const*, T const*, true>();
  check_iterator_unwrapping<T const*, T const*, false>();

  check_iterator_unwrapping<thrust::device_ptr<T>, T*, true>();
  check_iterator_unwrapping<thrust::device_ptr<T const>, T const*, true>();
  check_iterator_unwrapping<typename std::vector<T>::iterator, T*, true>();
  check_iterator_unwrapping<typename std::vector<T>::reverse_iterator, T*, false>();
  check_iterator_unwrapping<typename std::array<T, 1>::iterator, T*, true>();
  check_iterator_unwrapping<typename std::array<T const, 1>::iterator, T const*, true>();
  check_iterator_unwrapping<typename std::list<T>::iterator, T*, false>();
  check_iterator_unwrapping<typename std::deque<T>::iterator, T*, false>();
  check_iterator_unwrapping<typename std::set<T>::iterator, T*, false>();
  check_iterator_unwrapping<typename std::multiset<T>::iterator, T*, false>();
  check_iterator_unwrapping<typename std::map<T, T>::iterator, std::pair<T const, T>*, false>();
  check_iterator_unwrapping<typename std::multimap<T, T>::iterator, std::pair<T const, T>*, false>();
  check_iterator_unwrapping<typename std::unordered_set<T>::iterator, T*, false>();
  check_iterator_unwrapping<typename std::unordered_multiset<T>::iterator, T*, false>();
  check_iterator_unwrapping<typename std::unordered_map<T, T>::iterator, std::pair<T const, T>*, false>();
  check_iterator_unwrapping<typename std::unordered_multimap<T, T>::iterator, std::pair<T const, T>*, false>();
  check_iterator_unwrapping<std::istream_iterator<T>, T*, false>();
  check_iterator_unwrapping<std::ostream_iterator<T>, void, false>();
}
