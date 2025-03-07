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

#include <unittest/unittest.h>

static_assert(thrust::is_contiguous_iterator<std::string::iterator>::value);
static_assert(thrust::is_contiguous_iterator<std::wstring::iterator>::value);
static_assert(thrust::is_contiguous_iterator<std::string_view::iterator>::value);
static_assert(thrust::is_contiguous_iterator<std::wstring_view::iterator>::value);
static_assert(!thrust::is_contiguous_iterator<std::vector<bool>::iterator>::value);

template <typename T>
_CCCL_HOST void test_is_contiguous_iterator()
{
  static_assert(thrust::is_contiguous_iterator<T*>::value);
  static_assert(thrust::is_contiguous_iterator<T const*>::value);
  static_assert(thrust::is_contiguous_iterator<thrust::device_ptr<T>>::value);
  static_assert(thrust::is_contiguous_iterator<typename std::vector<T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::vector<T>::reverse_iterator>::value);
  static_assert(thrust::is_contiguous_iterator<typename std::array<T, 1>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::list<T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::deque<T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::set<T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::multiset<T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::map<T, T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::multimap<T, T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::unordered_set<T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::unordered_multiset<T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::unordered_map<T, T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<typename std::unordered_multimap<T, T>::iterator>::value);
  static_assert(!thrust::is_contiguous_iterator<std::istream_iterator<T>>::value);
  static_assert(!thrust::is_contiguous_iterator<std::ostream_iterator<T>>::value);
}
DECLARE_GENERIC_UNITTEST(test_is_contiguous_iterator);

template <typename Vector>
_CCCL_HOST void test_is_contiguous_iterator_vectors()
{
  static_assert(thrust::is_contiguous_iterator<typename Vector::iterator>::value);
}
DECLARE_VECTOR_UNITTEST(test_is_contiguous_iterator_vectors);

struct expect_pointer
{};
struct expect_passthrough
{};

template <typename IteratorT, typename PointerT, typename expected_unwrapped_type /* = expect_[pointer|passthrough] */>
struct check_unwrapped_iterator
{
  using unwrapped_t =
    ::cuda::std::remove_reference_t<decltype(thrust::try_unwrap_contiguous_iterator(cuda::std::declval<IteratorT>()))>;

  static constexpr bool value =
    std::is_same<expected_unwrapped_type, expect_pointer>::value
      ? std::is_same<unwrapped_t, PointerT>::value
      : std::is_same<unwrapped_t, IteratorT>::value;
};

template <typename T>
void test_try_unwrap_contiguous_iterator()
{
  // Raw pointers should pass whether expecting pointers or passthrough.
  static_assert(check_unwrapped_iterator<T*, T*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<T*, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<T const*, T const*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<T const*, T const*, expect_passthrough>::value);

  static_assert(check_unwrapped_iterator<thrust::device_ptr<T>, T*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<thrust::device_ptr<T const>, T const*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<typename std::vector<T>::iterator, T*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<typename std::vector<T>::reverse_iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::array<T, 1>::iterator, T*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<typename std::array<T const, 1>::iterator, T const*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<typename std::list<T>::iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::deque<T>::iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::set<T>::iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::multiset<T>::iterator, T*, expect_passthrough>::value);
  static_assert(
    check_unwrapped_iterator<typename std::map<T, T>::iterator, std::pair<T const, T>*, expect_passthrough>::value);
  static_assert(
    check_unwrapped_iterator<typename std::multimap<T, T>::iterator, std::pair<T const, T>*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::unordered_set<T>::iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::unordered_multiset<T>::iterator, T*, expect_passthrough>::value);
  static_assert(
    check_unwrapped_iterator<typename std::unordered_map<T, T>::iterator, std::pair<T const, T>*, expect_passthrough>::
      value);
  static_assert(check_unwrapped_iterator<typename std::unordered_multimap<T, T>::iterator,
                                         std::pair<T const, T>*,
                                         expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<std::istream_iterator<T>, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<std::ostream_iterator<T>, void, expect_passthrough>::value);
}
DECLARE_GENERIC_UNITTEST(test_try_unwrap_contiguous_iterator);
