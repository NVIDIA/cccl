#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <vector>

#include <unittest/unittest.h>

template <>
inline constexpr bool thrust::detail::is_proxy_reference_v<std::vector<bool>::reference> = true;

void TestRawReferenceCast()
{
  using ::cuda::std::is_same_v;

  {
    [[maybe_unused]] int i        = 42;
    [[maybe_unused]] const int ci = 42;
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(i)), int&>);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(ci)), const int&>);
  }
  {
    [[maybe_unused]] thrust::host_vector<int> vec(1);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(*vec.begin())), int&>);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(*vec.cbegin())), const int&>);

    [[maybe_unused]] auto zip = thrust::make_zip_iterator(vec.begin(), vec.begin());
    static_assert(
      is_same_v<decltype(thrust::raw_reference_cast(*zip)), thrust::detail::tuple_of_iterator_references<int&, int&>>);

    [[maybe_unused]] auto zip2 = thrust::make_zip_iterator(zip, zip);
    static_assert(
      is_same_v<decltype(thrust::raw_reference_cast(*zip2)),
                thrust::detail::tuple_of_iterator_references<thrust::detail::tuple_of_iterator_references<int&, int&>,
                                                             thrust::detail::tuple_of_iterator_references<int&, int&>>>);
  }
  {
    [[maybe_unused]] thrust::device_vector<int> vec(1);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(*vec.begin())), int&>);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(*vec.cbegin())), const int&>);

    [[maybe_unused]] auto zip = thrust::make_zip_iterator(vec.begin(), vec.begin());
    static_assert(
      is_same_v<decltype(thrust::raw_reference_cast(*zip)), thrust::detail::tuple_of_iterator_references<int&, int&>>);

    [[maybe_unused]] auto zip2 = thrust::make_zip_iterator(zip, zip);
    static_assert(
      is_same_v<decltype(thrust::raw_reference_cast(*zip2)),
                thrust::detail::tuple_of_iterator_references<thrust::detail::tuple_of_iterator_references<int&, int&>,
                                                             thrust::detail::tuple_of_iterator_references<int&, int&>>>);
  }

  // proxy references
  {
    [[maybe_unused]] std::vector<bool> vb;
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(vb[0])), std::vector<bool>::reference>);
  }
}
DECLARE_UNITTEST(TestRawReferenceCast);
