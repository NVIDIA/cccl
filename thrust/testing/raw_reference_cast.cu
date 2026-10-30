#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuda/std/memory>

#include <vector>

#include <unittest/unittest.h>

template <>
inline constexpr bool thrust::detail::is_proxy_reference_v<std::vector<bool>::reference> = true;

TEST_CASE("TestRawReferenceCast", "[raw_reference_cast]")
{
  using ::cuda::std::is_same_v;

  {
    // NOLINTNEXTLINE(misc-const-correctness): the non-const overload is under test
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
    [[maybe_unused]] std::vector<bool> vb; // NOLINT(misc-const-correctness)
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(vb[0])), std::vector<bool>::reference>);
  }
}

struct overloaded_address_of
{
  _CCCL_HOST_DEVICE overloaded_address_of* operator&()
  {
    return this + 1;
  }

  _CCCL_HOST_DEVICE const overloaded_address_of* operator&() const
  {
    return this + 1;
  }
};

TEST_CASE("TestRawReferenceCastOverloadedAddressOf", "[raw_reference_cast]")
{
  // NOLINTNEXTLINE(misc-const-correctness): the non-const overload is under test
  overloaded_address_of values[2]{};
  const overloaded_address_of const_values[2]{};

  // An overloaded operator& must not be used for references that are already raw
  REQUIRE(cuda::std::addressof(thrust::raw_reference_cast(values[0])) == cuda::std::addressof(values[0]));
  REQUIRE(cuda::std::addressof(thrust::raw_reference_cast(const_values[0])) == cuda::std::addressof(const_values[0]));
}
