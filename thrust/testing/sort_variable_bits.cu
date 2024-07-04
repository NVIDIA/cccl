#include <thrust/sort.h>

#include <algorithm>

#include <unittest/unittest.h>

using namespace unittest;

using UnsignedIntegerTypes =
  unittest::type_list<unittest::uint8_t, unittest::uint16_t, unittest::uint32_t, unittest::uint64_t>;

template <typename T>
struct TestSortVariableBits
{
  void operator()(const size_t n)
  {
    for (size_t num_bits = 0; num_bits < 8 * sizeof(T); num_bits += 3)
    {
      thrust::host_vector<T> h_keys = unittest::random_integers<T>(n);

      size_t mask = (1 << num_bits) - 1;
      for (size_t i = 0; i < n; i++)
      {
        h_keys[i] &= mask;
      }

      thrust::host_vector<T> reference = h_keys;
      thrust::device_vector<T> d_keys  = h_keys;

      std::sort(reference.begin(), reference.end());

      thrust::sort(h_keys.begin(), h_keys.end());
      thrust::sort(d_keys.begin(), d_keys.end());

      ASSERT_EQUAL(reference, h_keys);
      ASSERT_EQUAL(h_keys, d_keys);
    }
  }
};
VariableUnitTest<TestSortVariableBits, UnsignedIntegerTypes> TestSortVariableBitsInstance;
