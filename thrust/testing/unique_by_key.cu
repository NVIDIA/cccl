#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/unique.h>

#include <unittest/unittest.h>

template <typename ValueT>
struct index_to_value_t
{
  template <typename IndexT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ValueT operator()(IndexT index)
  {
    if (static_cast<std::uint64_t>(index) == 4300000000ULL)
    {
      return static_cast<ValueT>(1);
    }
    else
    {
      return static_cast<ValueT>(0);
    }
  }
};

template <typename ForwardIterator1, typename ForwardIterator2>
thrust::pair<ForwardIterator1, ForwardIterator2>
unique_by_key(my_system& system, ForwardIterator1 keys_first, ForwardIterator1, ForwardIterator2 values_first)
{
  system.validate_dispatch();
  return thrust::make_pair(keys_first, values_first);
}

void TestUniqueByKeyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::unique_by_key(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestUniqueByKeyDispatchExplicit);

template <typename ForwardIterator1, typename ForwardIterator2>
thrust::pair<ForwardIterator1, ForwardIterator2>
unique_by_key(my_tag, ForwardIterator1 keys_first, ForwardIterator1, ForwardIterator2 values_first)
{
  *keys_first = 13;
  return thrust::make_pair(keys_first, values_first);
}

void TestUniqueByKeyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::unique_by_key(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestUniqueByKeyDispatchImplicit);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator1, typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
  my_system& system,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  OutputIterator1 keys_output,
  OutputIterator2 values_output)
{
  system.validate_dispatch();
  return thrust::make_pair(keys_output, values_output);
}

void TestUniqueByKeyCopyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::unique_by_key_copy(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestUniqueByKeyCopyDispatchExplicit);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator1, typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(
  my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator1 keys_output, OutputIterator2 values_output)
{
  *keys_output = 13;
  return thrust::make_pair(keys_output, values_output);
}

void TestUniqueByKeyCopyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::unique_by_key_copy(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestUniqueByKeyCopyDispatchImplicit);

template <typename T>
struct is_equal_div_10_unique
{
  _CCCL_HOST_DEVICE bool operator()(const T x, const T& y) const
  {
    return ((int) x / 10) == ((int) y / 10);
  }
};

template <typename Vector>
void initialize_keys(Vector& keys)
{
  keys.resize(9);
  keys = {11, 11, 21, 20, 21, 21, 21, 37, 37};
}

template <typename Vector>
void initialize_values(Vector& values)
{
  values.resize(9);
  values = {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

template <typename Vector>
void TestUniqueByKeySimple()
{
  using T = typename Vector::value_type;

  Vector keys;
  Vector values;

  typename thrust::pair<typename Vector::iterator, typename Vector::iterator> new_last;

  // basic test
  initialize_keys(keys);
  initialize_values(values);

  new_last = thrust::unique_by_key(keys.begin(), keys.end(), values.begin());

  ASSERT_EQUAL(new_last.first - keys.begin(), 5);
  ASSERT_EQUAL(new_last.second - values.begin(), 5);
  keys.resize(5);
  values.resize(5);
  Vector keys_ref{11, 21, 20, 21, 37};
  ASSERT_EQUAL(keys, keys_ref);

  Vector values_ref{0, 2, 3, 4, 7};
  ASSERT_EQUAL(values, values_ref);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  new_last = thrust::unique_by_key(keys.begin(), keys.end(), values.begin(), is_equal_div_10_unique<T>());

  ASSERT_EQUAL(new_last.first - keys.begin(), 3);
  ASSERT_EQUAL(new_last.second - values.begin(), 3);
  keys_ref.resize(3);
  keys.resize(3);
  keys_ref = {11, 21, 37};
  ASSERT_EQUAL(keys, keys_ref);

  values.resize(3);
  values_ref.resize(3);
  values_ref = {0, 2, 7};
  ASSERT_EQUAL(values, values_ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestUniqueByKeySimple);

template <typename Vector>
void TestUniqueCopyByKeySimple()
{
  using T = typename Vector::value_type;

  Vector keys;
  Vector values;

  typename thrust::pair<typename Vector::iterator, typename Vector::iterator> new_last;

  // basic test
  initialize_keys(keys);
  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  new_last =
    thrust::unique_by_key_copy(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  output_keys.resize(5);
  output_values.resize(5);
  Vector keys_ref{11, 21, 20, 21, 37};
  ASSERT_EQUAL(output_keys, keys_ref);

  Vector values_ref{0, 2, 3, 4, 7};
  ASSERT_EQUAL(output_values, values_ref);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  new_last = thrust::unique_by_key_copy(
    keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_unique<T>());

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  output_keys.resize(3);
  output_values.resize(3);
  keys_ref = {11, 21, 37};
  ASSERT_EQUAL(output_keys, keys_ref);

  values_ref.resize(3);
  values_ref = {0, 2, 7};
  ASSERT_EQUAL(output_values, values_ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestUniqueCopyByKeySimple);

template <typename K>
struct TestUniqueByKey
{
  void operator()(const size_t n)
  {
    using V = unsigned int; // ValueType

    thrust::host_vector<K> h_keys   = unittest::random_integers<bool>(n);
    thrust::host_vector<V> h_vals   = unittest::random_integers<V>(n);
    thrust::device_vector<K> d_keys = h_keys;
    thrust::device_vector<V> d_vals = h_vals;

    using HostKeyIterator   = typename thrust::host_vector<K>::iterator;
    using HostValIterator   = typename thrust::host_vector<V>::iterator;
    using DeviceKeyIterator = typename thrust::device_vector<K>::iterator;
    using DeviceValIterator = typename thrust::device_vector<V>::iterator;

    using HostIteratorPair   = typename thrust::pair<HostKeyIterator, HostValIterator>;
    using DeviceIteratorPair = typename thrust::pair<DeviceKeyIterator, DeviceValIterator>;

    HostIteratorPair h_last   = thrust::unique_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
    DeviceIteratorPair d_last = thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    ASSERT_EQUAL(h_last.first - h_keys.begin(), d_last.first - d_keys.begin());
    ASSERT_EQUAL(h_last.second - h_vals.begin(), d_last.second - d_vals.begin());

    size_t N = h_last.first - h_keys.begin();

    h_keys.resize(N);
    h_vals.resize(N);
    d_keys.resize(N);
    d_vals.resize(N);

    ASSERT_EQUAL(h_keys, d_keys);
    ASSERT_EQUAL(h_vals, d_vals);
  }
};
VariableUnitTest<TestUniqueByKey, IntegralTypes> TestUniqueByKeyInstance;

template <typename K>
struct TestUniqueCopyByKey
{
  void operator()(const size_t n)
  {
    using V = unsigned int; // ValueType

    thrust::host_vector<K> h_keys   = unittest::random_integers<bool>(n);
    thrust::host_vector<V> h_vals   = unittest::random_integers<V>(n);
    thrust::device_vector<K> d_keys = h_keys;
    thrust::device_vector<V> d_vals = h_vals;

    thrust::host_vector<K> h_keys_output(n);
    thrust::host_vector<V> h_vals_output(n);
    thrust::device_vector<K> d_keys_output(n);
    thrust::device_vector<V> d_vals_output(n);

    using HostKeyIterator   = typename thrust::host_vector<K>::iterator;
    using HostValIterator   = typename thrust::host_vector<V>::iterator;
    using DeviceKeyIterator = typename thrust::device_vector<K>::iterator;
    using DeviceValIterator = typename thrust::device_vector<V>::iterator;

    using HostIteratorPair   = typename thrust::pair<HostKeyIterator, HostValIterator>;
    using DeviceIteratorPair = typename thrust::pair<DeviceKeyIterator, DeviceValIterator>;

    HostIteratorPair h_last = thrust::unique_by_key_copy(
      h_keys.begin(), h_keys.end(), h_vals.begin(), h_keys_output.begin(), h_vals_output.begin());
    DeviceIteratorPair d_last = thrust::unique_by_key_copy(
      d_keys.begin(), d_keys.end(), d_vals.begin(), d_keys_output.begin(), d_vals_output.begin());

    ASSERT_EQUAL(h_last.first - h_keys_output.begin(), d_last.first - d_keys_output.begin());
    ASSERT_EQUAL(h_last.second - h_vals_output.begin(), d_last.second - d_vals_output.begin());

    size_t N = h_last.first - h_keys_output.begin();

    h_keys_output.resize(N);
    h_vals_output.resize(N);
    d_keys_output.resize(N);
    d_vals_output.resize(N);

    ASSERT_EQUAL(h_keys_output, d_keys_output);
    ASSERT_EQUAL(h_vals_output, d_vals_output);
  }
};
VariableUnitTest<TestUniqueCopyByKey, IntegralTypes> TestUniqueCopyByKeyInstance;

template <typename K>
struct TestUniqueCopyByKeyToDiscardIterator
{
  void operator()(const size_t n)
  {
    using V = unsigned int; // ValueType

    thrust::host_vector<K> h_keys   = unittest::random_integers<bool>(n);
    thrust::host_vector<V> h_vals   = unittest::random_integers<V>(n);
    thrust::device_vector<K> d_keys = h_keys;
    thrust::device_vector<V> d_vals = h_vals;

    thrust::host_vector<V> h_vals_output(n);
    thrust::device_vector<V> d_vals_output(n);

    thrust::host_vector<K> h_keys_output(n);
    thrust::device_vector<K> d_keys_output(n);

    thrust::host_vector<K> h_unique_keys = h_keys;
    h_unique_keys.erase(thrust::unique(h_unique_keys.begin(), h_unique_keys.end()), h_unique_keys.end());

    size_t num_unique_keys = h_unique_keys.size();

    // mask both outputs
    thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> h_result1 = thrust::unique_by_key_copy(
      h_keys.begin(), h_keys.end(), h_vals.begin(), thrust::make_discard_iterator(), thrust::make_discard_iterator());

    thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> d_result1 = thrust::unique_by_key_copy(
      d_keys.begin(), d_keys.end(), d_vals.begin(), thrust::make_discard_iterator(), thrust::make_discard_iterator());

    thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> reference1 =
      thrust::make_pair(thrust::make_discard_iterator(num_unique_keys), thrust::make_discard_iterator(num_unique_keys));

    ASSERT_EQUAL_QUIET(reference1, h_result1);
    ASSERT_EQUAL_QUIET(reference1, d_result1);

    // mask values output
    thrust::pair<typename thrust::host_vector<K>::iterator, thrust::discard_iterator<>> h_result2 =
      thrust::unique_by_key_copy(
        h_keys.begin(), h_keys.end(), h_vals.begin(), h_keys_output.begin(), thrust::make_discard_iterator());

    thrust::pair<typename thrust::device_vector<K>::iterator, thrust::discard_iterator<>> d_result2 =
      thrust::unique_by_key_copy(
        d_keys.begin(), d_keys.end(), d_vals.begin(), d_keys_output.begin(), thrust::make_discard_iterator());

    thrust::pair<typename thrust::host_vector<K>::iterator, thrust::discard_iterator<>> h_reference2 =
      thrust::make_pair(h_keys_output.begin() + num_unique_keys, thrust::make_discard_iterator(num_unique_keys));

    thrust::pair<typename thrust::device_vector<K>::iterator, thrust::discard_iterator<>> d_reference2 =
      thrust::make_pair(d_keys_output.begin() + num_unique_keys, thrust::make_discard_iterator(num_unique_keys));

    ASSERT_EQUAL(h_keys_output, d_keys_output);
    ASSERT_EQUAL_QUIET(h_reference2, h_result2);
    ASSERT_EQUAL_QUIET(d_reference2, d_result2);

    // mask keys output
    thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<V>::iterator> h_result3 =
      thrust::unique_by_key_copy(
        h_keys.begin(), h_keys.end(), h_vals.begin(), thrust::make_discard_iterator(), h_vals_output.begin());

    thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<V>::iterator> d_result3 =
      thrust::unique_by_key_copy(
        d_keys.begin(), d_keys.end(), d_vals.begin(), thrust::make_discard_iterator(), d_vals_output.begin());

    thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<V>::iterator> h_reference3 =
      thrust::make_pair(thrust::make_discard_iterator(num_unique_keys), h_vals_output.begin() + num_unique_keys);

    thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<V>::iterator> d_reference3 =
      thrust::make_pair(thrust::make_discard_iterator(num_unique_keys), d_vals_output.begin() + num_unique_keys);

    ASSERT_EQUAL(h_vals_output, d_vals_output);
    ASSERT_EQUAL_QUIET(h_reference3, h_result3);
    ASSERT_EQUAL_QUIET(d_reference3, d_result3);
  }
};
VariableUnitTest<TestUniqueCopyByKeyToDiscardIterator, IntegralTypes> TestUniqueCopyByKeyToDiscardIteratorInstance;

// OpenMP has issues with these tests, NVIDIA/cccl#1715
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP

template <typename K>
struct TestUniqueCopyByKeyLargeInput
{
  void operator()()
  {
    using type       = K;
    using index_type = std::int64_t;

    const std::size_t num_items = 4400000000ULL;
    thrust::host_vector<type> reference_keys{static_cast<type>(0), static_cast<type>(1), static_cast<type>(0)};
    thrust::host_vector<index_type> reference_values{0, 4300000000ULL, 4300000001ULL};

    auto keys_in   = thrust::make_transform_iterator(thrust::make_counting_iterator(0ULL), index_to_value_t<type>{});
    auto values_in = thrust::make_counting_iterator(0ULL);
    thrust::device_vector<type> keys_out(reference_keys.size());
    thrust::device_vector<index_type> values_out(reference_values.size());

    // Run test
    const auto selected_aut_end =
      thrust::unique_by_key_copy(keys_in, keys_in + num_items, values_in, keys_out.begin(), values_out.begin());

    // Ensure that we created the correct output
    auto const num_selected_out = thrust::distance(keys_out.begin(), selected_aut_end.first);
    ASSERT_EQUAL(reference_keys.size(), static_cast<std::size_t>(num_selected_out));
    ASSERT_EQUAL(num_selected_out, thrust::distance(values_out.begin(), selected_aut_end.second));
    keys_out.resize(num_selected_out);
    values_out.resize(num_selected_out);
    ASSERT_EQUAL(reference_keys, keys_out);
    ASSERT_EQUAL(reference_values, values_out);
  }
};
SimpleUnitTest<TestUniqueCopyByKeyLargeInput, IntegralTypes> TestUniqueCopyByKeyLargeInputInstance;

template <typename K>
struct TestUniqueCopyByKeyLargeOutCount
{
  void operator()()
  {
    constexpr std::size_t num_items = 4400000000ULL;

    auto keys_in   = thrust::make_counting_iterator(0ULL);
    auto values_in = thrust::make_counting_iterator(0ULL);

    // Run test
    auto keys_out   = thrust::make_discard_iterator();
    auto values_out = thrust::make_discard_iterator();
    const auto selected_aut_end =
      thrust::unique_by_key_copy(thrust::device, keys_in, keys_in + num_items, values_in, keys_out, values_out);

    // Ensure that we created the correct output
    auto const num_selected_out = thrust::distance(keys_out, selected_aut_end.first);
    ASSERT_EQUAL(num_items, static_cast<std::size_t>(num_selected_out));
    ASSERT_EQUAL(num_selected_out, thrust::distance(values_out, selected_aut_end.second));
  }
};
SimpleUnitTest<TestUniqueCopyByKeyLargeOutCount, IntegralTypes> TestUniqueCopyByKeyLargeOutCountInstance;

#endif // non-OpenMP backend

// This test fails only on GCC 6
#if !defined(__GNUC__) || __GNUC__ != 6

// Based on GitHub issue: https://github.com/NVIDIA/cccl/issues/1956
namespace
{
struct CompareFirst
{
  template <typename T>
  _CCCL_HOST_DEVICE bool operator()(T const& lhs, T const& rhs) const
  {
    return lhs.first == rhs.first;
  }
};
struct Entry
{
  std::int32_t a;
  float b;
};
} // namespace

void TestKeysWithoutEqualityOperator()
{
  using Key = thrust::pair<std::int32_t, Entry>;

  const auto k1 = Key{1, {}};
  const auto k2 = Key{2, {}};
  const thrust::device_vector<Key> keys{k1, k1, k1, k2, k2};
  thrust::device_vector<Entry> data{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};

  thrust::device_vector<Key> unique_keys(5);
  thrust::device_vector<Entry> unique_data(5);

  const auto result = thrust::unique_by_key_copy(
    thrust::device, keys.cbegin(), keys.cend(), data.begin(), unique_keys.begin(), unique_data.begin(), CompareFirst{});

  unique_keys.erase(result.first, unique_keys.end());
  unique_data.erase(result.second, unique_data.end());

  auto unique_keys_h = thrust::host_vector<Key>(unique_keys);
  auto unique_data_h = thrust::host_vector<Entry>(unique_data);

  ASSERT_EQUAL(unique_keys_h[0].first, k1.first);
  ASSERT_EQUAL(unique_keys_h[0].second.a, k1.second.a);
  ASSERT_EQUAL(unique_keys_h[0].second.b, k1.second.b);
  ASSERT_EQUAL(unique_keys_h[1].first, k2.first);
  ASSERT_EQUAL(unique_keys_h[1].second.a, k2.second.a);
  ASSERT_EQUAL(unique_keys_h[1].second.b, k2.second.b);

  ASSERT_EQUAL(unique_data_h[0].a, 0);
  ASSERT_EQUAL(unique_data_h[0].b, 0);
  ASSERT_EQUAL(unique_data_h[1].a, 3);
  ASSERT_EQUAL(unique_data_h[1].b, 3);
}
DECLARE_UNITTEST(TestKeysWithoutEqualityOperator);
#endif // !defined(__GNUC__) || __GNUC__ != 6
