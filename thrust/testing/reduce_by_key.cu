#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>

#include <unittest/unittest.h>

template <typename T>
struct is_equal_div_10_reduce
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
void TestReduceByKeySimple()
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
    thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  output_keys.resize(new_last.first - output_keys.begin());
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  output_values.resize(new_last.second - output_values.begin());
  Vector ref_keys{11, 21, 20, 21, 37};
  ASSERT_EQUAL(output_keys, ref_keys);

  Vector ref_values{1, 2, 3, 15, 15};
  ASSERT_EQUAL(output_values, ref_values);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  new_last = thrust::reduce_by_key(
    keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>());

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 3);
  output_keys.resize(new_last.first - output_keys.begin());
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  output_values.resize(new_last.second - output_values.begin());

  ref_keys = {11, 21, 37};
  ASSERT_EQUAL(output_keys, ref_keys);

  ref_values = {1, 20, 15};
  ASSERT_EQUAL(output_values, ref_values);

  // test BinaryFunction
  initialize_keys(keys);
  initialize_values(values);

  output_keys.resize(keys.size());
  output_values.resize(values.size());

  new_last = thrust::reduce_by_key(
    keys.begin(),
    keys.end(),
    values.begin(),
    output_keys.begin(),
    output_values.begin(),
    ::cuda::std::equal_to<T>(),
    ::cuda::std::plus<T>());

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  output_keys.resize(new_last.first - output_keys.begin());
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  output_values.resize(new_last.second - output_values.begin());

  ref_keys = {11, 21, 20, 21, 37};
  ASSERT_EQUAL(output_keys, ref_keys);

  ref_values = {1, 2, 3, 15, 15};
  ASSERT_EQUAL(output_values, ref_values);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestReduceByKeySimple);

template <typename K>
struct TestReduceByKey
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

    HostIteratorPair h_last =
      thrust::reduce_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_keys_output.begin(), h_vals_output.begin());
    DeviceIteratorPair d_last =
      thrust::reduce_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_keys_output.begin(), d_vals_output.begin());

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
VariableUnitTest<TestReduceByKey, IntegralTypes> TestReduceByKeyInstance;

template <typename K>
struct TestReduceByKeyToDiscardIterator
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

    thrust::host_vector<K> unique_keys = h_keys;
    unique_keys.erase(thrust::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());

    // discard key output
    size_t h_size =
      thrust::reduce_by_key(
        h_keys.begin(), h_keys.end(), h_vals.begin(), thrust::make_discard_iterator(), h_vals_output.begin())
        .second
      - h_vals_output.begin();

    size_t d_size =
      thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(), d_vals.begin(), thrust::make_discard_iterator(), d_vals_output.begin())
        .second
      - d_vals_output.begin();

    h_vals_output.resize(h_size);
    d_vals_output.resize(d_size);

    ASSERT_EQUAL(h_vals_output.size(), unique_keys.size());
    ASSERT_EQUAL(d_vals_output.size(), unique_keys.size());
    ASSERT_EQUAL(d_vals_output.size(), h_vals_output.size());
  }
};
VariableUnitTest<TestReduceByKeyToDiscardIterator, IntegralTypes> TestReduceByKeyToDiscardIteratorInstance;

template <typename InputIterator1, typename InputIterator2, typename OutputIterator1, typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> reduce_by_key(
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

void TestReduceByKeyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::reduce_by_key(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestReduceByKeyDispatchExplicit);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator1, typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator1 keys_output, OutputIterator2 values_output)
{
  *keys_output = 13;
  return thrust::make_pair(keys_output, values_output);
}

void TestReduceByKeyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::reduce_by_key(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestReduceByKeyDispatchImplicit);
