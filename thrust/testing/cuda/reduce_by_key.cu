#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <cuda/iterator>
#include <cuda/std/functional>

#include <cstdint>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Iterator5>
__global__ void reduce_by_key_kernel(
  ExecutionPolicy exec,
  Iterator1 keys_first,
  Iterator1 keys_last,
  Iterator2 values_first,
  Iterator3 keys_result,
  Iterator4 values_result,
  Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result);
}

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename BinaryPredicate,
          typename Iterator5>
__global__ void reduce_by_key_kernel(
  ExecutionPolicy exec,
  Iterator1 keys_first,
  Iterator1 keys_last,
  Iterator2 values_first,
  Iterator3 keys_result,
  Iterator4 values_result,
  BinaryPredicate pred,
  Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, pred);
}

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename BinaryPredicate,
          typename BinaryFunction,
          typename Iterator5>
__global__ void reduce_by_key_kernel(
  ExecutionPolicy exec,
  Iterator1 keys_first,
  Iterator1 keys_last,
  Iterator2 values_first,
  Iterator3 keys_result,
  Iterator4 values_result,
  BinaryPredicate pred,
  BinaryFunction binary_op,
  Iterator5 result)
{
  *result =
    thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, pred, binary_op);
}
#endif

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
  keys[0] = 11;
  keys[1] = 11;
  keys[2] = 21;
  keys[3] = 20;
  keys[4] = 21;
  keys[5] = 21;
  keys[6] = 21;
  keys[7] = 37;
  keys[8] = 37;
}

template <typename Vector>
void initialize_values(Vector& values)
{
  values.resize(9);
  values[0] = 0;
  values[1] = 1;
  values[2] = 2;
  values[3] = 3;
  values[4] = 4;
  values[5] = 5;
  values[6] = 6;
  values[7] = 7;
  values[8] = 8;
}

// Checks whether the equality operator is ever invoked on out-of-bounds items
struct check_valid_item_op
{
  cuda::std::uint32_t* error_counter{};
  int expected_upper_bound{};

  __device__ bool operator()(const int lhs, const int rhs) const
  {
    if (lhs > expected_upper_bound || rhs > expected_upper_bound)
    {
      if (error_counter)
      {
        atomicAdd(error_counter, 1);
      }
      return false;
    }
    return lhs == rhs;
  }
};

template <typename ReturnedT>
struct check_accumulator_t_op
{
  cuda::std::uint32_t* error_counter{};

  template <typename T,
            typename U,
            typename = cuda::std::enable_if_t<cuda::std::is_same_v<T, U> && !cuda::std::is_same_v<T, ReturnedT>>>
  _CCCL_DEVICE ReturnedT operator()(const T lhs, const U& rhs) const
  {
    return static_cast<ReturnedT>(lhs + rhs);
  }

  template <typename T>
  _CCCL_DEVICE ReturnedT operator()(const ReturnedT lhs, const T& rhs) const
  {
    atomicAdd(error_counter, 1);
    return lhs + static_cast<ReturnedT>(rhs);
  }
};

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy>
void TestReduceByKeyDevice(ExecutionPolicy exec)
{
  using T = int;

  thrust::device_vector<T> keys;
  thrust::device_vector<T> values;

  using iterator_pair =
    typename cuda::std::pair<typename thrust::device_vector<T>::iterator, typename thrust::device_vector<T>::iterator>;

  thrust::device_vector<iterator_pair> new_last_vec(1);
  iterator_pair new_last;

  // basic test
  initialize_keys(keys);
  initialize_values(values);

  thrust::device_vector<T> output_keys(keys.size());
  thrust::device_vector<T> output_values(values.size());

  reduce_by_key_kernel<<<1, 1>>>(
    exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);

  ASSERT_EQUAL(output_values[0], 1);
  ASSERT_EQUAL(output_values[1], 2);
  ASSERT_EQUAL(output_values[2], 3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  reduce_by_key_kernel<<<1, 1>>>(
    exec,
    keys.begin(),
    keys.end(),
    values.begin(),
    output_keys.begin(),
    output_values.begin(),
    is_equal_div_10_reduce<T>(),
    new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);

  ASSERT_EQUAL(output_values[0], 1);
  ASSERT_EQUAL(output_values[1], 20);
  ASSERT_EQUAL(output_values[2], 15);

  // test BinaryFunction
  initialize_keys(keys);
  initialize_values(values);

  reduce_by_key_kernel<<<1, 1>>>(
    exec,
    keys.begin(),
    keys.end(),
    values.begin(),
    output_keys.begin(),
    output_values.begin(),
    ::cuda::std::equal_to<T>(),
    ::cuda::std::plus<T>(),
    new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);

  ASSERT_EQUAL(output_values[0], 1);
  ASSERT_EQUAL(output_values[1], 2);
  ASSERT_EQUAL(output_values[2], 3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
}

void TestReduceByKeyDeviceSeq()
{
  TestReduceByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceSeq);

void TestReduceByKeyDeviceDevice()
{
  TestReduceByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceDevice);

void TestReduceByKeyDeviceNoSync()
{
  TestReduceByKeyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestReduceByKeyCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector keys;
  Vector values;

  cuda::std::pair<Vector::iterator, Vector::iterator> new_last;

  // basic test
  initialize_keys(keys);
  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::reduce_by_key(
    streampolicy, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);

  ASSERT_EQUAL(output_values[0], 1);
  ASSERT_EQUAL(output_values[1], 2);
  ASSERT_EQUAL(output_values[2], 3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  new_last = thrust::reduce_by_key(
    streampolicy,
    keys.begin(),
    keys.end(),
    values.begin(),
    output_keys.begin(),
    output_values.begin(),
    is_equal_div_10_reduce<T>());

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);

  ASSERT_EQUAL(output_values[0], 1);
  ASSERT_EQUAL(output_values[1], 20);
  ASSERT_EQUAL(output_values[2], 15);

  // test BinaryFunction
  initialize_keys(keys);
  initialize_values(values);

  new_last = thrust::reduce_by_key(
    streampolicy,
    keys.begin(),
    keys.end(),
    values.begin(),
    output_keys.begin(),
    output_values.begin(),
    ::cuda::std::equal_to<T>(),
    ::cuda::std::plus<T>());

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);

  ASSERT_EQUAL(output_values[0], 1);
  ASSERT_EQUAL(output_values[1], 2);
  ASSERT_EQUAL(output_values[2], 3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);

  cudaStreamDestroy(s);
}

void TestReduceByKeyCudaStreamsSync()
{
  TestReduceByKeyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestReduceByKeyCudaStreamsSync);

void TestReduceByKeyCudaStreamsNoSync()
{
  TestReduceByKeyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestReduceByKeyCudaStreamsNoSync);

// Maps indices to key ids
class div_op
{
  std::int64_t m_divisor;

public:
  _CCCL_HOST div_op(std::int64_t divisor)
      : m_divisor(divisor)
  {}

  _CCCL_HOST_DEVICE std::int64_t operator()(std::int64_t x) const
  {
    return x / m_divisor;
  }
};

// Produces unique sequence for key
class mod_op
{
  std::int64_t m_divisor;

public:
  _CCCL_HOST mod_op(std::int64_t divisor)
      : m_divisor(divisor)
  {}

  _CCCL_HOST_DEVICE std::int64_t operator()(std::int64_t x) const
  {
    // div: 2
    // idx: 0 1   2 3   4 5
    // key: 0 0 | 1 1 | 2 2
    // mod: 0 1 | 0 1 | 0 1
    // ret: 0 1   1 2   2 3
    return (x % m_divisor) + (x / m_divisor);
  }
};

void TestReduceByKeyWithBigIndexesHelper(int magnitude)
{
  const std::int64_t key_size_magnitude = 8;
  ASSERT_EQUAL(true, key_size_magnitude < magnitude);

  const std::int64_t num_items       = 1ll << magnitude;
  const std::int64_t num_unique_keys = 1ll << key_size_magnitude;

  // Size of each key group
  const std::int64_t key_size = num_items / num_unique_keys;

  using counting_it      = thrust::counting_iterator<std::int64_t>;
  using transform_key_it = thrust::transform_iterator<div_op, counting_it>;
  using transform_val_it = thrust::transform_iterator<mod_op, counting_it>;

  counting_it count_begin(0ll);
  counting_it count_end = count_begin + num_items;
  ASSERT_EQUAL(static_cast<std::int64_t>(::cuda::std::distance(count_begin, count_end)), num_items);

  transform_key_it keys_begin(count_begin, div_op{key_size});
  transform_key_it keys_end(count_end, div_op{key_size});

  transform_val_it values_begin(count_begin, mod_op{key_size});

  thrust::device_vector<std::int64_t> output_keys(num_unique_keys);
  thrust::device_vector<std::int64_t> output_values(num_unique_keys);

  // example:
  //  items:        6
  //  unique_keys:  2
  //  key_size:     3
  //  keys:         0 0 0 | 1 1 1
  //  values:       0 1 2 | 1 2 3
  //  result:       3       6     = sum(range(key_size)) + key_size * key_id
  thrust::reduce_by_key(keys_begin, keys_end, values_begin, output_keys.begin(), output_values.begin());

  ASSERT_EQUAL(true, thrust::equal(output_keys.begin(), output_keys.end(), count_begin));

  thrust::host_vector<std::int64_t> result = output_values;

  const std::int64_t sum = (key_size - 1) * key_size / 2;
  for (std::int64_t key_id = 0; key_id < num_unique_keys; key_id++)
  {
    ASSERT_EQUAL(result[key_id], sum + key_id * key_size);
  }
}

void TestReduceByKeyWithBigIndexes()
{
  TestReduceByKeyWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestReduceByKeyWithBigIndexesHelper(31);
  TestReduceByKeyWithBigIndexesHelper(32);
  TestReduceByKeyWithBigIndexesHelper(33);
#endif
}
DECLARE_UNITTEST(TestReduceByKeyWithBigIndexes);

void TestReduceByKeyWithCustomEqualityOp()
{
  using key_vector_t = thrust::device_vector<cuda::std::int32_t>;
  using val_vector_t = thrust::device_vector<cuda::std::int32_t>;
  using key_t        = key_vector_t::value_type;
  using val_t        = val_vector_t::value_type;

  auto constexpr num_items = 1000;
  auto keys                = cuda::make_counting_iterator(key_t{0});
  auto values              = cuda::make_counting_iterator(val_t{42});

  thrust::device_vector<cuda::std::uint32_t> error_counter(1, 0);
  auto const error_counter_ptr = thrust::raw_pointer_cast(error_counter.data());

  key_vector_t unique_out(num_items);
  val_vector_t aggregates_out(num_items);
  auto [unique_out_end, aggregates_out_end] = thrust::reduce_by_key(
    keys,
    keys + num_items,
    values,
    unique_out.begin(),
    aggregates_out.begin(),
    check_valid_item_op{error_counter_ptr, num_items - 1});

  // Verify that the number of unique keys is correct
  const auto num_unique_out     = cuda::std::distance(unique_out.begin(), unique_out_end);
  const auto num_aggregates_out = cuda::std::distance(aggregates_out.begin(), aggregates_out_end);
  ASSERT_EQUAL(num_unique_out, num_items);
  ASSERT_EQUAL(num_aggregates_out, num_items);

  // Verify that the equality operator was never invoked on out-of-bounds items
  ASSERT_EQUAL(error_counter[0], cuda::std::uint32_t{0});

  // Verify that unique keys are correct
  bool all_keys_correct = thrust::equal(unique_out.cbegin(), unique_out.cend(), keys);
  ASSERT_EQUAL(all_keys_correct, true);

  // Verify that the aggregates are correct
  bool all_values_correct = thrust::equal(aggregates_out.cbegin(), aggregates_out.cend(), values);
  ASSERT_EQUAL(all_values_correct, true);
}

DECLARE_UNITTEST(TestReduceByKeyWithCustomEqualityOp);

void TestReduceByKeyWithDifferentAccumulatorT()
{
  using key_t          = cuda::std::uint32_t;
  using val_t          = cuda::std::uint8_t;
  using reduction_op_t = check_accumulator_t_op<cuda::std::uint32_t>;

  auto constexpr num_items            = 20000;
  auto constexpr expected_num_uniques = 1;
  constexpr auto unique_key           = key_t{42U};
  auto keys                           = cuda::make_constant_iterator(unique_key);
  auto values                         = cuda::make_counting_iterator(val_t{0});

  thrust::device_vector<cuda::std::uint32_t> error_counter(1, 0);
  auto const error_counter_ptr = thrust::raw_pointer_cast(error_counter.data());

  thrust::device_vector<key_t> unique_out(expected_num_uniques);
  thrust::device_vector<val_t> aggregates_out(expected_num_uniques);
  auto [unique_out_end, aggregates_out_end] = thrust::reduce_by_key(
    keys,
    keys + num_items,
    values,
    unique_out.begin(),
    aggregates_out.begin(),
    cuda::std::equal_to<>{},
    reduction_op_t{error_counter_ptr});

  // Verify that the number of unique keys is correct
  auto num_unique_out     = cuda::std::distance(unique_out.begin(), unique_out_end);
  auto num_aggregates_out = cuda::std::distance(aggregates_out.begin(), aggregates_out_end);
  ASSERT_EQUAL(num_unique_out, expected_num_uniques);
  ASSERT_EQUAL(num_aggregates_out, expected_num_uniques);

  // Verify that the equality operator was never invoked on out-of-bounds items
  ASSERT_EQUAL(error_counter[0], cuda::std::uint32_t{0});

  // Verify that the unique key is correct
  ASSERT_EQUAL(unique_out[0], unique_key);

  // // Verify that the aggregate is correct
  constexpr auto mod_val            = 0x01 << cuda::std::numeric_limits<val_t>::digits;
  constexpr auto sum                = ((num_items * (num_items - 1)) / 2);
  constexpr auto expected_aggregate = static_cast<val_t>(sum % mod_val);
  ASSERT_EQUAL(aggregates_out[0], expected_aggregate);
}

DECLARE_UNITTEST(TestReduceByKeyWithDifferentAccumulatorT);
