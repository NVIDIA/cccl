#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/unique.h>

#include <unittest/unittest.h>

template <typename T>
struct div_n_equality_op
{
  T div;
  __host__ __device__ bool operator()(const T x, const T& y) const
  {
    return (x / div) == (y / div);
  }
};

template <typename T>
struct multiply_n
{
  T multiplier;
  __host__ __device__ T operator()(T x)
  {
    return x * multiplier;
  }
};

struct check_valid_item_op
{
  ::cuda::std::uint32_t* error_counter{};
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

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void unique_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::unique(exec, first, last);
}

template <typename ExecutionPolicy, typename Iterator1, typename BinaryPredicate, typename Iterator2>
__global__ void
unique_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::unique(exec, first, last, pred);
}

template <typename ExecutionPolicy>
void TestUniqueDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  unique_kernel<<<1, 1>>>(exec, data.begin(), data.end(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 7);
  data.erase(new_last, data.end());
  Vector ref{11, 12, 20, 29, 21, 31, 37}; // should we consider calculating ref from std::algorithm if exists?
  ASSERT_EQUAL(data, ref);

  unique_kernel<<<1, 1>>>(exec, data.begin(), new_last, div_n_equality_op<T>{10}, new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 3);
  data.erase(new_last, data.end());
  ref = {11, 20, 31};
  ASSERT_EQUAL(data, ref);
}

void TestUniqueDeviceSeq()
{
  TestUniqueDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueDeviceSeq);

void TestUniqueDeviceDevice()
{
  TestUniqueDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueDeviceDevice);

void TestUniqueDeviceNoSync()
{
  TestUniqueDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestUniqueCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::unique(streampolicy, data.begin(), data.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 7);
  data.erase(new_last, data.end());
  Vector ref{11, 12, 20, 29, 21, 31, 37};
  ASSERT_EQUAL(data, ref);

  new_last = thrust::unique(streampolicy, data.begin(), new_last, div_n_equality_op<T>{10});
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 3);
  data.erase(new_last, data.end());
  ref = {11, 20, 31};
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}

void TestUniqueCudaStreamsSync()
{
  TestUniqueCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueCudaStreamsSync);

void TestUniqueCudaStreamsNoSync()
{
  TestUniqueCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCudaStreamsNoSync);

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void
unique_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Iterator3 result2)
{
  *result2 = thrust::unique_copy(exec, first, last, result1);
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename BinaryPredicate, typename Iterator3>
__global__ void unique_copy_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, BinaryPredicate pred, Iterator3 result2)
{
  *result2 = thrust::unique_copy(exec, first, last, result1, pred);
}

template <typename ExecutionPolicy>
void TestUniqueCopyDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  Vector output(10, -1);

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  unique_copy_kernel<<<1, 1>>>(exec, data.begin(), data.end(), output.begin(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - output.begin(), 7);
  output.erase(new_last, output.end());
  Vector ref{11, 12, 20, 29, 21, 31, 37};
  ASSERT_EQUAL(output, ref);

  unique_copy_kernel<<<1, 1>>>(
    exec, output.begin(), new_last, data.begin(), div_n_equality_op<T>{10}, new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 3);
  data.erase(new_last, data.end());
  ref = {11, 20, 31};
  ASSERT_EQUAL(data, ref);
}

void TestUniqueCopyDeviceSeq()
{
  TestUniqueCopyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueCopyDeviceSeq);

void TestUniqueCopyDeviceDevice()
{
  TestUniqueCopyDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueCopyDeviceDevice);

void TestUniqueCopyDeviceNoSync()
{
  TestUniqueCopyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCopyDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestUniqueCopyCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  Vector output(10, -1);

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::unique_copy(streampolicy, data.begin(), data.end(), output.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - output.begin(), 7);
  output.erase(new_last, output.end());
  Vector ref{11, 12, 20, 29, 21, 31, 37};
  ASSERT_EQUAL(output, ref);

  new_last = thrust::unique_copy(streampolicy, output.begin(), new_last, data.begin(), div_n_equality_op<T>{10});
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 3);
  data.erase(new_last, data.end());
  ref = {11, 20, 31};
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}

void TestUniqueCopyCudaStreamsSync()
{
  TestUniqueCopyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueCopyCudaStreamsSync);

void TestUniqueCopyCudaStreamsNoSync()
{
  TestUniqueCopyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCopyCudaStreamsNoSync);

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void unique_count_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::unique_count(exec, first, last);
}

template <typename ExecutionPolicy, typename Iterator1, typename BinaryPredicate, typename Iterator2>
__global__ void
unique_count_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::unique_count(exec, first, last, pred);
}

template <typename ExecutionPolicy>
void TestUniqueCountDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  Vector output(1, -1);

  unique_count_kernel<<<1, 1>>>(exec, data.begin(), data.end(), output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(output[0], 7);

  unique_count_kernel<<<1, 1>>>(exec, data.begin(), data.end(), div_n_equality_op<T>{10}, output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(output[0], 3);
}

void TestUniqueCountDeviceSeq()
{
  TestUniqueCountDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueCountDeviceSeq);

void TestUniqueCountDeviceDevice()
{
  TestUniqueCountDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueCountDeviceDevice);

void TestUniqueCountDeviceNoSync()
{
  TestUniqueCountDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCountDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestUniqueCountCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  int result = thrust::unique_count(streampolicy, data.begin(), data.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(result, 7);

  result = thrust::unique_count(streampolicy, data.begin(), data.end(), div_n_equality_op<T>{10});
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(result, 3);

  cudaStreamDestroy(s);
}

void TestUniqueCountCudaStreamsSync()
{
  TestUniqueCountCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueCountCudaStreamsSync);

void TestUniqueCountCudaStreamsNoSync()
{
  TestUniqueCountCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCountCudaStreamsNoSync);

void TestUniqueWithMagnitude(int magnitude)
{
  using offset_t      = std::int64_t;
  using equality_op_t = div_n_equality_op<offset_t>;

  offset_t run_length_of_equal_items = offset_t{10};
  equality_op_t equality_op          = equality_op_t{run_length_of_equal_items};

  // Prepare input
  offset_t num_items = offset_t{1ull} << magnitude;
  thrust::counting_iterator<offset_t> begin(offset_t{0});
  auto end = begin + num_items;
  ASSERT_EQUAL(static_cast<offset_t>(cuda::std::distance(begin, end)), num_items);

  offset_t expected_num_unique = ::cuda::ceil_div(num_items, offset_t{10});
  thrust::device_vector<offset_t> unique_out(expected_num_unique);
  auto unique_out_end = thrust::unique_copy(begin, end, unique_out.begin(), equality_op);

  // Ensure number of selected items are correct
  offset_t num_selected_out = static_cast<offset_t>(cuda::std::distance(unique_out.begin(), unique_out_end));
  ASSERT_EQUAL(num_selected_out, expected_num_unique);
  unique_out.resize(expected_num_unique);

  // Ensure selected items are correct
  auto expected_out_it     = thrust::make_transform_iterator(begin, multiply_n<offset_t>{run_length_of_equal_items});
  bool all_results_correct = thrust::equal(unique_out.begin(), unique_out.end(), expected_out_it);
  ASSERT_EQUAL(all_results_correct, true);
}

void TestUniqueWithLargeNumberOfItems()
{
  for (int mag : {30, 31, 32, 33})
  {
    TestUniqueWithMagnitude(mag);
  }
}
DECLARE_UNITTEST(TestUniqueWithLargeNumberOfItems);

void TestUniqueWithCustomEqualityOp()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  auto constexpr num_items = 1000;
  auto data                = thrust::make_counting_iterator(T{0});

  thrust::device_vector<::cuda::std::uint32_t> error_counter(1, 0);
  auto const error_counter_ptr = thrust::raw_pointer_cast(error_counter.data());

  Vector unique_out(num_items);
  auto unique_out_end = thrust::unique_copy(
    data, data + num_items, unique_out.begin(), check_valid_item_op{error_counter_ptr, num_items - 1});

  auto num_selected_out = cuda::std::distance(unique_out.begin(), unique_out_end);
  ASSERT_EQUAL(num_selected_out, num_items);
  ASSERT_EQUAL(error_counter[0], ::cuda::std::uint32_t{0});
  bool all_results_correct = thrust::equal(unique_out.cbegin(), unique_out.cend(), data);
  ASSERT_EQUAL(all_results_correct, true);
}

DECLARE_UNITTEST(TestUniqueWithCustomEqualityOp);

template <typename F>
struct NonConstAdapter
{
  F f;
  NonConstAdapter(const F& func)
      : f(func)
  {}

  template <typename... Args>
  __device__ auto operator()(Args&&... args) -> decltype(f(cuda::std::forward<Args>(args)...))
  {
    return f(cuda::std::forward<Args>(args)...);
  }
};

void TestUniqueWithCustomEqualityOpMutable()
{
  using Vector = thrust::device_vector<int>;

  thrust::device_vector<int> in = {1, 1, 2, 3, 4, 4, 5};
  thrust::unique(thrust::cuda::par, in.begin(), in.end(), NonConstAdapter(cuda::std::equal_to<>{}));
}

DECLARE_UNITTEST(TestUniqueWithCustomEqualityOpMutable);
