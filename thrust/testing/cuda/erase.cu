#include <thrust/erase.h>
#include <thrust/execution_policy.h>

#include <unittest/unittest.h>

struct tracked_int
{
  int value;

  _CCCL_HOST_DEVICE tracked_int(int v = 0)
      : value(v)
  {}

  tracked_int(const tracked_int&)            = default;
  tracked_int& operator=(const tracked_int&) = default;

  _CCCL_HOST_DEVICE ~tracked_int() {}

  _CCCL_HOST_DEVICE bool operator==(const tracked_int& other) const
  {
    return value == other.value;
  }

  friend std::ostream& operator<<(std::ostream& os, const tracked_int& t)
  {
    return os << t.value;
  }
};

template <typename T>
struct is_even
{
  _CCCL_HOST_DEVICE bool operator()(T x) const
  {
    return (static_cast<unsigned int>(x) & 1) == 0;
  }
};

struct is_even_tracked_int
{
  _CCCL_HOST_DEVICE bool operator()(const tracked_int& x) const
  {
    return (x.value & 1) == 0;
  }
};

void TestEraseCudaStreams()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  cudaStream_t s;
  cudaStreamCreate(&s);

  size_t removed = thrust::erase(thrust::cuda::par.on(s), data, (T) 2);

  cudaStreamSynchronize(s);

  Vector ref{1, 1, 3};
  ASSERT_EQUAL(removed, 2ul);
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}

DECLARE_UNITTEST(TestEraseCudaStreams);

void TestEraseIfCudaStreams()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  cudaStream_t s;
  cudaStreamCreate(&s);

  size_t removed = thrust::erase_if(thrust::cuda::par.on(s), data, is_even<T>());

  cudaStreamSynchronize(s);

  Vector ref{1, 1, 3};
  ASSERT_EQUAL(removed, 2ul);
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestEraseIfCudaStreams);

void TestEraseIfCudaStreamsNonTrivialType()
{
  using Vector = thrust::device_vector<tracked_int>;

  Vector data{1, 2, 1, 3, 2};

  cudaStream_t s;
  cudaStreamCreate(&s);

  size_t removed = thrust::erase_if(thrust::cuda::par.on(s), data, is_even_tracked_int());

  cudaStreamSynchronize(s);

  Vector ref{1, 1, 3};
  ASSERT_EQUAL(removed, 2ul);
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestEraseIfCudaStreamsNonTrivialType);
