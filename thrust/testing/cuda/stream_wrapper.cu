#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

#include <cuda/stream_ref>

#include <unittest/unittest.h>

// Simple non-owning stream wrapper that allows implicit conversion to cudaStream_t.
struct stream_wrapper
{
  stream_wrapper(cudaStream_t s)
      : stream(s)
  {}

  operator cudaStream_t() const
  {
    return stream;
  }

  cudaStream_t stream;
};

// Simple non-owning stream wrapper that allows implicit conversion to cudaStream_t and cuda::stream_ref.
struct stream_wrapper_ref
{
  stream_wrapper_ref(cudaStream_t s)
      : stream(s)
  {}

  operator cudaStream_t() const
  {
    return stream;
  }
  operator cuda::stream_ref() const
  {
    return cuda::stream_ref(stream);
  }

  cudaStream_t stream;
};

template <typename Wrapper, typename ExecutionPolicy>
void TestOnStream(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;

  Vector v(3);
  v[0] = 1;
  v[1] = -2;
  v[2] = 3;

  cudaStream_t s;
  cudaStreamCreate(&s);

  Wrapper wrapper(s);

  auto streampolicy = policy.on(wrapper);

  ASSERT_EQUAL(thrust::reduce(streampolicy, v.begin(), v.end()), 2);

  cudaStreamDestroy(s);
}

void TestCudartStreamSync()
{
  TestOnStream<stream_wrapper>(thrust::cuda::par);
}
DECLARE_UNITTEST(TestCudartStreamSync);

void TestCudartStreamNoSync()
{
  TestOnStream<stream_wrapper>(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestCudartStreamNoSync);

void TestCudaStreamRefSync()
{
  TestOnStream<stream_wrapper_ref>(thrust::cuda::par);
}
DECLARE_UNITTEST(TestCudaStreamRefSync);

void TestCudaStreamRefNoSync()
{
  TestOnStream<stream_wrapper_ref>(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestCudaStreamRefNoSync);
