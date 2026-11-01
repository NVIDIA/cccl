#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cuda/stream>

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
void test_on_stream(ExecutionPolicy policy)
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

  REQUIRE(thrust::reduce(streampolicy, v.begin(), v.end()) == 2);

  cudaStreamDestroy(s);
}

TEST_CASE("TestCudartStreamSync", "[stream_wrapper]")
{
  test_on_stream<stream_wrapper>(thrust::cuda::par);
}

TEST_CASE("TestCudartStreamNoSync", "[stream_wrapper]")
{
  test_on_stream<stream_wrapper>(thrust::cuda::par_nosync);
}

TEST_CASE("TestCudaStreamRefSync", "[stream_wrapper]")
{
  test_on_stream<stream_wrapper_ref>(thrust::cuda::par);
}

TEST_CASE("TestCudaStreamRefNoSync", "[stream_wrapper]")
{
  test_on_stream<stream_wrapper_ref>(thrust::cuda::par_nosync);
}
