#include <thrust/execution_policy.h>
#include <thrust/mismatch.h>

#include <cuda/std/atomic>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void
mismatch_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 result)
{
  *result = thrust::mismatch(exec, first1, last1, first2);
}

template <typename ExecutionPolicy>
void TestMismatchDevice(ExecutionPolicy exec)
{
  thrust::device_vector<int> a = {1, 2, 3, 4};
  thrust::device_vector<int> b = {1, 2, 4, 3};

  using pair_type =
    cuda::std::pair<typename thrust::device_vector<int>::iterator, typename thrust::device_vector<int>::iterator>;

  thrust::device_vector<pair_type> d_result(1);

  mismatch_kernel<<<1, 1>>>(exec, a.begin(), a.end(), b.begin(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);
  }

  REQUIRE(2 == ((pair_type) d_result[0]).first - a.begin());
  REQUIRE(2 == ((pair_type) d_result[0]).second - b.begin());

  b[2] = 3;

  mismatch_kernel<<<1, 1>>>(exec, a.begin(), a.end(), b.begin(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);
  }

  REQUIRE(3 == ((pair_type) d_result[0]).first - a.begin());
  REQUIRE(3 == ((pair_type) d_result[0]).second - b.begin());

  b[3] = 4;

  mismatch_kernel<<<1, 1>>>(exec, a.begin(), a.end(), b.begin(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);
  }

  REQUIRE(4 == ((pair_type) d_result[0]).first - a.begin());
  REQUIRE(4 == ((pair_type) d_result[0]).second - b.begin());
}

void TestMismatchDeviceSeq()
{
  TestMismatchDevice(thrust::seq);
}
TEST_CASE("TestMismatchDeviceSeq", "[mismatch]")
{
  TestMismatchDeviceSeq();
}

void TestMismatchDeviceDevice()
{
  TestMismatchDevice(thrust::device);
}
TEST_CASE("TestMismatchDeviceDevice", "[mismatch]")
{
  TestMismatchDeviceDevice();
}
#endif

void TestMismatchCudaStreams()
{
  using Vector = thrust::device_vector<int>;

  Vector a = {1, 2, 3, 4};
  Vector b = {1, 2, 4, 3};

  cudaStream_t s;
  cudaStreamCreate(&s);

  REQUIRE(thrust::mismatch(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin()).first - a.begin() == 2);
  REQUIRE(thrust::mismatch(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin()).second - b.begin() == 2);

  b[2] = 3;

  REQUIRE(thrust::mismatch(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin()).first - a.begin() == 3);
  REQUIRE(thrust::mismatch(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin()).second - b.begin() == 3);

  b[3] = 4;

  REQUIRE(thrust::mismatch(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin()).first - a.begin() == 4);
  REQUIRE(thrust::mismatch(thrust::cuda::par.on(s), a.begin(), a.end(), b.begin()).second - b.begin() == 4);

  cudaStreamDestroy(s);
}
TEST_CASE("TestMismatchCudaStreams", "[mismatch]")
{
  TestMismatchCudaStreams();
}

// see https://github.com/NVIDIA/cccl/issues/3591
template <typename T>
class Wrapper
{
public:
  Wrapper()
  {
    ++my_count;
  }

  _CCCL_HOST_DEVICE bool operator==(const Wrapper&) const
  {
    return true;
  }

  ~Wrapper()
  {
    --my_count;
  }

private:
  static cuda::std::atomic<size_t> my_count;
  T dummy;
};

void TestMismatchBug3591()
{
  using T = Wrapper<int32_t>;
  T* p    = nullptr;
  thrust::mismatch(thrust::device, p, p, p, cuda::std::equal_to<T>());
}
TEST_CASE("TestMismatchBug3591", "[mismatch]")
{
  TestMismatchBug3591();
}
