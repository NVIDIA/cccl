#include <thrust/universal_vector.h>

#include <unittest/unittest.h>

template <class VecInT, class VecOutT>
_CCCL_HOST_DEVICE void universal_vector_access(VecInT& in, VecOutT& out)
{
  const int expected_front = 4;
  const int expected_back  = 2;

  out[0] = in.size() == 2 && //
           in[0] == expected_front && //
           in.front() == expected_front && //
         *in.data() == expected_front && //
           in[1] == expected_back && //
           in.back() == expected_back;
}

#if defined(THRUST_TEST_DEVICE_SIDE)
template <class VecInT, class VecOutT>
__global__ void universal_vector_device_access_kernel(VecInT& vec, VecOutT& out)
{
  universal_vector_access(vec, out);
}

template <class VecInT, class VecOutT>
void test_universal_vector_access(VecInT& vec, VecOutT& out)
{
  universal_vector_device_access_kernel<<<1, 1>>>(vec, out);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  ASSERT_EQUAL(out[0], true);
}
#else
template <class VecInT, class VecOutT>
void test_universal_vector_access(VecInT& vec, VecOutT& out)
{
  universal_vector_access(vec, out);
  ASSERT_EQUAL(out[0], true);
}
#endif

template <typename UniversalIntVector, typename UniversalBoolVector>
void TestDeviceAccess()
{
  using in_vector_t  = UniversalIntVector;
  using out_vector_t = UniversalBoolVector;

  in_vector_t* in_ptr{};
  cudaMallocManaged(&in_ptr, sizeof(*in_ptr));
  new (in_ptr) in_vector_t(1);

  auto& in = *in_ptr;
  in.resize(2);
  in = {4, 2};

  out_vector_t* out_ptr{};
  cudaMallocManaged(&out_ptr, sizeof(*out_ptr));
  new (out_ptr) out_vector_t(1);
  auto& out = *out_ptr;

  out.resize(1);
  out[0] = false;

  test_universal_vector_access(in, out);
  const auto& const_in = *in_ptr;
  test_universal_vector_access(const_in, out);

  cudaFree(in_ptr);
  cudaFree(out_ptr);
}
DECLARE_UNITTEST_WITH_NAME((TestDeviceAccess<thrust::universal_vector<int>, thrust::universal_vector<bool>>),
                           TestUniversalVectorDeviceAccess);
DECLARE_UNITTEST_WITH_NAME(
  (TestDeviceAccess<thrust::universal_host_pinned_vector<int>, thrust::universal_host_pinned_vector<bool>>),
  TestUniversalHPVectorDeviceAccess);
