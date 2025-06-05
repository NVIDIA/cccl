#include <thrust/device_vector.h>

#include <unittest/unittest.h>

#if _CCCL_HAS_CTK()
void TestConversionToCudeviceptr()
{
  thrust::device_vector<int> vec(3);
  int* p = thrust::raw_pointer_cast(vec.data());

  CUdeviceptr cdevptr = static_cast<CUdeviceptr>(vec.data());
  ASSERT_EQUAL(cdevptr, reinterpret_cast<CUdeviceptr>(p));
}
DECLARE_UNITTEST(TestConversionToCudeviceptr);
#endif
