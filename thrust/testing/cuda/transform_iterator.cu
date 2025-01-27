#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

#include <unittest/unittest.h>

// see also: https://github.com/NVIDIA/cccl/issues/3541
template <class Vector>
void TestTransformWithLambda()
{
  using T = typename Vector::value_type;
  Vector A{1, 2, 3, 4, 5, 6, 7};
  const auto result = thrust::any_of(A.begin(), A.end(), [] __host__ __device__(T v) {
    return v < 4;
  });
  ASSERT_EQUAL(result, true);
}

DECLARE_INTEGRAL_VECTOR_UNITTEST(TestTransformWithLambda);
