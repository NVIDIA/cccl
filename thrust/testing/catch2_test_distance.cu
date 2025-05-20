#include <thrust/distance.h>

#include "catch2_test_helper.h"

_CCCL_SUPPRESS_DEPRECATED_PUSH
TEMPLATE_LIST_TEST_CASE("TestDistance", "[distance]", vector_list)
{
  using Vector   = TestType;
  using Iterator = typename Vector::iterator;

  Vector v(100);
  Iterator i = v.begin();
  CHECK(thrust::distance(i, v.end()) == 100);

  i++;
  CHECK(thrust::distance(i, v.end()) == 99);

  i += 49;
  CHECK(thrust::distance(i, v.end()) == 50);
  CHECK(thrust::distance(i, i) == 0);
}
_CCCL_SUPPRESS_DEPRECATED_POP
