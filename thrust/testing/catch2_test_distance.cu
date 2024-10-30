#include <thrust/distance.h>

#include "catch2_test_helper.h"

C2H_TEST("TestDistance", "[distance]", vector_list)
{
  using Vector = typename c2h::get<0, TestType>;
  ;
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
