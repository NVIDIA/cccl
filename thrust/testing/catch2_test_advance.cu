#include <thrust/advance.h>
#include <thrust/sequence.h>

#include "catch2_test_helper.h"

// TODO expand this with other iterator types (forward, bidirectional, etc.)

C2H_TEST("TestAdvance", "[advance]", vector_list)
{
  using Vector   = typename c2h::get<0, TestType>;
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector v(10);
  thrust::sequence(v.begin(), v.end());

  Iterator i = v.begin();
  thrust::advance(i, 1);
  CHECK(*i == T(1));

  thrust::advance(i, 8);
  CHECK(*i == T(9));

  thrust::advance(i, -4);
  CHECK(*i == T(5));
}

C2H_TEST("TestNext", "[next]", vector_list)
{
  using Vector   = typename c2h::get<0, TestType>;
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector v(10);
  thrust::sequence(v.begin(), v.end());

  Iterator const i0 = v.begin();
  Iterator const i1 = thrust::next(i0);
  CHECK(*i0 == T(0));
  CHECK(*i1 == T(1));

  Iterator const i2 = thrust::next(i1, 8);
  CHECK(*i0 == T(0));
  CHECK(*i1 == T(1));
  CHECK(*i2 == T(9));

  Iterator const i3 = thrust::next(i2, -4);
  CHECK(*i0 == T(0));
  CHECK(*i1 == T(1));
  CHECK(*i2 == T(9));
  CHECK(*i3 == T(5));
}

C2H_TEST("TestPrev", "[prev]", vector_list)
{
  using Vector   = typename c2h::get<0, TestType>;
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector v(10);
  thrust::sequence(v.begin(), v.end());

  Iterator const i0 = v.end();
  Iterator const i1 = thrust::prev(i0);
  CHECK((i0 == v.end()));
  CHECK(*i1 == T(9));

  Iterator const i2 = thrust::prev(i1, 8);
  CHECK((i0 == v.end()));
  CHECK(*i1 == T(9));
  CHECK(*i2 == T(1));

  Iterator const i3 = thrust::prev(i2, -4);
  CHECK((i0 == v.end()));
  CHECK(*i1 == T(9));
  CHECK(*i2 == T(1));
  CHECK(*i3 == T(5));
}
