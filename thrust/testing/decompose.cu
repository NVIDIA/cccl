#include <thrust/system/detail/internal/decompose.h>

#include <unittest/unittest.h>

void TestUniformDecomposition()
{
  using thrust::system::detail::internal::uniform_decomposition;

  {
    const uniform_decomposition<int> ud(10, 10, 1);

    // [0,10)
    REQUIRE(ud.size() == 1);
    REQUIRE(ud[0].begin() == 0);
    REQUIRE(ud[0].end() == 10);
    REQUIRE(ud[0].size() == 10);
  }

  {
    const uniform_decomposition<int> ud(10, 20, 1);

    // [0,10)
    REQUIRE(ud.size() == 1);
    REQUIRE(ud[0].begin() == 0);
    REQUIRE(ud[0].end() == 10);
    REQUIRE(ud[0].size() == 10);
  }

  {
    const uniform_decomposition<int> ud(8, 5, 2);

    // [0,5)[5,8)
    REQUIRE(ud.size() == 2);
    REQUIRE(ud[0].begin() == 0);
    REQUIRE(ud[0].end() == 5);
    REQUIRE(ud[0].size() == 5);
    REQUIRE(ud[1].begin() == 5);
    REQUIRE(ud[1].end() == 8);
    REQUIRE(ud[1].size() == 3);
  }

  {
    const uniform_decomposition<int> ud(8, 5, 3);

    // [0,5)[5,8)
    REQUIRE(ud.size() == 2);
    REQUIRE(ud[0].begin() == 0);
    REQUIRE(ud[0].end() == 5);
    REQUIRE(ud[0].size() == 5);
    REQUIRE(ud[1].begin() == 5);
    REQUIRE(ud[1].end() == 8);
    REQUIRE(ud[1].size() == 3);
  }

  {
    const uniform_decomposition<int> ud(10, 1, 2);

    // [0,5)[5,10)
    REQUIRE(ud.size() == 2);
    REQUIRE(ud[0].begin() == 0);
    REQUIRE(ud[0].end() == 5);
    REQUIRE(ud[0].size() == 5);
    REQUIRE(ud[1].begin() == 5);
    REQUIRE(ud[1].end() == 10);
    REQUIRE(ud[1].size() == 5);
  }

  {
    // [0,4)[4,8)[8,10)
    const uniform_decomposition<int> ud(10, 2, 3);

    REQUIRE(ud.size() == 3);
    REQUIRE(ud[0].begin() == 0);
    REQUIRE(ud[0].end() == 4);
    REQUIRE(ud[0].size() == 4);
    REQUIRE(ud[1].begin() == 4);
    REQUIRE(ud[1].end() == 8);
    REQUIRE(ud[1].size() == 4);
    REQUIRE(ud[2].begin() == 8);
    REQUIRE(ud[2].end() == 10);
    REQUIRE(ud[2].size() == 2);
  }
}
DECLARE_UNITTEST(TestUniformDecomposition);
