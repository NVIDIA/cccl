#include <cuda/std/type_traits>

#include <cstdio>

#include <c2h/catch2_test_helper.h>

C2H_TEST("libcudacxx can be used", "")
{
  printf("CCCL version: %d.%d.%d\n", CCCL_MAJOR_VERSION, CCCL_MINOR_VERSION, CCCL_PATCH_VERSION);
  REQUIRE(cuda::std::true_type::value);
}
