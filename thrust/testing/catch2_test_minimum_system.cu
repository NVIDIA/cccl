#include <thrust/iterator/detail/minimum_system.h>

#include <cuda/std/type_traits>

#include "catch2_test_helper.h"

template <typename System, typename SeqSystem>
void check()
{
  STATIC_REQUIRE(cuda::std::is_convertible_v<System, SeqSystem>);
  STATIC_REQUIRE(!cuda::std::is_convertible_v<SeqSystem, System>);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::detail::minimum_system_t<SeqSystem, System>, SeqSystem>);
  STATIC_REQUIRE(cuda::std::is_same_v<thrust::detail::minimum_system_t<System, SeqSystem>, SeqSystem>);
}

TEST_CASE("host and device systems convert to sequential", "[minimum_system]")
{
  using seq      = decltype(thrust::seq);
  using seq_tag  = decltype(thrust::seq)::tag_type;
  using dev      = decltype(thrust::device);
  using dev_tag  = thrust::device_system_tag;
  using host     = decltype(thrust::host);
  using host_tag = thrust::host_system_tag;

  check<dev, seq>();
  check<dev_tag, seq>();
  check<dev, seq_tag>();
  check<dev_tag, seq_tag>();
  check<host, seq_tag>();
  check<host_tag, seq_tag>();
}
