#include <thrust/equal.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestEqual(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector v1{5, 2, 0, 0, 0};
  Vector v2{5, 2, 0, 6, 1};

  {
    REQUIRE(thrust::equal(exec, v1.begin(), v1.end(), v1.begin()) == true);
    REQUIRE(thrust::equal(exec, v1.begin(), v1.end(), v2.begin()) == false);

    REQUIRE(thrust::equal(exec, v1.begin(), v1.begin() + 0, v1.begin()) == true);
    REQUIRE(thrust::equal(exec, v1.begin(), v1.begin() + 1, v1.begin()) == true);
    REQUIRE(thrust::equal(exec, v1.begin(), v1.begin() + 3, v2.begin()) == true);
    REQUIRE(thrust::equal(exec, v1.begin(), v1.begin() + 4, v2.begin()) == false);
  }
  {
    REQUIRE(thrust::equal(exec, v1.begin(), v1.end(), v2.begin(), thrust::less_equal<T>()) == true);
    REQUIRE(thrust::equal(exec, v1.begin(), v1.end(), v2.begin(), thrust::greater<T>()) == false);
  }
}

TEST_CASE("TestEqualPar", "[hpx]")
{
  TestEqual(thrust::hpx::par);
}

TEST_CASE("TestEqualParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestEqual(thrust::hpx::par.on(exec));
}

TEST_CASE("TestEqualParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestEqual(thrust::hpx::par.on(exec));
}

TEST_CASE("TestEqualParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestEqual(thrust::hpx::par.with(acs));
}

TEST_CASE("TestEqualParUnseq", "[hpx]")
{
  TestEqual(thrust::hpx::par_unseq);
}

TEST_CASE("TestEqualParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestEqual(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestEqualParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestEqual(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestEqualParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestEqual(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestEqualSeq", "[hpx]")
{
  TestEqual(thrust::hpx::seq);
}

TEST_CASE("TestEqualSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestEqual(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestEqualSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestEqual(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestEqualSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestEqual(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestEqualUnseq", "[hpx]")
{
  TestEqual(thrust::hpx::unseq);
}

TEST_CASE("TestEqualUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestEqual(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestEqualUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestEqual(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestEqualUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestEqual(thrust::hpx::unseq.with(acs));
}

_CCCL_DIAG_POP
