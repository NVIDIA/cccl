#include <thrust/count.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestCount(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector data{1, 6, 1, 9, 2};

  {
    REQUIRE(thrust::count(exec, data.begin(), data.end(), 0) == 0);
    REQUIRE(thrust::count(exec, data.begin(), data.end(), 1) == 2);
    REQUIRE(thrust::count(exec, data.begin(), data.end(), 2) == 1);
  }
  {
    auto greater_than_five = [](const T& x) {
      return x > 5;
    };
    REQUIRE(thrust::count_if(exec, data.begin(), data.end(), greater_than_five) == 2);
  }
}

TEST_CASE("TestCountPar", "[hpx]")
{
  TestCount(thrust::hpx::par);
}

TEST_CASE("TestCountParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestCount(thrust::hpx::par.on(exec));
}

TEST_CASE("TestCountParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestCount(thrust::hpx::par.on(exec));
}

TEST_CASE("TestCountParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCount(thrust::hpx::par.with(acs));
}

TEST_CASE("TestCountParUnseq", "[hpx]")
{
  TestCount(thrust::hpx::par_unseq);
}

TEST_CASE("TestCountParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestCount(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestCountParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestCount(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestCountParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCount(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestCountSeq", "[hpx]")
{
  TestCount(thrust::hpx::seq);
}

TEST_CASE("TestCountSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestCount(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestCountSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestCount(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestCountSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCount(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestCountUnseq", "[hpx]")
{
  TestCount(thrust::hpx::unseq);
}

TEST_CASE("TestCountUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestCount(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestCountUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestCount(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestCountUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCount(thrust::hpx::unseq.with(acs));
}

_CCCL_DIAG_POP
