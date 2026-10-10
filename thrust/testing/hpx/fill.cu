#include <thrust/fill.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestFill(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector ref{1, 1, 1, 1, 1};

  {
    Vector v{0, 1, 2, 3, 4};

    thrust::fill(exec, v.begin(), v.end(), (T) 1);

    REQUIRE(v == ref);
  }
  {
    Vector v{0, 1, 2, 3, 4};

    thrust::fill_n(exec, v.begin(), v.size(), (T) 1);

    REQUIRE(v == ref);
  }
}

TEST_CASE("TestFillPar", "[hpx]")
{
  TestFill(thrust::hpx::par);
}

TEST_CASE("TestFillParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestFill(thrust::hpx::par.on(exec));
}

TEST_CASE("TestFillParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestFill(thrust::hpx::par.on(exec));
}

TEST_CASE("TestFillParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFill(thrust::hpx::par.with(acs));
}

TEST_CASE("TestFillParUnseq", "[hpx]")
{
  TestFill(thrust::hpx::par_unseq);
}

TEST_CASE("TestFillParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestFill(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestFillParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestFill(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestFillParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFill(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestFillSeq", "[hpx]")
{
  TestFill(thrust::hpx::seq);
}

TEST_CASE("TestFillSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestFill(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestFillSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestFill(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestFillSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFill(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestFillUnseq", "[hpx]")
{
  TestFill(thrust::hpx::unseq);
}

TEST_CASE("TestFillUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestFill(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestFillUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestFill(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestFillUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestFill(thrust::hpx::unseq.with(acs));
}

_CCCL_DIAG_POP
