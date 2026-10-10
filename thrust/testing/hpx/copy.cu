#include <thrust/copy.h>

#include "test_executor.h"
#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename ExecutionPolicy>
void TestCopy(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector ref{0, 1, 2, 3, 4};

  {
    Vector v{1, 1, 1, 1, 1};

    thrust::copy(exec, ref.cbegin(), ref.cend(), v.begin());

    REQUIRE(v == ref);
  }
  {
    Vector v{1, 1, 1, 1, 1};

    thrust::copy_n(exec, ref.cbegin(), ref.size(), v.begin());

    REQUIRE(v == ref);
  }
}

TEST_CASE("TestCopyPar", "[hpx]")
{
  TestCopy(thrust::hpx::par);
}

TEST_CASE("TestCopyParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestCopy(thrust::hpx::par.on(exec));
}

TEST_CASE("TestCopyParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestCopy(thrust::hpx::par.on(exec));
}

TEST_CASE("TestCopyParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopy(thrust::hpx::par.with(acs));
}

TEST_CASE("TestCopyParUnseq", "[hpx]")
{
  TestCopy(thrust::hpx::par_unseq);
}

TEST_CASE("TestCopyParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestCopy(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestCopyParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestCopy(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestCopyParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopy(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestCopySeq", "[hpx]")
{
  TestCopy(thrust::hpx::seq);
}

TEST_CASE("TestCopySeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestCopy(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestCopySeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestCopy(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestCopySeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopy(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestCopyUnseq", "[hpx]")
{
  TestCopy(thrust::hpx::unseq);
}

TEST_CASE("TestCopyUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestCopy(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestCopyUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestCopy(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestCopyUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestCopy(thrust::hpx::unseq.with(acs));
}

_CCCL_DIAG_POP
