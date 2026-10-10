#include <thrust/merge.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestMerge(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  const Vector a{0, 2, 4}, b{0, 3, 3, 4};
  const Vector ref{0, 0, 2, 3, 3, 4, 4};

  Vector result(7);
  const auto end = thrust::merge(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin());

  REQUIRE(result.end() == end);
  REQUIRE(ref == result);
}

TEST_CASE("TestMergePar", "[hpx]")
{
  TestMerge(thrust::hpx::par);
}

TEST_CASE("TestMergeParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestMerge(thrust::hpx::par.on(exec));
}

TEST_CASE("TestMergeParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestMerge(thrust::hpx::par.on(exec));
}

TEST_CASE("TestMergeParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMerge(thrust::hpx::par.with(acs));
}

TEST_CASE("TestMergeParUnseq", "[hpx]")
{
  TestMerge(thrust::hpx::par_unseq);
}

TEST_CASE("TestMergeParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestMerge(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestMergeParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestMerge(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestMergeParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMerge(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestMergeSeq", "[hpx]")
{
  TestMerge(thrust::hpx::seq);
}

TEST_CASE("TestMergeSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestMerge(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestMergeSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestMerge(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestMergeSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMerge(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestMergeUnseq", "[hpx]")
{
  TestMerge(thrust::hpx::unseq);
}

TEST_CASE("TestMergeUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestMerge(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestMergeUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestMerge(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestMergeUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMerge(thrust::hpx::unseq.with(acs));
}
