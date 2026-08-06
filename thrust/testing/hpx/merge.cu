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

  ASSERT_EQUAL_QUIET(result.end(), end);
  ASSERT_EQUAL(ref, result);
}

void TestMergePar()
{
  TestMerge(thrust::hpx::par);
}
DECLARE_UNITTEST(TestMergePar);

void TestMergeParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestMerge(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestMergeParOnSequencedExecutor);

void TestMergeParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestMerge(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestMergeParOnTestSyncExecutor);

void TestMergeParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMerge(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestMergeParWithAutoChunkSize);

void TestMergeParUnseq()
{
  TestMerge(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestMergeParUnseq);

void TestMergeParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestMerge(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestMergeParUnseqOnSequencedExecutor);

void TestMergeParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestMerge(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestMergeParUnseqOnTestSyncExecutor);

void TestMergeParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMerge(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestMergeParUnseqWithAutoChunkSize);

void TestMergeSeq()
{
  TestMerge(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestMergeSeq);

void TestMergeSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestMerge(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestMergeSeqOnSequencedExecutor);

void TestMergeSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestMerge(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestMergeSeqOnTestSyncExecutor);

void TestMergeSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMerge(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestMergeSeqWithAutoChunkSize);

void TestMergeUnseq()
{
  TestMerge(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestMergeUnseq);

void TestMergeUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestMerge(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestMergeUnseqOnSequencedExecutor);

void TestMergeUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestMerge(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestMergeUnseqOnTestSyncExecutor);

void TestMergeUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestMerge(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestMergeUnseqWithAutoChunkSize);
