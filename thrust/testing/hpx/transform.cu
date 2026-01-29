#include <thrust/transform.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestTransform(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  {
    Vector input{1, -2, 3};
    Vector output(3);
    Vector result{-1, 2, -3};

    typename Vector::iterator iter = thrust::transform(exec, input.begin(), input.end(), output.begin(), thrust::negate<T>());

    ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
    ASSERT_EQUAL(output, result);
  }
  {
    Vector input1{1, -2, 3};
    Vector input2{-4, 5, 6};
    Vector output(3);
    Vector result{5, -7, -3};

    typename Vector::iterator iter =
      thrust::transform(exec, input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>());

    ASSERT_EQUAL(std::size_t(iter - output.begin()), input1.size());
    ASSERT_EQUAL(output, result);
  }
}

void TestTransformPar()
{
  TestTransform(thrust::hpx::par);
}
DECLARE_UNITTEST(TestTransformPar);

void TestTransformParOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTransform(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestTransformParOnSequencedExecutor);

void TestTransformParOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTransform(thrust::hpx::par.on(exec));
}
DECLARE_UNITTEST(TestTransformParOnTestSyncExecutor);

void TestTransformParWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransform(thrust::hpx::par.with(acs));
}
DECLARE_UNITTEST(TestTransformParWithAutoChunkSize);

void TestTransformParUnseq()
{
  TestTransform(thrust::hpx::par_unseq);
}
DECLARE_UNITTEST(TestTransformParUnseq);

void TestTransformParUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTransform(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestTransformParUnseqOnSequencedExecutor);

void TestTransformParUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTransform(thrust::hpx::par_unseq.on(exec));
}
DECLARE_UNITTEST(TestTransformParUnseqOnTestSyncExecutor);

void TestTransformParUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransform(thrust::hpx::par_unseq.with(acs));
}
DECLARE_UNITTEST(TestTransformParUnseqWithAutoChunkSize);

void TestTransformSeq()
{
  TestTransform(thrust::hpx::seq);
}
DECLARE_UNITTEST(TestTransformSeq);

void TestTransformSeqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTransform(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestTransformSeqOnSequencedExecutor);

void TestTransformSeqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTransform(thrust::hpx::seq.on(exec));
}
DECLARE_UNITTEST(TestTransformSeqOnTestSyncExecutor);

void TestTransformSeqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransform(thrust::hpx::seq.with(acs));
}
DECLARE_UNITTEST(TestTransformSeqWithAutoChunkSize);

void TestTransformUnseq()
{
  TestTransform(thrust::hpx::unseq);
}
DECLARE_UNITTEST(TestTransformUnseq);

void TestTransformUnseqOnSequencedExecutor()
{
  hpx::execution::sequenced_executor exec{};

  TestTransform(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestTransformUnseqOnSequencedExecutor);

void TestTransformUnseqOnTestSyncExecutor()
{
  test_sync_executor exec{};

  TestTransform(thrust::hpx::unseq.on(exec));
}
DECLARE_UNITTEST(TestTransformUnseqOnTestSyncExecutor);

void TestTransformUnseqWithAutoChunkSize()
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransform(thrust::hpx::unseq.with(acs));
}
DECLARE_UNITTEST(TestTransformUnseqWithAutoChunkSize);
