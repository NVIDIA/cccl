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

    typename Vector::iterator iter =
      thrust::transform(exec, input.begin(), input.end(), output.begin(), thrust::negate<T>());

    REQUIRE(std::size_t(iter - output.begin()) == input.size());
    REQUIRE(output == result);
  }
  {
    Vector input1{1, -2, 3};
    Vector input2{-4, 5, 6};
    Vector output(3);
    Vector result{5, -7, -3};

    typename Vector::iterator iter =
      thrust::transform(exec, input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>());

    REQUIRE(std::size_t(iter - output.begin()) == input1.size());
    REQUIRE(output == result);
  }
}

TEST_CASE("TestTransformPar", "[hpx]")
{
  TestTransform(thrust::hpx::par);
}

TEST_CASE("TestTransformParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestTransform(thrust::hpx::par.on(exec));
}

TEST_CASE("TestTransformParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestTransform(thrust::hpx::par.on(exec));
}

TEST_CASE("TestTransformParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransform(thrust::hpx::par.with(acs));
}

TEST_CASE("TestTransformParUnseq", "[hpx]")
{
  TestTransform(thrust::hpx::par_unseq);
}

TEST_CASE("TestTransformParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestTransform(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestTransformParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestTransform(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestTransformParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransform(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestTransformSeq", "[hpx]")
{
  TestTransform(thrust::hpx::seq);
}

TEST_CASE("TestTransformSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestTransform(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestTransformSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestTransform(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestTransformSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransform(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestTransformUnseq", "[hpx]")
{
  TestTransform(thrust::hpx::unseq);
}

TEST_CASE("TestTransformUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestTransform(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestTransformUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestTransform(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestTransformUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestTransform(thrust::hpx::unseq.with(acs));
}
