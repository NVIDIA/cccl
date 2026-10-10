#include <thrust/adjacent_difference.h>

#include "test_executor.h"
#include <unittest/unittest.h>

template <typename ExecutionPolicy>
void TestAdjacentDifference(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  {
    Vector input{1, 4, 6, 13, 20};
    Vector output(5);
    Vector ref{1, 3, 2, 7, 7};

    auto result = thrust::adjacent_difference(exec, input.begin(), input.end(), output.begin());

    REQUIRE(output == ref);
    REQUIRE(result == output.end());
  }
  {
    Vector input{1, 4, 6, 13, 20};
    Vector output(5);
    Vector ref{1, 5, 10, 19, 33};

    auto result = thrust::adjacent_difference(exec, input.begin(), input.end(), output.begin(), thrust::plus<T>());

    REQUIRE(output == ref);
    REQUIRE(result == output.end());
  }
}

TEST_CASE("TestAdjacentDifferencePar", "[hpx]")
{
  TestAdjacentDifference(thrust::hpx::par);
}

TEST_CASE("TestAdjacentDifferenceParOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestAdjacentDifference(thrust::hpx::par.on(exec));
}

TEST_CASE("TestAdjacentDifferenceParOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestAdjacentDifference(thrust::hpx::par.on(exec));
}

TEST_CASE("TestAdjacentDifferenceParWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestAdjacentDifference(thrust::hpx::par.with(acs));
}

TEST_CASE("TestAdjacentDifferenceParUnseq", "[hpx]")
{
  TestAdjacentDifference(thrust::hpx::par_unseq);
}

TEST_CASE("TestAdjacentDifferenceParUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestAdjacentDifference(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestAdjacentDifferenceParUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestAdjacentDifference(thrust::hpx::par_unseq.on(exec));
}

TEST_CASE("TestAdjacentDifferenceParUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestAdjacentDifference(thrust::hpx::par_unseq.with(acs));
}

TEST_CASE("TestAdjacentDifferenceSeq", "[hpx]")
{
  TestAdjacentDifference(thrust::hpx::seq);
}

TEST_CASE("TestAdjacentDifferenceSeqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestAdjacentDifference(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestAdjacentDifferenceSeqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestAdjacentDifference(thrust::hpx::seq.on(exec));
}

TEST_CASE("TestAdjacentDifferenceSeqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestAdjacentDifference(thrust::hpx::seq.with(acs));
}

TEST_CASE("TestAdjacentDifferenceUnseq", "[hpx]")
{
  TestAdjacentDifference(thrust::hpx::unseq);
}

TEST_CASE("TestAdjacentDifferenceUnseqOnSequencedExecutor", "[hpx]")
{
  hpx::execution::sequenced_executor exec{};

  TestAdjacentDifference(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestAdjacentDifferenceUnseqOnTestSyncExecutor", "[hpx]")
{
  test_sync_executor exec{};

  TestAdjacentDifference(thrust::hpx::unseq.on(exec));
}

TEST_CASE("TestAdjacentDifferenceUnseqWithAutoChunkSize", "[hpx]")
{
  hpx::execution::experimental::auto_chunk_size acs{};

  TestAdjacentDifference(thrust::hpx::unseq.with(acs));
}
