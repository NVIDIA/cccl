//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/internal/dot.cuh>
#include <cuda/experimental/stf.cuh>

#include <cstdlib> // For rand() and srand()
#include <numeric> // accumulate

using namespace cuda::experimental::stf;

enum test_id
{
  TRIVIAL = 0,
  STENCIL = 1,
  FFT     = 2,
  SWEEP   = 3,
  TREE    = 4,
  RANDOM  = 5,
};

std::string test_name(test_id id)
{
  switch (id)
  {
    case TRIVIAL:
      return "TRIVIAL";
    case STENCIL:
      return "STENCIL";
    case FFT:
      return "FFT";
    case SWEEP:
      return "SWEEP";
    case TREE:
      return "TREE";
    case RANDOM:
      return "RANDOM";
    default:
      return "unknown";
  }
}

int log2Int(int n)
{
  int result = 0;
  while (n >>= 1)
  { // Divide n by 2 until n becomes 0
    ++result;
  }
  return result;
}

#if defined(_CCCL_COMPILER_MSVC)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code
#endif // _CCCL_COMPILER_MSVC
bool skip_task(test_id id, int t, int i, int /*W*/)
{
  switch (id)
  {
    case TRIVIAL:
    case STENCIL:
    case FFT:
    case RANDOM:
      return false;
    case SWEEP:
      // return (i <= t) && (t - i < W);
      return (i <= t);
    case TREE: {
      if (t == 0)
      {
        return false;
      }
      int stride = 1 << (t);
      return (i % stride != 0);
    }
    default:
      abort();
  }

  // should not be reached
  abort();
  return true;
}
#if defined(_CCCL_COMPILER_MSVC)
_CCCL_DIAG_POP
#endif // _CCCL_COMPILER_MSVC

std::vector<int> input_deps(test_id id, int t, int i, int W)
{
  std::vector<int> res;

  // no input deps for the first step
  if (t == 0)
  {
    return res;
  }

  switch (id)
  {
    case TRIVIAL:
      // D(t,i) = NIL
      break;
    case STENCIL:
      // D(t, i) = {i, i-1, i+1}
      res.push_back(i);
      if (i > 0)
      {
        res.push_back(i - 1);
      }
      if (i < W - 1)
      {
        res.push_back(i + 1);
      }
      break;
    case FFT:
      // D(t,i) = {i, i - 2^t, i+2^t}
      res.push_back(i);
      {
        if (t < 32)
        {
          int two_t1 = 1 << (t - 1);
          if (i - two_t1 >= 0)
          {
            res.push_back(i - two_t1);
          }
          if (i + two_t1 < W)
          {
            res.push_back(i + two_t1);
          }
        }
      }

      break;
    case SWEEP:
      // D(t,i) = (i, i-1)
      res.push_back(i);
      if (i > 0)
      {
        res.push_back(i - 1);
      }
      break;
    case TREE:
      // D(t,i) = (t <= log2(W)) {i - 2^(-t)W(i mod 2^(-t+1)W)} else {i, i + 2^(t-1)*W^-1}
      {
        int stride = 1 << (t - 1);
        res.push_back(i);
        if (i + stride < W)
        {
          res.push_back(i + stride);
        }
      }
      break;
    case RANDOM:
      // Differs from TaskBench ( D(t,i) = {i | 0 <= i < W && random() < 0.5))
      // TaskBench topology assumes there can be arbitrarily large numbers of deps
      // for (int j = 0; j < W; j++) {
      //     double r = static_cast<double>(rand()) / RAND_MAX;
      //     if (r < 0.5)
      //         res.push_back(j);
      // }
      for (int k = 0; k < 2; k++)
      {
        res.push_back(rand() % W);
      }
      break;
    default:
      abort();
  }

  return res;
}

void bench(context& ctx, test_id id, size_t width, size_t nsteps, size_t repeat_cnt)
{
  std::chrono::steady_clock::time_point start, stop;

  const size_t b = nsteps;

  std::vector<logical_data<slice<int>>> data(width * b);
  const size_t data_size = 128;

  /*
   * Note :  CUDASTF_DOT_REMOVE_DATA_DEPS=1 CUDASTF_DOT_NO_FENCE=1 ACUDASTF_DOT_DISPLAY_EPOCHS=1
   * CUDASTF_DOT_FILE=pif.dot build/nvcc/tests/stress/task_bench 8 6 1 3; dot -Tpdf pif.dot -o pif.pdf
   */
  ctx.get_dot()->set_tracing(false);

  for (size_t t = 0; t < b; t++)
  {
    for (size_t i = 0; i < width; i++)
    {
      auto d = ctx.logical_data<int>(data_size);
      ctx.task(d.write())->*[](cudaStream_t, auto) {};
      data[t * width + i] = d;
    }
  }
  ctx.get_dot()->set_tracing(true);

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));
  ctx.change_epoch(); // for better DOT rendering

  std::vector<double> tv;
  const int niter = 10;
  size_t task_cnt;
  size_t deps_cnt;

  for (size_t iter = 0; iter < niter; iter++)
  {
    task_cnt = 0;
    deps_cnt = 0;
    start    = std::chrono::steady_clock::now();

    for (size_t k = 0; k < repeat_cnt; k++)
    {
      for (size_t t = 0; t < nsteps; t++)
      {
        for (size_t i = 0; i < width; i++)
        {
          if (!skip_task(id, t, i, width))
          {
            auto tsk = ctx.task();
            tsk.add_deps(data[(t % b) * width + i].rw());

            auto deps = input_deps(id, t, i, width);
            for (int d : deps)
            {
              tsk.add_deps(data[((t - 1 + b) % b) * width + d].read());
              deps_cnt++;
            }

            tsk.set_symbol(std::to_string(t) + "," + std::to_string(i));
            tsk->*[](cudaStream_t) {};
            task_cnt++;
          }
        }
      }
    }

    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));
    ctx.change_epoch(); // for better DOT rendering
    stop = std::chrono::steady_clock::now();

    std::chrono::duration<double> duration = stop - start;
    tv.push_back(duration.count() * 1000000.0 / (task_cnt));
  }

  // Compute the mean (average)
  double sum  = ::std::accumulate(tv.begin(), tv.end(), 0.0);
  double mean = sum / tv.size();

  // Compute the standard deviation
  double sq_sum            = ::std::accumulate(tv.begin(), tv.end(), 0.0, [mean](double acc, double val) {
    return acc + std::pow(val - mean, 2);
  });
  double variance          = sq_sum / tv.size();
  double standardDeviation = ::std::sqrt(variance);

  fprintf(stderr,
          "[%s] Elapsed: %.3lf+-%.4lf us per task (%zu tasks, %zu deps, %lf deps/task (avg)\n",
          test_name(id).c_str(),
          mean,
          standardDeviation,
          task_cnt,
          deps_cnt,
          (1.0 * deps_cnt) / task_cnt);
}

int main(int argc, char** argv)
{
  context ctx;

  size_t width = 8;
  if (argc > 1)
  {
    width = atol(argv[1]);
  }

  size_t nsteps = width;
  if (argc > 2)
  {
    nsteps = atol(argv[2]);
  }

  size_t repeat_cnt = 10;
  if (argc > 3)
  {
    repeat_cnt = atol(argv[3]);
  }

  int id = -1; // all
  if (argc > 4)
  {
    id = atoi(argv[4]);
  }

  if (id == -1)
  {
    bench(ctx, TRIVIAL, width, nsteps, repeat_cnt);
    bench(ctx, STENCIL, width, nsteps, repeat_cnt);
    bench(ctx, FFT, width, nsteps, repeat_cnt);
    bench(ctx, SWEEP, width, nsteps, repeat_cnt);
    bench(ctx, TREE, width, nsteps, repeat_cnt);
    bench(ctx, RANDOM, width, nsteps, repeat_cnt);
  }
  else
  {
    bench(ctx, test_id(id), width, nsteps, repeat_cnt);
  }

  ctx.finalize();
}
