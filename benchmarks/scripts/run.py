#!/usr/bin/env python3

import os
import sys
import math
import cccl.bench


def elapsed_time_looks_good(x):
  if isinstance(x, float):
    if math.isfinite(x):
      return True
  return False


def get_largest_problem_size(rt_values):
  # Small problem sizes do not utilize entire GPU.
  # Benchmarking small problem sizes in environments where we do not control
  # distributions comparison, e.g. CI, is not useful because of stability issues.
  elements = []
  for element in rt_values:
    if element.isdigit():
      elements.append(int(element))
  return [str(max(elements))]

def filter_runtime_workloads_for_ci(rt_values):
  for subbench in rt_values:
    for axis in rt_values[subbench]:
      if axis.startswith('Elements') and axis.endswith('[pow2]'):
        rt_values[subbench][axis] = get_largest_problem_size(rt_values[subbench][axis])

  return rt_values


class BaseRunner:
  def __init__(self):
    self.estimator = cccl.bench.MedianCenterEstimator()

  def __call__(self, algname, ct_workload_space, rt_values):
    failure_occured = False
    rt_values = filter_runtime_workloads_for_ci(rt_values)

    for ct_workload in ct_workload_space:
      bench = cccl.bench.BaseBench(algname)
      if bench.build(): # might throw
        results = bench.run(ct_workload, rt_values, self.estimator, False)
        for subbench in results:
          for point in results[subbench]:
            bench_name = "{}.{}-{}".format(bench.algorithm_name(), subbench, point)
            bench_name = bench_name.replace(' ', '___')
            bench_name = "".join(c if c.isalnum() else "_" for c in bench_name)
            elapsed_time = results[subbench][point]
            if elapsed_time_looks_good(elapsed_time):
              print("&&&& PERF {} {} -sec".format(bench_name, elapsed_time))
      else:
        failure_occured = True
        print("&&&& FAILED {}".format(algname))

    if failure_occured:
      sys.exit(1)


def main():
  print("&&&& RUNNING bench")
  os.environ["CUDA_MODULE_LOADING"] = "EAGER"
  cccl.bench.search(BaseRunner())
  print("&&&& PASSED bench")


if __name__ == "__main__":
  main()
