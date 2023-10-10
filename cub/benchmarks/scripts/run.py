#!/usr/bin/env python3

import os
import sys
import math
import cub.bench


def elapsed_time_look_good(x):
  if isinstance(x, float):
    if math.isfinite(x):
      return True
  return False


class BaseRunner:
  def __init__(self):
    self.estimator = cub.bench.MedianCenterEstimator()

  def __call__(self, algname, ct_workload_space, rt_values):
    for ct_workload in ct_workload_space:
      bench = cub.bench.BaseBench(algname)
      if bench.build():
        results = bench.run(ct_workload, rt_values, self.estimator, False)
        for subbench in results:
          for point in results[subbench]:
            bench_name = "{}.{}-{}".format(bench.algorithm_name(), subbench, point)
            bench_name = bench_name.replace(' ', '___')
            bench_name = "".join(c if c.isalnum() else "_" for c in bench_name)
            elapsed_time = results[subbench][point]
            if elapsed_time_look_good(elapsed_time):
              print("&&&& PERF {} {} -sec".format(bench_name, elapsed_time))
      else:
        print("&&&& FAILED bench")
        sys.exit(-1)


def main():
  print("&&&& RUNNING bench")
  os.environ["CUDA_MODULE_LOADING"] = "EAGER"
  cub.bench.search(BaseRunner())
  print("&&&& PASSED bench")


if __name__ == "__main__":
  main()
