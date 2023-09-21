#!/usr/bin/env python3

import os
import sys
import cub.bench


class BaseRunner:
  def __init__(self):
    self.estimator = cub.bench.MedianCenterEstimator()

  def __call__(self, algname, ct_workload_space, rt_values):
    print("&&&& RUNNING bench")
    for ct_workload in ct_workload_space:
      bench = cub.bench.BaseBench(algname)
      if bench.build():
        results = bench.run(ct_workload, rt_values, self.estimator, False)
        for subbench in results:
          for point in results[subbench]:
            bench_name = "{}.{}-{}".format(bench.algorithm_name(), subbench, point)
            bench_name = "".join(c if c.isalnum() else "_" for c in bench_name)
            print("&&&& PERF {} {} -sec".format(bench_name, results[subbench][point]))
      else:
        print("&&&& FAILED bench")
        sys.exit(-1)
    print("&&&& PASSED bench")


def main():
  os.environ["CUDA_MODULE_LOADING"] = "EAGER"
  cub.bench.search(BaseRunner())


if __name__ == "__main__":
  main()
