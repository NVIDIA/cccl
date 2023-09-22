#!/usr/bin/env python3

import cccl.bench as bench


# TODO:
# - driver version
# - host compiler + version
# - gpu clocks / pm
# - ecc


def main():
    center_estimator = bench.MedianCenterEstimator()
    bench.search(bench.BruteForceSeeker(center_estimator, center_estimator))


if __name__ == "__main__":
    main()
