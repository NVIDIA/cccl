import re
import argparse
import numpy as np

from .bench import Bench, BaseBench
from .config import Config
from .storage import Storage
from .cmake import CMake


def list_axes(benchmarks, sub_space):
    print("### Axes")

    axes = {}

    for algname in benchmarks:
        bench = BaseBench(algname)
        for axis in bench.axes_values(sub_space, True) + bench.axes_values(sub_space, False):
            for point in axis:
                axis, value = point.split('=')
                if axis in axes:
                    axes[axis].add(value)
                else:
                    axes[axis] = {value}

    for axis in axes:
        print("  * `{}`".format(axis))

        for value in axes[axis]:
            print("    * `{}`".format(value))


def list_benches():
    print("### Benchmarks")

    config = Config()

    for algname in config.benchmarks:
        space_size = config.variant_space_size(algname)
        print("  * `{}`: {} variants: ".format(algname, space_size))

        for param_space in config.benchmarks[algname]:
            param_name = param_space.label
            param_rng = (param_space.low, param_space.high, param_space.step)
            print("    * `{}`: {}".format(param_name, param_rng))


def parse_sub_space(args):
    sub_space = {}
    for axis in args:
        name, value = axis.split('=')

        if '[' in value:
            value = value.replace('[', '').replace(']', '')
            values = value.split(',')
        else:
            values = [value]
        sub_space[name] = values

    return sub_space


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Runs benchmarks and stores results in a database.")
    parser.add_argument('-R', type=str, default='.*',
                        help="Regex for benchmarks selection.")
    parser.add_argument('-a', '--args', action='append',
                        type=str, help="Parameter in the format `Param=Value`.")
    parser.add_argument(
        '--list-axes', action=argparse.BooleanOptionalAction, help="Show available parameters.")
    parser.add_argument(
        '--list-benches', action=argparse.BooleanOptionalAction, help="Show available benchmarks.")
    return parser.parse_args()


def run_benches(benchmarks, workload_sub_space, regex, seeker):
    pattern = re.compile(regex)

    for algname in benchmarks:
        if pattern.match(algname):
            bench = BaseBench(algname)
            ct_workload_space = bench.ct_workload_space(workload_sub_space)
            rt_workload_space = bench.rt_workload_space(workload_sub_space)
            seeker(algname, ct_workload_space, rt_workload_space)


def search(seeker):
    args = parse_arguments()

    if not Storage().exists():
        CMake().clean()

    config = Config()
    print("ctk: ", config.ctk)
    print("cub: ", config.cub)

    workload_sub_space = {}

    if args.args:
        workload_sub_space = parse_sub_space(args.args)

    if args.list_axes:
        list_axes(config.benchmarks, workload_sub_space)
        return

    if args.list_benches:
        list_benches()
        return

    run_benches(config.benchmarks, workload_sub_space, args.R, seeker)


class MedianCenterEstimator:
    def __init__(self):
        pass

    def __call__(self, samples):
        if len(samples) == 0:
            return float("inf")

        return float(np.median(samples))


class BruteForceSeeker:
    def __init__(self, base_center_estimator, variant_center_estimator):
        self.base_center_estimator = base_center_estimator
        self.variant_center_estimator = variant_center_estimator

    def __call__(self, algname, ct_workload_space, rt_workload_space):
        variants = Config().variant_space(algname)

        for ct_workload in ct_workload_space:
            for variant in variants:
                bench = Bench(algname, variant, list(ct_workload))
                if bench.build():
                    score = bench.score(ct_workload, 
                                        rt_workload_space, 
                                        self.base_center_estimator, 
                                        self.variant_center_estimator)

                    print(bench.label(), score)
