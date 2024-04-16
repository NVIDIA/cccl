import re
import argparse
import numpy as np

from .bench import Bench, BaseBench
from .config import Config
from .storage import Storage
from .cmake import CMake


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
        '--list-benches', action=argparse.BooleanOptionalAction, help="Show available benchmarks.")
    parser.add_argument('--num-shards', type=int, default=1, help='Split benchmarks into M pieces and only run one')
    parser.add_argument('--run-shard', type=int, default=0, help='Run shard N / M of benchmarks')
    parser.add_argument('-P0', action=argparse.BooleanOptionalAction, help="Run P0 benchmarks")
    return parser.parse_args()


def run_benches(algnames, sub_space, seeker):
    for algname in algnames:
        try:
            bench = BaseBench(algname)
            ct_space = bench.ct_workload_space(sub_space)
            rt_values = bench.rt_axes_values(sub_space)
            seeker(algname, ct_space, rt_values)
        except Exception as e:
            print("#### ERROR exception occured while running {}: '{}'".format(algname, e))


def filter_benchmarks_by_regex(benchmarks, R):
    pattern = re.compile(R)
    return list(filter(lambda x: pattern.match(x), benchmarks))


def filter_benchmarks(benchmarks, args):
    if args.run_shard >= args.num_shards:
        raise ValueError('run-shard must be less than num-shards')

    algnames = filter_benchmarks_by_regex(benchmarks.keys(), args.R)
    if args.P0:
        algnames = filter_benchmarks_by_regex(algnames, '^(?!.*segmented).*(scan|reduce|select|sort).*')
    algnames.sort()

    if args.num_shards > 1:
        algnames = np.array_split(algnames, args.num_shards)[args.run_shard].tolist()
        return algnames

    return algnames


def search(seeker):
    args = parse_arguments()

    if not Storage().exists():
        CMake().clean()

    config = Config()
    print(" ctk: ", config.ctk)
    print("cccl: ", config.cccl)

    workload_sub_space = {}

    if args.args:
        workload_sub_space = parse_sub_space(args.args)

    if args.list_benches:
        list_benches()
        return

    run_benches(filter_benchmarks(config.benchmarks, args), workload_sub_space, seeker)


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

    def __call__(self, algname, ct_workload_space, rt_values):
        variants = Config().variant_space(algname)

        for ct_workload in ct_workload_space:
            for variant in variants:
                bench = Bench(algname, variant, list(ct_workload))
                if bench.build():
                    score = bench.score(ct_workload,
                                        rt_values,
                                        self.base_center_estimator,
                                        self.variant_center_estimator)

                    print(bench.label(), score)
