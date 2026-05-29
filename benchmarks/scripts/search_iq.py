#!/usr/bin/env python3

import cccl.bench as bench
import compileiq.search_spaces.base as ss
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration


def pool_cull_sizes(num_genes, num_objectives, variant_space_size, cull=0.75):
    min_pool_size = 128 if variant_space_size > 10000 else 32
    target = (2 * num_objectives) + 1
    poolsize = int(target / (1 - cull))
    poolsize = max(max(poolsize, min_pool_size), 2 * num_genes)
    poolsize = poolsize if poolsize % 2 == 0 else poolsize + 1
    cullsize = int(poolsize * cull)
    cullsize = cullsize if cullsize % 2 == 0 else cullsize - 1
    return poolsize, cullsize


def get_num_expected_runs(num_genes, num_objectives, variant_space_size):
    poolsize, _ = pool_cull_sizes(num_genes, num_objectives, variant_space_size)
    return poolsize * 5


def build_search_space(parameter_space):
    search_space = {}
    for rng in parameter_space:
        search_space[rng.label] = ss.range(
            start=rng.low, end=rng.high - 1, step=rng.step
        )
    return search_space


INVALID_SCORE = "*"


def make_objective(algname, ct_workload, rt_workload_space, parameter_space):
    center_estimator = bench.MedianCenterEstimator()

    def objective(config):
        print(f"Evaluating variant {config}: ", end="")

        range_points = []
        for rng in parameter_space:
            value = int(config[rng.label])
            range_points.append(bench.RangePoint(rng.definition, rng.label, value))

        variant = bench.VariantPoint(range_points)
        b = bench.Bench(algname, variant, list(ct_workload))

        if not b.build():
            print("Build failed")
            return INVALID_SCORE

        score = b.score(
            ct_workload, rt_workload_space, center_estimator, center_estimator
        )

        if score == float("inf") or score == float("-inf"):
            print("Infinite store")
            return INVALID_SCORE

        print(score)
        return score

    return objective


class CompileIQSeeker:
    def __init__(self):
        self.base_center_estimator = bench.MedianCenterEstimator()
        self.variant_center_estimator = bench.MedianCenterEstimator()
        self.bruteforce_seeker = bench.BruteForceSeeker(
            self.base_center_estimator, self.variant_center_estimator
        )

    def iq_search(self, algname, ct_workload_space, rt_workload_space):
        config = bench.Config()
        parameter_space = config.benchmarks[algname]
        variant_space_size = config.variant_space_size(algname)

        num_genes = len(parameter_space)
        num_objectives = 1
        poolsize, cullsize = pool_cull_sizes(
            num_genes, num_objectives, variant_space_size
        )
        num_generations = 50 if variant_space_size > 10000 else 25

        search_space = build_search_space(parameter_space)

        search_config = SearchConfiguration(
            problem_type="max",
            num_objectives=num_objectives,
            generations=num_generations,
            pool_size=poolsize,
            cull_size=cullsize,
            mutate_rate=0.15,
            init_with_true_random_threshold=0.90,
        )

        for ct_workload in ct_workload_space:
            objective = make_objective(
                algname, ct_workload, rt_workload_space, parameter_space
            )

            tuner = Search(
                objective_function=objective,
                search_space=search_space,
                search_config=search_config,
            )

            results = tuner.start(num_workers=1)
            best = results.get_best_result()
            print("Best for {} {}: {}".format(algname, ct_workload, best))

    def bf_search(self, algname, ct_workload_space, rt_workload_space):
        self.bruteforce_seeker(algname, ct_workload_space, rt_workload_space)

    def __call__(self, algname, ct_workload_space, rt_workload_space):
        config = bench.Config()
        num_genes = len(config.benchmarks[algname])
        variant_space_size = config.variant_space_size(algname)
        num_rt_workloads = len(rt_workload_space)
        expected_runs = get_num_expected_runs(
            num_genes, num_rt_workloads, variant_space_size
        )

        if expected_runs < 128:
            self.bf_search(algname, ct_workload_space, rt_workload_space)
        else:
            self.iq_search(algname, ct_workload_space, rt_workload_space)


def main():
    bench.search(CompileIQSeeker())


if __name__ == "__main__":
    main()
