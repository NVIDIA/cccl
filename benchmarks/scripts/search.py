#!/usr/bin/env python3

import collections
import contextlib
import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile

# compileiq defaults to forkserver, which pickles the objective into a clean
# interpreter. Fork lets workers inherit the parent, which is what the pid-keyed
# lane claim below relies on. Set before compileiq.worker picks a start method.
os.environ.setdefault("CIQ_PROCESS_MODE", "fork")

import cccl.bench as bench  # noqa: E402
import compileiq.search_spaces.base as ss  # noqa: E402
from compileiq.ciq import Search  # noqa: E402
from compileiq.types import SearchConfiguration  # noqa: E402
from compileiq.worker import MultiProcessWorker  # noqa: E402

INVALID_SCORE = "*"

# Lanes per GPU: the pipelining depth that lets one evaluation build while
# another benchmarks on the same GPU. See search_multi_gpu.md.
LANES_PER_GPU = 4

TUNING_PRESET = "cub-tune"

SCRIPT_PATH = os.path.abspath(__file__)
PARENT_PID = os.getpid()

Lane = collections.namedtuple("Lane", ["gpu", "directory", "lock"])


def pool_cull_sizes(num_genes, num_objectives, variant_space_size, cull=0.75):
    if not (0.05 <= cull <= 0.95):
        raise ValueError("cull must be between 0.05 and 0.95, got {}".format(cull))
    min_pool_size = 128 if variant_space_size > 10000 else 32
    target = (2 * num_objectives) + 1
    poolsize = int(target / (1 - cull))
    poolsize = max(max(poolsize, min_pool_size), 2 * num_genes)
    poolsize = poolsize + (poolsize % 2)
    cullsize = int(poolsize * cull)
    cullsize = cullsize - (cullsize % 2)
    return poolsize, cullsize


def build_search_space(parameter_space):
    search_space = {}
    for search_range in parameter_space:
        search_space[search_range.label] = ss.range(
            start=search_range.low, end=search_range.high - 1, step=search_range.step
        )
    return search_space


def visible_gpus():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [gpu.strip() for gpu in visible.split(",") if gpu.strip()]

    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
    )
    return [line.strip() for line in listing.splitlines() if line.strip()]


def build_lanes(gpus, lanes_per_gpu):
    """Lane i runs on gpu i % len(gpus), keeping GPUs balanced for any lane count."""
    root = os.getcwd()
    lanes = []
    for index in range(len(gpus) * lanes_per_gpu):
        gpu = gpus[index % len(gpus)]
        gpu_dir = os.path.join(root, "build", "gpu{}".format(gpu))
        lanes.append(
            Lane(
                gpu=gpu,
                directory=os.path.join(gpu_dir, "lane{}".format(index // len(gpus))),
                lock=os.path.join(gpu_dir, ".lock"),
            )
        )
    return lanes


def configure_lanes(lanes, cmake_args):
    processes = []
    for lane in lanes:
        os.makedirs(lane.directory, exist_ok=True)
        cmd = [
            "cmake",
            "-S",
            ".",
            "-B",
            lane.directory,
            "--preset",
            TUNING_PRESET,
        ] + cmake_args
        print("configuring {}".format(lane.directory))
        # Configure against the lane's own GPU so CMAKE_CUDA_ARCHITECTURES=native
        # resolves per lane rather than against whichever GPU the driver holds.
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=lane.gpu)
        processes.append(
            (
                lane,
                subprocess.Popen(
                    cmd,
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ),
            )
        )

    for lane, process in processes:
        if process.wait() != 0:
            raise Exception("Unable to configure {}".format(lane.directory))


def separate_args(argv):
    """Split cmake -D options and the lane count out of the search arguments."""
    cmake_args = []
    search_args = []
    lanes_per_gpu = LANES_PER_GPU

    argv = list(argv)
    while argv:
        arg = argv.pop(0)
        if arg.startswith("-D"):
            cmake_args.append(arg)
        elif arg == "--lanes-per-gpu":
            lanes_per_gpu = int(argv.pop(0))
        else:
            search_args.append(arg)

    return cmake_args, search_args, lanes_per_gpu


_lane_index = None
_lane_pid = None


def claim_lane(lane_queue, lanes):
    """Bind this process to a lane once, for its whole life.

    compileiq spawns anonymous workers, so the id is minted here: the parent is
    always lane 0 and every other process draws one. Keying the cache on the pid
    is what makes a forked child notice it inherited the parent's lane.
    """
    global _lane_index, _lane_pid

    if _lane_pid != os.getpid():
        _lane_index = 0 if os.getpid() == PARENT_PID else lane_queue.get()
        _lane_pid = os.getpid()

    return lanes[_lane_index]


class LaneWorker(MultiProcessWorker):
    """Hands out lane ids to the process pool compileiq builds each generation."""

    lane_queue = None
    num_lanes = 1

    def run(self, **kwargs):
        while not LaneWorker.lane_queue.empty():
            LaneWorker.lane_queue.get()
        for index in range(1, LaneWorker.num_lanes):
            LaneWorker.lane_queue.put(index)

        return super().run(**kwargs)


@contextlib.contextmanager
def gpu_lock(path):
    """Serialize benchmarks on one GPU. Builds run outside of this."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def evaluate(request_path):
    """Score one variant. Runs in a fresh process, bound to one lane."""
    with open(request_path) as request_file:
        request = json.load(request_file)

    range_points = [
        bench.RangePoint(definition, label, value)
        for definition, label, value in request["variant"]
    ]
    variant = bench.Bench(
        request["algname"], bench.VariantPoint(range_points), request["ct_workload"]
    )

    # The base binary answers the device query behind every cache lookup, and is
    # a no-op once this lane has built it.
    if not variant.get_base().build():
        print("SCORE {}".format(INVALID_SCORE))
        return

    estimator = bench.MedianCenterEstimator()
    ct_workload = request["ct_workload"]
    rt_workload_space = request["rt_workload_space"]

    # Ask for the score first: a variant this lane already measured is served
    # from its database, so there is nothing to compile and nothing to run.
    if variant.is_cached(ct_workload, rt_workload_space):
        score = variant.score(ct_workload, rt_workload_space, estimator, estimator)
    else:
        if not variant.build():
            print("SCORE {}".format(INVALID_SCORE))
            return

        with gpu_lock(request["lock"]):
            score = variant.score(ct_workload, rt_workload_space, estimator, estimator)

    if score in (float("inf"), float("-inf")):
        score = INVALID_SCORE

    print("SCORE {}".format(score))


class LaneObjective:
    """Spawns one evaluation in the lane this worker owns."""

    def __init__(
        self,
        algname,
        ct_workload,
        rt_workload_space,
        parameter_space,
        lanes,
        lane_queue,
    ):
        self.algname = algname
        self.ct_workload = list(ct_workload)
        self.rt_workload_space = rt_workload_space
        self.parameter_space = parameter_space
        self.lanes = lanes
        self.lane_queue = lane_queue

    def request(self, lane, config):
        return {
            "algname": self.algname,
            "ct_workload": self.ct_workload,
            "rt_workload_space": self.rt_workload_space,
            "lock": lane.lock,
            "variant": [
                (
                    search_range.definition,
                    search_range.label,
                    int(config[search_range.label]),
                )
                for search_range in self.parameter_space
            ],
        }

    def __call__(self, config):
        lane = claim_lane(self.lane_queue, self.lanes)
        request = self.request(lane, config)

        handle, request_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(handle, "w") as request_file:
            json.dump(request, request_file)

        try:
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=lane.gpu)
            completed = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--evaluate", request_path],
                cwd=lane.directory,
                env=environment,
                capture_output=True,
                text=True,
            )
        finally:
            os.remove(request_path)

        score = INVALID_SCORE
        for line in completed.stdout.splitlines():
            if line.startswith("SCORE "):
                score = line.split(" ", 1)[1].strip()

        if not any(line.startswith("SCORE ") for line in completed.stdout.splitlines()):
            # The shim died before scoring; without this its traceback is lost and
            # every variant just looks invalid.
            sys.stderr.write(
                "evaluation failed in {}:\n{}".format(lane.directory, completed.stderr)
            )

        if score != INVALID_SCORE:
            score = float(score)

        # Flush: workers are forked, so buffered output would be inherited and
        # re-emitted by children.
        print("gpu {} {}: {}".format(lane.gpu, config, score), flush=True)
        return score


class CompileIQSeeker:
    def __init__(self, lanes, lane_queue):
        self.lanes = lanes
        self.lane_queue = lane_queue

    def __call__(self, algname, ct_workload_space, rt_workload_space):
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
            objective = LaneObjective(
                algname,
                ct_workload,
                rt_workload_space,
                parameter_space,
                self.lanes,
                self.lane_queue,
            )

            tuner = Search(
                objective_function=objective,
                search_space=search_space,
                search_config=search_config,
                worker_type=LaneWorker,
            )

            results = tuner.start(num_workers=len(self.lanes))
            best = results.get_best_result()
            print("Best for {} {}: {}".format(algname, ct_workload, best))


def main():
    if "--evaluate" in sys.argv:
        evaluate(sys.argv[sys.argv.index("--evaluate") + 1])
        return

    cmake_args, search_args, lanes_per_gpu = separate_args(sys.argv)
    sys.argv = search_args

    gpus = visible_gpus()
    lanes = build_lanes(gpus, lanes_per_gpu)
    print("{} gpus x {} lanes".format(len(gpus), lanes_per_gpu))

    configure_lanes(lanes, cmake_args)
    os.chdir(lanes[0].directory)

    LaneWorker.lane_queue = multiprocessing.Manager().Queue()
    LaneWorker.num_lanes = len(lanes)

    bench.search(CompileIQSeeker(lanes, LaneWorker.lane_queue))


if __name__ == "__main__":
    main()
