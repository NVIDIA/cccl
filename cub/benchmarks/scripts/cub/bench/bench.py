import os
import json
import time
import fpzip
import signal
import itertools
import subprocess
import numpy as np

from .cmake import CMake
from .config import *
from .storage import Storage
from .score import *
from .logger import *


class JsonCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.bench_cache = {}
            cls._instance.device_cache = {}
        return cls._instance

    def get_bench(self, algname):
        if algname not in self.bench_cache:
            result = subprocess.check_output(
                [os.path.join('.', 'bin', algname + '.base'), "--jsonlist-benches"])
            self.bench_cache[algname] = json.loads(result)
        return self.bench_cache[algname]

    def get_device(self, algname):
        if algname not in self.device_cache:
            result = subprocess.check_output(
                [os.path.join('.', 'bin', algname + '.base'), "--jsonlist-devices"])
            devices = json.loads(result)["devices"]

            if len(devices) != 1:
                raise Exception(
                    "NVBench doesn't work well with multiple GPUs, use `CUDA_VISIBLE_DEVICES`")

            self.device_cache[algname] = devices[0]

        return self.device_cache[algname]


def json_benches(algname):
    return JsonCache().get_bench(algname)


def create_benches_tables(conn, bench_axes):
    for algorithm_name in bench_axes:
        axes = bench_axes[algorithm_name]
        columns = ", ".join(["\"{}\" TEXT".format(name) for name in axes])

        if axes:
            columns = ", " + columns

        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS "{0}" (
                ctk TEXT NOT NULL,
                cub TEXT NOT NULL,
                gpu TEXT NOT NULL,
                variant TEXT NOT NULL,
                elapsed REAL,
                center REAL,
                samples BLOB
                {1}
            );
            """.format(algorithm_name, columns))


def read_json(filename):
    with open(filename, "r") as f:
        file_root = json.load(f)
    return file_root


def extract_filename(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "filename", summary_data))
    assert (value_data["type"] == "string")
    return value_data["value"]


def extract_size(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "size", summary_data))
    assert (value_data["type"] == "int64")
    return int(value_data["value"])


def parse_samples_meta(state):
    summaries = state["summaries"]
    if not summaries:
        return None, None

    summary = next(filter(lambda s: s["tag"] == "nv/json/bin:nv/cold/sample_times",
                          summaries),
                   None)
    if not summary:
        return None, None

    sample_filename = extract_filename(summary)
    sample_count = extract_size(summary)
    return sample_count, sample_filename


def parse_samples(state):
    sample_count, samples_filename = parse_samples_meta(state)
    if not sample_count or not samples_filename:
        return np.array([], dtype=np.float32)

    with open(samples_filename, "rb") as f:
        samples = np.fromfile(f, "<f4")

    samples.sort()

    assert (sample_count == len(samples))
    return samples


def read_samples(json_path):
    result = {}

    try:
        meta = read_json(json_path)
        benches = meta["benchmarks"]

        if len(benches) != 1:
            raise Exception("Executable should contain exactly one benchmark")

        for bench in benches:
            bench_name = bench["name"]
            result[bench_name] = {}
            states = bench["states"]

            if len(states) != 1:
                return np.array([], dtype=np.float32)

            for state in states:
                return parse_samples(state)

    except Exception:
        print("??")
        pass

    return np.array([], dtype=np.float32)


def device_json(algname):
    return JsonCache().get_device(algname)


def get_device_name(device):
    gpu_name = device["name"]
    bw = device["global_memory_bus_width"]
    sms = device["number_of_sms"]
    ecc = "eccon" if device["ecc_state"] else "eccoff"
    name = "{} ({}, {}, {})".format(gpu_name, bw, sms, ecc)
    return name.replace('NVIDIA ', '')


class BenchCache:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.existing_tables = set()

        return cls._instance
    

    def create_table_if_not_exists(self, conn, bench):
        bench_base = bench.get_base()
        if bench_base.algorithm_name() not in self.existing_tables:
            create_benches_tables(conn, {bench_base.algorithm_name(): bench_base.axes_names()})
            self.existing_tables.add(bench_base.algorithm_name())


    def push_bench(self, bench, workload_point, elapsed, distribution_samples, distribution_center):
        config = Config()
        ctk = config.ctk
        cub = config.cub
        gpu = get_device_name(device_json(bench.algname))
        conn = Storage().connection()

        self.create_table_if_not_exists(conn, bench)

        columns = ""
        placeholders = ""
        values = []

        if workload_point:
            for axis in workload_point:
                name, value = axis.split('=')
                columns = columns + ", \"{}\"".format(name)
                placeholders = placeholders + ", ?"
                values.append(value)

        values = tuple(values)
        samples = fpzip.compress(distribution_samples)
        to_insert = (ctk, cub, gpu, bench.variant_name(),
                     elapsed, distribution_center, samples) + values

        with conn:
            query = """
            INSERT INTO "{0}" (ctk, cub, gpu, variant, elapsed, center, samples {1})
            VALUES (?, ?, ?, ?, ?, ?, ? {2})
            """.format(bench.algorithm_name(), columns, placeholders)

            conn.execute(query, to_insert)

    def pull_column(self, column, bench, workload_point):
        config = Config()
        ctk = config.ctk
        cub = config.cub
        gpu = get_device_name(device_json(bench.algname))
        conn = Storage().connection()

        self.create_table_if_not_exists(conn, bench)

        with conn:
            point_checks = ""
            if workload_point:
                for axis in workload_point:
                    name, value = axis.split('=')
                    point_checks = point_checks + \
                        " AND \"{}\" = \"{}\"".format(name, value)

            query = """
            SELECT {0} FROM "{1}" WHERE ctk = ? AND cub = ? AND gpu = ? AND variant = ?{2};
            """.format(column, bench.algorithm_name(), point_checks)

            return conn.execute(query, (ctk, cub, gpu, bench.variant_name())).fetchone()

    def pull_center(self, bench, workload_point):
        return self.pull_column('center', bench, workload_point)

    def pull_elapsed(self, bench, workload_point):
        return self.pull_column('elapsed', bench, workload_point)


class Bench:
    def __init__(self, algorithm_name, variant, ct_workload):
        self.algname = algorithm_name
        self.variant = variant
        self.ct_workload = ct_workload

    def label(self):
        return self.algname + '.' + self.variant.label()

    def variant_name(self):
        return self.variant.label()

    def algorithm_name(self):
        return self.algname

    def is_base(self):
        return self.variant.is_base()

    def get_base(self):
        return BaseBench(self.algorithm_name())
    
    def exe_name(self):
        if self.is_base():
            return self.algorithm_name() + '.base'
        return self.algorithm_name() + '.variant'

    def axes_names(self):
        result = json_benches(self.algname)

        if len(result["benchmarks"]) != 1:
            raise Exception("Executable should contain exactly one benchmark")

        names = []
        for axis in result["benchmarks"][0]["axes"]:
            name = axis["name"]
            if axis["flags"]:
                name = name + "[{}]".format(axis["flags"])
            names.append(name)

        return names

    def axes_values(self, sub_space, ct):
        result = json_benches(self.algname)

        if len(result["benchmarks"]) != 1:
            raise Exception("Executable should contain exactly one benchmark")

        space = []
        for axis in result["benchmarks"][0]["axes"]:
            name = axis["name"]

            if ct:
                if not '{ct}' in name:
                    continue
            else:
                if '{ct}' in name:
                    continue

            if axis["flags"]:
                name = name + "[{}]".format(axis["flags"])

            axis_space = []
            if name in sub_space:
                for value in sub_space[name]:
                    axis_space.append("{}={}".format(name, value))
            else:
                for value in axis["values"]:
                    axis_space.append("{}={}".format(name, value["input_string"]))

            space.append(axis_space)

        return space
    
    def axes_value_descriptions(self):
        result = json_benches(self.algname)

        if len(result["benchmarks"]) != 1:
            raise Exception("Executable should contain exactly one benchmark")

        descriptions = {}
        for axis in result["benchmarks"][0]["axes"]:
            name = axis["name"]
            if axis["flags"]:
                name = name + "[{}]".format(axis["flags"])
            descriptions[name] = {}
            for value in axis["values"]:
                descriptions[name][value["input_string"]] = value["description"]

        return descriptions

    
    def axis_values(self, axis_name):
        result = json_benches(self.algname)

        if len(result["benchmarks"]) != 1:
            raise Exception("Executable should contain exactly one benchmark")

        for axis in result["benchmarks"][0]["axes"]:
            name = axis["name"]

            if axis["flags"]:
                name = name + "[{}]".format(axis["flags"])

            if name != axis_name:
                continue

            values = []
            for value in axis["values"]:
                values.append(value["input_string"])

            return values

        return []

    def build(self):
        if not self.is_base():
            self.get_base().build()
        build = CMake().build(self)
        return build.code == 0
    
    def definitions(self):
        definitions = self.variant.tuning()
        definitions = definitions + "\n"

        descriptions = self.axes_value_descriptions()
        for ct_component in self.ct_workload:
            ct_axis_name, ct_value = ct_component.split('=')
            description = descriptions[ct_axis_name][ct_value]
            ct_axis_name = ct_axis_name.replace('{ct}', '')
            definitions = definitions + "#define TUNE_{} {}\n".format(ct_axis_name, description)

        return definitions

    def do_run(self, point, timeout):
        logger = Logger()

        try:
            result_path = 'result.json'
            if os.path.exists(result_path):
                os.remove(result_path)

            bench_path = os.path.join('.', 'bin', self.exe_name())
            cmd = [bench_path]

            for value in point:
                cmd.append('-a')
                cmd.append(value)

            cmd.append('--jsonbin')
            cmd.append(result_path)

            # Allow noise because we rely on min samples
            cmd.append("--max-noise")
            cmd.append("100")

            # Need at least 70 samples
            cmd.append("--min-samples")
            cmd.append("70")

            # NVBench is currently broken for multiple GPUs, use `CUDA_VISIBLE_DEVICES`
            cmd.append("-d")
            cmd.append("0")

            logger.info("starting benchmark {} with {}: {}".format(self.label(), point, " ".join(cmd)))

            begin = time.time()
            p = subprocess.Popen(cmd,
                                 start_new_session=True,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            p.wait(timeout=timeout)
            elapsed = time.time() - begin

            logger.info("finished benchmark {} with {} ({}) in {}s".format(self.label(), point, p.returncode, elapsed))

            return read_samples(result_path), elapsed
        except subprocess.TimeoutExpired:
            logger.info("benchmark {} with {} reached timeout of {}s".format(self.label(), point, timeout))
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            return np.array([], dtype=np.float32), float('inf')

    def ct_workload_space(self, sub_space):
        if not self.build():
            raise Exception("Unable to build benchmark: " + self.label())

        return list(itertools.product(*self.axes_values(sub_space, True)))

    def rt_workload_space(self, sub_space):
        if not self.build():
            raise Exception("Unable to build benchmark: " + self)

        return list(itertools.product(*self.axes_values(sub_space, False)))
    
    def elapsed(self, workload_point, estimator):
        self.run(workload_point, estimator)

        cache = BenchCache()
        cached_elapsed = cache.pull_elapsed(self, workload_point)

        if cached_elapsed:
            return float(cached_elapsed[0])

        return float('inf')

    def run(self, workload_point, estimator):
        logger = Logger()

        cache = BenchCache()
        cached_center = cache.pull_center(self, workload_point)

        if cached_center:
            logger.info("found benchmark {} ({}) in cache".format(self.label(), workload_point))
            return float(cached_center[0])

        timeout = None

        if not self.is_base():
            base = self.get_base()
            base_elapsed = base.elapsed(workload_point, estimator)
            timeout = base_elapsed * 50

        distribution_samples, elapsed = self.do_run(workload_point, timeout)
        distribution_center = estimator(distribution_samples)
        cache.push_bench(self, workload_point, elapsed,
                         distribution_samples, distribution_center)
        return distribution_center
    
    def speedup(self, workload_point, base_estimator, variant_estimator):
        if self.is_base():
            return 1.0

        base = self.get_base()
        base_center = base.run(workload_point, base_estimator)
        self_center = self.run(workload_point, variant_estimator)

        return base_center / self_center
    
    def score(self, ct_workload, rt_workload_space, base_estimator, variant_estimator):
        if self.is_base():
            return 1.0

        # Score should not be cached in the database because the number of values
        # on a given axis can change between runs, which would affect the score.

        rt_axes_values = {}
        for rt_workload in rt_workload_space:
            for pair in rt_workload:
                name, value = pair.split('=')

                if not name in rt_axes_values:
                    rt_axes_values[name] = set()
                rt_axes_values[name].add(value)

        for rt_axis in rt_axes_values:
            filtered_values = rt_axes_values[rt_axis]
            rt_axes_values[rt_axis] = []
            for value in self.axis_values(rt_axis):
                if value in filtered_values:
                    rt_axes_values[rt_axis].append(value)

        rt_axes_ids = compute_axes_ids(rt_axes_values)
        weight_matrix = compute_weight_matrix(rt_axes_values, rt_axes_ids)

        score = 0
        for rt_workload in rt_workload_space:
            weight = get_workload_weight(rt_workload, rt_axes_values, rt_axes_ids, weight_matrix)
            speedup = self.speedup(ct_workload + rt_workload, base_estimator, variant_estimator)
            score = score + weight * speedup

        return score


class BaseBench(Bench):
    def __init__(self, algname):
        super().__init__(algname, BasePoint(), [])
