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
from .storage import Storage, get_bench_table_name
from .score import *
from .logger import *


def first_val(my_dict):
    values = list(my_dict.values())
    first_value = values[0]

    if not all(value == first_value for value in values):
        raise ValueError('All values in the dictionary are not equal. First value: {} All values: {}'.format(first_value, values))

    return first_value


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


def create_benches_tables(conn, subbench, bench_axes):
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS subbenches (
            algorithm TEXT NOT NULL,
            bench TEXT NOT NULL,
            UNIQUE(algorithm, bench)
        );
        """)

        for algorithm_name in bench_axes:
            axes = bench_axes[algorithm_name]
            column_names = ", ".join(["\"{}\"".format(name) for name in axes])
            columns = ", ".join(["\"{}\" TEXT".format(name) for name in axes])

            conn.execute("""
            INSERT INTO subbenches (algorithm, bench)
            VALUES (?, ?)
            ON CONFLICT DO NOTHING;
            """, (algorithm_name, subbench))

            if axes:
                columns = ", " + columns
                column_names = ", " + column_names
                conn.execute("""
                CREATE TABLE IF NOT EXISTS "{0}" (
                    ctk TEXT NOT NULL,
                    cccl TEXT NOT NULL,
                    gpu TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    elapsed REAL,
                    center REAL,
                    bw REAL,
                    samples BLOB
                    {1}
                    , UNIQUE(ctk, cccl, gpu, variant {2})
                );
                """.format(get_bench_table_name(subbench, algorithm_name), columns, column_names))


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


def extract_bw(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "value", summary_data))
    assert (value_data["type"] == "float64")
    return float(value_data["value"])


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


def parse_bw(state):
    bwutil = next(filter(lambda s: s["tag"] == "nv/cold/bw/global/utilization",
                         state['summaries']), None)
    if not bwutil:
        return None

    return extract_bw(bwutil)

class SubBenchState:
    def __init__(self, state, axes_names, axes_values):
        self.samples = parse_samples(state)
        self.bw = parse_bw(state)

        self.point = {}
        for axis in state["axis_values"]:
            name = axes_names[axis['name']]
            value = axes_values[axis['name']][axis['value']]
            self.point[name] = value

    def __repr__(self):
        return str(self.__dict__)

    def name(self):
        return ' '.join(f'{k}={v}' for k, v in self.point.items())

    def center(self, estimator):
        return estimator(self.samples)

class SubBenchResult:
    def __init__(self, bench):
        axes_names = {}
        axes_values = {}
        for axis in bench["axes"]:
            short_name = axis["name"]
            full_name = get_axis_name(axis)
            axes_names[short_name] = full_name
            axes_values[short_name] = {}
            for value in axis["values"]:
                if "value" in value:
                    axes_values[axis["name"]][str(value["value"])] = value["input_string"]
                else:
                    axes_values[axis["name"]][value["input_string"]] = value["input_string"]

        self.states = []
        for state in bench["states"]:
            if not state["is_skipped"]:
                self.states.append(SubBenchState(state, axes_names, axes_values))

    def __repr__(self):
        return str(self.__dict__)

    def centers(self, estimator):
        result = {}
        for state in self.states:
            result[state.name()] = state.center(estimator)
        return result


class BenchResult:
    def __init__(self, json_path, code, elapsed):
        self.code = code
        self.elapsed = elapsed

        if json_path:
            self.subbenches = {}
            if code == 0:
                for bench in read_json(json_path)["benchmarks"]:
                    self.subbenches[bench["name"]] = SubBenchResult(bench)

    def __repr__(self):
        return str(self.__dict__)

    def centers(self, estimator):
        result = {}
        for subbench in self.subbenches:
            result[subbench] = self.subbenches[subbench].centers(estimator)
        return result


def device_json(algname):
    return JsonCache().get_device(algname)


def get_device_name(device):
    gpu_name = device["name"]
    bw = device["global_memory_bus_width"]
    sms = device["number_of_sms"]
    ecc = "eccon" if device["ecc_state"] else "eccoff"
    name = "{} ({}, {}, {})".format(gpu_name, bw, sms, ecc)
    return name.replace('NVIDIA ', '')


def is_ct_axis(name):
    return '{ct}' in name


def state_to_rt_workload(bench, state):
    rt_workload = []
    for param in state.split(' '):
        name, value = param.split('=')
        if is_ct_axis(name):
            continue
        rt_workload.append("{}={}".format(name, value))
    return rt_workload


def create_runs_table(conn):
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            ctk TEXT NOT NULL,
            cccl TEXT NOT NULL,
            bench TEXT NOT NULL,
            code TEXT NOT NULL,
            elapsed REAL
        );
        """)


class RunsCache:
  _instance = None

  def __new__(cls, *args, **kwargs):
      if cls._instance is None:
          cls._instance = super().__new__(cls, *args, **kwargs)
          create_runs_table(Storage().connection())
      return cls._instance

  def pull_run(self, bench):
      config = Config()
      ctk = config.ctk
      cccl = config.cccl
      conn = Storage().connection()

      with conn:
          query = "SELECT code, elapsed FROM runs WHERE ctk = ? AND cccl = ? AND bench = ?;"
          result = conn.execute(query, (ctk, cccl, bench.label())).fetchone()

          if result:
              code, elapsed = result
              return int(code), float(elapsed)

          return result

  def push_run(self, bench, code, elapsed):
      config = Config()
      ctk = config.ctk
      cccl = config.cccl
      conn = Storage().connection()

      with conn:
          conn.execute("INSERT INTO runs (ctk, cccl, bench, code, elapsed) VALUES (?, ?, ?, ?, ?);",
                       (ctk, cccl, bench.label(), code, elapsed))


class BenchCache:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.existing_tables = set()

        return cls._instance


    def create_table_if_not_exists(self, conn, bench):
        bench_base = bench.get_base()
        alg_name = bench_base.algorithm_name()

        if alg_name not in self.existing_tables:
            subbench_axes_names = bench_base.axes_names()
            for subbench in subbench_axes_names:
                create_benches_tables(conn, subbench, {alg_name: subbench_axes_names[subbench]})
                self.existing_tables.add(alg_name)


    def push_bench_centers(self, bench, result, estimator):
        config = Config()
        ctk = config.ctk
        cccl = config.cccl
        gpu = get_device_name(device_json(bench.algname))
        conn = Storage().connection()

        self.create_table_if_not_exists(conn, bench)

        centers = {}
        with conn:
            for subbench in result.subbenches:
                centers[subbench] = {}
                for state in result.subbenches[subbench].states:
                    table_name = get_bench_table_name(subbench, bench.algorithm_name())
                    columns = ""
                    placeholders = ""
                    values = []

                    for name in state.point:
                        value = state.point[name]
                        columns = columns + ", \"{}\"".format(name)
                        placeholders = placeholders + ", ?"
                        values.append(value)

                    values = tuple(values)
                    samples = fpzip.compress(state.samples)
                    center = estimator(state.samples)
                    to_insert = (ctk, cccl, gpu, bench.variant_name(),
                                 result.elapsed, center, state.bw, samples) + values

                    query = """
                    INSERT INTO "{0}" (ctk, cccl, gpu, variant, elapsed, center, bw, samples {1})
                    VALUES (?, ?, ?, ?, ?, ?, ?, ? {2})
                    ON CONFLICT(ctk, cccl, gpu, variant {1}) DO NOTHING;
                    """.format(table_name, columns, placeholders)

                    conn.execute(query, to_insert)
                    centers[subbench][state.name()] = center

        return centers

    def pull_bench_centers(self, bench, ct_workload_point, rt_values):
        config = Config()
        ctk = config.ctk
        cccl = config.cccl
        gpu = get_device_name(device_json(bench.algname))
        conn = Storage().connection()

        self.create_table_if_not_exists(conn, bench)

        centers = {}

        with conn:
            for subbench in rt_values:
                centers[subbench] = {}
                table_name = get_bench_table_name(subbench, bench.algorithm_name())

                for rt_point in values_to_space(rt_values[subbench]):
                    point_map = {}
                    point_checks = ""
                    workload_point = list(ct_workload_point) + list(rt_point)
                    for axis in workload_point:
                        name, value = axis.split('=')
                        point_map[name] = value
                        point_checks = point_checks + " AND \"{}\" = \"{}\"".format(name, value)

                    query = """
                    SELECT center FROM "{0}" WHERE ctk = ? AND cccl = ? AND gpu = ? AND variant = ?{1};
                    """.format(table_name, point_checks)

                    result = conn.execute(query, (ctk, cccl, gpu, bench.variant_name())).fetchone()
                    if result is None:
                        return None

                    state_name = ' '.join(f'{k}={v}' for k, v in point_map.items())
                    centers[subbench][state_name] = float(result[0])

        return centers

def get_axis_name(axis):
    name = axis["name"]
    if axis["flags"]:
        name = name + "[{}]".format(axis["flags"])
    return name


def speedup(base, variant):
    # If one of the runs failed, dict is empty
    if not base or not variant:
        return {}

    benchmarks = set(base.keys())
    if benchmarks != set(variant.keys()):
        raise Exception("Benchmarks do not match.")

    result = {}
    for bench in benchmarks:
        base_states = base[bench]
        variant_states = variant[bench]

        state_names = set(base_states.keys())
        if state_names != set(variant_states.keys()):
            raise Exception("States do not match.")

        result[bench] = {}
        for state in state_names:
            result[bench][state] = base_states[state] / variant_states[state]

    return result


def values_to_space(axes):
    result = []
    for axis in axes:
        result.append(["{}={}".format(axis, value) for value in axes[axis]])
    return list(itertools.product(*result))


class ProcessRunner:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(ProcessRunner, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.process = None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def new_process(self, cmd):
        self.process = subprocess.Popen(cmd,
                                        start_new_session=True,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
        return self.process

    def signal_handler(self, signum, frame):
        self.kill_process()
        raise SystemExit('search was interrupted')

    def kill_process(self):
        if self.process is not None:
            self.process.kill()


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

    def bench_names(self):
        return [bench['name'] for bench in json_benches(self.algname)["benchmarks"]]

    def axes_names(self):
        subbench_names = {}
        for bench in json_benches(self.algname)["benchmarks"]:
            names = []
            for axis in bench["axes"]:
                names.append(get_axis_name(axis))

            subbench_names[bench['name']] = names
        return subbench_names

    def axes_values(self, sub_space, ct):
        subbench_space = {}
        for bench in json_benches(self.algname)["benchmarks"]:
            space = {}
            for axis in bench["axes"]:
                name = get_axis_name(axis)

                if ct:
                    if not '{ct}' in name:
                        continue
                else:
                    if '{ct}' in name:
                        continue

                axis_space = []
                if name in sub_space:
                    for value in sub_space[name]:
                        axis_space.append(value)
                else:
                    for value in axis["values"]:
                        axis_space.append(value["input_string"])

                space[name] = axis_space

            subbench_space[bench['name']] = space
        return subbench_space

    def ct_axes_value_descriptions(self):
        subbench_descriptions = {}
        for bench in json_benches(self.algname)["benchmarks"]:
            descriptions = {}
            for axis in bench["axes"]:
                name = axis["name"]
                if not '{ct}' in name:
                    continue
                if axis["flags"]:
                    name = name + "[{}]".format(axis["flags"])
                descriptions[name] = {}
                for value in axis["values"]:
                    descriptions[name][value["input_string"]] = value["description"]

            subbench_descriptions[bench["name"]] = descriptions
        return first_val(subbench_descriptions)


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

        descriptions = self.ct_axes_value_descriptions()
        for ct_component in self.ct_workload:
            ct_axis_name, ct_value = ct_component.split('=')
            description = descriptions[ct_axis_name][ct_value]
            ct_axis_name = ct_axis_name.replace('{ct}', '')
            definitions = definitions + "#define TUNE_{} {}\n".format(ct_axis_name, description)

        return definitions

    def do_run(self, ct_point, rt_values, timeout, is_search=True):
        logger = Logger()

        try:
            result_path = 'result.json'
            if os.path.exists(result_path):
                os.remove(result_path)

            bench_path = os.path.join('.', 'bin', self.exe_name())
            cmd = [bench_path]

            for value in ct_point:
                cmd.append('-a')
                cmd.append(value)

            cmd.append('--jsonbin')
            cmd.append(result_path)

            cmd.append("--stopping-criterion")
            cmd.append("entropy")

            # NVBench is currently broken for multiple GPUs, use `CUDA_VISIBLE_DEVICES`
            cmd.append("-d")
            cmd.append("0")

            for bench in rt_values:
                cmd.append('-b')
                cmd.append(bench)

                for axis in rt_values[bench]:
                    cmd.append('-a')
                    cmd.append("{}=[{}]".format(axis, ",".join(rt_values[bench][axis])))

            logger.info("starting benchmark {} with {}: {}".format(self.label(), ct_point, " ".join(cmd)))

            begin = time.time()
            p = ProcessRunner().new_process(cmd)
            p.wait(timeout=timeout)
            elapsed = time.time() - begin

            logger.info("finished benchmark {} with {} ({}) in {:.3f}s".format(self.label(), ct_point, p.returncode, elapsed))

            return BenchResult(result_path, p.returncode, elapsed)
        except subprocess.TimeoutExpired:
            logger.info("benchmark {} with {} reached timeout of {:.3f}s".format(self.label(), ct_point, timeout))
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            return BenchResult(None, 42, float('inf'))

    def ct_workload_space(self, sub_space):
        if not self.build():
            raise Exception("Unable to build benchmark: " + self.label())

        return values_to_space(first_val(self.axes_values(sub_space, True)))

    def rt_axes_values(self, sub_space):
        if not self.build():
            raise Exception("Unable to build benchmark: " + self.label())

        return self.axes_values(sub_space, False)

    def run(self, ct_workload_point, rt_values, estimator, is_search=True):
        logger = Logger()
        bench_cache = BenchCache()
        runs_cache = RunsCache()
        cached_centers = bench_cache.pull_bench_centers(self, ct_workload_point, rt_values)
        if cached_centers:
            logger.info("found benchmark {} in cache".format(self.label()))
            return cached_centers

        timeout = None

        if not self.is_base():
            code, elapsed = runs_cache.pull_run(self.get_base())
            if code != 0:
                raise Exception("Base bench return code = " + code)
            timeout = elapsed * 50

        result = self.do_run(ct_workload_point, rt_values, timeout, is_search)
        runs_cache.push_run(self, result.code, result.elapsed)
        return bench_cache.push_bench_centers(self, result, estimator)

    def speedup(self, ct_workload_point, rt_values, base_estimator, variant_estimator):
        if self.is_base():
            return 1.0

        base = self.get_base()
        base_center = base.run(ct_workload_point, rt_values, base_estimator)
        self_center = self.run(ct_workload_point, rt_values, variant_estimator)
        return speedup(base_center, self_center)

    def score(self, ct_workload, rt_values, base_estimator, variant_estimator):
        if self.is_base():
            return 1.0

        speedups = self.speedup(ct_workload, rt_values, base_estimator, variant_estimator)

        if not speedups:
            return float('-inf')

        rt_axes_ids = compute_axes_ids(rt_values)
        weight_matrices = compute_weight_matrices(rt_values, rt_axes_ids)

        score = 0
        for bench in speedups:
            for state in speedups[bench]:
                rt_workload = state_to_rt_workload(bench, state)
                weights = weight_matrices[bench]
                weight = get_workload_weight(rt_workload, rt_values[bench], rt_axes_ids[bench], weights)
                speedup = speedups[bench][state]
                score = score + weight * speedup

        return score


class BaseBench(Bench):
    def __init__(self, algname):
        super().__init__(algname, BasePoint(), [])
