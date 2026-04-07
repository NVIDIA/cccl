import json

import numpy as np


def read_json(filename):
    with open(filename, "r") as f:
        file_root = json.load(f)
    return file_root


def extract_filename(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "filename", summary_data))
    assert value_data["type"] == "string"
    return value_data["value"]


def extract_size(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "size", summary_data))
    assert value_data["type"] == "int64"
    return int(value_data["value"])


def extract_bw(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "value", summary_data))
    assert value_data["type"] == "float64"
    return float(value_data["value"])


def parse_samples_meta(state):
    summaries = state["summaries"]
    if not summaries:
        return None, None

    summary = next(
        filter(lambda s: s["tag"] == "nv/json/bin:nv/cold/sample_times", summaries),
        None,
    )
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

    assert sample_count == len(samples)
    return samples


def parse_bw(state):
    bwutil = next(
        filter(
            lambda s: s["tag"] == "nv/cold/bw/global/utilization", state["summaries"]
        ),
        None,
    )
    if not bwutil:
        return None

    return extract_bw(bwutil)


def get_axis_name(axis):
    name = axis["name"]
    if af := axis["flags"]:
        name = name + f"[{af}]"
    return name


class SubBenchState:
    def __init__(self, state, axes_names, axes_values):
        self.samples = parse_samples(state)
        self.bw = parse_bw(state)

        self.point = {}
        for axis in state["axis_values"]:
            axis_name = axis["name"]
            name = axes_names[axis_name]
            value = axes_values[axis_name][axis["value"]]
            self.point[name] = value

    def __repr__(self):
        return str(self.__dict__)

    def name(self):
        return " ".join(f"{k}={v}" for k, v in self.point.items())

    def center(self, estimator):
        return estimator(self.samples)


class SubBenchResult:
    def __init__(self, bench):
        axes_names = {}
        axes_values = {}
        for axis in bench["axes"]:
            short_name = axis["name"]
            full_name = get_axis_name(axis)
            this_axis_values = {}
            for value in axis["values"]:
                if "value" in value:
                    this_axis_values[str(value["value"])] = value["input_string"]
                else:
                    this_axis_values[value["input_string"]] = value["input_string"]
            axes_names[short_name] = full_name
            axes_values[short_name] = this_axis_values

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
