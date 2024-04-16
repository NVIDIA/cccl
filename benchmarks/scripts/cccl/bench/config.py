import os
import sys
import random
import itertools


def randomized_cartesian_product(list_of_lists):
    length = 1
    for l in list_of_lists:
        length *= len(l)

    visited = set()
    while len(visited) < length:
        variant = tuple(map(random.choice, list_of_lists))
        if variant not in visited:
            visited.add(variant)
            yield variant


class Range:
    def __init__(self, definition, label, low, high, step):
        self.definition = definition
        self.label = label
        self.low = low
        self.high = high
        self.step = step


class RangePoint:
    def __init__(self, definition, label, value):
        self.definition = definition
        self.label = label
        self.value = value


class VariantPoint:
    def __init__(self, range_points):
        self.range_points = range_points

    def label(self):
        if self.is_base():
            return 'base'
        return '.'.join(["{}_{}".format(point.label, point.value) for point in self.range_points])

    def is_base(self):
        return len(self.range_points) == 0

    def tuning(self):
        if self.is_base():
            return ""

        tuning = "#pragma once\n\n"
        for point in self.range_points:
            tuning += "#define {} {}\n".format(point.definition, point.value)
        return tuning


class BasePoint(VariantPoint):
    def __init__(self):
        VariantPoint.__init__(self, [])


def parse_ranges(columns):
    ranges = []
    for column in columns:
        definition, label_range = column.split('|')
        label, range = label_range.split('=')
        start, end, step = [int(x) for x in range.split(':')]
        ranges.append(Range(definition, label, start, end + 1, step))

    return ranges


def parse_meta():
    if not os.path.isfile("cccl_meta_bench.csv"):
        print("cccl_meta_bench.csv not found", file=sys.stderr)
        print("make sure to run the script from the CUB build directory",
              file=sys.stderr)

    benchmarks = {}
    ctk_version = "0.0.0"
    cccl_revision = "0.0-0-0000"
    with open("cccl_meta_bench.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            if ',' in line:
                columns = line.split(',')
            else:
                columns = [' '.join(line.split())]

            name = columns[0]

            if name == "ctk_version":
                ctk_version = columns[1].rstrip()
            elif name == "cccl_revision":
                cccl_revision = columns[1].rstrip()
            else:
                if len(columns) > 1:
                    benchmarks[name] = parse_ranges(columns[1:])
                else:
                    benchmarks[name] = []

    return ctk_version, cccl_revision, benchmarks


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.ctk, cls._instance.cccl, cls._instance.benchmarks = parse_meta()
        return cls._instance

    def label_to_variant_point(self, algname, label):
        if label == "base":
            return BasePoint()

        label_to_definition = {}
        for param_space in self.benchmarks[algname]:
            label_to_definition[param_space.label] = param_space.definition

        points = []
        for point in label.split('.'):
            label, value = point.split('_')
            points.append(RangePoint(label_to_definition[label], label, int(value)))

        return VariantPoint(points)

    def variant_space(self, algname):
        variants = []
        for param_space in self.benchmarks[algname]:
            variants.append([])
            for value in range(param_space.low, param_space.high, param_space.step):
                variants[-1].append(RangePoint(param_space.definition, param_space.label, value))

        return (VariantPoint(points) for points in randomized_cartesian_product(variants))

    def variant_space_size(self, algname):
        num_variants = 1
        for param_space in self.benchmarks[algname]:
            num_variants = num_variants * len(range(param_space.low, param_space.high, param_space.step))
        return num_variants
