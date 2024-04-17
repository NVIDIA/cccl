import math
import numpy as np


def importance_function(x):
    return 1 - math.exp(-x)


def x_by_importance(y):
    return -math.log(1 - y)


def compute_weights(num_values):
    least_importance = 0.6
    most_importance = 0.999

    assert(least_importance < most_importance)
    assert(least_importance >= 0 and least_importance < 1)
    assert(most_importance > 0 and most_importance < 1)

    begin = x_by_importance(least_importance)
    end = x_by_importance(most_importance)

    rng = end - begin
    step = rng / num_values

    return np.array([importance_function(begin + x * step) for x in range(num_values)])


def io_weights(values):
    return compute_weights(len(values))


def ei_weights(values):
    return np.ones(len(values))


def compute_axes_ids(rt_axes_values):
    result = {}
    for bench in rt_axes_values:
        rt_axes_ids = {}

        axis_id = 0
        for rt_axis in rt_axes_values[bench]:
            rt_axes_ids[rt_axis] = axis_id
            axis_id = axis_id + 1
        result[bench] = rt_axes_ids
    return result

def compute_raw_weight_matrix(rt_axes_values, rt_axes_ids):
    rt_axes_weights = {}

    first_rt_axis = True
    first_rt_axis_name = None
    for rt_axis in rt_axes_values:
        if first_rt_axis:
            first_rt_axis_name = rt_axis
            first_rt_axis = False
        values = rt_axes_values[rt_axis]
        rt_axes_values[rt_axis] = values
        if '{io}' in rt_axis:
            rt_axes_weights[rt_axis] = io_weights(values)
        else:
            rt_axes_weights[rt_axis] = ei_weights(values)

    num_rt_axes = len(rt_axes_ids)
    for rt_axis in rt_axes_weights:
        shape = [1] * num_rt_axes
        shape[rt_axes_ids[rt_axis]] = -1
        rt_axes_weights[rt_axis] = rt_axes_weights[rt_axis].reshape(*shape)

    weights_matrix = rt_axes_weights[first_rt_axis_name]
    for rt_axis in rt_axes_weights:
        if rt_axis == first_rt_axis_name:
            continue

        weights_matrix = weights_matrix * rt_axes_weights[rt_axis]

    return weights_matrix

def compute_weight_matrices(rt_axes_values, rt_axes_ids):
    matrices = {}
    aggregate = 0.0
    for bench in rt_axes_values:
        matrices[bench] = compute_raw_weight_matrix(rt_axes_values[bench], rt_axes_ids[bench])
        aggregate = aggregate + np.sum(matrices[bench])
    for bench in rt_axes_values:
        matrices[bench] = matrices[bench] / aggregate
    return matrices


def get_workload_coordinates(rt_workload, rt_axes_values, rt_axes_ids):
    coordinates = [0] * len(rt_axes_ids)
    for point in rt_workload:
        rt_axis, rt_value = point.split('=')
        coordinates[rt_axes_ids[rt_axis]] = rt_axes_values[rt_axis].index(rt_value)
    return coordinates

def get_workload_weight(rt_workload, rt_axes_values, rt_axes_ids, weights_matrix):
    coordinates = get_workload_coordinates(rt_workload, rt_axes_values, rt_axes_ids)
    return weights_matrix[tuple(coordinates)]
