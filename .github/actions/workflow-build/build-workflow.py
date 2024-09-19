#!/usr/bin/env python3

"""
Concepts:
- matrix_job: an entry of a workflow matrix, converted from matrix.yaml["workflow"][id] into a JSON object.
  Example:
  {
    "jobs": [
      "test"
    ],
    "project": [
      "libcudacxx",
      "cub",
      "thrust"
    ],
    "ctk": "11.1",
    "cudacxx": 'nvcc',
    "cxx": 'gcc10',
    "sm": "75-real",
    "std": 17
    "cpu": "amd64",
    "gpu": "t4",
  }

Matrix jobs are read from the matrix.yaml file and converted into a JSON object and passed to matrix_job_to_dispatch_group, where
the matrix job is turned into one or more dispatch groups consisting of potentially many jobs.

- dispatch_group_json: A json object used in conjunction with the ci-dispatch-groups.yml GHA workflow.
  Example:
  {
    "<group name>": {
      "standalone": [ {<job_json>}, ... ]
      "two_stage": [ {<two_stage_json>}, ]
    }
  }

- two_stage_json: A json object that represents bulk-synchronous producer/consumer jobs, used with ci-dispatch-two-stage.yml.
  Example:
  {
    "id": "<unique id>", # Used as a compact unique name for the GHA dispatch workflows.
    "producers": [ {<job_json>}, ... ],
    "consumers": [ {<job_json>}, ... ]
  }

- job_json: A json object that represents a single job in a workflow. Used with ci-dispatch-job.yml.
  Example:
  {
    "id": "<unique id>", # Used as a compact unique name for the GHA dispatch workflows.
    "name": "...",
    "runner": "...",
    "image": "...",
    "command": "..." },
  }
"""

import argparse
import base64
import copy
import functools
import json
import os
import re
import struct
import sys
import yaml


matrix_yaml = None


# Decorators to cache static results of functions:
# static_result: function has no args, same result each invocation.
# memoize_result: result depends on args.
def static_result(func): return functools.lru_cache(maxsize=1)(func)
def memoize_result(func): return functools.lru_cache(maxsize=None)(func)


def generate_guids():
    """
    Simple compact global unique ID generator.
    Produces up to 65535 unique IDs between 1-3 characters in length.
    Throws an exception once exhausted.
    """
    i = 0
    while True:
        # Generates a base64 hash of an incrementing 16-bit integer:
        hash = base64.b64encode(struct.pack(">H", i)).decode('ascii')
        # Strips off up-to 2 leading 'A' characters and a single trailing '=' characters, if they exist:
        guid = re.sub(r'^A{0,2}', '', hash).removesuffix("=")
        yield guid
        i += 1
        if i >= 65535:
            raise Exception("GUID generator exhausted.")


guid_generator = generate_guids()


def write_json_file(filename, json_object):
    with open(filename, 'w') as f:
        json.dump(json_object, f, indent=2)


def write_text_file(filename, text):
    with open(filename, 'w') as f:
        print(text, file=f)


def error_message_with_matrix_job(matrix_job, message):
    return f"{matrix_job['origin']['workflow_location']}: {message}\n  Input: {matrix_job['origin']['original_matrix_job']}"


@memoize_result
def canonicalize_ctk_version(ctk_string):
    if ctk_string in matrix_yaml['ctk_versions']:
        return ctk_string

    # Check for aka's:
    for ctk_key, ctk_value in matrix_yaml['ctk_versions'].items():
        if 'aka' in ctk_value and ctk_string == ctk_value['aka']:
            return ctk_key

    raise Exception(f"Unknown CTK version '{ctk_string}'")


def get_ctk(ctk_string):
    result = matrix_yaml['ctk_versions'][ctk_string]
    result["version"] = ctk_string
    return result


@memoize_result
def parse_cxx_string(cxx_string):
    "Returns (id, version) tuple. Version may be None if not present."
    return re.match(r'^([a-z]+)-?([\d\.]+)?$', cxx_string).groups()


@memoize_result
def canonicalize_host_compiler_name(cxx_string):
    """
    Canonicalize the host compiler cxx_string.

    Valid input formats: 'gcc', 'gcc10', or 'gcc-12'.
    Output format: 'gcc12'.

    If no version is specified, the latest version is used.
    """
    id, version = parse_cxx_string(cxx_string)

    if not id in matrix_yaml['host_compilers']:
        raise Exception(
            f"Unknown host compiler '{id}'. Valid options are: {', '.join(matrix_yaml['host_compilers'].keys())}")

    hc_def = matrix_yaml['host_compilers'][id]
    hc_versions = hc_def['versions']

    if not version:
        version = max(hc_def['versions'].keys(), key=lambda x: tuple(map(int, x.split('.'))))

    # Check for aka's:
    if not version in hc_def['versions']:
        for version_key, version_data in hc_def['versions'].items():
            if 'aka' in version_data and version == version_data['aka']:
                version = version_key

    if not version in hc_def['versions']:
        raise Exception(
            f"Unknown version '{version}' for host compiler '{id}'.")

    cxx_string = f"{id}{version}"

    return cxx_string


@memoize_result
def get_host_compiler(cxx_string):
    "Expects a canonicalized cxx_string."
    id, version = parse_cxx_string(cxx_string)

    if not id in matrix_yaml['host_compilers']:
        raise Exception(
            f"Unknown host compiler '{id}'. Valid options are: {', '.join(matrix_yaml['host_compilers'].keys())}")

    hc_def = matrix_yaml['host_compilers'][id]

    if not version in hc_def['versions']:
        raise Exception(
            f"Unknown version '{version}' for host compiler '{id}'. Valid options are: {', '.join(hc_def['versions'].keys())}")

    version_def = hc_def['versions'][version]

    result = {'id': id,
              'name': hc_def['name'],
              'version': version,
              'container_tag': hc_def['container_tag'],
              'exe': hc_def['exe']}

    for key, value in version_def.items():
        result[key] = value

    return result


def get_device_compiler(matrix_job):
    id = matrix_job['cudacxx']
    if not id in matrix_yaml['device_compilers'].keys():
        raise Exception(
            f"Unknown device compiler '{id}'. Valid options are: {', '.join(matrix_yaml['device_compilers'].keys())}")
    result = matrix_yaml['device_compilers'][id]
    result['id'] = id

    if id == 'nvcc':
        ctk = get_ctk(matrix_job['ctk'])
        result['version'] = ctk['version']
        result['stds'] = ctk['stds']
    elif id == 'clang':
        host_compiler = get_host_compiler(matrix_job['cxx'])
        result['version'] = host_compiler['version']
        result['stds'] = host_compiler['stds']
    else:
        raise Exception(f"Cannot determine version/std info for device compiler '{id}'")

    return result


@memoize_result
def get_gpu(gpu_string):
    if not gpu_string in matrix_yaml['gpus']:
        raise Exception(
            f"Unknown gpu '{gpu_string}'. Valid options are: {', '.join(matrix_yaml['gpus'].keys())}")

    result = matrix_yaml['gpus'][gpu_string]
    result['id'] = gpu_string

    if not 'testing' in result:
        result['testing'] = False

    return result


@memoize_result
def get_project(project):
    if not project in matrix_yaml['projects'].keys():
        raise Exception(
            f"Unknown project '{project}'. Valid options are: {', '.join(matrix_yaml['projects'].keys())}")

    result = matrix_yaml['projects'][project]
    result['id'] = project

    if not 'name' in result:
        result['name'] = project

    if not 'job_map' in result:
        result['job_map'] = {}

    return result


@memoize_result
def get_job_type_info(job):
    if not job in matrix_yaml['jobs'].keys():
        raise Exception(
            f"Unknown job '{job}'. Valid options are: {', '.join(matrix_yaml['jobs'].keys())}")

    result = matrix_yaml['jobs'][job]
    result['id'] = job

    if not 'name' in result:
        result['name'] = job.capitalize()
    if not 'gpu' in result:
        result['gpu'] = False
    if not 'needs' in result:
        result['needs'] = None
    if not 'invoke' in result:
        result['invoke'] = {}
    if not 'prefix' in result['invoke']:
        result['invoke']['prefix'] = job
    if not 'args' in result['invoke']:
        result['invoke']['args'] = ""

    return result


@memoize_result
def get_tag_info(tag):
    if not tag in matrix_yaml['tags'].keys():
        raise Exception(
            f"Unknown tag '{tag}'. Valid options are: {', '.join(matrix_yaml['tags'].keys())}")

    result = matrix_yaml['tags'][tag]
    result['id'] = tag

    if 'required' not in result:
        result['required'] = False

    if 'default' in result:
        result['required'] = False
    else:
        result['default'] = None


    return result


@static_result
def get_all_matrix_job_tags_sorted():
    all_tags = set(matrix_yaml['tags'].keys())

    # Sorted using a highly subjective opinion on importance:
    # Always first, information dense:
    sorted_important_tags = ['project', 'jobs', 'cudacxx', 'cxx', 'ctk', 'gpu', 'std', 'sm', 'cpu']

    # Always last, derived:
    sorted_noise_tags = ['origin']

    # In between?
    sorted_tags = set(sorted_important_tags + sorted_noise_tags)
    sorted_meh_tags = sorted(list(all_tags - sorted_tags))

    return sorted_important_tags + sorted_meh_tags + sorted_noise_tags


def lookup_supported_stds(matrix_job):
    stds = set(matrix_yaml['all_stds'])
    if 'ctk' in matrix_job:
        ctk = get_ctk(matrix_job['ctk'])
        stds = stds & set(ctk['stds'])
    if 'cxx' in matrix_job:
        host_compiler = get_host_compiler(matrix_job['cxx'])
        stds = stds & set(host_compiler['stds'])
    if 'cudacxx' in matrix_job:
        device_compiler = get_device_compiler(matrix_job)
        stds = stds & set(device_compiler['stds'])
    if 'project' in matrix_job:
        project = get_project(matrix_job['project'])
        stds = stds & set(project['stds'])
    if len(stds) == 0:
        raise Exception(error_message_with_matrix_job(matrix_job, "No supported stds found."))
    return sorted(list(stds))


def is_windows(matrix_job):
    host_compiler = get_host_compiler(matrix_job['cxx'])
    return host_compiler['container_tag'] == 'cl'


def generate_dispatch_group_name(matrix_job):
    project = get_project(matrix_job['project'])
    ctk = matrix_job['ctk']
    device_compiler = get_device_compiler(matrix_job)
    host_compiler = get_host_compiler(matrix_job['cxx'])

    compiler_info = ""
    if device_compiler['id'] == 'nvcc':
        compiler_info = f"{device_compiler['name']} {host_compiler['name']}"
    elif device_compiler['id'] == 'clang':
        compiler_info = f"{device_compiler['name']}"
    else:
        compiler_info = f"{device_compiler['name']}-{device_compiler['version']} {host_compiler['name']}"

    return f"{project['name']} CTK{ctk} {compiler_info}"


def generate_dispatch_job_name(matrix_job, job_type):
    job_info = get_job_type_info(job_type)
    std_str = ("C++" + str(matrix_job['std']) + " ") if 'std' in matrix_job else ''
    cpu_str = matrix_job['cpu']
    gpu_str = (', ' + matrix_job['gpu'].upper()) if job_info['gpu'] else ""
    cuda_compile_arch = (" sm{" + str(matrix_job['sm']) + "}") if 'sm' in matrix_job else ""
    cmake_options = (' ' + matrix_job['cmake_options']) if 'cmake_options' in matrix_job else ""

    host_compiler = get_host_compiler(matrix_job['cxx'])

    config_tag = f"{std_str}{host_compiler['name']}{host_compiler['version']}"

    extra_info = f":{cuda_compile_arch}{cmake_options}" if cuda_compile_arch or cmake_options else ""

    return f"[{config_tag}] {job_info['name']}({cpu_str}{gpu_str}){extra_info}"


def generate_dispatch_job_runner(matrix_job, job_type):
    runner_os = "windows" if is_windows(matrix_job) else "linux"
    cpu = matrix_job['cpu']

    job_info = get_job_type_info(job_type)
    if not job_info['gpu']:
        return f"{runner_os}-{cpu}-cpu16"

    gpu = get_gpu(matrix_job['gpu'])
    suffix = "-testing" if gpu['testing'] else ""

    return f"{runner_os}-{cpu}-gpu-{gpu['id']}-latest-1{suffix}"


def generate_dispatch_job_ctk_version(matrix_job, job_type):
    ".devcontainers/launch.sh --cuda option:"
    return matrix_job['ctk']


def generate_dispatch_job_host_compiler(matrix_job, job_type):
    ".devcontainers/launch.sh --host option:"
    host_compiler = get_host_compiler(matrix_job['cxx'])
    return host_compiler['container_tag'] + host_compiler['version']


def generate_dispatch_job_image(matrix_job, job_type):
    devcontainer_version = matrix_yaml['devcontainer_version']
    ctk = matrix_job['ctk']
    host_compiler = generate_dispatch_job_host_compiler(matrix_job, job_type)

    if is_windows(matrix_job):
        return f"rapidsai/devcontainers:{devcontainer_version}-cuda{ctk}-{host_compiler}"

    return f"rapidsai/devcontainers:{devcontainer_version}-cpp-{host_compiler}-cuda{ctk}"


def generate_dispatch_job_command(matrix_job, job_type):
    script_path = "./ci/windows" if is_windows(matrix_job) else "./ci"
    script_ext = ".ps1" if is_windows(matrix_job) else ".sh"

    job_info = get_job_type_info(job_type)
    job_prefix = job_info['invoke']['prefix']
    job_args = job_info['invoke']['args']

    project = get_project(matrix_job['project'])
    script_name = f"{script_path}/{job_prefix}_{project['id']}{script_ext}"

    std_str = str(matrix_job['std']) if 'std' in matrix_job else ''

    device_compiler = get_device_compiler(matrix_job)

    cuda_compile_arch = matrix_job['sm'] if 'sm' in matrix_job else ''
    cmake_options = matrix_job['cmake_options'] if 'cmake_options' in matrix_job else ''

    command = f"\"{script_name}\""
    if job_args:
        command += f" {job_args}"
    if std_str:
        command += f" -std \"{std_str}\""
    if cuda_compile_arch:
        command += f" -arch \"{cuda_compile_arch}\""
    if device_compiler['id'] != 'nvcc':
        command += f" -cuda \"{device_compiler['exe']}\""
    if cmake_options:
        command += f" -cmake-options \"{cmake_options}\""

    return command


def generate_dispatch_job_origin(matrix_job, job_type):
    # Already has silename, line number, etc:
    origin = matrix_job['origin'].copy()

    origin_job = matrix_job.copy()
    del origin_job['origin']

    job_info = get_job_type_info(job_type)

    # The origin tags are used to build the execution summary for the CI PR comment.
    # Use the human readable job label for the execution summary:
    origin_job['jobs'] = job_info['name']

    # Replace some of the clunkier tags with a summary-friendly version:
    if 'cxx' in origin_job:
        host_compiler = get_host_compiler(matrix_job['cxx'])
        del origin_job['cxx']

        origin_job['cxx'] = host_compiler['name'] + host_compiler['version']
        origin_job['cxx_family'] = host_compiler['name']

    if 'cudacxx' in origin_job:
        device_compiler = get_device_compiler(matrix_job)
        del origin_job['cudacxx']

        origin_job['cudacxx'] = device_compiler['name'] + device_compiler['version']
        origin_job['cudacxx_family'] = device_compiler['name']

    origin['matrix_job'] = origin_job

    return origin


def generate_dispatch_job_json(matrix_job, job_type):
    return {
        'cuda': generate_dispatch_job_ctk_version(matrix_job, job_type),
        'host': generate_dispatch_job_host_compiler(matrix_job, job_type),
        'name': generate_dispatch_job_name(matrix_job, job_type),
        'runner': generate_dispatch_job_runner(matrix_job, job_type),
        'image': generate_dispatch_job_image(matrix_job, job_type),
        'command': generate_dispatch_job_command(matrix_job, job_type),
        'origin': generate_dispatch_job_origin(matrix_job, job_type)
    }


# Create a single build producer, and a separate consumer for each test_job_type:
def generate_dispatch_two_stage_json(matrix_job, producer_job_type, consumer_job_types):
    producer_json = generate_dispatch_job_json(matrix_job, producer_job_type)

    consumers_json = []
    for consumer_job_type in consumer_job_types:
        consumers_json.append(generate_dispatch_job_json(matrix_job, consumer_job_type))

    return {
        "producers": [producer_json],
        "consumers": consumers_json
    }


def generate_dispatch_group_jobs(matrix_job):
    dispatch_group_jobs = {
        "standalone": [],
        "two_stage": []
    }

    # The jobs tag is left unexploded to optimize scheduling here.
    job_types = set(matrix_job['jobs'])

    # Add all dpendencies to the job_types set:
    standalone = set([])
    two_stage = {}  # {producer: set([consumer, ...])}
    for job_type in job_types:
        job_info = get_job_type_info(job_type)
        dep = job_info['needs']
        if dep:
            if dep in two_stage:
                two_stage[dep].add(job_type)
            else:
                two_stage[dep] = set([job_type])
        else:
            standalone.add(job_type)

    standalone.difference_update(two_stage.keys())

    for producer, consumers in two_stage.items():
        dispatch_group_jobs['two_stage'].append(
            generate_dispatch_two_stage_json(matrix_job, producer, list(consumers)))

    for job_type in standalone:
        dispatch_group_jobs['standalone'].append(generate_dispatch_job_json(matrix_job, job_type))

    return dispatch_group_jobs


def matrix_job_to_dispatch_group(matrix_job, group_prefix=""):
    return {group_prefix + generate_dispatch_group_name(matrix_job):
            generate_dispatch_group_jobs(matrix_job)}


def merge_dispatch_groups(accum_dispatch_groups, new_dispatch_groups):
    for group_name, group_json in new_dispatch_groups.items():
        if group_name not in accum_dispatch_groups:
            accum_dispatch_groups[group_name] = group_json
        else:
            # iterate standalone and two_stage:
            for key, value in group_json.items():
                accum_dispatch_groups[group_name][key] += value


def compare_dispatch_jobs(job1, job2):
    "Compare two dispatch job specs for equality. Considers only name/runner/image/command."
    # Ignores the 'origin' key, which may vary between identical job specifications.
    return (job1['name'] == job2['name'] and
            job1['runner'] == job2['runner'] and
            job1['image'] == job2['image'] and
            job1['command'] == job2['command'])


def dispatch_job_in_container(job, container):
    "Check if a dispatch job is in a container, using compare_dispatch_jobs."
    for job2 in container:
        if compare_dispatch_jobs(job, job2):
            return True
    return False


def remove_dispatch_job_from_container(job, container):
    "Remove a dispatch job from a container, using compare_dispatch_jobs."
    for i, job2 in enumerate(container):
        if compare_dispatch_jobs(job, job2):
            del container[i]
            return True
    return False


def index_of_dispatch_job_in_container(job, container):
    "Find the index of a dispatch job in a container, using compare_dispatch_jobs."
    for idx, job2 in enumerate(container):
        if compare_dispatch_jobs(job, job2):
            return idx
    return None


def finalize_workflow_dispatch_groups(workflow_dispatch_groups_orig):
    workflow_dispatch_groups = copy.deepcopy(workflow_dispatch_groups_orig)

    # Check to see if any .two_stage.producers arrays have more than 1 job, which is not supported.
    # See ci-dispatch-two-stage.yml for details.
    for group_name, group_json in workflow_dispatch_groups.items():
        if 'two_stage' in group_json:
            for two_stage_json in group_json['two_stage']:
                num_producers = len(two_stage_json['producers'])
                if num_producers > 1:
                    producer_names = ""
                    for job in two_stage_json['producers']:
                        producer_names += f" - {job['name']}\n"
                    error_message = f"ci-dispatch-two-stage.yml currently only supports a single producer. "
                    error_message += f"Found {num_producers} producers in '{group_name}':\n{producer_names}"
                    print(f"::error file=ci/matrix.yaml::{error_message}", file=sys.stderr)
                    raise Exception(error_message)

    # Merge consumers for any two_stage arrays that have the same producer(s). Print a warning.
    for group_name, group_json in workflow_dispatch_groups.items():
        if not 'two_stage' in group_json:
            continue
        two_stage_json = group_json['two_stage']
        merged_producers = []
        merged_consumers = []
        for two_stage in two_stage_json:
            producers = two_stage['producers']
            consumers = two_stage['consumers']

            # Make sure this gets updated if we add support for multiple producers:
            assert (len(producers) == 1)
            producer = producers[0]

            if dispatch_job_in_container(producer, merged_producers):
                producer_index = index_of_dispatch_job_in_container(producer, merged_producers)
                matching_consumers = merged_consumers[producer_index]

                producer_name = producer['name']
                print(f"::notice::Merging consumers for duplicate producer '{producer_name}' in '{group_name}'",
                      file=sys.stderr)
                consumer_names = ", ".join([consumer['name'] for consumer in matching_consumers])
                print(f"::notice::Original consumers: {consumer_names}", file=sys.stderr)
                consumer_names = ", ".join([consumer['name'] for consumer in consumers])
                print(f"::notice::Duplicate consumers: {consumer_names}", file=sys.stderr)
                # Merge if unique:
                for consumer in consumers:
                    if not dispatch_job_in_container(consumer, matching_consumers):
                        matching_consumers.append(consumer)
                consumer_names = ", ".join([consumer['name'] for consumer in matching_consumers])
                print(f"::notice::Merged consumers: {consumer_names}", file=sys.stderr)
            else:
                merged_producers.append(producer)
                merged_consumers.append(consumers)
        # Update with the merged lists:
        two_stage_json = []
        for producer, consumers in zip(merged_producers, merged_consumers):
            two_stage_json.append({'producers': [producer], 'consumers': consumers})
        group_json['two_stage'] = two_stage_json

    # Check for any duplicate jobs in standalone arrays. Warn and remove duplicates.
    for group_name, group_json in workflow_dispatch_groups.items():
        standalone_jobs = group_json['standalone'] if 'standalone' in group_json else []
        unique_standalone_jobs = []
        for job_json in standalone_jobs:
            if dispatch_job_in_container(job_json, unique_standalone_jobs):
                print(f"::notice::Removing duplicate standalone job '{job_json['name']}' in '{group_name}'",
                      file=sys.stderr)
            else:
                unique_standalone_jobs.append(job_json)

        # If any producer/consumer jobs exist in standalone arrays, warn and remove the standalones.
        two_stage_jobs = group_json['two_stage'] if 'two_stage' in group_json else []
        for two_stage_job in two_stage_jobs:
            for producer in two_stage_job['producers']:
                if remove_dispatch_job_from_container(producer, unique_standalone_jobs):
                    print(f"::notice::Removing standalone job '{producer['name']}' " +
                          f"as it appears as a producer in '{group_name}'",
                          file=sys.stderr)
            for consumer in two_stage_job['consumers']:
                if remove_dispatch_job_from_container(producer, unique_standalone_jobs):
                    print(f"::notice::Removing standalone job '{consumer['name']}' " +
                          f"as it appears as a consumer in '{group_name}'",
                          file=sys.stderr)
        standalone_jobs = list(unique_standalone_jobs)
        group_json['standalone'] = standalone_jobs

        # If any producer or consumer job appears more than once, warn and leave as-is.
        all_two_stage_jobs = []
        duplicate_jobs = {}
        for two_stage_job in two_stage_jobs:
            for job in two_stage_job['producers'] + two_stage_job['consumers']:
                if dispatch_job_in_container(job, all_two_stage_jobs):
                    duplicate_jobs[job['name']] = duplicate_jobs.get(job['name'], 1) + 1
                else:
                    all_two_stage_jobs.append(job)
        for job_name, count in duplicate_jobs.items():
            print(f"::warning file=ci/matrix.yaml::" +
                  f"Job '{job_name}' appears {count} times in '{group_name}'.",
                  f"Cannot remove duplicate while resolving dependencies. This job WILL execute {count} times.",
                  file=sys.stderr)

    # Remove all named values that contain an empty list of jobs:
    for group_name, group_json in workflow_dispatch_groups.items():
        if not group_json['standalone'] and not group_json['two_stage']:
            del workflow_dispatch_groups[group_name]
        elif not group_json['standalone']:
            del group_json['standalone']
        elif not group_json['two_stage']:
            del group_json['two_stage']

    # Natural sort impl (handles embedded numbers in strings, case insensitive)
    def natural_sort_key(key):
        return [(int(text) if text.isdigit() else text.lower()) for text in re.split('(\d+)', key)]

    # Sort the dispatch groups by name:
    workflow_dispatch_groups = dict(sorted(workflow_dispatch_groups.items(), key=lambda x: natural_sort_key(x[0])))

    # Sort the jobs within each dispatch group:
    for group_name, group_json in workflow_dispatch_groups.items():
        if 'standalone' in group_json:
            group_json['standalone'] = sorted(group_json['standalone'], key=lambda x: natural_sort_key(x['name']))
        if 'two_stage' in group_json:
            group_json['two_stage'] = sorted(
                group_json['two_stage'], key=lambda x: natural_sort_key(x['producers'][0]['name']))

    # Assign unique IDs in appropriate locations.
    # These are used to give "hidden" dispatch jobs a short, unique name,
    # otherwise GHA generates a long, cluttered name.
    for group_name, group_json in workflow_dispatch_groups.items():
        if 'standalone' in group_json:
            for job_json in group_json['standalone']:
                job_json['id'] = next(guid_generator)
        if 'two_stage' in group_json:
            for two_stage_json in group_json['two_stage']:
                two_stage_json['id'] = next(guid_generator)
                for job_json in two_stage_json['producers'] + two_stage_json['consumers']:
                    job_json['id'] = next(guid_generator)

    return workflow_dispatch_groups


def find_workflow_line_number(workflow_name):
    regex = re.compile(f"^( )*{workflow_name}:", re.IGNORECASE)
    line_number = 0
    with open(matrix_yaml['filename'], 'r') as f:
        for line in f:
            line_number += 1
            if regex.match(line):
                return line_number
    raise Exception(
        f"Workflow '{workflow_name}' not found in {matrix_yaml['filename]']} (could not match regex: {regex})")


def get_matrix_job_origin(matrix_job, workflow_name, workflow_location):
    filename = matrix_yaml['filename']
    original_matrix_job = json.dumps(matrix_job, indent=None, separators=(', ', ': '))
    original_matrix_job = original_matrix_job.replace('"', '')
    return {
        'filename': filename,
        'workflow_name': workflow_name,
        'workflow_location': workflow_location,
        'original_matrix_job': original_matrix_job
    }


@static_result
def get_excluded_matrix_jobs():
    return parse_workflow_matrix_jobs(None, 'exclude')


def apply_matrix_job_exclusion(matrix_job, exclusion):
    # Excluded tags to remove from unexploded tag categories: { tag: [exluded_value1, excluded_value2] }
    update_dict = {}

    for tag, excluded_values in exclusion.items():
        # Not excluded if a specified tag isn't even present:
        if not tag in matrix_job:
            return matrix_job

        # Some tags are left unexploded (e.g. 'jobs') to optimize scheduling,
        # so the values can be either a list or a single value.
        # Standardize to a list for comparison:
        if type(excluded_values) != list:
            excluded_values = [excluded_values]
        matrix_values = matrix_job[tag]
        if type(matrix_values) != list:
            matrix_values = [matrix_values]

        # Identify excluded values that are present in the matrix job for this tag:
        matched_tag_values = [value for value in matrix_values if value in excluded_values]
        # Not excluded if no values match for a tag:
        if not matched_tag_values:
            return matrix_job

        # If there is only a partial match to the matrix values, record the matches in the update_dict.
        # If the match is complete, do nothing.
        if len(matched_tag_values) < len(matrix_values):
            update_dict[tag] = matched_tag_values

    # If we get here, the matrix job matches and should be updated or excluded entirely.
    # If all tag matches are complete, then update_dict will be empty and the job should be excluded entirely
    if not update_dict:
        return None

    # If update_dict is populated, remove the matched values from the matrix job and return it.
    new_matrix_job = copy.deepcopy(matrix_job)
    for tag, values in update_dict.items():
        for value in values:
            new_matrix_job[tag].remove(value)

    return new_matrix_job


def remove_excluded_jobs(matrix_jobs):
    '''Remove jobs that match all tags in any of the exclusion matrix jobs.'''
    excluded = get_excluded_matrix_jobs()
    filtered_matrix_jobs = []
    for matrix_job_orig in matrix_jobs:
        matrix_job = copy.deepcopy(matrix_job_orig)
        for exclusion in excluded:
            matrix_job = apply_matrix_job_exclusion(matrix_job, exclusion)
            if not matrix_job:
                break
        if matrix_job:
            filtered_matrix_jobs.append(matrix_job)
    return filtered_matrix_jobs


def validate_tags(matrix_job, ignore_required=False):
    all_tags = matrix_yaml['tags'].keys()

    if not ignore_required:
        for tag in all_tags:
            tag_info = get_tag_info(tag)
            if tag not in matrix_job:
                if tag_info['required']:
                    raise Exception(error_message_with_matrix_job(matrix_job, f"Missing required tag '{tag}'"))
        if 'cudacxx' in matrix_job:
            if matrix_job['cudacxx'] == 'clang' and ('cxx' not in matrix_job or 'clang' not in matrix_job['cxx']):
                raise Exception(error_message_with_matrix_job(matrix_job, f"cudacxx=clang requires cxx=clang."))

    for tag in matrix_job:
        if tag == 'origin':
            continue
        if tag not in all_tags:
            raise Exception(error_message_with_matrix_job(matrix_job, f"Unknown tag '{tag}'"))

    if 'gpu' in matrix_job and matrix_job['gpu'] not in matrix_yaml['gpus'].keys():
        raise Exception(error_message_with_matrix_job(matrix_job, f"Unknown gpu '{matrix_job['gpu']}'"))


def set_default_tags(matrix_job):
    all_tags = matrix_yaml['tags'].keys()
    for tag in all_tags:
        if tag in matrix_job:
            continue

        tag_info = get_tag_info(tag)
        if tag_info['default']:
            matrix_job[tag] = tag_info['default']


def canonicalize_tags(matrix_job):
    if 'ctk' in matrix_job:
        matrix_job['ctk'] = canonicalize_ctk_version(matrix_job['ctk'])
    if 'cxx' in matrix_job:
        matrix_job['cxx'] = canonicalize_host_compiler_name(matrix_job['cxx'])


def set_derived_tags(matrix_job):
    if 'sm' in matrix_job and matrix_job['sm'] == 'gpu':
        if not 'gpu' in matrix_job:
            raise Exception(error_message_with_matrix_job(matrix_job, f"\"sm: 'gpu'\" requires tag 'gpu'."))
        gpu = get_gpu(matrix_job['gpu'])
        matrix_job['sm'] = gpu['sm']

    if 'std' in matrix_job:
        if matrix_job['std'] == 'all':
            matrix_job['std'] = lookup_supported_stds(matrix_job)
        elif matrix_job['std'] == 'min':
            matrix_job['std'] = min(lookup_supported_stds(matrix_job))
        elif matrix_job['std'] == 'max':
            matrix_job['std'] = max(lookup_supported_stds(matrix_job))
        elif matrix_job['std'] == 'minmax':
            stds = lookup_supported_stds(matrix_job)
            if len(stds) == 1:
                matrix_job['std'] = stds[0]
            else:
                matrix_job['std'] = [min(stds), max(stds)]


    # Add all deps before applying project job maps:
    for job in matrix_job['jobs']:
        job_info = get_job_type_info(job)
        dep = job_info['needs']
        if dep and dep not in matrix_job['jobs']:
            matrix_job['jobs'].append(dep)

    # Apply project job map:
    project = get_project(matrix_job['project'])
    for original_job, expanded_jobs in project['job_map'].items():
        if original_job in matrix_job['jobs']:
            matrix_job['jobs'].remove(original_job)
            matrix_job['jobs'] += expanded_jobs


def next_explode_tag(matrix_job):
    non_exploded_tags = ['jobs']

    for tag in matrix_job:
        if not tag in non_exploded_tags and isinstance(matrix_job[tag], list):
            return tag
    return None


def explode_tags(matrix_job, explode_tag=None):
    if not explode_tag:
        explode_tag = next_explode_tag(matrix_job)

    if not explode_tag:
        return [matrix_job]

    result = []
    for value in matrix_job[explode_tag]:
        new_job = copy.deepcopy(matrix_job)
        new_job[explode_tag] = value
        result.extend(explode_tags(new_job))

    return result


def preprocess_matrix_jobs(matrix_jobs, is_exclusion_matrix=False):
    result = []
    if is_exclusion_matrix:
        for matrix_job in matrix_jobs:
            validate_tags(matrix_job, ignore_required=True)
            for job in explode_tags(matrix_job):
                canonicalize_tags(job)
                result.append(job)
    else:
        for matrix_job in matrix_jobs:
            validate_tags(matrix_job)
            set_default_tags(matrix_job)
            for job in explode_tags(matrix_job):
                canonicalize_tags(job)
                set_derived_tags(job)
                # The derived tags may need to be exploded again:
                result.extend(explode_tags(job))
    return result


def parse_workflow_matrix_jobs(args, workflow_name):
    # Special handling for exclusion matrix: don't validate, add default, etc. Only explode.
    is_exclusion_matrix = (workflow_name == 'exclude')

    if not workflow_name in matrix_yaml['workflows']:
        if (is_exclusion_matrix):  # Valid, no exclusions if not defined
            return []
        raise Exception(f"Workflow '{workflow_name}' not found in matrix file '{matrix_yaml['filename']}'")

    matrix_jobs = matrix_yaml['workflows'][workflow_name]
    if not matrix_jobs or len(matrix_jobs) == 0:
        return []

    workflow_line_number = find_workflow_line_number(workflow_name)

    # Tag with the original matrix info, location, etc. for error messages and post-processing.
    # Do this first so the original tags / order /idx match the inpt object exactly.
    if not is_exclusion_matrix:
        for idx, matrix_job in enumerate(matrix_jobs):
            workflow_location = f"{matrix_yaml['filename']}:{workflow_line_number} (job {idx + 1})"
            matrix_job['origin'] = get_matrix_job_origin(matrix_job, workflow_name, workflow_location)

    # Fill in default values, explode lists.
    matrix_jobs = preprocess_matrix_jobs(matrix_jobs, is_exclusion_matrix)

    if args:
        if args.dirty_projects != None:  # Explicitly check for None, as an empty list is valid:
            matrix_jobs = [job for job in matrix_jobs if job['project'] in args.dirty_projects]

    # Don't remove excluded jobs if we're currently parsing them:
    if not is_exclusion_matrix:
        matrix_jobs = remove_excluded_jobs(matrix_jobs)

    # Sort the tags by, *ahem*, "importance":
    sorted_tags = get_all_matrix_job_tags_sorted()
    matrix_jobs = [{tag: matrix_job[tag] for tag in sorted_tags if tag in matrix_job} for matrix_job in matrix_jobs]

    return matrix_jobs


def parse_workflow_dispatch_groups(args, workflow_name):
    # Add origin information to each matrix job, explode, filter, add defaults, etc.
    # The resulting matrix_jobs list is a complete and standardized list of jobs for the dispatch_group builder.
    matrix_jobs = parse_workflow_matrix_jobs(args, workflow_name)

    # If we're printing multiple workflows, add a prefix to the group name to differentiate them.
    group_prefix = f"[{workflow_name}] " if len(args.workflows) > 1 else ""

    # Convert the matrix jobs into a dispatch group object:
    workflow_dispatch_groups = {}
    for matrix_job in matrix_jobs:
        matrix_job_dispatch_group = matrix_job_to_dispatch_group(matrix_job, group_prefix)
        merge_dispatch_groups(workflow_dispatch_groups, matrix_job_dispatch_group)

    return workflow_dispatch_groups


def write_outputs(final_workflow):
    job_list = []
    runner_counts = {}
    id_to_full_job_name = {}

    total_jobs = 0

    def process_job_array(group_name, array_name, parent_json):
        nonlocal job_list
        nonlocal runner_counts
        nonlocal total_jobs

        job_array = parent_json[array_name] if array_name in parent_json else []
        for job_json in job_array:
            total_jobs += 1
            job_list.append(f"{total_jobs:4} id: {job_json['id']:<4}   {array_name:13} {job_json['name']}")
            id_to_full_job_name[job_json['id']] = f"{group_name} {job_json['name']}"
            runner = job_json['runner']
            runner_counts[runner] = runner_counts.get(runner, 0) + 1

    for group_name, group_json in final_workflow.items():
        job_list.append(f"{'':4} {group_name}:")
        process_job_array(group_name, 'standalone', group_json)
        if 'two_stage' in group_json:
            for two_stage_json in group_json['two_stage']:
                process_job_array(group_name, 'producers', two_stage_json)
                process_job_array(group_name, 'consumers', two_stage_json)

    # Sort by descending counts:
    runner_counts = {k: v for k, v in sorted(runner_counts.items(), key=lambda item: item[1], reverse=True)}

    runner_heading = f"🏃‍ Runner counts (total jobs: {total_jobs})"

    runner_counts_table = f"| {'#':^4} | Runner\n"
    runner_counts_table += "|------|------\n"
    for runner, count in runner_counts.items():
        runner_counts_table += f"| {count:4} | `{runner}`\n"

    runner_json = {"heading": runner_heading, "body": runner_counts_table}

    os.makedirs("workflow", exist_ok=True)
    write_json_file("workflow/workflow.json", final_workflow)
    write_json_file("workflow/job_ids.json", id_to_full_job_name)
    write_text_file("workflow/job_list.txt", "\n".join(job_list))
    write_json_file("workflow/runner_summary.json", runner_json)


def write_override_matrix(override_matrix):
    os.makedirs("workflow", exist_ok=True)
    write_json_file("workflow/override.json", override_matrix)


def print_gha_workflow(args):
    workflow_names = args.workflows
    if args.allow_override and 'override' in matrix_yaml['workflows']:
        override_matrix = matrix_yaml['workflows']['override']
        if override_matrix and len(override_matrix) > 0:
            print(f"::notice::Using 'override' workflow instead of '{workflow_names}'")
            workflow_names = ['override']
            write_override_matrix(override_matrix)

    final_workflow = {}
    for workflow_name in workflow_names:
        workflow_dispatch_groups = parse_workflow_dispatch_groups(args, workflow_name)
        merge_dispatch_groups(final_workflow, workflow_dispatch_groups)

    final_workflow = finalize_workflow_dispatch_groups(final_workflow)

    write_outputs(final_workflow)


def print_devcontainer_info(args):
    devcontainer_version = matrix_yaml['devcontainer_version']

    matrix_jobs = []

    # Remove the `exclude` and `override` entries:
    ignored_matrix_keys = ['exclude', 'override']
    workflow_names = [key for key in matrix_yaml['workflows'].keys() if key not in ignored_matrix_keys]
    for workflow_name in workflow_names:
        matrix_jobs.extend(parse_workflow_matrix_jobs(args, workflow_name))

    # Remove all but the following keys from the matrix jobs:
    keep_keys = ['ctk', 'cxx']
    combinations = [{key: job[key] for key in keep_keys} for job in matrix_jobs]

    # Remove duplicates and filter out windows jobs:
    unique_combinations = []
    for combo in combinations:
        if not is_windows(combo) and combo not in unique_combinations:
            unique_combinations.append(combo)

    for combo in unique_combinations:
        host_compiler = get_host_compiler(combo['cxx'])
        del combo['cxx']
        combo['compiler_name'] = host_compiler['container_tag']
        combo['compiler_version'] = host_compiler['version']
        combo['compiler_exe'] = host_compiler['exe']

        combo['cuda'] = combo['ctk']
        del combo['ctk']

    devcontainer_json = {'devcontainer_version': devcontainer_version, 'combinations': unique_combinations}

    # Pretty print the devcontainer json to stdout:
    print(json.dumps(devcontainer_json, indent=2))


def preprocess_matrix_yaml(matrix):
    # Make all CTK version keys into strings:
    new_ctk = {}
    for version, attrs in matrix['ctk_versions'].items():
        new_ctk[str(version)] = attrs
    matrix['ctk_versions'] = new_ctk

    # Make all compiler version keys into strings:
    for id, hc_def in matrix['host_compilers'].items():
        new_versions = {}
        for version, attrs in hc_def['versions'].items():
            new_versions[str(version)] = attrs
        hc_def['versions'] = new_versions

    return matrix


def main():
    parser = argparse.ArgumentParser(description='Compute matrix for workflow')
    parser.add_argument('matrix_file', help='Path to the matrix YAML file')
    parser_mode_group = parser.add_argument_group('Output Mode', "Must specify one of these options.")
    parser_mode = parser_mode_group.add_mutually_exclusive_group(required=True)
    parser_mode.add_argument('--workflows', nargs='+',
                             help='Print GHA workflow with jobs from [pull_request, nightly, weekly, etc]')
    parser_mode.add_argument('--devcontainer-info', action='store_true',
                             help='Print devcontainer info instead of GHA workflows.')
    parser.add_argument('--dirty-projects', nargs='*', help='Filter jobs to only these projects')
    parser.add_argument('--allow-override', action='store_true',
                        help='If a non-empty "override" workflow exists, it will be used instead of those in --workflows.')
    args = parser.parse_args()

    # Check if the matrix file exists
    if not os.path.isfile(args.matrix_file):
        print(f"Error: Matrix file '{args.matrix_file}' does not exist.")
        sys.exit(1)

    with open(args.matrix_file, 'r') as f:
        global matrix_yaml
        matrix_yaml = yaml.safe_load(f)
        matrix_yaml = preprocess_matrix_yaml(matrix_yaml)
        matrix_yaml['filename'] = args.matrix_file

    if args.workflows:
        print_gha_workflow(args)
    elif args.devcontainer_info:
        print_devcontainer_info(args)
    else:
        parser.print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
