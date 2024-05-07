#!/usr/bin/env python3

"""
Concepts:
- matrix_job: an entry of a workflow matrix, converted from matrix.yaml["workflow"][id] into a JSON object.
  Example:
  {
    "jobs": [
      "test"
    ],
    "ctk": "11.1",
    "gpu": "t4",
    "sm": "75-real",
    "cxx": {
      "name": "llvm",
      "version": "9",
      "exe": "clang++"
    },
    "std": [
      17
    ],
    "project": [
      "libcudacxx",
      "cub",
      "thrust"
    ],
    "os": "ubuntu18.04"
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
import json
import os
import re
import struct
import sys
import yaml

matrix_yaml = None


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


def get_all_matrix_job_tags_sorted():
    required_tags = set(matrix_yaml['required_tags'])
    defaulted_tags = set(matrix_yaml['defaulted_tags'])
    optional_tags = set(matrix_yaml['optional_tags'])
    all_tags = required_tags | defaulted_tags | optional_tags

    # Sorted using a highly subjective opinion on importance:
    # Always first, information dense:
    sorted_important_tags = ['project', 'jobs', 'cudacxx', 'cxx', 'ctk', 'gpu', 'std', 'sm', 'cpu']

    # Always last, derived:
    sorted_noise_tags = ['os', 'origin']

    # In between?
    sorted_tags = set(sorted_important_tags + sorted_noise_tags)
    sorted_meh_tags = sorted(list(all_tags - sorted_tags))

    return sorted_important_tags + sorted_meh_tags + sorted_noise_tags


def lookup_os(ctk, host_compiler):
    key = f'ctk{ctk}-{host_compiler["name"]}{host_compiler["version"]}'
    if not key in matrix_yaml['default_os_lookup']:
        raise Exception(f"Missing matrix.yaml `default_os_lookup` entry for key `{key}`")
    return matrix_yaml['default_os_lookup'][key]


def lookup_supported_stds(device_compiler=None, host_compiler=None):
    stds = set(matrix_yaml['all_stds'])
    if device_compiler:
        key = f"{device_compiler['name']}{device_compiler['version']}"
        if not key in matrix_yaml['lookup_cudacxx_supported_stds']:
            raise Exception(f"Missing matrix.yaml 'lookup_cudacxx_supported_stds' entry for key '{key}'")
        stds = stds & set(matrix_yaml['lookup_cudacxx_supported_stds'][key])
    if host_compiler:
        key = f"{host_compiler['name']}{host_compiler['version']}"
        if not key in matrix_yaml['lookup_cxx_supported_stds']:
            raise Exception(f"Missing matrix.yaml 'lookup_cxx_supported_stds' entry for key '{key}'")
        stds = stds & set(matrix_yaml['lookup_cxx_supported_stds'][key])
    return sorted(list(stds))


def get_formatted_project_name(project_name):
    if project_name in matrix_yaml['formatted_project_names']:
        return matrix_yaml['formatted_project_names'][project_name]
    return project_name


def get_formatted_host_compiler_name(host_compiler):
    config_name = host_compiler['name']
    if config_name in matrix_yaml['formatted_cxx_names']:
        return matrix_yaml['formatted_cxx_names'][config_name]
    return config_name


def get_formatted_job_type(job_type):
    if job_type in matrix_yaml['formatted_jobs']:
        return matrix_yaml['formatted_jobs'][job_type]
    # Return with first letter capitalized:
    return job_type.capitalize()


def is_windows(matrix_job):
    return matrix_job['os'].startswith('windows')


def generate_dispatch_group_name(matrix_job):
    project_name = get_formatted_project_name(matrix_job['project'])
    ctk = matrix_job['ctk']
    device_compiler = matrix_job['cudacxx']
    host_compiler_name = get_formatted_host_compiler_name(matrix_job['cxx'])

    compiler_info = ""
    if device_compiler['name'] == 'nvcc':
        compiler_info = f"nvcc {host_compiler_name}"
    elif device_compiler['name'] == 'llvm':
        compiler_info = f"clang-cuda"
    else:
        compiler_info = f"{device_compiler['name']}-{device_compiler['version']} {host_compiler_name}"

    return f"{project_name} {compiler_info} CTK{ctk}"


def generate_dispatch_job_name(matrix_job, job_type):
    std_str = ("C++" + str(matrix_job['std']) + " ") if 'std' in matrix_job else ''
    cpu_str = matrix_job['cpu']
    gpu_str = (', ' + matrix_job['gpu'].upper()) if job_type in matrix_yaml['gpu_required_jobs'] else ""
    cuda_compile_arch = (" sm{" + matrix_job['sm'] + "}") if 'sm' in matrix_job else ""
    cmake_options = (' ' + matrix_job['cmake_options']) if 'cmake_options' in matrix_job else ""

    host_compiler_name = get_formatted_host_compiler_name(matrix_job['cxx'])
    host_compiler_info = f"{host_compiler_name}{matrix_job['cxx']['version']}"

    config_tag = f"{std_str}{host_compiler_info}"

    formatted_job_type = get_formatted_job_type(job_type)

    extra_info = f":{cuda_compile_arch}{cmake_options}" if cuda_compile_arch or cmake_options else ""

    return f"[{config_tag}] {formatted_job_type}({cpu_str}{gpu_str}){extra_info}"


def generate_dispatch_job_runner(matrix_job, job_type):
    runner_os = "windows" if is_windows(matrix_job) else "linux"
    cpu = matrix_job['cpu']

    if not job_type in matrix_yaml['gpu_required_jobs']:
        return f"{runner_os}-{cpu}-cpu16"

    gpu = matrix_job['gpu']
    suffix = "-testing" if gpu in matrix_yaml['testing_pool_gpus'] else ""

    return f"{runner_os}-{cpu}-gpu-{gpu}-latest-1{suffix}"


def generate_dispatch_job_image(matrix_job, job_type):
    devcontainer_version = matrix_yaml['devcontainer_version']
    ctk = matrix_job['ctk']
    image_os = matrix_job['os']
    host_compiler = matrix_job['cxx']['name'] + matrix_job['cxx']['version']

    if is_windows(matrix_job):
        return f"rapidsai/devcontainers:{devcontainer_version}-cuda{ctk}-{host_compiler}-{image_os}"

    return f"rapidsai/devcontainers:{devcontainer_version}-cpp-{host_compiler}-cuda{ctk}-{image_os}"


def generate_dispatch_job_command(matrix_job, job_type):
    script_path = "ci/windows" if is_windows(matrix_job) else "ci"
    script_ext = ".ps1" if is_windows(matrix_job) else ".sh"
    script_job_type = job_type
    script_project = matrix_job['project']
    script_name = f"{script_path}/{script_job_type}_{script_project}{script_ext}"

    std_str = str(matrix_job['std']) if 'std' in matrix_job else ''

    host_compiler_exe = matrix_job['cxx']['exe']
    device_compiler_name = matrix_job['cudacxx']['name']
    device_compiler_exe = matrix_job['cudacxx']['exe']

    cuda_compile_arch = matrix_job['sm'] if 'sm' in matrix_job else ''
    cmake_options = matrix_job['cmake_options'] if 'cmake_options' in matrix_job else ''

    command = f"\"{script_name}\""
    if std_str:
        command += f" -std \"{std_str}\""
    if cuda_compile_arch:
        command += f" -arch \"{cuda_compile_arch}\""
    if device_compiler_name != 'nvcc':
        command += f" -cuda \"{device_compiler_exe}\""
    if cmake_options:
        command += f" -cmake-options \"{cmake_options}\""

    return command


def generate_dispatch_job_origin(matrix_job, job_type):
    origin = matrix_job['origin'].copy()

    matrix_job = matrix_job.copy()
    del matrix_job['origin']

    matrix_job['jobs'] = job_type

    if 'cxx' in matrix_job:
        host_compiler = matrix_job['cxx']
        formatted_name = get_formatted_host_compiler_name(host_compiler)
        matrix_job['cxx_name'] = formatted_name
        matrix_job['cxx_full'] = formatted_name + host_compiler['version']
        del matrix_job['cxx']

    if 'cudacxx' in matrix_job:
        device_compiler = matrix_job['cudacxx']
        formatted_name = 'clang-cuda' if device_compiler['name'] == 'llvm' else device_compiler['name']
        matrix_job['cudacxx_name'] = formatted_name
        matrix_job['cudacxx_full'] = formatted_name + device_compiler['version']
        del matrix_job['cudacxx']

    origin['matrix_job'] = matrix_job

    return origin


def generate_dispatch_job_json(matrix_job, job_type):
    return {
        'name': generate_dispatch_job_name(matrix_job, job_type),
        'runner': generate_dispatch_job_runner(matrix_job, job_type),
        'image': generate_dispatch_job_image(matrix_job, job_type),
        'command': generate_dispatch_job_command(matrix_job, job_type),
        'origin': generate_dispatch_job_origin(matrix_job, job_type)
    }


# Create a single build producer, and a separate consumer for each test_job_type:
def generate_dispatch_build_and_test_json(matrix_job, build_job_type, test_job_types):
    build_json = generate_dispatch_job_json(matrix_job, build_job_type)

    test_json = []
    for test_job_type in test_job_types:
        test_json.append(generate_dispatch_job_json(matrix_job, test_job_type))

    return {
        "producers": [build_json],
        "consumers": test_json
    }


def generate_dispatch_group_jobs(matrix_job):
    dispatch_group_jobs = {
        "standalone": [],
        "two_stage": []
    }

    job_types = set(matrix_job['jobs'])

    build_required = set(matrix_yaml['build_required_jobs']) & job_types
    has_build_and_test = len(build_required) > 0
    job_types -= build_required

    has_standalone_build = 'build' in job_types and not has_build_and_test
    job_types -= {'build'}

    if has_standalone_build:
        dispatch_group_jobs['standalone'].append(generate_dispatch_job_json(matrix_job, "build"))
    elif has_build_and_test:
        dispatch_group_jobs['two_stage'].append(
            generate_dispatch_build_and_test_json(matrix_job, "build", build_required))

    # Remaining jobs are assumed to be standalone (e.g. nvrtc):
    for job_type in job_types:
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


def finalize_workflow_dispatch_groups(workflow_dispatch_groups_orig):
    workflow_dispatch_groups = copy.deepcopy(workflow_dispatch_groups_orig)

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
            if producers in merged_producers:
                producer_index = merged_producers.index(producers)
                matching_consumers = merged_consumers[producer_index]

                producer_names = ", ".join([producer['name'] for producer in producers])
                print(f"::notice file=ci/matrix.yaml::Merging consumers for duplicate producer '{producer_names}' in '{group_name}'",
                      file=sys.stderr)
                consumer_names = ", ".join([consumer['name'] for consumer in matching_consumers])
                print(f"::notice file=ci/matrix.yaml::Original consumers: {consumer_names}", file=sys.stderr)
                consumer_names = ", ".join([consumer['name'] for consumer in consumers])
                print(f"::notice file=ci/matrix.yaml::Duplicate consumers: {consumer_names}", file=sys.stderr)
                # Merge if unique:
                for consumer in consumers:
                    if consumer not in matching_consumers:
                        matching_consumers.append(consumer)
                consumer_names = ", ".join([consumer['name'] for consumer in matching_consumers])
                print(f"::notice file=ci/matrix.yaml::Merged consumers: {consumer_names}", file=sys.stderr)
            else:
                merged_producers.append(producers)
                merged_consumers.append(consumers)
        # Update with the merged lists:
        two_stage_json = []
        for producers, consumers in zip(merged_producers, merged_consumers):
            two_stage_json.append({'producers': producers, 'consumers': consumers})
        group_json['two_stage'] = two_stage_json

    # Check for any duplicate jobs in standalone arrays. Warn and remove duplicates.
    for group_name, group_json in workflow_dispatch_groups.items():
        standalone_jobs = group_json['standalone'] if 'standalone' in group_json else []
        unique_standalone_jobs = []
        for job_json in standalone_jobs:
            if job_json in unique_standalone_jobs:
                print(f"::notice file=ci/matrix.yaml::Removing duplicate standalone job '{job_json['name']}' in '{group_name}'",
                      file=sys.stderr)
            else:
                unique_standalone_jobs.append(job_json)

        # If any producer/consumer jobs exist in standalone arrays, warn and remove the standalones.
        two_stage_jobs = group_json['two_stage'] if 'two_stage' in group_json else []
        for two_stage_job in two_stage_jobs:
            for producer in two_stage_job['producers']:
                if producer in unique_standalone_jobs:
                    print(f"::notice file=ci/matrix.yaml::Removing standalone job '{producer['name']}' " +
                          f"as it appears as a producer in '{group_name}'",
                          file=sys.stderr)
                    unique_standalone_jobs.remove(producer)
            for consumer in two_stage_job['consumers']:
                if consumer in unique_standalone_jobs:
                    print(f"::notice file=ci/matrix.yaml::Removing standalone job '{consumer['name']}' " +
                          f"as it appears as a consumer in '{group_name}'",
                          file=sys.stderr)
                    unique_standalone_jobs.remove(consumer)
        standalone_jobs = list(unique_standalone_jobs)

        # If any producer or consumer job appears more than once, warn and leave as-is.
        all_two_stage_jobs = []
        duplicate_jobs = {}
        for two_stage_job in two_stage_jobs:
            for job in two_stage_job['producers'] + two_stage_job['consumers']:
                if job in all_two_stage_jobs:
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


def remove_skip_test_jobs(matrix_jobs):
    '''Remove jobs defined in `matrix_file.skip_test_jobs`.'''
    new_matrix_jobs = []
    for matrix_job in matrix_jobs:
        jobs = matrix_job['jobs']
        new_jobs = set()
        for job in jobs:
            if job in matrix_yaml['skip_test_jobs']:
                # If a skipped test job is a build_required_job, replace it with the 'build' job.
                if job in matrix_yaml['build_required_jobs']:
                    # Replace with the prerequisite build job:
                    new_jobs.add('build')
                    # If a skipped test job is not a build_required_job, ignore it.
                else:
                    pass  # Ignore the job
            else:
                new_jobs.add(job)
        # If no jobs remain, skip this matrix job.
        if new_jobs:
            new_matrix_job = copy.deepcopy(matrix_job)
            new_matrix_job['jobs'] = list(new_jobs)
            new_matrix_jobs.append(new_matrix_job)
    return new_matrix_jobs


def validate_required_tags(matrix_job):
    for tag in matrix_yaml['required_tags']:
        if tag not in matrix_job:
            raise Exception(error_message_with_matrix_job(matrix_job, f"Missing required tag '{tag}'"))

    all_tags = get_all_matrix_job_tags_sorted()
    for tag in matrix_job:
        if tag not in all_tags:
            raise Exception(error_message_with_matrix_job(matrix_job, f"Unknown tag '{tag}'"))

    if 'gpu' in matrix_job and matrix_job['gpu'] not in matrix_yaml['gpus']:
        raise Exception(error_message_with_matrix_job(matrix_job, f"Unknown gpu '{matrix_job['gpu']}'"))


def set_default_tags(matrix_job):
    generic_defaults = set(matrix_yaml['defaulted_tags'])
    generic_defaults -= set(['os'])  # handled specially.

    for tag in generic_defaults:
        if tag not in matrix_job:
            matrix_job[tag] = matrix_yaml['default_'+tag]


def set_derived_tags(matrix_job):
    if 'os' not in matrix_job:
        matrix_job['os'] = lookup_os(matrix_job['ctk'], matrix_job['cxx'])

    # Expand nvcc device compiler shortcut:
    if matrix_job['cudacxx'] == 'nvcc':
        matrix_job['cudacxx'] = {'name': 'nvcc', 'version': matrix_job['ctk'], 'exe': 'nvcc'}

    if 'sm' in matrix_job and matrix_job['sm'] == 'gpu':
        if not 'gpu' in matrix_job:
            raise Exception(error_message_with_matrix_job(matrix_job, f"\"sm: 'gpu'\" requires tag 'gpu'."))
        if not matrix_job['gpu'] in matrix_yaml['gpu_sm']:
            raise Exception(error_message_with_matrix_job(matrix_job,
                                                          f"Missing matrix.yaml 'gpu_sm' entry for gpu '{matrix_job['gpu']}'"))
        matrix_job['sm'] = matrix_yaml['gpu_sm'][matrix_job['gpu']]

    if 'std' in matrix_job and matrix_job['std'] == 'all':
        host_compiler = matrix_job['cxx'] if 'cxx' in matrix_job else None
        device_compiler = matrix_job['cudacxx'] if 'cudacxx' in matrix_job else None
        matrix_job['std'] = lookup_supported_stds(device_compiler, host_compiler)


def next_explode_tag(matrix_job):
    for tag in matrix_job:
        if not tag in matrix_yaml['non_exploded_tags'] and isinstance(matrix_job[tag], list):
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


def preprocess_matrix_jobs(matrix_jobs):
    result = []
    for matrix_job in matrix_jobs:
        validate_required_tags(matrix_job)
        set_default_tags(matrix_job)
        for job in explode_tags(matrix_job):
            set_derived_tags(job)
            # The derived tags may need to be exploded again:
            result.extend(explode_tags(job))
    return result


def parse_workflow_matrix_jobs(args, workflow_name):
    if not workflow_name in matrix_yaml['workflows']:
        raise Exception(f"Workflow '{workflow_name}' not found in matrix file '{matrix_yaml['filename']}'")

    matrix_jobs = matrix_yaml['workflows'][workflow_name]
    workflow_line_number = find_workflow_line_number(workflow_name)

    # Tag with the original matrix info, location, etc. for error messages and post-processing.
    # Do this first so the original tags / order /idx match the inpt object exactly.
    for idx, matrix_job in enumerate(matrix_jobs):
        workflow_location = f"{matrix_yaml['filename']}:{workflow_line_number} (job {idx + 1})"
        matrix_job['origin'] = get_matrix_job_origin(matrix_job, workflow_name, workflow_location)

    # Fill in default values, explode lists.
    matrix_jobs = preprocess_matrix_jobs(matrix_jobs)

    if args.skip_tests:
        matrix_jobs = remove_skip_test_jobs(matrix_jobs)
    if args.dirty_projects:
        matrix_jobs = [job for job in matrix_jobs if job['project'] in args.dirty_projects]

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

    return finalize_workflow_dispatch_groups(workflow_dispatch_groups)


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
    write_json_file("workflow/workflow_keys.json", list(final_workflow.keys()))
    write_json_file("workflow/job_ids.json", id_to_full_job_name)
    write_text_file("workflow/job_list.txt", "\n".join(job_list))
    write_json_file("workflow/runner_summary.json", runner_json)


def print_gha_workflow(args):
    final_workflow = {}
    for workflow_name in args.workflows:
        workflow_dispatch_groups = parse_workflow_dispatch_groups(args, workflow_name)
        merge_dispatch_groups(final_workflow, workflow_dispatch_groups)

    write_outputs(final_workflow)


def print_devcontainer_info(args):
    devcontainer_version = matrix_yaml['devcontainer_version']

    matrix_jobs = []
    for workflow in matrix_yaml['workflows']:
        matrix_jobs.extend(parse_workflow_matrix_jobs(args, workflow))

    # Remove all but the following keys from the matrix jobs:
    keep_keys = ['ctk', 'cxx', 'os']
    combinations = [{key: job[key] for key in keep_keys} for job in matrix_jobs]

    # Remove duplicates and filter out windows jobs:
    unique_combinations = []
    for combo in combinations:
        if not is_windows(combo) and combo not in unique_combinations:
            unique_combinations.append(combo)

    for combo in unique_combinations:
        combo['compiler_name'] = combo['cxx']['name']
        combo['compiler_version'] = combo['cxx']['version']
        combo['compiler_exe'] = combo['cxx']['exe']
        del combo['cxx']

        combo['cuda'] = combo['ctk']
        del combo['ctk']

    devcontainer_json = {'devcontainer_version': devcontainer_version, 'combinations': unique_combinations}

    # Pretty print the devcontainer json to stdout:
    print(json.dumps(devcontainer_json, indent=2))


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
    parser.add_argument('--skip-tests', action='store_true',
                        help='Remove jobs defined in `matrix_file.skip_test_jobs`.')
    args = parser.parse_args()

    # Check if the matrix file exists
    if not os.path.isfile(args.matrix_file):
        print(f"Error: Matrix file '{args.matrix_file}' does not exist.")
        sys.exit(1)

    with open(args.matrix_file, 'r') as f:
        global matrix_yaml
        matrix_yaml = yaml.safe_load(f)
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
