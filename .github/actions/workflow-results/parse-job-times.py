#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import sys


def get_jobs_json(jobs_file):
    # Return the contents of jobs.json
    with open(jobs_file) as f:
        result = json.load(f)

    return result


def get_workflow_json(workflow_file):
    # Return the contents of ~/cccl/.local/tmp/workflow.json
    with open(workflow_file) as f:
        return json.load(f)


def write_json(filepath, json_object):
    with open(filepath, 'w') as f:
        json.dump(json_object, f, indent=4)


def generate_job_id_map(workflow):
    '''Map full job name to job id'''
    job_id_map = {}
    for group_name, group_json in workflow.items():
        standalone = group_json['standalone'] if 'standalone' in group_json else []
        for job in standalone:
            name = f"{group_name} / s.{job['id']} / {job['name']}"
            job_id_map[name] = job['id']
        two_stage = group_json['two_stage'] if 'two_stage' in group_json else []
        for pc in two_stage:
            producers = pc['producers']
            for job in producers:
                name = f"{group_name} / t.{pc['id']} / p.{job['id']} / {job['name']}"
                job_id_map[name] = job['id']
            consumers = pc['consumers']
            for job in consumers:
                name = f"{group_name} / t.{pc['id']} / c.{job['id']} / {job['name']}"
                job_id_map[name] = job['id']

    return job_id_map


def main():
    # Accept two command line arguments: <workflow.json> <jobs.json>
    parser = argparse.ArgumentParser(description='Parse job times')
    parser.add_argument('workflow', type=str, help='Path to workflow.json')
    parser.add_argument('jobs', type=str, help='Path to jobs.json')
    args = parser.parse_args()

    jobs = get_jobs_json(args.jobs)
    workflow = get_workflow_json(args.workflow)

    # Converts full github job names into job ids:
    job_id_map = generate_job_id_map(workflow)

    # Map of id -> { <job stats> }
    result = {}

    for job in jobs:
        name = job['name']

        # Build workflow, verify devcontainers, etc:
        if name not in job_id_map:
            print(f"Skipping {name}")
            continue

        id = job_id_map[name]

        # Job times are 2024-05-09T06:52:20Z
        started_at = job['started_at']
        started_time = datetime.datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ")

        completed_at = job['completed_at']
        completed_time = datetime.datetime.strptime(completed_at, "%Y-%m-%dT%H:%M:%SZ")

        job_seconds = (completed_time - started_time).total_seconds()
        job_duration = str(datetime.timedelta(seconds=job_seconds))

        result[id] = {}
        result[id]['name'] = name
        result[id]['started_at'] = started_at
        result[id]['completed_at'] = completed_at
        result[id]['job_duration'] = job_duration
        result[id]['job_seconds'] = job_seconds

        # Find the "Run command" step and record its duration:
        command_seconds = 0
        for step in job['steps']:
            if step['name'] == "Run command":
                step_started_at = step['started_at']
                step_started_time = datetime.datetime.strptime(step_started_at, "%Y-%m-%dT%H:%M:%SZ")
                step_completed_at = step['completed_at']
                step_completed_time = datetime.datetime.strptime(step_completed_at, "%Y-%m-%dT%H:%M:%SZ")
                command_seconds = (step_completed_time - step_started_time).total_seconds()
                break

        command_duration = str(datetime.timedelta(seconds=command_seconds))

        result[id]['command_seconds'] = command_seconds
        result[id]['command_duration'] = command_duration

    os.makedirs("results", exist_ok=True)
    write_json("results/job_times.json", result)

    print("::group::Job times")
    for id, stats in result.items():
        print(f"{stats['job_duration']:10} {stats['command_duration']:10} {stats['command_seconds'] * 100 / stats['job_seconds']:3f} {stats['name']}")
    print("::endgroup::")


if __name__ == "__main__":
    main()
