#!/usr/bin/env python3

"""
This script prepares a full workflow for GHA dispatch.

To avoid skipped jobs from cluttering the GHA UI, this script splits the full workflow.json into multiple workflows
that don't require large numbers of skipped jobs in the workflow implementation.
"""

import argparse
import json
import os
import sys


def write_json_file(filename, json_object):
    with open(filename, 'w') as f:
        json.dump(json_object, f, indent=2)


def is_windows(job):
    return job['runner'].startswith('windows')


def split_workflow(workflow):
    linux_standalone = {}
    linux_two_stage = {}
    windows_standalone = {}
    windows_two_stage = {}

    def strip_extra_info(job):
        del job['origin']

    for group_name, group_json in workflow.items():
        standalone = group_json['standalone'] if 'standalone' in group_json else []
        two_stage = group_json['two_stage'] if 'two_stage' in group_json else []

        if len(standalone) > 0:
            for job in standalone:
                strip_extra_info(job)

            if is_windows(standalone[0]):
                windows_standalone[group_name] = standalone
            else:
                linux_standalone[group_name] = standalone

        if len(two_stage) > 0:
            for ts in two_stage:
                for job in ts['producers']:
                    strip_extra_info(job)
                for job in ts['consumers']:
                    strip_extra_info(job)

            if is_windows(two_stage[0]['producers'][0]):
                windows_two_stage[group_name] = two_stage
            else:
                linux_two_stage[group_name] = two_stage

    dispatch = {
        'linux_standalone': {
            'keys': list(linux_standalone.keys()),
            'jobs': linux_standalone},
        'linux_two_stage': {
            'keys': list(linux_two_stage.keys()),
            'jobs': linux_two_stage},
        'windows_standalone': {
            'keys': list(windows_standalone.keys()),
            'jobs': windows_standalone},
        'windows_two_stage': {
            'keys': list(windows_two_stage.keys()),
            'jobs': windows_two_stage}
    }

    os.makedirs('workflow', exist_ok=True)
    write_json_file('workflow/dispatch.json', dispatch)


def main():
    parser = argparse.ArgumentParser(description='Prepare a full workflow for GHA dispatch.')
    parser.add_argument('workflow_json', help='Path to the full workflow.json file')
    args = parser.parse_args()

    # Check if the workflow file exists
    if not os.path.isfile(args.workflow_json):
        print(f"Error: Matrix file '{args.workflow_json}' not found.")
        sys.exit(1)

    with open(args.workflow_json) as f:
        workflow = json.load(f)

    split_workflow(workflow)


if __name__ == '__main__':
    main()
