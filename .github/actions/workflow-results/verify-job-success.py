#!/usr/bin/env python3

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id_map", type=argparse.FileType('r'))
    args = parser.parse_args()

    job_id_map = json.load(args.job_id_map)

    # For each job id, verify that the success artifact exists
    success = True
    for job_id, job_name in job_id_map.items():
        success_file = f'jobs/{job_id}/success'
        print(f'Verifying job with id "{job_id}": "{job_name}"')
        if not os.path.exists(success_file):
            print(f'Failed: Artifact "{success_file}" not found')
            success = False

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
