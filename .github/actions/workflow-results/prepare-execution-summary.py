#!/usr/bin/env python3


import argparse
import json
import os
import re
import sys


def job_succeeded(job):
    # The job was successful if the artifact file 'dispatch-job-success/dispatch-job-success-<job_id>' exists:
    return os.path.exists(f'dispatch-job-success/{job["id"]}')


def natural_sort_key(key):
    # Natural sort impl (handles embedded numbers in strings, case insensitive)
    return [(int(text) if text.isdigit() else text.lower()) for text in re.split('(\d+)', key)]


# Print the prepared text summary to the file at the given path
def write_text(filepath, summary):
    with open(filepath, 'w') as f:
        print(summary, file=f)


# Print the prepared JSON object to the file at the given path
def write_json(filepath, json_object):
    with open(filepath, 'w') as f:
        json.dump(json_object, f, indent=4)


def extract_jobs(workflow):
    jobs = []
    for group_name, group in workflow.items():
        if "standalone" in group:
            jobs += group["standalone"]
        if "two_stage" in group:
            for two_stage in group["two_stage"]:
                jobs += two_stage["producers"]
                jobs += two_stage["consumers"]
    return jobs


def build_summary(jobs):
    summary = {'passed': 0, 'failed': 0, 'projects': {}}
    projects = summary['projects']

    for job in jobs:
        success = job_succeeded(job)

        if success:
            summary['passed'] += 1
        else:
            summary['failed'] += 1

        matrix_job = job["origin"]["matrix_job"]

        project = matrix_job["project"]
        if not project in projects:
            projects[project] = {'passed': 0, 'failed': 0, 'tags': {}}

        if success:
            projects[project]['passed'] += 1
        else:
            projects[project]['failed'] += 1

        tags = projects[project]['tags']
        for tag in matrix_job.keys():
            if tag == 'project':
                continue

            if not tag in tags:
                tags[tag] = {'passed': 0, 'failed': 0, 'values': {}}

            value = str(matrix_job[tag])
            values = tags[tag]['values']

            if not value in values:
                values[value] = {'passed': 0, 'failed': 0}

            if success:
                tags[tag]['passed'] += 1
                values[value]['passed'] += 1
            else:
                tags[tag]['failed'] += 1
                values[value]['failed'] += 1

    # Natural sort the value strings within each tag:
    for project, project_summary in projects.items():
        for tag, tag_summary in project_summary['tags'].items():
            tag_summary['values'] = dict(sorted(tag_summary['values'].items(),
                                         key=lambda item: natural_sort_key(item[0])))

    # Sort the tags within each project so that:
    # - "Likely culprits" come first. These are tags that have multiple values, but only one has failures.
    # - The remaining tags with failures come next.
    # - Tags with no failures come last.
    def rank_tag(tag_summary):
        num_failing_values = sum(1 for value_summary in tag_summary['values'].values() if value_summary['failed'] > 0)

        if len(tag_summary['values']) > 1 and num_failing_values == 1:
            return 0
        elif len(tag_summary['values']) > 1 and tag_summary['failed'] > 0:
            return 1
        elif tag_summary['failed'] > 0:
            return 2
        return 3
    for project, project_summary in projects.items():
        project_summary['tags'] = dict(sorted(project_summary['tags'].items(),
                                       key=lambda item: (rank_tag(item[1]), item[0])))

    return summary


def get_summary_heading(summary):
    passed = summary['passed']
    failed = summary['failed']
    total = passed + failed

    if passed == 0:
        flag = '🟥'
    elif failed > 0:
        flag = '🟨'
    else:
        flag = '🟩'

    return f'{flag} CI Results [ Failed: {failed} | Passed: {passed} | Total: {total} ]'


def get_project_heading(project, project_summary):
    if project_summary['passed'] == 0:
        flag = '🟥'
    elif project_summary['failed'] > 0:
        flag = '🟨'
    else:
        flag = '🟩'

    passed = project_summary['passed']
    failed = project_summary['failed']
    total = project_summary['failed'] + project_summary['passed']

    return f'{flag} Project {project} [ Failed: {failed} | Passed: {passed} | Total: {total} ]'


def get_tag_line(tag, tag_summary):
    passed = tag_summary['passed']
    failed = tag_summary['failed']
    values = tag_summary['values']

    # Find the value with an failure rate that matches the tag's failure rate:
    suspicious = None
    if len(values) > 1 and failed > 0:
        for value, value_summary in values.items():
            if value_summary['failed'] == failed:
                suspicious = value_summary
                suspicious['name'] = value
                break

    # Did any jobs with this value pass?
    likely_culprit = suspicious if suspicious and suspicious['passed'] == 0 else None

    note = ''
    if likely_culprit:
        flag = '🚨'
        note = f': {likely_culprit["name"]} {flag}'
    elif suspicious:
        flag = '🔍'
        note = f': {suspicious["name"]} {flag}'
    elif passed == 0:
        flag = '🟥'
    elif failed > 0:
        flag = '🟨'
    else:
        flag = '🟩'

    return f'{flag} {tag}{note}'


def get_value_line(value, value_summary, tag_summary):
    passed = value_summary['passed']
    failed = value_summary['failed']
    total = passed + failed

    parent_size = len(tag_summary['values'])
    parent_failed = tag_summary['failed']

    is_suspicious = failed > 0 and failed == parent_failed and parent_size > 1
    is_likely_culprit = is_suspicious and passed == 0

    if is_likely_culprit:
        flag = '🔥'
    elif is_suspicious:
        flag = '🔍'
    elif passed == 0:
        flag = '🟥'
    elif failed > 0:
        flag = '🟨'
    else:
        flag = '🟩'

    percent = int(100 * failed / total)
    left_aligned = f"{flag} {value} ({percent}% Fail)"
    return f'  {left_aligned:<30} Failed: {failed:^3} -- Passed: {passed:^3} -- Total: {total:^3}'


def get_project_summary_body(project, project_summary):
    body = ['```']
    for tag, tag_summary in project_summary['tags'].items():
        body.append(get_tag_line(tag, tag_summary))
        for value, value_summary in tag_summary['values'].items():
            body.append(get_value_line(value, value_summary, tag_summary))
    body.append('```')
    return "\n".join(body)


def write_project_summary(project, project_summary):
    heading = get_project_heading(project, project_summary)
    body = get_project_summary_body(project, project_summary)

    summary = {'heading': heading, 'body': body}

    write_json(f'execution/projects/{project}_summary.json', summary)


def write_workflow_summary(workflow):
    summary = build_summary(extract_jobs(workflow))

    os.makedirs('execution/projects', exist_ok=True)

    write_text('execution/heading.txt', get_summary_heading(summary))

    for project, project_summary in summary['projects'].items():
        write_project_summary(project, project_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workflow', type=argparse.FileType('r'))
    args = parser.parse_args()

    workflow = json.load(args.workflow)
    write_workflow_summary(workflow)


if __name__ == '__main__':
    main()
