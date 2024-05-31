#!/usr/bin/env python3


import argparse
import functools
import json
import os
import re
import sys


def job_succeeded(job):
    # The job was successful if the success file exists:
    return os.path.exists(f'jobs/{job["id"]}/success')


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


@functools.lru_cache(maxsize=None)
def get_sccache_stats(job_id):
    sccache_file = f'jobs/{job_id}/sccache_stats.json'
    if os.path.exists(sccache_file):
        with open(sccache_file) as f:
            return json.load(f)
    return None


def update_summary_entry(entry, job, job_times=None):
    if 'passed' not in entry:
        entry['passed'] = 0
    if 'failed' not in entry:
        entry['failed'] = 0

    if job_succeeded(job):
        entry['passed'] += 1
    else:
        entry['failed'] += 1

    if job_times:
        time_info = job_times[job["id"]]
        job_time = time_info["job_seconds"]
        command_time = time_info["command_seconds"]

        if not 'job_time' in entry:
            entry['job_time'] = 0
        if not 'command_time' in entry:
            entry['command_time'] = 0
        if not 'max_job_time' in entry:
            entry['max_job_time'] = 0

        entry['job_time'] += job_time
        entry['command_time'] += command_time
        entry['max_job_time'] = max(entry['max_job_time'], job_time)

    sccache_stats = get_sccache_stats(job["id"])
    if sccache_stats:
        sccache_stats = sccache_stats['stats']
        requests = sccache_stats.get('compile_requests', 0)
        hits = 0
        if 'cache_hits' in sccache_stats:
            cache_hits = sccache_stats['cache_hits']
            if 'counts' in cache_hits:
                counts = cache_hits['counts']
                for lang, lang_hits in counts.items():
                    hits += lang_hits
        if 'sccache' not in entry:
            entry['sccache'] = {'requests': requests, 'hits': hits}
        else:
            entry['sccache']['requests'] += requests
            entry['sccache']['hits'] += hits

    return entry


def build_summary(jobs, job_times=None):
    summary = {'projects': {}}
    projects = summary['projects']

    for job in jobs:
        update_summary_entry(summary, job, job_times)

        matrix_job = job["origin"]["matrix_job"]

        project = matrix_job["project"]
        if not project in projects:
            projects[project] = {'tags': {}}
        tags = projects[project]['tags']

        update_summary_entry(projects[project], job, job_times)

        for tag in matrix_job.keys():
            if tag == 'project':
                continue

            if not tag in tags:
                tags[tag] = {'values': {}}
            values = tags[tag]['values']

            update_summary_entry(tags[tag], job, job_times)

            value = str(matrix_job[tag])

            if not value in values:
                values[value] = {}
            update_summary_entry(values[value], job, job_times)

    # Natural sort the value strings within each tag:
    for project, project_summary in projects.items():
        for tag, tag_summary in project_summary['tags'].items():
            tag_summary['values'] = dict(sorted(tag_summary['values'].items(),
                                         key=lambda item: natural_sort_key(item[0])))

    # Sort the tags within each project so that:
    # - "Likely culprits" come first. These are tags that have multiple values, but only one has failures.
    # - Tags with multiple values and mixed pass/fail results come next.
    # - Tags with all failing values come next.
    # - Tags with no failures are last.
    def rank_tag(tag_summary):
        tag_failures = tag_summary['failed']
        num_values = len(tag_summary['values'])
        num_failing_values = sum(1 for value_summary in tag_summary['values'].values() if value_summary['failed'] > 0)

        if num_values > 1:
            if num_failing_values == 1:
                return 0
            elif num_failing_values > 0 and num_failing_values < num_values:
                return 1
        elif tag_failures > 0:
            return 2
        return 3
    for project, project_summary in projects.items():
        project_summary['tags'] = dict(sorted(project_summary['tags'].items(),
                                       key=lambda item: (rank_tag(item[1]), item[0])))

    return summary


def get_walltime(job_times):
    "Return the walltime for all jobs in seconds."
    start = None
    end = None
    for job_id, job_time in job_times.items():
        job_start_timestamp = job_time['started_epoch_secs']
        job_end_timestamp = job_time['completed_epoch_secs']
        if not start or job_start_timestamp < start:
            start = job_start_timestamp
        if not end or job_end_timestamp > end:
            end = job_end_timestamp
    return end - start


def format_seconds(seconds):
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    days = int(days)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    if (days > 0):
        return f'{days}d {hours:02}h'
    elif (hours > 0):
        return f'{hours}h {minutes:02}m'
    else:
        return f'{minutes}m {seconds:02}s'


def get_summary_stats(summary):
    passed = summary['passed']
    failed = summary['failed']
    total = passed + failed

    percent = int(100 * passed / total) if total > 0 else 0
    pass_string = f'Pass: {percent:>3}%/{total}'

    stats = f'{pass_string:<14}'

    if 'job_time' in summary and total > 0 and summary['job_time'] > 0:
        job_time = summary['job_time']
        max_job_time = summary['max_job_time']
        total_job_duration = format_seconds(job_time)
        avg_job_duration = format_seconds(job_time / total)
        max_job_duration = format_seconds(max_job_time)
        stats += f' | Total: {total_job_duration:>7} | Avg: {avg_job_duration:>7} | Max: {max_job_duration:>7}'

    if 'sccache' in summary:
        sccache = summary['sccache']
        requests = sccache["requests"]
        hits = sccache["hits"]
        hit_percent = int(100 * hits / requests) if requests > 0 else 0
        hit_string = f'Hits: {hit_percent:>3}%/{requests}'
        stats += f' | {hit_string:<17}'

    return stats


def get_summary_heading(summary, walltime):
    passed = summary['passed']
    failed = summary['failed']

    if summary['passed'] == 0:
        flag = '🟥'
    elif summary['failed'] > 0:
        flag = '🟨'
    else:
        flag = '🟩'

    return f'{flag} CI finished in {walltime}: {get_summary_stats(summary)}'


def get_project_heading(project, project_summary):
    if project_summary['passed'] == 0:
        flag = '🟥'
    elif project_summary['failed'] > 0:
        flag = '🟨'
    else:
        flag = '🟩'

    return f'{flag} {project}: {get_summary_stats(project_summary)}'


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

    left_aligned = f"{flag} {value}"
    return f'  {left_aligned:<20} {get_summary_stats(value_summary)}'


def get_project_summary_body(project, project_summary):
    body = ['```']
    for tag, tag_summary in project_summary['tags'].items():
        body.append(get_tag_line(tag, tag_summary))
        for value, value_summary in tag_summary['values'].items():
            body.append(get_value_line(value, value_summary, tag_summary))
    body.append('```')
    return "\n".join(body)


def write_project_summary(idx, project, project_summary):
    heading = get_project_heading(project, project_summary)
    body = get_project_summary_body(project, project_summary)

    summary = {'heading': heading, 'body': body}

    write_json(f'execution/projects/{idx:03}_{project}_summary.json', summary)


def write_workflow_summary(workflow, job_times=None):
    summary = build_summary(extract_jobs(workflow), job_times)
    walltime = format_seconds(get_walltime(job_times)) if job_times else '[unknown]'

    os.makedirs('execution/projects', exist_ok=True)

    write_text('execution/heading.txt', get_summary_heading(summary, walltime))

    # Sort summary projects so that projects with failures come first, and ties
    # are broken by the total number of jobs:
    def sort_project_key(project_summary):
        failed = project_summary[1]['failed']
        total = project_summary[1]['passed'] + failed
        return (-failed, -total)

    for i, (project, project_summary) in enumerate(sorted(summary['projects'].items(), key=sort_project_key)):
        write_project_summary(i, project, project_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workflow', type=argparse.FileType('r'))
    parser.add_argument('job_times', type=argparse.FileType('r'))
    args = parser.parse_args()

    workflow = json.load(args.workflow)

    # The timing file is not required.
    try:
        job_times = json.load(args.job_times)
    except:
        job_times = None

    write_workflow_summary(workflow, job_times)


if __name__ == '__main__':
    main()
