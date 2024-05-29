#!/usr/bin/env python3

import json
import os
import re
import sys


def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read().rstrip("\n ")


def print_text_file(filepath):
    if os.path.exists(filepath):
        print(read_file(filepath) + "\n\n")


def print_json_summary(summary, heading_level):
    print(f"<details><summary><h{heading_level}>{summary['heading']}</h{heading_level}></summary>\n")
    print(summary["body"] + "\n")
    print("</details>\n")


def print_summary_file(filepath, heading_level):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            print_json_summary(json.load(f), heading_level)


def print_json_file(filepath, heading):
    if os.path.exists(filepath):
        json_data = json.load(open(filepath))
        print(f"<details><summary><h3>{heading}</h3></summary>\n")
        print('```json')
        print(json.dumps(json_data, indent=2))
        print('```')
        print("</details>\n")


def main():
    # Parse project summaries and sort them by the number of failed jobs:
    projects = []
    project_file_regex = "[0-9]+_.+_summary.json"
    for filename in sorted(os.listdir("execution/projects")):
        match = re.match(project_file_regex, filename)
        if match:
            with open(f"execution/projects/{filename}", 'r') as f:
                projects.append(json.load(f))

    print(f"<details><summary>{read_file('execution/heading.txt')}</summary>\n")

    print("<ul>")
    for project in projects:
        print("<li>")
        print_json_summary(project, 3)
    print("</ul>\n")

    print_json_file('workflow/override.json', '🛠️ Override Matrix')
    print_text_file('workflow/changes.md')
    print_summary_file("workflow/runner_summary.json", 2)

    print("</details>")


if __name__ == '__main__':
    main()
