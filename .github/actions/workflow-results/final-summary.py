#!/usr/bin/env python3

import json
import os
import re
import sys


def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read().rstrip("\n ")

def print_file_if_present(filepath):
    if os.path.exists(filepath):
        print(read_file(filepath) + "\n\n")


def print_summary_file(filepath, heading_level):
    summary_json = json.load(open(filepath))
    print(f"<details><summary><h{heading_level}>{summary_json['heading']}</h{heading_level}></summary>\n")
    print(summary_json["body"] + "\n")
    print("</details>\n")


def main():
    # List of all projects detected in 'execution/projects/{project}_summary.json':
    projects = []
    project_file_regex="(.*)_summary.json"
    for filename in os.listdir("execution/projects"):
        match = re.match(project_file_regex, filename)
        if match:
            projects.append(match.group(1))

    print(f"<details><summary>{read_file('execution/heading.txt')}</summary>\n")

    print("<ul>")
    for project in projects:
      print("<li>")
      print_summary_file(f"execution/projects/{project}_summary.json", 3)
    print("</ul>\n")

    print_summary_file("workflow/runner_summary.json", 2)
    print_file_if_present('workflow/changes.md')

    print("</details>")



if __name__ == '__main__':
    main()
