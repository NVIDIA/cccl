#!/usr/bin/env python3

import json
import os
import re


def read_file(filepath):
    with open(filepath, "r") as f:
        return f.read().rstrip("\n ")


def print_text_file(filepath):
    if os.path.exists(filepath):
        print(read_file(filepath) + "\n\n")


def print_json_summary(summary, heading_level):
    print(
        f"<details><summary><h{heading_level}>{summary['heading']}</h{heading_level}></summary>\n"
    )
    print(summary["body"] + "\n")
    print("</details>\n")


def main():
    # Parse project summaries and sort them by the number of failed jobs:
    projects = []
    project_file_regex = "[0-9]+_.+_summary.json"
    for filename in sorted(os.listdir("execution/projects")):
        match = re.match(project_file_regex, filename)
        if match:
            with open(f"execution/projects/{filename}", "r") as f:
                projects.append(json.load(f))

    print(
        f"<details><summary><h3>{read_file('execution/heading.txt')}</h3></summary>\n"
    )

    for project in projects:
        print_json_summary(project, 3)

    print("</details>")


if __name__ == "__main__":
    main()
