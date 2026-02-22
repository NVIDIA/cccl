#!/usr/bin/env python

# Copied from nvbench/scripts

import json


def read_file(filename):
    with open(filename, "r") as f:
        file_root = json.load(f)
    check_file_version(filename, file_root)
    return file_root


def find_device_by_id(device_id, all_devices):
    """Find device info by ID."""
    for device in all_devices:
        if device["id"] == device_id:
            return device
    return None


file_version = (1, 0, 0)

file_version_string = "{}.{}.{}".format(
    file_version[0], file_version[1], file_version[2]
)


def check_file_version(filename, root_node):
    try:
        version_node = root_node["meta"]["version"]["json"]
    except KeyError:
        print("WARNING:")
        print("  {} is written in an older, unversioned format. ".format(filename))
        print("  It may not read correctly.")
        print("  Reader expects JSON file version {}.".format(file_version_string))
        return

    # TODO We could do something fancy here using semantic versioning, but
    # for now just warn on mismatch.
    if version_node["string"] != file_version_string:
        print("WARNING:")
        print(
            "  {} was written using a different NVBench JSON file version.".format(
                filename
            )
        )
        print("  It may not read correctly.")
        print(
            "  (file version: {} reader version: {})".format(
                version_node["string"], file_version_string
            )
        )
