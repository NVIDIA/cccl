#!/usr/bin/env python3

"""
Print the matrices of a CI_MATRICES profile as "name ref path" lines.

CI_MATRICES is read from --ci-matrices, or from stdin if that is not given. It must
be a JSON object mapping profile names to objects of "path@ref" specs. Empty spec
parts use the given defaults. Each path/ref pair must have one name in the
selected profile. Nothing is printed if the profile is not defined.
"""

import argparse
import json
import re
import sys

# Restrict profile and matrix identifiers to the allowed characters.
NAME_PATTERN = re.compile(r"[-_A-Za-z0-9]+")


def fail(message):
    sys.exit(f"::error::{message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="nightly")
    parser.add_argument("--default-ref", default="main")
    parser.add_argument("--default-path", default="ci/matrix.yaml")
    parser.add_argument("--ci-matrices")
    args = parser.parse_args()

    if not NAME_PATTERN.fullmatch(args.profile):
        fail(f"Invalid profile name: '{args.profile}'.")

    try:
        profiles: dict[str, dict[str, str]] = (
            json.loads(args.ci_matrices)
            if args.ci_matrices is not None
            else json.load(sys.stdin)
        )
    except json.JSONDecodeError as e:
        fail(f"CI_MATRICES is not valid JSON: {e}")

    # Any other JSON shape lacks .get(), .items(), or .partition().
    try:
        selected = profiles.get(args.profile, {})
        matrices = {name: spec.partition("@")[::2] for name, spec in selected.items()}
    except AttributeError:
        fail(
            "CI_MATRICES must be a JSON object of the form {profile: {name: 'path@ref'}}."
        )

    # Resolve defaults and reject different names for the same matrix file/ref.
    matrix_files = {}
    for name, (path, ref) in matrices.items():
        if not NAME_PATTERN.fullmatch(name):
            fail(f"Invalid name: '{name}'.")
        resolved_ref = ref.strip() or args.default_ref
        resolved_path = path.strip() or args.default_path
        identity = (resolved_path, resolved_ref)
        if identity in matrix_files:
            fail(
                f"Conflicting matrix names for '{resolved_path}@{resolved_ref}': "
                f"'{matrix_files[identity]}' and '{name}'."
            )
        matrix_files[identity] = name

    # Emit the resolved selection only after all definitions have been validated.
    for (path, ref), name in matrix_files.items():
        print(f"Found {name}='{path}@{ref}'", file=sys.stderr)
        print(name, ref, path)


if __name__ == "__main__":
    main()
