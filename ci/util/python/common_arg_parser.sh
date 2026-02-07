#!/bin/bash

# Argument parser for Python CI scripts.
parse_python_args() {
    # Initialize variables
    py_version=""
    set_git_safe_directory="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -py-version=*)
                py_version="${1#*=}"
                shift
                ;;
            -py-version)
                if [[ $# -lt 2 ]]; then
                    echo "Error: -py-version requires a value" >&2
                    return 1
                fi
                py_version="$2"
                shift 2
                ;;
            -set-git-safe-directory)
                set_git_safe_directory=true
                shift
                ;;
            *)
                # Unknown argument, ignore
                shift
                ;;
        esac
    done

    # Export for use by calling script
    export py_version
    export set_git_safe_directory
}

require_py_version() {
    if [[ -z "$py_version" ]]; then
        echo "Error: -py-version is required" >&2
        [[ -n "$1" ]] && echo "$1" >&2
        return 1
    fi
}
