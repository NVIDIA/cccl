#!/bin/bash

# Argument parser for Python CI scripts.
parse_python_args() {
    # Initialize variables
    py_version=""
    cuda_version=""

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
            -cuda-version=*)
                cuda_version="${1#*=}"
                shift
                ;;
            -cuda-version)
                if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                    cuda_version="$2"
                    shift 2
                else
                    shift 1
                fi
                ;;
            *)
                # Unknown argument, ignore
                shift
                ;;
        esac
    done

    # Export for use by calling script
    export py_version
    export cuda_version
}

require_py_version() {
    if [[ -z "$py_version" ]]; then
        echo "Error: -py-version is required" >&2
        [[ -n "$1" ]] && echo "$1" >&2
        return 1
    fi
}
