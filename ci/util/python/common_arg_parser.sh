# Argument parser for Python CI scripts.
parse_python_args() {
    # Initialize variables
    py_version=""
    # ctk_mode is a plain shell var the calling script forwards to
    # pin_cuda_toolkit / ctk_extra_flavor. Empty means the default ("pinned").
    ctk_mode=""

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
            -ctk-mode=*)
                ctk_mode="${1#*=}"
                shift
                ;;
            -ctk-mode)
                if [[ $# -lt 2 ]]; then
                    echo "Error: -ctk-mode requires a value" >&2
                    return 1
                fi
                ctk_mode="$2"
                shift 2
                ;;
            *)
                # Unknown argument, ignore
                shift
                ;;
        esac
    done

    # Export for use by calling script
    export py_version
}

require_py_version() {
    if [[ -z "$py_version" ]]; then
        echo "Error: -py-version is required" >&2
        [[ -n "$1" ]] && echo "$1" >&2
        return 1
    fi
}
