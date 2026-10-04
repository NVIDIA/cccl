# Argument parser for Python CI scripts.
parse_python_args() {
    # Initialize variables
    py_version=""
    # ctk_mode carries the -ctk-mode value; empty means the default ("pinned").
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
                # Reject an explicit-but-empty value (e.g. `-ctk-mode=`): a lane
                # that wants the default omits the flag entirely, so an empty
                # value signals a malformed generated argument -- fail loudly.
                if [[ -z "${ctk_mode}" ]]; then
                    echo "Error: -ctk-mode requires a value" >&2
                    return 1
                fi
                shift
                ;;
            -ctk-mode)
                if [[ $# -lt 2 || -z "$2" ]]; then
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

    # Export for use by the calling script (py_version and ctk_mode are its inputs).
    export py_version ctk_mode
}

require_py_version() {
    if [[ -z "$py_version" ]]; then
        echo "Error: -py-version is required" >&2
        [[ -n "$1" ]] && echo "$1" >&2
        return 1
    fi
}
