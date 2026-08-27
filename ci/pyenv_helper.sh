setup_python_env() {
    local py_version=$1
    # Optional venv directory name (under $HOME). Callers whose job runs
    # several build scripts (e.g. the combined cuda-stf producer) pass their
    # own name so the setups don't collide on one venv: uv errors on an
    # existing venv instead of reusing it.
    local venv_name="${2:-.cccl-venv}"
    local venv_path="${HOME}/${venv_name}"

    # Source pretty_printing.sh for begin_group/end_group helpers
    local script_dir
    script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # shellcheck source=ci/pretty_printing.sh
    source "${script_dir}/pretty_printing.sh"

    begin_group "🐍 Setting up Python ${py_version} (uv)"

    # The minimal Python containers ship pip but no curl.
    if ! command -v uv &> /dev/null; then
        if command -v curl &> /dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        else
            python3 -m pip install --quiet --root-user-action=ignore uv
        fi
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Create a venv with the requested Python version.
    # uv downloads a pre-built CPython binary automatically — no compilation needed.
    uv venv --seed --python "${py_version}" "${venv_path}"

    # Windows venvs use Scripts/, Linux/macOS use bin/
    if [[ -f "${venv_path}/Scripts/activate" ]]; then
        #shellcheck disable=SC1091
        source "${venv_path}/Scripts/activate"
    else
        #shellcheck disable=SC1091
        source "${venv_path}/bin/activate"
    fi

    end_group "🐍 Setting up Python ${py_version} (uv)"
}

# Pin the cuda-toolkit wheels to the container's CTK major.minor
# via PIP_CONSTRAINT when the mode ($1) is "pinned" (the default; empty also means
# pinned). "latest" and "sysctk" leave it unpinned; any other value is a hard
# error. This is the lane's mode gate -- it runs before ctk_extra_flavor in every
# script, so ctk_extra_flavor can assume the mode is already valid. Also sets and
# exports cuda_version / cuda_major_version; the caller uses cuda_major_version in
# the pip-extra name (e.g. minimal-cu${cuda_major_version}).
pin_cuda_toolkit() {
    # nvcc is the source of truth wherever it exists; `sysctk` mode depends on
    # matching the toolkit actually installed. The minimal containers have no
    # toolkit, so they get the version forwarded in via CCCL_CUDA_VERSION.
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1-2 | cut -d 'V' -f 2)
    elif [[ -n "${CCCL_CUDA_VERSION:-}" ]]; then
        cuda_version="${CCCL_CUDA_VERSION}"
    else
        echo "ERROR: cannot determine the CUDA version: no nvcc on PATH and CCCL_CUDA_VERSION is unset" >&2
        return 1
    fi
    cuda_major_version=$(echo "$cuda_version" | cut -d '.' -f 1)
    export cuda_version cuda_major_version

    local mode="${1:-pinned}"
    case "${mode,,}" in
        pinned)
            export PIP_CONSTRAINT="${TMPDIR:-/tmp}/ctk-constraint.txt"
            echo "cuda-toolkit==${cuda_version}.*" > "${PIP_CONSTRAINT}"
            ;;
        latest | sysctk)
            # No pin. Clear any inherited constraint so it cannot affect the
            # resolve (latest tests the newest minor; sysctk installs no
            # cuda-toolkit wheel at all).
            unset PIP_CONSTRAINT
            ;;
        *)
            echo "ERROR: invalid ctk mode '${mode}' (expected pinned|latest|sysctk)" >&2
            return 1
            ;;
    esac
}

# Echoes the pip-extra toolkit "flavor" for the mode ($1): "sysctk" when the mode
# is sysctk (rely on the system-provided CUDA toolkit) or "cu" otherwise
# (pip-installed toolkit). The mode is validated by pin_cuda_toolkit, which every
# lane calls first. Combine with the CUDA major, e.g.
# "minimal-$(ctk_extra_flavor "${ctk_mode}")${cuda_major_version}" -> minimal-sysctk12.
ctk_extra_flavor() {
    local mode="${1:-}"
    if [[ "${mode,,}" == "sysctk" ]]; then
        echo "sysctk"
    else
        echo "cu"
    fi
}

# Echoes the path of the cuda_cccl wheel the entry point staged in wheelhouse/,
# erroring unless there is exactly one (a bare glob yields "" or two paths, and
# either goes on to be pip-installed as-is).
cuda_cccl_wheel_path() {
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local matches=()
    while IFS= read -r match; do
        matches+=("${match}")
    done < <(find "${repo_root}/wheelhouse" -maxdepth 1 -name 'cuda_cccl-*.whl' 2>/dev/null | sort)

    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "ERROR: expected exactly one cuda_cccl wheel in ${repo_root}/wheelhouse, found ${#matches[@]}" >&2
        return 1
    fi
    echo "${matches[0]}"
}

# Brings up what every test payload needs: the lane's arguments, the cuda-toolkit
# pin, and the interpreter. Sets py_version, ctk_mode, cuda_version and
# cuda_major_version for the caller.
python_payload_init() {
    local ci_dir
    ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # shellcheck source=ci/util/python/common_arg_parser.sh
    source "${ci_dir}/util/python/common_arg_parser.sh"
    parse_python_args "$@"
    pin_cuda_toolkit "${ctk_mode}"
    setup_python_env "${py_version}"
}

# Stages the cuda_cccl wheel, then hands off to the lane's test payload ($1),
# forwarding the remaining arguments. The payload runs in a minimal sibling
# container except in `sysctk` mode, which exists to test a system-provided CUDA
# toolkit, or when CCCL_MINIMAL_CONTAINER=0. Expects parse_python_args to have
# run. See "Testing Python in a minimal container" in
# docs/infrastructure/ci/references/ci_overview.rst.
dispatch_python_lane() {
    local payload="$1"
    shift

    local ci_dir repo_root
    ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${ci_dir}/.." && pwd)"

    # Needs `gh` (or docker, for a local build), which the minimal container lacks.
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        local wheel_artifact_name
        wheel_artifact_name="$("${ci_dir}/util/workflow/get_wheel_artifact_name.sh")"
        "${ci_dir}/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
    else
        "${ci_dir}/build_cuda_cccl_python.sh" -py-version "${py_version}"
    fi

    if [[ "${ctk_mode,,}" != "sysctk" && "${CCCL_MINIMAL_CONTAINER:-1}" != "0" ]]; then
        exec "${ci_dir}/util/python/run_in_minimal_container.sh" "${payload}" "$@"
    else
        exec "${repo_root}/${payload}" "$@"
    fi
}
