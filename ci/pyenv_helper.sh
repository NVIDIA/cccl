setup_python_env() {
    local py_version=$1

    # Source pretty_printing.sh for begin_group/end_group helpers
    local script_dir
    script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # shellcheck source=ci/pretty_printing.sh
    source "${script_dir}/pretty_printing.sh"

    begin_group "🐍 Setting up Python ${py_version} (uv)"

    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Create a venv with the requested Python version.
    # uv downloads a pre-built CPython binary automatically — no compilation needed.
    uv venv --seed --python "${py_version}" "${HOME}/.cccl-venv"

    # Windows venvs use Scripts/, Linux/macOS use bin/
    if [[ -f "${HOME}/.cccl-venv/Scripts/activate" ]]; then
        #shellcheck disable=SC1091
        source "${HOME}/.cccl-venv/Scripts/activate"
    else
        #shellcheck disable=SC1091
        source "${HOME}/.cccl-venv/bin/activate"
    fi

    end_group "🐍 Setting up Python ${py_version} (uv)"
}

# Validates and normalizes a CTK test mode passed as $1 (forwarded by the caller
# from the -ctk-mode arg / the ${ctk_mode} shell var):
#   pinned  (default; empty means pinned) -- pip-install cuda-toolkit pinned to
#           the container's CTK minor (reproducible; the usual CI test).
#   latest  -- pip-install cuda-toolkit at the newest available minor (a canary
#           for breakage against the latest CTK).
#   sysctk  -- do NOT pip-install a toolkit; use the system-provided one (the
#           scenario a CuPy user without cuda-cccl[cuNN] hits).
# Any other value is a hard error. Echoes the (lowercased) mode.
ctk_test_mode() {
    local mode="${1:-pinned}"
    case "${mode,,}" in
        pinned | latest | sysctk) echo "${mode,,}" ;;
        *)
            echo "ERROR: invalid ctk mode '${mode}' (expected pinned|latest|sysctk)" >&2
            return 1
            ;;
    esac
}

# Pin the cuda-toolkit wheels to the container's CTK major.minor (read from nvcc)
# via PIP_CONSTRAINT when the mode ($1) is "pinned" (the default); "latest" and
# "sysctk" leave it unpinned (see ctk_test_mode). Sets and exports cuda_version
# and cuda_major_version for the caller (e.g. the [<flavor>-<major>] extra).
pin_cuda_toolkit() {
    cuda_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1-2 | cut -d 'V' -f 2)
    cuda_major_version=$(echo "$cuda_version" | cut -d '.' -f 1)
    export cuda_version cuda_major_version

    local mode
    mode="$(ctk_test_mode "${1:-}")" || return 1

    if [[ "${mode}" == "pinned" ]]; then
        export PIP_CONSTRAINT="${TMPDIR:-/tmp}/ctk-constraint.txt"
        echo "cuda-toolkit==${cuda_version}.*" > "${PIP_CONSTRAINT}"
    else
        # latest / sysctk: no pin. Clear any inherited constraint so it cannot
        # affect the resolve (latest tests the newest minor; sysctk installs no
        # cuda-toolkit wheel at all).
        unset PIP_CONSTRAINT
    fi
}

# Echoes the pip-extra toolkit "flavor" for the mode ($1): "sysctk" when the mode
# is sysctk (rely on the system-provided CUDA toolkit) or "cu" otherwise
# (pip-installed toolkit). Combine with the CUDA major, e.g.
# "minimal-$(ctk_extra_flavor "${ctk_mode}")${cuda_major_version}" -> minimal-sysctk12.
ctk_extra_flavor() {
    if [[ "$(ctk_test_mode "${1:-}")" == "sysctk" ]]; then
        echo "sysctk"
    else
        echo "cu"
    fi
}
