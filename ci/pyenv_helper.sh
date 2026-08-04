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

# Pin the cuda-toolkit wheels to the container's CTK major.minor (read from nvcc)
# via PIP_CONSTRAINT when the mode ($1) is "pinned" (the default; empty also means
# pinned). "latest" and "sysctk" leave it unpinned; any other value is a hard
# error. This is the lane's mode gate -- it runs before ctk_extra_flavor in every
# script, so ctk_extra_flavor can assume the mode is already valid. Also sets and
# exports cuda_version / cuda_major_version; the caller uses cuda_major_version in
# the pip-extra name (e.g. minimal-cu${cuda_major_version}).
pin_cuda_toolkit() {
    cuda_version=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',' | cut -d '.' -f 1-2 | cut -d 'V' -f 2)
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

# Resolve exactly one wheel with the supplied distribution filename prefix.
# Unlike command substitution around `ls`, this keeps the explicit diagnostic
# reachable under `set -e` when there are no matches.
get_one_python_wheel() {
    local wheelhouse=$1
    local wheel_prefix=$2
    local -a matches=()

    mapfile -d '' -t matches < <(
        find "${wheelhouse}" -maxdepth 1 -type f -name "${wheel_prefix}-*.whl" -print0 2> /dev/null
    )
    if (( ${#matches[@]} != 1 )); then
        echo "Expected exactly one ${wheel_prefix}-*.whl under ${wheelhouse}; found ${#matches[@]}." >&2
        if (( ${#matches[@]} > 0 )); then
            printf '  %s\n' "${matches[@]}" >&2
        fi
        return 1
    fi

    printf '%s\n' "${matches[0]}"
}

# Constrain transitive package resolution to coordinated wheels in the local
# wheelhouse. Arguments after the output path are distribution/path pairs.
write_python_wheel_constraints() {
    local output_path=$1
    shift

    "${PYTHON:-python}" - "${output_path}" "$@" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1])
arguments = sys.argv[2:]
if len(arguments) % 2:
    raise ValueError("wheel constraints require distribution/path pairs")

lines = [
    f"{distribution} @ {Path(wheel).resolve().as_uri()}"
    for distribution, wheel in zip(arguments[::2], arguments[1::2])
]
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}
