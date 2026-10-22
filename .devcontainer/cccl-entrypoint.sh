#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::group::Initializing devcontainer..."
fi

# The CUDA forward compatibility (cuda-compat) libraries are only supported on
# data center GPUs; activating them on other GPUs breaks CUDA. GPU jobs set
# DATA_CENTER_GPU ('1' or '0') via the CI matrix, based on the runner's GPU.
# Local users with a data center GPU can opt in with:
#   .devcontainer/launch.sh --env DATA_CENTER_GPU=1
if [[ "${DATA_CENTER_GPU:-0}" = "1" ]]; then
    setup-cuda-compat;
else
    echo "DATA_CENTER_GPU is '${DATA_CENTER_GPU:-0}', skipping cuda-compat configuration.";
fi
devcontainer-utils-post-create-command;
devcontainer-utils-init-git;
devcontainer-utils-post-attach-command;
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::endgroup::"
fi

if ! dpkg -s ca-certificates > /dev/null 2>&1; then
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::group::Installing ca-certificates..."
    fi
    sudo apt-get update
    sudo apt-get install -y ca-certificates
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::endgroup::"
    fi
else
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::group::Updating ca-certificates..."
    fi
    sudo update-ca-certificates
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::endgroup::"
    fi
fi

cd /home/coder/cccl/

if test $# -gt 0; then
    exec "$@";
else
    exec /bin/bash -li;
fi
