#!/usr/bin/env bash

# shellcheck disable=SC1091

set -e;

devcontainer-utils-post-create-command;
devcontainer-utils-init-git;
devcontainer-utils-post-attach-command;

sudo apt-get update && sudo apt-get install -y ca-certificates

cd /home/coder/cccl/

# Check if docker CLI is available, and if not, install docker-outside-of-docker feature if specified in devcontainer.json
if ! command -v docker >/dev/null 2>&1; then
    # Find the devcontainer.json file (search up from current dir)
    search_dir="$(pwd)"
    found_json=""
    while [[ "$search_dir" != "/" ]]; do
        if [[ -f "$search_dir/.devcontainer/devcontainer.json" ]]; then
            found_json="$search_dir/.devcontainer/devcontainer.json"
            break
        fi
        search_dir="$(dirname "$search_dir")"
    done
    if [[ -n "$found_json" ]] && grep -q 'docker-outside-of-docker' "$found_json"; then
        echo "Installing docker-outside-of-docker feature..."
        git clone --depth=1 https://github.com/devcontainers/features.git /tmp/features
        cd /tmp/features/src/docker-outside-of-docker
        chmod +x install.sh
        sudo SOURCE_SOCKET=/var/run/docker.sock TARGET_SOCKET=/var/run/docker.sock MOBY=false ./install.sh || { echo "docker-outside-of-docker install failed"; exit 1; }
        cd -

        # Install nvidia-container-toolkit
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
    fi
fi

if test $# -gt 0; then
    exec "$@";
else
    exec /bin/bash -li;
fi
