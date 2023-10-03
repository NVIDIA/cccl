#!/usr/bin/env bash

set -euo pipefail

# Ensure the script is being executed in the cccl/ root
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

print_help() {
    echo "Usage: $0 [-c|--cuda <CUDA version>] [-H|--host <Host compiler>] [-d|--docker]"
    echo "Launch a development container. If no CUDA version or Host compiler are specified,"
    echo "the top-level devcontainer in .devcontainer/devcontainer.json will be used."
    echo ""
    echo "Options:"
    echo "  -c, --cuda       Specify the CUDA version. E.g., 12.2"
    echo "  -H, --host       Specify the host compiler. E.g., gcc12"
    echo "  -d, --docker     Launch the development environment in Docker directly without using VSCode."
    echo "  -h, --help       Display this help message and exit."
}

parse_options() {
    local OPTIONS=c:H:dh
    local LONG_OPTIONS=cuda:,host:,docker,help
    local PARSED_OPTIONS=$(getopt -n "$0" -o "${OPTIONS}" --long "${LONG_OPTIONS}" -- "$@")

    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    eval set -- "${PARSED_OPTIONS}"

    while true; do
        case "$1" in
            -c|--cuda)
                cuda_version="$2"
                shift 2
                ;;
            -H|--host)
                host_compiler="$2"
                shift 2
                ;;
            -d|--docker)
                docker_mode=true
                shift
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Invalid option: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

launch_docker() {
    DOCKER_IMAGE=$(grep "image" "${path}/devcontainer.json" | sed 's/.*: "\(.*\)",/\1/')
    echo "Found image: ${DOCKER_IMAGE}"
    docker pull ${DOCKER_IMAGE}
    docker run   \
        -it --rm \
        --user coder \
        --workdir /home/coder/cccl \
        --mount type=bind,src="$(pwd)",dst='/home/coder/cccl' \
        ${DOCKER_IMAGE} \
        /bin/bash
}

launch_vscode() {
    # Since Visual Studio Code allows only one instance per `devcontainer.json`,
    # this code prepares a unique temporary directory structure for each launch of a devcontainer.
    # By doing so, it ensures that multiple instances of the same environment can be run
    # simultaneously. The script replicates the `devcontainer.json` from the desired CUDA
    # and compiler environment into this temporary directory, adjusting paths to ensure the
    # correct workspace is loaded. A special URL is then generated to instruct VSCode to
    # launch the development container using this temporary configuration.
    local workspace="$(basename "$(pwd)")"
    local tmpdir="$(mktemp -d)/${workspace}"
    mkdir -p "${tmpdir}"
    mkdir -p "${tmpdir}/.devcontainer"
    cp -arL "${path}/devcontainer.json" "${tmpdir}/.devcontainer"
    sed -i 's@\\${localWorkspaceFolder}@$(pwd)@g' "${tmpdir}/.devcontainer/devcontainer.json"
    local path="${tmpdir}"
    local hash="$(echo -n "${path}" | xxd -pu - | tr -d '[:space:]')"
    local url="vscode://vscode-remote/dev-container+${hash}/home/coder/cccl"

    local launch=""
    if type open >/dev/null 2>&1; then
        launch="open"
    elif type xdg-open >/dev/null 2>&1; then
        launch="xdg-open"
    fi

    if [ -n "${launch}" ]; then
        echo "Launching VSCode Dev Container URL: ${url}"
        code --new-window "${tmpdir}"
        exec "${launch}" "${url}" >/dev/null 2>&1
    fi
}

main() {
    parse_options "$@"

    # If no CTK/Host compiler are provided, just use the default environment
    if [[ -z ${cuda_version:-} ]] && [[ -z ${host_compiler:-} ]]; then
        path=".devcontainer"
    else
        path=".devcontainer/cuda${cuda_version}-${host_compiler}"
        if [[ ! -f "${path}/devcontainer.json" ]]; then
            echo "Unknown CUDA [${cuda_version}] compiler [${host_compiler}] combination"
            echo "Requested devcontainer ${path}/devcontainer.json does not exist"
            exit 1
        fi
    fi

    if ${docker_mode:-'false'}; then
        launch_docker
    else
        launch_vscode
    fi
}

main "$@"

