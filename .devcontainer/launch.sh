#!/usr/bin/env bash

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

# Parse options using getopt
OPTIONS=c:H:dh
LONG_OPTIONS=cuda:,host:,docker,help
PARSED_OPTIONS=$(getopt -n "$0" -o "${OPTIONS}" --long "${LONG_OPTIONS}" -- "$@")

# If the parsing failed, exit
if [[ $? -ne 0 ]]; then
    exit 1
fi

eval set -- "${PARSED_OPTIONS}"

docker_mode=false

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

if [[ -z ${cuda_version} ]] && [[ -z ${host_compiler} ]]; then
    # Use the top-level devcontainer
    path=".devcontainer"
else
    path=".devcontainer/cuda${cuda_version}-${host_compiler}"
    if [[ ! -f "${path}/devcontainer.json" ]]; then
        echo "Unknown CUDA [${cuda_version}] compiler [${host_compiler}] combination"
        echo "Requested devcontainer ${path}/devcontainer.json does not exist"
        exit 1
    fi
fi

if $docker_mode; then
    # Extract Docker image and run it
    DOCKER_IMAGE=$(grep "image" "${path}/devcontainer.json" | sed 's/.*: "\(.*\)",/\1/')
    echo "Found image: ${DOCKER_IMAGE}"
    docker pull ${DOCKER_IMAGE}
    docker run -it --rm -v $(pwd):/workspace ${DOCKER_IMAGE} /bin/bash
else
    # Launch devcontainer in Visual Studio Code
    local workspace="$(basename "$(pwd)")"
    local tmpdir="$(mktemp -d)/${workspace}"
    mkdir -p "${tmpdir}"
    mkdir -p "${tmpdir}/.devcontainer"
    cp -arL "${path}/devcontainer.json" "${tmpdir}/.devcontainer"
    sed -i "s@\\${localWorkspaceFolder}@$(pwd)@g" "${tmpdir}/.devcontainer/devcontainer.json"
    path="${tmpdir}"

    local hash="$(echo -n "${path}" | xxd -pu - | tr -d '[:space:]')"
    local url="vscode://vscode-remote/dev-container+${hash}/home/coder/cccl"
    echo "devcontainer URL: ${url}"

    local launch=""
    if type open >/dev/null 2>&1; then
        launch="open"
    elif type xdg-open >/dev/null 2>&1; then
        launch="xdg-open"
    fi

    if [ -n "${launch}" ]; then
        code --new-window "${tmpdir}"
        exec "${launch}" "${url}" >/dev/null 2>&1
    fi
fi
