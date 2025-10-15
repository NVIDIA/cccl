# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Sourced from https://github.com/NVIDIA/cuda-python

# Install the driver
function Install-Driver {

    # Set the correct URL, filename, and arguments to the installer
    # This driver is picked to support Windows 11 & CUDA 12.8
    $version = '581.15'

    # extract gpu type from gha runner label:
    # Labels are in the form: <os>-<cpu>-gpu-<gpu>-<driver>-<n>
    $gha_runner_label=$env:RUNNER
    $gpu_type = $gha_runner_label.Split('-')[3]

    $data_center_gpus = @('a100', 'h100', 'l4', 't4', 'v100', 'rtxa6000', 'rtx6000ada')
    $desktop_gpus = @('rtx2080', 'rtx4090')

    if ($data_center_gpus -contains $gpu_type) {
        Write-Output "Data center GPU detected: $gpu_type"
        $filename="$version-data-center-tesla-desktop-winserver-2022-2025-dch-international.exe"
        $server_path="tesla/$version"
    } elseif ($desktop_gpus -contains $gpu_type) {
        Write-Output "Desktop GPU detected: $gpu_type"
        $filename="$version-desktop-win10-win11-64bit-international-dch-whql.exe"
        $server_path="Windows/$version"
    } else {
        Write-Output "Unknown GPU type: $gpu_type"
        exit 1
    }

    $url="https://us.download.nvidia.com/$server_path/$filename"
    $filepath="C:\NVIDIA-Driver\$filename"

    Write-Output "Installing NVIDIA driver version $version for GPU type $gpu_type"
    Write-Output "Download URL: $url"

    # Silent install arguments
    $install_args = '/s /noeula /noreboot';

    # Create the folder for the driver download
    if (!(Test-Path -Path 'C:\NVIDIA-Driver')) {
        New-Item -Path 'C:\' -Name 'NVIDIA-Driver' -ItemType 'directory' | Out-Null
    }

    # Download the file to a specified directory
    # Disabling progress bar due to https://github.com/GoogleCloudPlatform/compute-gpu-installation/issues/29
    $ProgressPreference_tmp = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    Write-Output 'Downloading the driver installer...'
    Invoke-WebRequest $url -OutFile $filepath
    $ProgressPreference = $ProgressPreference_tmp
    Write-Output 'Download complete!'

    # Install the file with the specified path from earlier
    Write-Output 'Running the driver installer...'
    Start-Process -FilePath $filepath -ArgumentList $install_args -Wait
    Write-Output 'Done!'

    # TCC -> MCDM on data center GPUs:
    if ($data_center_gpus -contains $gpu_type) {
        nvidia-smi -fdm 2
        pnputil /disable-device /class Display
        pnputil /enable-device /class Display
        # Give it a minute to settle:
        Start-Sleep -Seconds 5
    }
}

# Run the functions
Install-Driver
