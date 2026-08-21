#!/usr/bin/env bash

# Source this file with the physical GPU index shown by nvidia-smi:
#   source examples/libero/activate_hgpu.sh 7

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source examples/libero/activate_hgpu.sh <gpu-id>"
  exit 1
fi

RETAIN_GPU_ID="${1:-0}"
RETAIN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RETAIN_NVIDIA_EGL_HOME="/shared/.cache/retain/nvidia-egl"

source "${RETAIN_PROJECT_ROOT}/examples/libero/.venv/bin/activate"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${RETAIN_GPU_ID}"
export LIBERO_CONFIG_PATH=/shared/.cache/retain/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# MuJoCo EGL uses the physical index and does not apply CUDA's index remapping.
export MUJOCO_EGL_DEVICE_ID="${RETAIN_GPU_ID}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${RETAIN_NVIDIA_EGL_HOME}/driver-595.71.05/10_nvidia.json"
export LD_LIBRARY_PATH="${RETAIN_NVIDIA_EGL_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "RETAIN LIBERO environment active on physical GPU ${RETAIN_GPU_ID}"
