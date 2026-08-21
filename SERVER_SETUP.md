# hgpu server setup

This project is installed on `hgpu` at `/root/RETAIN_code`.

The installation is isolated from VLS: it does not activate, install into, or
modify any VLS environment. Keep using the project-local virtual environments
documented below.

## Main OpenPI environment

```bash
ssh hgpu
cd /root/RETAIN_code
source .venv/bin/activate
export UV_CACHE_DIR=/shared/.cache/retain/uv
export HF_HOME=/shared/.cache/retain/huggingface
export OPENPI_DATA_HOME=/shared/.cache/retain/openpi
```

Use `python ...` after activation, or use `uv run --no-sync ...`. Always retain
the `--no-sync` flag on this server: it avoids re-fetching the pinned LeRobot
Git dependency and preserves the Blackwell-compatible PyTorch override.

To rebuild the main environment from the server-side cache:

```bash
cd /root/RETAIN_code
GIT_LFS_SKIP_SMUDGE=1 UV_CACHE_DIR=/shared/.cache/retain/uv \
  /root/.local/bin/uv sync --frozen --python 3.11 --no-install-package lerobot
UV_CACHE_DIR=/shared/.cache/retain/uv /root/.local/bin/uv pip install \
  --python .venv/bin/python --no-deps \
  /shared/.cache/retain/sources/lerobot-0cf864870cf29f4738d3ade893e6fd13fbd7cdb5.tar.gz
UV_CACHE_DIR=/shared/.cache/retain/uv /root/.local/bin/uv pip install \
  --python .venv/bin/python --link-mode=copy \
  -r requirements-hgpu-sm120.txt
```

The cached LeRobot archive has SHA-256
`2455b2ed303778f8bcd570caceae0b22c97908eaaf029dff868d9f237b495c00`.

The upstream lock resolves PyTorch 2.6.0 with CUDA 12.4, which cannot execute
on the server's `sm_120` GPUs. `requirements-hgpu-sm120.txt` upgrades only the
server environment to PyTorch 2.7.1 and torchvision 0.22.1 with CUDA 12.8; both
remain within this project's declared minimum-version constraints.

## LIBERO evaluation environment

LIBERO uses its own Python 3.8 and CUDA 11.3-compatible environment:

```bash
ssh hgpu
cd /root/RETAIN_code
source examples/libero/activate_hgpu.sh 7  # replace 7 with a physical GPU index
```

LIBERO datasets should be stored under `/shared/.cache/retain/libero/datasets`.
The project-specific config is `/shared/.cache/retain/libero/config.yaml`; do not
replace `/root/.libero/config.yaml`, which belongs to the existing VLS setup.

To rebuild the LIBERO environment without touching VLS:

```bash
cd /root/RETAIN_code
/root/.local/bin/uv venv --python 3.8 examples/libero/.venv
UV_CACHE_DIR=/shared/.cache/retain/uv /root/.local/bin/uv pip install \
  --python examples/libero/.venv/bin/python cmake==3.31.6
UV_CACHE_DIR=/shared/.cache/retain/uv /root/.local/bin/uv pip install \
  --python examples/libero/.venv/bin/python \
  -r examples/libero/requirements-server.lock.txt \
  --index-strategy=unsafe-best-match
UV_CACHE_DIR=/shared/.cache/retain/uv /root/.local/bin/uv pip install \
  --python examples/libero/.venv/bin/python \
  -e packages/openpi-client -e third_party/libero
printf '%s\n' /root/RETAIN_code/third_party/libero > \
  examples/libero/.venv/lib/python3.8/site-packages/retain-libero.pth
```

`cmake==3.31.6` is installed only in this virtual environment because the
`egl-probe` build used by LIBERO is incompatible with CMake 4 policy defaults.

The server had only Mesa EGL libraries, so NVIDIA EGL 595.71.05 was extracted
without system installation under `/shared/.cache/retain/nvidia-egl`. The
activation script exposes only the minimum EGL libraries to LIBERO and does not
change the system linker or VLS. Its cached Ubuntu source archive has SHA-256
`9b82372f9fdc696d67f568d578a5f05a59c3535a55de4c30ba227cf5eb05a911`.

## GPU validation

GPU 7 was used with memory preallocation disabled where supported. The
following checks completed successfully on an NVIDIA RTX 6000D (`sm_120`):

- JAX 0.5.0 CUDA matrix multiplication.
- PyTorch 2.7.1+cu128 matrix multiplication with native `sm_120` kernels.
- TensorFlow 2.18.1 matrix multiplication. TensorFlow JIT-compiles PTX for
  `sm_120`, so the first use of a new kernel can be slower.
- LIBERO/MuJoCo EGL initialization, task loading, initial-state loading, and
  two 64x64 off-screen camera renders.

LIBERO's isolated Python 3.8 environment intentionally keeps its upstream
PyTorch 1.11 dependency, which does not contain `sm_120` CUDA kernels. Run
policy inference in the main JAX environment; the LIBERO client and EGL
renderer do not require PyTorch CUDA.

## Project-specific notes

- The code's LIBERO config is named `pi0_libero_MINE`, while several README
  examples still say `pi0_libero`.
- The default LIBERO entry in `scripts/serve_policy.py` refers to
  `pi0_fast_libero`, which is not present in this fork. Use the explicit
  `policy:checkpoint` form and a config that exists in
  `src/openpi/training/config.py`.
- Some training configs still contain the original author's absolute paths
  under `/raid/users/yajatyadav`. Override checkpoint and dataset paths for
  this server before launching those experiments.
- Model checkpoints and datasets are intentionally not pre-downloaded. They are
  task-specific and much larger than the software environment.
- To repeat a lightweight JAX GPU check without memory preallocation:

```bash
cd /root/RETAIN_code
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -c 'import jax; print(jax.devices())'
```
