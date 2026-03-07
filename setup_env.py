"""
Environment setup helper for rtd-gdes.

Detects whether the current machine is an NVIDIA Jetson and installs
dependencies from the Jetson AI Lab index when appropriate. The target
platform can also be set via the TARGET environment variable, or overridden
with CLI flags.

Usage
-----
# Auto-detect:
    python setup_env.py

# Via env var (preferred):
    TARGET=jetson pip install .        # or: TARGET=jetson python setup_env.py
    TARGET=amd-gfx1151 pip install .   # or: TARGET=amd-gfx1151 python setup_env.py

# Via CLI flags:
    python setup_env.py --jetson
    python setup_env.py --amd-gfx1151
    python setup_env.py --no-jetson    # force standard mode

# Also install dev extras:
    TARGET=amd-gfx1151 python setup_env.py --dev

Note: pip itself does not support custom --index-url injection from env vars,
so this script must be invoked directly. The TARGET env var is read here and
translated into the correct pip invocation with --index-url and arch settings.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

JETSON_INDEX = "https://pypi.jetson-ai-lab.io/jp6/cu126"
AMD_GFX1151_INDEX = "https://rocm.nightlies.amd.com/v2/gfx1151/"
REQUIREMENTS = "requirements.txt"

# Orin Nano uses the Ampere GA10B die — compute capability 8.7.
# Setting this stops PyTorch's JIT/Triton from building kernels for
# every other arch it would otherwise probe at runtime.
JETSON_TORCH_ARCH = "8.7"

# AMD GFX 1151 ROCm target architecture string (used by HIP/ROCm toolchain).
AMD_GFX_ARCH = "gfx1151"

# Valid TARGET env var values.
TARGET_JETSON = "jetson"
TARGET_AMD = "amd-gfx1151"


def is_jetson() -> bool:
    """
    Heuristically detect an NVIDIA Jetson device.

    Checks (in order):
    1. ``/etc/nv_tegra_release`` — present on all Jetson L4T images.
    2. ``/proc/device-tree/model`` — contains "NVIDIA Jetson" on Jetson hardware.
    3. The ``JETSON_MODEL_NAME`` env var — useful inside containers that strip
       the device tree but expose the variable.
    """
    if Path("/etc/nv_tegra_release").exists():
        return True

    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            content = model_path.read_text(errors="ignore")
            if "jetson" in content.lower():
                return True
        except OSError:
            pass

    if "jetson" in os.environ.get("JETSON_MODEL_NAME", "").lower():
        return True

    return False


def pip_install(
    primary_index: str | None,
    dev: bool,
    torch_arch: str | None,
    rocm_arch: str | None,
) -> None:
    """
    Run pip install with the appropriate index flags.

    When a ``primary_index`` is provided it is passed as ``--index-url`` so
    pip checks it *first*. PyPI is appended as ``--extra-index-url`` so it
    acts as a fallback when a wheel is not found in the primary index.

    When ``torch_arch`` is provided, ``TORCH_CUDA_ARCH_LIST`` is set in the
    subprocess environment so that any CUDA extension builds (Triton,
    bitsandbytes, etc.) target only that compute capability.

    When ``rocm_arch`` is provided, ``PYTORCH_ROCM_ARCH`` is set so that
    HIP/ROCm extension builds target only that architecture.

    Args:
        primary_index: Custom index URL to probe first (Jetson or AMD ROCm).
                       PyPI is used as fallback. ``None`` uses PyPI directly.
        dev:           If True, also install the ``[dev]`` optional group.
        torch_arch:    CUDA compute capability string, e.g. ``"8.7"`` for
                       Ampere (Orin Nano). ``None`` leaves the env var unset.
        rocm_arch:     ROCm/HIP architecture string, e.g. ``"gfx1151"``.
                       ``None`` leaves the env var unset.
    """
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]

    if dev:
        cmd[-1] = ".[dev]"

    if primary_index:
        # --index-url makes this the *primary* source; PyPI becomes the fallback.
        cmd += ["--index-url", primary_index]
        cmd += ["--extra-index-url", "https://pypi.org/simple"]
    # No primary_index → pip's default behaviour (PyPI only) is unchanged.

    env = os.environ.copy()

    if torch_arch:
        env["TORCH_CUDA_ARCH_LIST"] = torch_arch
        print(f"TORCH_CUDA_ARCH_LIST={torch_arch}  (Ampere sm_{torch_arch.replace('.', '')})")

    if rocm_arch:
        env["PYTORCH_ROCM_ARCH"] = rocm_arch
        print(f"PYTORCH_ROCM_ARCH={rocm_arch}")

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install rtd-gdes with the correct package index.")

    jetson_group = parser.add_mutually_exclusive_group()
    jetson_group.add_argument(
        "--jetson",
        action="store_true",
        default=False,
        help="Force Jetson installation mode (adds Jetson AI Lab index).",
    )
    jetson_group.add_argument(
        "--no-jetson",
        action="store_true",
        default=False,
        help="Force standard installation mode (skips Jetson detection).",
    )
    parser.add_argument(
        "--amd-gfx1151",
        action="store_true",
        default=False,
        help="Target AMD GFX 1151 (uses ROCm nightly index as primary source).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Also install development dependencies (pytest, ruff, black).",
    )

    args = parser.parse_args()

    if args.jetson and args.amd_gfx1151:
        parser.error("--jetson and --amd-gfx1151 are mutually exclusive.")

    # Read TARGET env var as the base signal, then let CLI flags override it.
    target_env = os.environ.get("TARGET", "").lower().strip()

    if args.no_jetson:
        use_jetson = False
        use_amd = False
        reason = "disabled via --no-jetson"
    elif args.jetson or target_env == TARGET_JETSON:
        use_jetson = True
        use_amd = False
        reason = "enabled via --jetson flag" if args.jetson else "set via TARGET env var"
    elif args.amd_gfx1151 or target_env == TARGET_AMD:
        use_jetson = False
        use_amd = True
        reason = "enabled via --amd-gfx1151 flag" if args.amd_gfx1151 else "set via TARGET env var"
    else:
        use_amd = False
        use_jetson = is_jetson()
        reason = "auto-detected" if use_jetson else "not detected"

    # Determine primary index and arch env vars.
    if use_jetson:
        primary_index = JETSON_INDEX
        torch_arch = JETSON_TORCH_ARCH
        rocm_arch = None
    elif use_amd:
        primary_index = AMD_GFX1151_INDEX
        torch_arch = None
        rocm_arch = AMD_GFX_ARCH
    else:
        primary_index = None
        torch_arch = None
        rocm_arch = None

    print(f"Platform : {platform.machine()} / {platform.system()}")
    print(f"Jetson   : {use_jetson} ({reason})")
    if use_jetson:
        print(f"Index    : {primary_index}  →  fallback: pypi.org")
        print(f"Arch     : sm_{JETSON_TORCH_ARCH.replace('.', '')} (Ampere / Orin Nano)")
    if use_amd:
        print(f"AMD GFX  : {AMD_GFX_ARCH}")
        print(f"Index    : {primary_index}  →  fallback: pypi.org")
    print()

    pip_install(
        primary_index=primary_index,
        dev=args.dev,
        torch_arch=torch_arch,
        rocm_arch=rocm_arch,
    )


if __name__ == "__main__":
    main()
