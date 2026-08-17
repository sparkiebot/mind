"""Report the local inference capabilities without downloading or loading a model."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Any


def _memory_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        return int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def report() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "memory_bytes": _memory_bytes(),
        "mps_available": False,
        "torch_version": None,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["mps_available"] = torch.backends.mps.is_available()
        result["mps_built"] = torch.backends.mps.is_built()
    except ImportError:
        result["torch_error"] = "PyTorch is not installed."
    return result


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
