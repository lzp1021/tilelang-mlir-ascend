from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Optional, Sequence


def run_target(target: Path) -> None:
    import torch

    original_set_device = torch.npu.set_device
    original_set_device(0)

    def set_device_zero(*_args, **_kwargs) -> None:
        original_set_device(0)

    torch.npu.set_device = set_device_zero
    original_argv = sys.argv
    sys.argv = [str(target)]
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = original_argv
        torch.npu.set_device = original_set_device


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one autotune script on NPU0")
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    run_target(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
