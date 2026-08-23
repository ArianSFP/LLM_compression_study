#!/usr/bin/env python3
"""Launch the existing frozen allocator at the legal 384-page/0.523478-bpw point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


path = Path("/workspace/run_all_layer_allocator_single.py")
spec = importlib.util.spec_from_file_location("allocator_0523", path)
if spec is None or spec.loader is None:
    raise RuntimeError(path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.BUDGET = 384
module.RATE = 0.523478
module.main()
