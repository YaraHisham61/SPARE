#!/usr/bin/env python
# AI-GENERATED - Check evaluation progress

import os
import json
from pathlib import Path
from datetime import datetime

exp_dir = Path("experiments/robustness_eval_2026-05-07")

print(f"\n{'='*70}")
print(f"EVALUATION PROGRESS CHECK - {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*70}\n")

if exp_dir.exists():
    files = list(exp_dir.glob("*.*"))
    print(f"✓ Results directory exists: {exp_dir}")
    print(f"  Files created: {len(files)}")
    
    for f in sorted(files):
        size = f.stat().st_size
        modified = datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S")
        print(f"  - {f.name:30s} ({size:10,d} bytes) - {modified}")
    
    if "robustness_results.csv" in [f.name for f in files]:
        with open(exp_dir / "robustness_results.csv") as f:
            lines = f.readlines()
        print(f"\n  CSV rows: {len(lines)-1}")  # Exclude header
else:
    print(f"✗ Results directory not yet created")

print(f"\n{'='*70}\n")
