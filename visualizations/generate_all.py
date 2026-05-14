"""Run all visualization scripts to populate report/figures/."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "visualizations/fig1_dataset.py",
    "visualizations/fig2_noise_types.py",
    "visualizations/fig3_preprocessing.py",
    "visualizations/fig4_results.py",
    "visualizations/fig5_visual.py",
    "visualizations/fig6_efficiency.py",
]

python = sys.executable

for script in SCRIPTS:
    print(f"\n=== Running {script} ===")
    result = subprocess.run([python, script], capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: {script} exited with code {result.returncode}")

print("\nAll visualization scripts completed.")
print("Figures saved to report/figures/")
