"""Run every figure script: python viz/run_all.py --csv "./*results*.csv" --out figures"""
import glob, os, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
for s in sorted(glob.glob(os.path.join(here, "fig*.py"))):
    print(f"== {os.path.basename(s)}")
    subprocess.run([sys.executable, s, *sys.argv[1:]], check=False)
