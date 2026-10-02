"""Thin CLI wrapper: `python scripts/run_phase1.py --csv data/aqi.csv`"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1.train_baselines import run_phase1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", default="results/phase1")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_phase1(args.csv, args.output_dir, args.test_frac, args.seed)
