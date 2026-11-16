"""One-command reproduction: train -> report.

    python run_all.py            # full run (~6-10 min on CPU)
    python run_all.py --fast     # smoke run (~2 min)
    python run_all.py --skip-report
"""

from __future__ import annotations

import argparse
import time

from src import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    train.main(fast=args.fast)
    if not args.skip_report:
        from src import report

        path = report.build()
        print(f"report -> {path}")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
