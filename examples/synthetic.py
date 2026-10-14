#!/usr/bin/env python3
"""Write three synthetic decision files with known calibration, for trying calibaudit.

    python3 examples/synthetic.py /tmp/calib_demo

honest.csv     the outcome really happens with probability p
probit15.csv   p = Phi(z), but the spread behind z is 1.5x what the forecast assumed: the truth is Phi(z / 1.5)
jumps.csv      p = Phi(z) with the spread right 90% of the time and 4x larger the other 10% (rare jumps):
               the truth is 0.9 Phi(z) + 0.1 Phi(z / 4), which flattens near 0.95 however large z gets
All three: 40 days x 150 decisions, seeded, p between 0.6 and 0.999.
"""
import csv
import os
import random
import sys
from statistics import NormalDist

ND = NormalDist()


def write(path, truth, seed):
    rng = random.Random(seed)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("p", "outcome", "day"))
        for d in range(40):
            for _ in range(150):
                p = round(rng.uniform(0.6, 0.999), 4)
                w.writerow((p, int(rng.random() < truth(p)), f"2026-{1 + d // 28:02d}-{1 + d % 28:02d}"))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, "honest.csv"), lambda p: p, 1)
    write(os.path.join(out, "probit15.csv"), lambda p: ND.cdf(ND.inv_cdf(p) / 1.5), 2)
    write(os.path.join(out, "jumps.csv"), lambda p: 0.9 * p + 0.1 * ND.cdf(ND.inv_cdf(p) / 4), 3)
    print(f"wrote honest.csv, probit15.csv and jumps.csv to {out}")
