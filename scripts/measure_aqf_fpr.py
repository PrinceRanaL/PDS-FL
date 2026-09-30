"""
Empirically measures the AQF authentication false-positive rate under
simulated unauthorised-client attacks, with a Wilson-score confidence
interval, for reporting alongside Table 2 / Section 4.2.1's stated
epsilon ~ 0.001. Useful evidence for reviewer requests about
authentication-attempt counts and confidence intervals.

Usage: python scripts/measure_aqf_fpr.py --n_legit 29 --n_attacks 100000
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase2.aqf import AdaptiveQuotientFilter, make_credential


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def run(n_legit: int, n_attacks: int, capacity: int, target_fpr: float, seed: int = 42):
    aqf = AdaptiveQuotientFilter(capacity=capacity, target_fpr=target_fpr)
    legit_ids = [make_credential(f"city_{i:03d}") for i in range(n_legit)]
    for cid in legit_ids:
        aqf.register(cid)

    attacker_ids = [make_credential(f"attacker_{seed}_{i:07d}") for i in range(n_attacks)]
    accepted = sum(aqf.authenticate(cid) for cid in attacker_ids)
    fpr = accepted / n_attacks
    lo, hi = wilson_ci(accepted, n_attacks)

    # sanity: every legitimate client must always authenticate (no false
    # negatives, matching the paper's 100% legitimate-authentication claim).
    legit_accept_rate = sum(aqf.authenticate(cid) for cid in legit_ids) / n_legit

    result = {
        "n_legitimate_clients": n_legit,
        "n_unauthorised_attack_attempts": n_attacks,
        "false_positives_accepted": accepted,
        "empirical_fpr": fpr,
        "wilson_95ci_low": lo,
        "wilson_95ci_high": hi,
        "theoretical_fpr": aqf.theoretical_fpr(),
        "target_fpr": target_fpr,
        "remainder_bits_final": aqf.remainder_bits,
        "n_rebuilds": aqf.n_rebuilds,
        "legitimate_client_acceptance_rate": legit_accept_rate,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_legit", type=int, default=29)
    parser.add_argument("--n_attacks", type=int, default=100000)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--target_fpr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run(args.n_legit, args.n_attacks, args.capacity, args.target_fpr, args.seed)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
