"""
Scalability test for the PDS security layer (Reviewer 2, Comment 7):
measures AQF registration/authentication time, SemBF per-round
query/lock time, and memory footprint, as the client population N
grows from the paper's evaluated N=29 up to N=10,000 synthetic
clients -- well beyond the 29-city federation actually deployed.

Usage: python scripts/measure_scalability.py --sizes 29 100 500 1000 5000 10000
"""
import argparse
import json
import sys
import os
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase2.aqf import AdaptiveQuotientFilter, make_credential
from src.phase2.sembf import SemaphoreBloomFilter


def measure_one(n_clients: int, n_auth_attempts: int = 5000, seed: int = 42):
    client_ids = [make_credential(f"client_{seed}_{i:07d}") for i in range(n_clients)]

    tracemalloc.start()
    aqf = AdaptiveQuotientFilter(capacity=max(64, n_clients * 2), target_fpr=0.001)
    t0 = time.perf_counter()
    for cid in client_ids:
        aqf.register(cid)
    t_register_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    for cid in client_ids[:n_auth_attempts]:
        aqf.authenticate(cid)
    t_auth_total = time.perf_counter() - t0
    _, aqf_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    sembf = SemaphoreBloomFilter(size=max(4096, n_clients * 16), num_hashes=4)
    sembf.reset()
    t0 = time.perf_counter()
    for cid in client_ids[:n_auth_attempts]:
        sembf.query_and_lock(cid, round_idx=1)
    t_sembf_total = time.perf_counter() - t0
    _, sembf_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_auth = min(n_auth_attempts, n_clients)
    return {
        "n_clients": n_clients,
        "aqf_register_total_s": t_register_total,
        "aqf_register_per_client_us": (t_register_total / n_clients) * 1e6,
        "aqf_auth_total_s": t_auth_total,
        "aqf_auth_per_call_us": (t_auth_total / n_auth) * 1e6,
        "aqf_peak_memory_kb": aqf_peak_mem / 1024,
        "aqf_remainder_bits": aqf.remainder_bits,
        "aqf_rebuilds": aqf.n_rebuilds,
        "sembf_query_total_s": t_sembf_total,
        "sembf_query_per_call_us": (t_sembf_total / n_auth) * 1e6,
        "sembf_peak_memory_kb": sembf_peak_mem / 1024,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+",
                         default=[29, 100, 500, 1000, 5000, 10000])
    parser.add_argument("--output", default="results/phase2/scalability.json")
    args = parser.parse_args()

    results = []
    for n in args.sizes:
        r = measure_one(n)
        print(json.dumps(r, indent=2))
        results.append(r)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
