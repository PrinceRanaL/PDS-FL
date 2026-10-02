"""
SemBF duplicate-submission test. For T rounds, each of N legitimate clients
submits once; in every round a random subset of clients also submits 1-3
extra duplicate updates (interleaved at random positions). Reports how many
duplicates were wrongly accepted (false negatives) and how many honest first
submissions were wrongly rejected (false positives), with Wilson 95% CIs.
Scope: duplicates within the same round only.
"""
import argparse, json, math, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phase2.sembf import SemaphoreBloomFilter


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((c - m) / d, (c + m) / d)


def run(n_clients, rounds, size, num_hashes, dup_frac, seed):
    rng = random.Random(seed)
    sem = SemaphoreBloomFilter(size=size, num_hashes=num_hashes)
    clients = [f"city_{i:03d}" for i in range(n_clients)]
    dup_total = dup_accepted = first_total = first_rejected = 0
    for t in range(1, rounds + 1):
        sem.reset()
        events = [(c, False) for c in clients]
        for c in clients:
            if rng.random() < dup_frac:
                events += [(c, True)] * rng.randint(1, 3)
        rng.shuffle(events)
        seen = set()
        for c, _ in events:
            accepted = sem.query_and_lock(c, t)
            if c in seen:                       # this is a duplicate
                dup_total += 1
                dup_accepted += accepted
            else:                               # first submission
                first_total += 1
                first_rejected += (not accepted)
                seen.add(c)
    res = {"n_clients": n_clients, "rounds": rounds, "bit_array_size": size,
           "num_hashes": num_hashes, "duplicates_submitted": dup_total,
           "duplicates_accepted_false_negatives": dup_accepted,
           "detection_rate": 1 - dup_accepted / dup_total,
           "honest_first_submissions": first_total,
           "honest_first_rejected_false_positives": first_rejected,
           "honest_false_positive_rate": first_rejected / first_total,
           "honest_fpr_wilson95": list(wilson(first_rejected, first_total))}
    print(json.dumps(res, indent=2)); return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_clients", type=int, default=29)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--num_hashes", type=int, default=4)
    ap.add_argument("--dup_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/phase2/sembf_dedup.json")
    a = ap.parse_args()
    r = run(a.n_clients, a.rounds, a.size, a.num_hashes, a.dup_frac, a.seed)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    json.dump(r, open(a.output, "w"), indent=2)
