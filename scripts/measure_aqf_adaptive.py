"""
Adaptive AQF sweep: enrol n clients for increasing n (remainder width r is
increased automatically whenever the bound 1-(1-2^-r)^n exceeds the target),
then measure the empirical false-positive rate against unregistered
attackers, with Wilson 95% CIs. Produces results/phase2/aqf_adaptive.json
and .png (theoretical bound vs. empirical FPR vs. n).
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phase2.aqf import AdaptiveQuotientFilter, make_credential
from scripts.measure_aqf_fpr import wilson_ci


def run(sizes, n_attacks, target, out):
    rows = []
    for n in sizes:
        aqf = AdaptiveQuotientFilter(capacity=64, target_fpr=target)
        for i in range(n):
            aqf.register(make_credential(f"c{i:06d}"))
        acc = sum(aqf.authenticate(make_credential(f"x{i:08d}")) for i in range(n_attacks))
        lo, hi = wilson_ci(acc, n_attacks)
        legit = all(aqf.authenticate(make_credential(f"c{i:06d}")) for i in range(n))
        rows.append({"n": n, "remainder_bits": aqf.remainder_bits, "rebuilds": aqf.n_rebuilds,
                     "theoretical_bound": aqf.theoretical_fpr(), "attacks": n_attacks,
                     "accepted": acc, "empirical_fpr": acc / n_attacks,
                     "wilson_low": lo, "wilson_high": hi, "all_legit_accepted": legit})
        print(rows[-1])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ns, [r["theoretical_bound"] for r in rows], "s--", color="#d62728", label="Theoretical bound $1-(1-2^{-r})^n$")
    emp = [max(r["empirical_fpr"], 1e-7) for r in rows]
    err = [[e - max(r["wilson_low"], 1e-7) for e, r in zip(emp, rows)], [r["wilson_high"] - e for e, r in zip(emp, rows)]]
    ax.errorbar(ns, emp, yerr=err, fmt="o-", color="#1f77b4", capsize=3, label="Empirical FPR (95% Wilson CI)")
    ax.axhline(target, color="gray", ls=":", label=f"Target {target}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Enrolled clients $n$"); ax.set_ylabel("False-positive rate")
    ax.set_title("Adaptive AQF: remainder width grows to hold FPR bound"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out.replace(".json", ".png"), dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[29, 64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--n_attacks", type=int, default=300000)
    ap.add_argument("--target", type=float, default=0.001)
    ap.add_argument("--output", default="results/phase2/aqf_adaptive.json")
    a = ap.parse_args(); run(a.sizes, a.n_attacks, a.target, a.output)
