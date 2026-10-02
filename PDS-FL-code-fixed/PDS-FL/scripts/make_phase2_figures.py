"""Figures for Section 5.3: centralised vs federated comparison and per-city results."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = "results/phase2"
p1 = json.load(open("results/phase1/phase1_results.json"))["LightGBM"]["test"]
bc = __import__("pandas").read_csv(f"{out}/convergence_best_checkpoint.csv")
fp = bc[bc.aggregation == "FedProx"].iloc[-1]
fed = {"r2": fp.best_r2, "rmse": fp.best_rmse, "mae": fp.best_mae}
fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
for ax, k, lab in zip(axes, ["r2", "rmse", "mae"], ["Test $R^2$", "Test RMSE", "Test MAE"]):
    v = [p1[k], fed[k]]
    ax.bar(["Centralised\nLightGBM", "Federated\n(FedProx, best ckpt)"], v, color=["#3182bd", "#e6550d"], width=.55)
    for i, x in enumerate(v): ax.text(i, x, f"{x:.3f}" if k == "r2" else f"{x:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(lab, fontsize=10); ax.grid(axis="y", alpha=.3); ax.tick_params(axis="x", labelsize=7)
fig.tight_layout(); fig.savefig(f"{out}/central_vs_federated.png", dpi=200); plt.close(fig)

rows = json.load(open(f"{out}/per_city.json"))["cities"]
rows = sorted(rows, key=lambda r: r["rmse"])
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
y = np.arange(len(rows))
axes[0].barh(y, [r["rmse"] for r in rows], color=plt.cm.viridis_r(np.array([r["r2"] for r in rows])))
axes[0].set_yticks(y); axes[0].set_yticklabels([r["city"].title() for r in rows], fontsize=7)
axes[0].set_xlabel("Test RMSE of federated model (AQI units)"); axes[0].set_title("Per-city test RMSE (colour = test $R^2$, darker = higher)", fontsize=9)
axes[0].grid(axis="x", alpha=.3)
tm = [r["train_mean_aqi"] for r in rows]; sm = [r["test_mean_aqi"] for r in rows]
axes[1].scatter(tm, sm, c="#3182bd", s=25)
lim = [40, 210]; axes[1].plot(lim, lim, "k--", lw=.8)
for r in rows:
    if r["rmse"] > 40: axes[1].annotate(r["city"].title(), (r["train_mean_aqi"], r["test_mean_aqi"]), fontsize=8, xytext=(4, -10), textcoords="offset points")
axes[1].set_xlabel("Mean AQI, training period"); axes[1].set_ylabel("Mean AQI, test period"); axes[1].set_title("Train-to-test shift per city", fontsize=9); axes[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{out}/per_city.png", dpi=200)
print("ok", fed)
