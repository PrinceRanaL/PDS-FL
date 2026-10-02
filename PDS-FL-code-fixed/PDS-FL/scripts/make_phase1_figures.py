"""Regenerates the Phase-1 figures from results/phase1/phase1_results.json."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/phase1/phase1_results.json"
out = sys.argv[2] if len(sys.argv) > 2 else "results/phase1/figures"
os.makedirs(out, exist_ok=True)
d = json.load(open(src))
order = ["LinearRegression", "SVR", "RandomForest", "LightGBM", "XGBoost"]
lab = {"LinearRegression": "LR", "SVR": "SVR", "RandomForest": "RF", "LightGBM": "LightGBM", "XGBoost": "XGBoost"}
col = {"LR": "#d9534f", "SVR": "#f0ad4e", "RF": "#5bc0de", "LightGBM": "#2e8b57", "XGBoost": "#337ab7"}
names = [lab[k] for k in order]

def bars(metric, fname, ylabel, title, both=True):
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    if both:
        tr = [d[k]["train"][metric] for k in order]; te = [d[k]["test"][metric] for k in order]
        ax.bar(x - w/2, tr, w, label="Train", color="#9ecae1"); ax.bar(x + w/2, te, w, label="Test", color="#3182bd")
        for i, (a, b) in enumerate(zip(tr, te)):
            ax.text(i - w/2, a, f"{a:.3f}" if metric == "r2" else f"{a:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w/2, b, f"{b:.3f}" if metric == "r2" else f"{b:.1f}", ha="center", va="bottom", fontsize=7)
        ax.legend()
    else:
        te = [d[k]["test"][metric] for k in order]
        ax.bar(x, te, 0.6, color=[col[n] for n in names])
        for i, b in enumerate(te): ax.text(i, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
    if metric == "r2": ax.set_ylim(0.6, 1.0)
    fig.tight_layout(); fig.savefig(os.path.join(out, fname), dpi=200); plt.close(fig)

bars("r2", "Phase1_R2.png", "Test $R^2$", "Test $R^2$ of ML models", both=False)
bars("r2", "Phase1_R2_TrainTest.png", "$R^2$", "Train vs test $R^2$")
bars("rmse", "Phase1_RMSE.png", "RMSE", "Train vs test RMSE")
bars("mae", "Phase1_MAE.png", "MAE", "Train vs test MAE")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for k in order:
    n = lab[k]; bias = d[k]["train"]["rmse"]; var = d[k]["test"]["rmse"] - d[k]["train"]["rmse"]
    ax.scatter(bias, var, s=140, color=col[n], edgecolor="k", zorder=3)
    ax.annotate(f"{n}\n(test R$^2$={d[k]['test']['r2']:.3f})", (bias, var), textcoords="offset points", xytext=((8, -26) if n == "XGBoost" else (8, 6)), fontsize=8)
ax.axhline(0, color="gray", lw=.8, ls="--")
ax.set_xlabel("Train RMSE (bias proxy; lower = lower bias)")
ax.set_ylabel("Test RMSE $-$ Train RMSE (gap proxy)")
ax.set_title("Measured bias-variance indicators (single run)", fontsize=10); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(out, "Phase1_BiasVariance.png"), dpi=200)
print("done")
