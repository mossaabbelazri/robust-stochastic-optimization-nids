"""
Visualization & Figure Generation: Heavy-Tailed Optimization on NIDS
=====================================================================
Generates high-resolution academic figures (300 DPI) using Matplotlib:
1. convergence_heavy_tailed.png: Excess risk / loss trajectories with confidence bands.
2. gradient_norm_distribution.png: Heavy-tailed gradient bursts vs clipping threshold.
3. clipping_threshold_ablation.png: Trade-off curve showing optimal clipping window.
4. tail_index_sensitivity.png: Breakdown of Standard SGD vs Clipped SGD as nu -> 1.0.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Set academic publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    results_path = os.path.join(root_dir, "results", "benchmark_results.json")
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}. Run run_experiments.py first.")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    conv = data["convergence"]
    
    # -------------------------------------------------------------
    # Figure 1: Convergence Trajectories under Heavy-Tailed Noise
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    
    colors = {
        "Standard SGD": "#D32F2F",            # Red (fails/diverges)
        "Clipped SGD (Fixed)": "#1976D2",     # Blue (solid)
        "Clipped SGD (Adaptive)": "#388E3C",  # Green (best theoretical)
        "Normalized SGD": "#7B1FA2"          # Purple
    }
    
    for opt_name, metrics in conv.items():
        steps = np.array(metrics["steps"])
        median = np.array(metrics["median"])
        q25 = np.array(metrics["q25"])
        q75 = np.array(metrics["q75"])
        
        color = colors.get(opt_name, "#333333")
        ax.plot(steps, median, label=opt_name, color=color)
        ax.fill_between(steps, q25, q75, color=color, alpha=0.18)
        
    ax.set_xlabel("Iteration Step ($t$)")
    ax.set_ylabel("Empirical Population Loss $\\mathcal{L}(w_t)$")
    ax.set_title("Stochastic Convergence under Heavy-Tailed Noise (Student-t, $\\nu = 1.35$)")
    ax.set_ylim(0.2, 1.2)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "convergence_heavy_tailed.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[Generated] {fig1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Heavy-Tailed Gradient Burst & Clipping Threshold
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    grad_norms = np.array(data["raw_gradient_norms"])
    
    steps_x = np.arange(1, len(grad_norms) + 1)
    ax.scatter(steps_x, grad_norms, color="#E65100", alpha=0.6, s=18, label=r"Stochastic Gradient Norm $\|g_t\|_2$")
    ax.axhline(y=1.2, color="#1976D2", linestyle="--", linewidth=2.2, label=r"Clipping Threshold $\tau = 1.2$")
    
    ax.set_yscale("log")
    ax.set_xlabel("Iteration Step ($t$)")
    ax.set_ylabel(r"Gradient Norm $\|g_t\|_2$ (Log Scale)")
    ax.set_title("Gradient Norm Explosions from Heavy-Tailed Noise (Infinite Variance)")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "gradient_norm_bursts.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[Generated] {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Sensitivity to Tail Index nu (Phase Transition)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sens = data["sensitivity"]
    nu_vals = sens["nu_values"]
    f1_std = [v * 100 for v in sens["Standard SGD"]]
    f1_clip = [v * 100 for v in sens["Clipped SGD (Fixed)"]]
    
    ax.plot(nu_vals, f1_clip, marker="o", color="#1976D2", linewidth=2.2, label="Clipped SGD ($\\tau = 1.2$)")
    ax.plot(nu_vals, f1_std, marker="s", color="#D32F2F", linewidth=2.2, linestyle="--", label="Standard SGD")
    
    ax.axvline(x=2.0, color="gray", linestyle=":", label="Finite Variance Boundary ($\\nu = 2.0$)")
    ax.set_xlabel("Degrees of Freedom / Tail Index $\\nu$")
    ax.set_ylabel("NIDS F1-Score (%)")
    ax.set_title("NIDS Robustness across Noise Tail Regimes")
    ax.set_ylim(40, 100)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "tail_index_sensitivity.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[Generated] {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Clipping Threshold Ablation (Bias-Variance Tradeoff)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ablation = data["threshold_ablation"]
    taus = ablation["thresholds"]
    f1_scores = [v * 100 for v in ablation["f1_scores"]]
    
    ax.plot(taus, f1_scores, marker="^", color="#2E7D32", linewidth=2.2)
    ax.set_xscale("log")
    ax.set_xlabel("Clipping Threshold $\\tau$ (Log Scale)")
    ax.set_ylabel("NIDS F1-Score (%)")
    ax.set_title("Bias-Variance Trade-Off: Impact of Clipping Threshold $\\tau$")
    
    # Annotate optimal zone
    best_idx = np.argmax(f1_scores)
    best_tau = taus[best_idx]
    best_score = f1_scores[best_idx]
    ax.annotate(f"Optimal $\\tau^* \\approx {best_tau}$ ({best_score:.1f}%)",
                xy=(best_tau, best_score),
                xytext=(best_tau * 1.5, best_score - 10),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6))
                
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "clipping_threshold_ablation.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"[Generated] {fig4_path}")

    print("\n[OK] All academic figures successfully generated!")

if __name__ == "__main__":
    main()
