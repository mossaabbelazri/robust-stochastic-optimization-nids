"""
Empirical Benchmark: Heavy-Tailed Stochastic Optimization on Network Intrusion Detection
========================================================================================
Runs controlled experiments comparing Standard SGD, Clipped SGD, and Normalized SGD.
Saves metrics for publication-grade visualization.
"""

import os
import json
import argparse
import numpy as np
from heavy_tailed_optimizers import StandardSGD, ClippedSGD, NormalizedSGD
from data_generator import load_dataset, sample_heavy_tailed_noise, compute_logistic_gradient

def run_single_optimization_trajectory(optimizer_name, opt_kwargs, X_train, y_train, 
                                      n_steps=1500, nu=1.4, noise_scale=2.0, seed=42):
    """
    Run one stochastic optimization trajectory under heavy-tailed gradient noise.
    """
    np.random.seed(seed)
    n_samples, dim = X_train.shape
    w = np.zeros(dim)
    
    if optimizer_name == "Standard SGD":
        opt = StandardSGD(**opt_kwargs)
    elif optimizer_name == "Clipped SGD (Fixed)":
        opt = ClippedSGD(**opt_kwargs)
    elif optimizer_name == "Clipped SGD (Adaptive)":
        opt = ClippedSGD(adaptive_threshold=True, **opt_kwargs)
    elif optimizer_name == "Normalized SGD":
        opt = NormalizedSGD(**opt_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    losses = []
    grad_norms = []
    
    for t in range(1, n_steps + 1):
        idx = np.random.randint(0, n_samples)
        x_i = X_train[idx]
        y_i = y_train[idx]
        
        # Exact stochastic gradient
        loss, grad_exact = compute_logistic_gradient(w, x_i, y_i)
        
        # Heavy-tailed noise perturbation: nu in (1, 2] implies infinite variance
        noise = sample_heavy_tailed_noise(dim, nu=nu, scale=noise_scale)
        stochastic_grad = grad_exact + noise
        
        grad_norm = np.linalg.norm(stochastic_grad)
        grad_norms.append(float(grad_norm))
        
        # Step
        w = opt.step(w, stochastic_grad, iteration=t)
        
        # Record population loss periodically
        if t % 10 == 0 or t == n_steps:
            # Evaluate test-sample average loss
            eval_indices = np.random.choice(n_samples, size=min(200, n_samples), replace=False)
            eval_loss = np.mean([compute_logistic_gradient(w, X_train[k], y_train[k])[0] for k in eval_indices])
            losses.append((t, float(eval_loss)))
            
    return w, losses, grad_norms


def evaluate_classifier(w, X_test, y_test):
    """
    Compute Accuracy, Precision, Recall, F1 for the binary classifier w^T x.
    """
    scores = np.dot(X_test, w)
    preds = (scores >= 0).astype(int)
    
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    tn = np.sum((preds == 0) & (y_test == 0))
    
    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def main():
    parser = argparse.ArgumentParser(description="Heavy-Tailed Stochastic Optimization Benchmark")
    parser.add_argument("--data", type=str, default=None, help="Optional path to real network CSV dataset")
    parser.add_argument("--seeds", type=int, default=15, help="Number of independent random seeds (default: 15)")
    args = parser.parse_args()

    print("=" * 70)
    print("Empirical Benchmark: Heavy-Tailed Optimization on Network Telemetry")
    print("=" * 70)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Dataset Loading (Real CSV or Controlled Synthetic)
    X, y = load_dataset(csv_path=args.data, n_samples=6000, seed=123)
    train_size = int(0.75 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Configuration
    n_seeds = args.seeds
    n_steps = 1200
    nu_heavy_tail = 1.35  # Nu in (1, 2]: Finite p-th moment for p < 1.35, infinite variance
    noise_scale = 1.5
    
    optimizers_to_test = [
        ("Standard SGD", {"lr": 0.08, "lr_decay": 0.5}),
        ("Clipped SGD (Fixed)", {"lr": 0.08, "clip_threshold": 1.2, "lr_decay": 0.5}),
        ("Clipped SGD (Adaptive)", {"lr": 0.08, "clip_threshold": 0.8, "lr_decay": 0.5, "p_moment": 1.35}),
        ("Normalized SGD", {"lr": 0.04, "lr_decay": 0.5})
    ]
    
    convergence_data = {}
    f1_results = {}
    all_grad_norms = {}
    
    print(f"\n--- Phase 1: Convergence Trajectories (nu = {nu_heavy_tail}, {n_seeds} seeds) ---")
    for opt_name, opt_kwargs in optimizers_to_test:
        print(f"Testing {opt_name}...")
        trajectories = []
        f1_list = []
        collected_norms = []
        
        for s in range(n_seeds):
            w_final, losses, grad_norms = run_single_optimization_trajectory(
                opt_name, opt_kwargs, X_train, y_train,
                n_steps=n_steps, nu=nu_heavy_tail, noise_scale=noise_scale, seed=42 + s
            )
            # losses is list of (step, loss_val)
            trajectories.append([val for _, val in losses])
            steps_axis = [step for step, _ in losses]
            
            metrics = evaluate_classifier(w_final, X_test, y_test)
            f1_list.append(metrics["f1"])
            if s == 0:
                collected_norms = grad_norms[:500]
                
        trajectories = np.array(trajectories) # shape: (n_seeds, n_eval_points)
        mean_loss = np.mean(trajectories, axis=0)
        median_loss = np.median(trajectories, axis=0)
        std_loss = np.std(trajectories, axis=0)
        q25 = np.percentile(trajectories, 25, axis=0)
        q75 = np.percentile(trajectories, 75, axis=0)
        
        convergence_data[opt_name] = {
            "steps": steps_axis,
            "mean": mean_loss.tolist(),
            "median": median_loss.tolist(),
            "std": std_loss.tolist(),
            "q25": q25.tolist(),
            "q75": q75.tolist()
        }
        f1_results[opt_name] = {
            "mean_f1": float(np.mean(f1_list)),
            "std_f1": float(np.std(f1_list)),
            "median_f1": float(np.median(f1_list)),
            "all_f1": f1_list
        }
        all_grad_norms[opt_name] = collected_norms
        
        print(f"  -> {opt_name} | Median Loss: {median_loss[-1]:.4f} | F1: {np.mean(f1_list)*100:.2f}% (+/- {np.std(f1_list)*100:.2f}%)")

    # 2. Phase 2: Sensitivity to Tail Index nu in [1.2, 2.5]
    print("\n--- Phase 2: Sensitivity to Heavy-Tail Index nu ---")
    nu_values = [1.2, 1.35, 1.5, 1.8, 2.2, 3.0]
    sensitivity_data = {"nu_values": nu_values, "Standard SGD": [], "Clipped SGD (Fixed)": []}
    
    for nu_val in nu_values:
        f1_std = []
        f1_clip = []
        for s in range(5):
            w_std, _, _ = run_single_optimization_trajectory(
                "Standard SGD", {"lr": 0.08, "lr_decay": 0.5}, X_train, y_train,
                n_steps=800, nu=nu_val, noise_scale=noise_scale, seed=100 + s
            )
            w_clip, _, _ = run_single_optimization_trajectory(
                "Clipped SGD (Fixed)", {"lr": 0.08, "clip_threshold": 1.2, "lr_decay": 0.5}, X_train, y_train,
                n_steps=800, nu=nu_val, noise_scale=noise_scale, seed=100 + s
            )
            f1_std.append(evaluate_classifier(w_std, X_test, y_test)["f1"])
            f1_clip.append(evaluate_classifier(w_clip, X_test, y_test)["f1"])
            
        sensitivity_data["Standard SGD"].append(float(np.mean(f1_std)))
        sensitivity_data["Clipped SGD (Fixed)"].append(float(np.mean(f1_clip)))
        print(f"  nu = {nu_val:4.1f} | Standard SGD F1: {np.mean(f1_std)*100:.2f}% | Clipped SGD F1: {np.mean(f1_clip)*100:.2f}%")

    # 3. Phase 3: Ablation of Clipping Threshold tau
    print("\n--- Phase 3: Clipping Threshold tau Ablation ---")
    thresholds = [0.1, 0.3, 0.8, 1.2, 2.5, 5.0, 15.0]
    threshold_f1 = []
    for tau in thresholds:
        f1_vals = []
        for s in range(5):
            w_tau, _, _ = run_single_optimization_trajectory(
                "Clipped SGD (Fixed)", {"lr": 0.08, "clip_threshold": tau, "lr_decay": 0.5}, X_train, y_train,
                n_steps=800, nu=1.35, noise_scale=noise_scale, seed=200 + s
            )
            f1_vals.append(evaluate_classifier(w_tau, X_test, y_test)["f1"])
        threshold_f1.append(float(np.mean(f1_vals)))
        print(f"  tau = {tau:4.1f} | F1 Score: {np.mean(f1_vals)*100:.2f}%")

    threshold_ablation = {
        "thresholds": thresholds,
        "f1_scores": threshold_f1
    }

    # Save all results
    payload = {
        "nu_heavy_tail": nu_heavy_tail,
        "n_seeds": n_seeds,
        "convergence": convergence_data,
        "f1_results": f1_results,
        "sensitivity": sensitivity_data,
        "threshold_ablation": threshold_ablation,
        "raw_gradient_norms": all_grad_norms["Standard SGD"][:300]
    }
    
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] Results successfully saved to: {results_path}")

if __name__ == "__main__":
    main()
