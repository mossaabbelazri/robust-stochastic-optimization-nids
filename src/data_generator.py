"""
Network Telemetry & Heavy-Tailed Synthetic Data Generator
=========================================================
Generates synthetic network intrusion detection datasets and simulates
stochastic gradient noise under heavy-tailed distributions:

1. Network Flow Characteristics:
   - Durations & packet counts follow Pareto / power-law distributions (bursty traffic).
   - Features aligned with CIC-IDS2017:
     * destination_port
     * flow_duration
     * total_fwd_packets
     * total_backward_packets

2. Heavy-Tailed Noise Injection:
   - Student-t distribution with degrees of freedom nu in (1, 2].
   - Note: For nu <= 2, the variance is INFINITE (E[||X||^2] = inf),
     yet E[||X||^p] < inf for p < nu.
   - Symmetric alpha-stable / Pareto gradient perturbations.
"""

import numpy as np

def generate_network_flows(n_samples=5000, seed=42):
    """
    Generate realistic network traffic flows exhibiting heavy-tailed behavior.
    Classes:
      0 = Benign traffic
      1 = Malicious / Intrusion flow (e.g. DDoS, Probe)
    """
    np.random.seed(seed)
    
    # 60% benign, 40% malicious
    n_benign = int(0.6 * n_samples)
    n_malicious = n_samples - n_benign
    
    # Benign flows: standard web/DNS/TLS traffic
    # Destination ports concentrated on 80, 443, 53
    ports_benign = np.random.choice([80, 443, 53, 8080], size=n_benign, p=[0.3, 0.5, 0.15, 0.05])
    # Flow duration (microseconds): lognormal
    duration_benign = np.random.lognormal(mean=6.0, sigma=1.2, size=n_benign)
    # Packet counts: mild Poisson/geometric
    fwd_pkts_benign = np.random.geometric(p=0.2, size=n_benign)
    bwd_pkts_benign = np.random.geometric(p=0.25, size=n_benign)
    
    # Malicious flows: heavy-tailed packet floods, scan bursts
    ports_malicious = np.random.choice([80, 443, 22, 21, 3389, 445], size=n_malicious)
    # Flow duration: heavy-tailed Pareto (bursts / slowloris)
    duration_malicious = (np.random.pareto(a=1.5, size=n_malicious) + 1.0) * 1000.0
    # Packet bursts: heavy-tailed Pareto (DDoS flood)
    fwd_pkts_malicious = np.random.pareto(a=1.3, size=n_malicious) * 50.0 + 1.0
    bwd_pkts_malicious = np.random.pareto(a=1.6, size=n_malicious) * 10.0
    
    X_benign = np.column_stack([ports_benign, duration_benign, fwd_pkts_benign, bwd_pkts_benign])
    y_benign = np.zeros(n_benign)
    
    X_malicious = np.column_stack([ports_malicious, duration_malicious, fwd_pkts_malicious, bwd_pkts_malicious])
    y_malicious = np.ones(n_malicious)
    
    X = np.vstack([X_benign, X_malicious])
    y = np.concatenate([y_benign, y_malicious])
    
    # Normalize features with robust scaling (median / IQR)
    medians = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-6
    X_scaled = (X - medians) / iqr
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    return X_scaled[indices], y[indices]


def load_dataset(csv_path=None, n_samples=5000, seed=42):
    """
    Unified dataset loader:
    - If csv_path is provided and exists: loads real tabular data (e.g. CIC-IDS2017 extract),
      extracts numeric features, standardizes using Median/IQR, and returns (X, y).
    - Otherwise: generates realistic synthetic bursty network traffic (Pareto / heavy-tailed).
    """
    if csv_path is not None:
        import os
        import pandas as pd
        if os.path.exists(csv_path):
            print(f"[Dataset] Loading real dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
            # Find label column
            label_col = None
            for col in ["label", "Label", "class", "Class", "target", "Target"]:
                if col in df.columns:
                    label_col = col
                    break
            if label_col is None:
                label_col = df.columns[-1]
            
            y = (df[label_col].astype(str).str.lower().str.contains("attack|ddos|malicious|1|anomaly")).astype(int).values
            X_df = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).fillna(0)
            X = X_df.values
            
            # Robust scaling
            medians = np.median(X, axis=0)
            iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-6
            X_scaled = (X - medians) / iqr
            return X_scaled, y

    # Default: Realistic synthetic bursty traffic
    return generate_network_flows(n_samples=n_samples, seed=seed)


def sample_heavy_tailed_noise(dim, nu=1.5, scale=1.0):
    """
    Sample heavy-tailed noise vector from a Student-t distribution with nu degrees of freedom.
    When nu in (1, 2], the second moment (variance) is unbounded:
      E[||Z||^2] = infinity.
    However, the p-th moment E[||Z||^p] is finite for all p < nu.
    """
    # Student-t sample in R^dim
    z = np.random.standard_t(df=nu, size=dim) * scale
    return z


def compute_logistic_gradient(w, x_i, y_i, lambda_reg=1e-4):
    """
    Compute binary logistic regression loss and exact subgradient on a single sample:
    Loss: l(w; x, y) = log(1 + exp(-y_tilde * w^T x)) + (lambda/2) * ||w||^2
    where y_tilde in {-1, +1}
    """
    y_tilde = 2.0 * y_i - 1.0 # convert 0,1 to -1,+1
    margin = y_tilde * np.dot(x_i, w)
    
    # Numerically stable sigmoid: sigma(-m) = 1 / (1 + exp(m))
    if margin >= 0:
        p = 1.0 / (1.0 + np.exp(-margin))
        prob_wrong = 1.0 - p
    else:
        exp_m = np.exp(margin)
        prob_wrong = 1.0 / (1.0 + exp_m)
        
    grad = - y_tilde * prob_wrong * x_i + lambda_reg * w
    loss = np.log1p(np.exp(-np.abs(margin))) + max(0, -margin) + 0.5 * lambda_reg * np.sum(w**2)
    return loss, grad
