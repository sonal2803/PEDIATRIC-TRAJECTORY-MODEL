
import torch
import numpy as np

# Feature index ranges matching the paper (0-indexed):
#   Structural  : cols  0 –  9
#   Cognitive   : cols 10 – 19
#   Epileptic   : cols 20 – 29
#   Genetic     : cols 30 – 39
#   Perinatal   : cols 40 – 49
STRUCTURAL_IDX = slice(0, 10)
COGNITIVE_IDX  = slice(10, 20)
EPILEPTIC_IDX  = slice(20, 30)
GENETIC_IDX    = slice(30, 40)
PERINATAL_IDX  = slice(40, 50)


def _apply_domain_prior(pred: torch.Tensor, domain: str, t: int) -> torch.Tensor:
    """
    Apply biologically informed progression priors (paper Eq. 22-24).

    Drift is applied per-feature-group, not uniformly, so only the
    clinically relevant dimensions are perturbed.

    Args:
        pred   : prediction tensor, shape [1, 1, features]
        domain : disease domain string or None
        t      : current simulation step index (0-based)

    Returns:
        pred   : adjusted tensor, same shape
    """
    pred = pred.clone()

    if domain == "neurodegenerative":
        # Eq. 22: δ_deg(t) = α(t+1) — progressive cognitive-structural worsening
        alpha = 0.012
        pred[0, 0, COGNITIVE_IDX]  += alpha * (t + 1)
        pred[0, 0, STRUCTURAL_IDX] += alpha * 0.6 * (t + 1)

    elif domain == "genetic_epileptic":
        # Epileptic instability: oscillating seizure burden
        pred[0, 0, EPILEPTIC_IDX] += 0.015 * np.sin(t * 0.9 + 0.5)
        pred[0, 0, GENETIC_IDX]   += 0.008 * (t + 1)

    elif domain == "neuroinflammatory":
        # Eq. 23: acute onset (t=0) then partial stabilisation
        if t == 0:
            pred[0, 0, STRUCTURAL_IDX] += 0.025
            pred[0, 0, COGNITIVE_IDX]  += 0.020
        else:
            decay = max(0.0, 0.010 - 0.002 * t)
            pred[0, 0, STRUCTURAL_IDX] -= decay
            pred[0, 0, COGNITIVE_IDX]  -= decay * 0.5

    elif domain == "structural":
        # Mostly stable; mild structural drift
        pred[0, 0, STRUCTURAL_IDX] += 0.005 * (t + 1)

    elif domain == "metabolic":
        # Slow global worsening if untreated
        pred[0, 0, COGNITIVE_IDX]  += 0.008 * (t + 1)
        pred[0, 0, STRUCTURAL_IDX] += 0.005 * (t + 1)

    elif domain == "vascular":
        # Initial ischaemic event, then plateau
        if t == 0:
            pred[0, 0, STRUCTURAL_IDX] += 0.030
            pred[0, 0, COGNITIVE_IDX]  += 0.015
        # Subsequent steps: minor residual decline
        else:
            pred[0, 0, STRUCTURAL_IDX] += 0.003
            pred[0, 0, COGNITIVE_IDX]  += 0.002

    elif domain == "demyelinating":
        # Fluctuating with mild upward drift
        drift = 0.005 * (t + 1)
        fluctuation = 0.008 * np.sin(t * 1.2)
        pred[0, 0, COGNITIVE_IDX]  += drift + fluctuation
        pred[0, 0, STRUCTURAL_IDX] += drift * 0.5

    # else (None / unknown): no domain prior applied

    return pred


def simulate_future(
    model,
    initial_sequence: torch.Tensor,
    domain: str = None,
    steps: int = 5,
    simulations: int = 50,
) -> list:
    """
    Generate Monte Carlo developmental trajectories.

    Args:
        model            : trained TrajectoryLSTM (on correct device)
        initial_sequence : Tensor shape [1, seq_len, features]
                           — the patient's observed history up to current stage
        domain           : disease domain string (or None)
        steps            : number of future developmental stages to simulate
        simulations      : number of Monte Carlo draws

    Returns:
        futures : list of np.ndarray, each shape [seq_len + steps, features]
                  The full trajectory (observed + simulated) for each draw.
    """
    # Enable dropout for stochastic Monte Carlo sampling
    model.train()

    futures = []
    for _ in range(simulations):
        seq = initial_sequence.clone()          # [1, seq_len, features]
        full_trajectory = [
            seq[0, i, :].cpu().numpy()
            for i in range(seq.shape[1])
        ]                                        # observed history

        for t in range(steps):
            with torch.no_grad():
                pred = model(seq)                # [1, features]

            pred = pred.unsqueeze(1)             # [1, 1, features]

            # ── Domain-specific biological prior ──────────────────
            pred = _apply_domain_prior(pred, domain, t)

            # ── Monte Carlo stochastic noise (paper Eq. 18-19) ────
            # noise_scale = 0.02 + 0.04 * tanh(t/2)  (step index t, not seq length)
            noise_scale = 0.02 + 0.04 * float(np.tanh(t / 2.0))
            noise = torch.randn_like(pred) * noise_scale
            pred = pred + noise

            # Clamp to valid normalised range [0, 1]
            pred = torch.clamp(pred, 0.0, 1.0)

            # Append new step to sequence (auto-rolling context window)
            seq = torch.cat([seq, pred], dim=1)  # [1, seq_len+t+1, features]

            full_trajectory.append(pred[0, 0, :].cpu().numpy())

        futures.append(np.array(full_trajectory))  # [seq_len + steps, features]

    return futures