import torch
import numpy as np


def simulate_future(model, initial_sequence, domain=None, steps=5, simulations=50):
    """
    Generate multiple possible developmental futures
    with domain-specific progression dynamics.

    Args:
        model: Trained LSTM model
        initial_sequence: Tensor (1, seq_len, features)
        domain: Disease domain classification (string or None)
        steps: Number of future stages to simulate
        simulations: Number of Monte Carlo trajectories

    Returns:
        List of simulated future sequences (numpy arrays)
    """

    # Enable dropout for stochasticity
    model.train()

    futures = []

    for _ in range(simulations):

        seq = initial_sequence.clone()
        generated = []

        for t in range(steps):

            with torch.no_grad():
                pred = model(seq)

            # ------------------------------------------------
            # DOMAIN-SPECIFIC PROGRESSION DYNAMICS (C behavior)
            # ------------------------------------------------
            if domain == "neurodegenerative":
                # Progressive worsening over time
                drift = 0.015 * (t + 1)
                pred = pred + drift

            elif domain == "genetic_epileptic":
                # Fluctuating instability
                fluctuation = 0.01 * torch.sin(torch.tensor(float(t)))
                pred = pred + fluctuation

            elif domain == "neuroinflammatory":
                # Early spike, then stabilization
                if t == 0:
                    pred = pred + 0.02
                else:
                    pred = pred - 0.01 * t

            elif domain == "structural":
                # Mostly stable with mild decline
                pred = pred - 0.005 * t

            elif domain == "metabolic":
                # Slow worsening if untreated
                pred = pred + 0.008 * (t + 1)

            elif domain == "vascular":
                # Initial event then stabilization
                if t == 0:
                    pred = pred + 0.015

            elif domain == "demyelinating":
                # Fluctuating with mild upward drift
                pred = pred + 0.005 * (t + 1)
                pred = pred + 0.008 * torch.sin(torch.tensor(float(t)))

            # ------------------------------------------------
            # STOCHASTIC UNCERTAINTY
            # ------------------------------------------------
            time_index = seq.shape[1]
            noise_scale = 0.02 + 0.04 * np.tanh(time_index / 2)
            noise = torch.randn_like(pred) * noise_scale
            pred = pred + noise

            # Clamp values to realistic bounds
            pred = torch.clamp(pred, 0.0, 1.0)

            # Append prediction to sequence
            pred = pred.unsqueeze(1)
            seq = torch.cat([seq, pred], dim=1)

            generated.append(pred.squeeze(0).squeeze(0).cpu().numpy())

        futures.append(np.array(generated))

    return futures