"""
SE Floor Experiment — v4
========================
(Renamed 2026-07-24 at the author's direction; apart from this header
and the figure banner, byte-identical to the version that produced
the published results.)
Training Dynamics + Multi-Seed Robustness

v4 tests the final untested pathway: training dynamics modulation.
Does F_c-structured learning rate scheduling outperform standard
cosine scheduling at maintaining spectral entropy?

Also validates the SE regularization finding across 3 random seeds.

  (a) Baseline: no intervention
  (b) SE Regularization: variance + covariance penalty
  (c) F_c LR Schedule: learning rate modulated by 3-6-9 harmonics
  (d) Cosine LR Schedule: standard ML technique (control)
  (e) SE Reg + F_c LR Schedule: combined

Then: multi-seed validation of (a) vs (b) with seeds 42, 137, 369.

J. David Mack & Claude (Opus 4.6)
World Tree Project — June 2026
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 256
LATENT_DIM = 100
INITIAL_EPOCHS = 20
RECURSIVE_EPOCHS = 5
NUM_GENERATIONS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PHI = (1 + np.sqrt(5)) / 2
OMEGA = 2.0 * np.pi
SEED = 42

LAMBDA_VAR = 1.0
LAMBDA_COV = 0.04

print(f"Device: {DEVICE}")
print(f"v4: Training Dynamics + Multi-Seed Robustness")
print()


# ============================================================
# AUTOENCODER
# ============================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


# ============================================================
# SE REGULARIZATION
# ============================================================
def se_regularization_loss(hidden_states, lambda_var=LAMBDA_VAR, lambda_cov=LAMBDA_COV):
    std = torch.std(hidden_states, dim=0)
    var_loss = torch.mean(torch.relu(1.0 - std))
    batch_size = hidden_states.shape[0]
    if batch_size < 2:
        return lambda_var * var_loss
    h_centered = hidden_states - hidden_states.mean(dim=0)
    cov = (h_centered.T @ h_centered) / (batch_size - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = torch.sum(off_diag ** 2) / hidden_states.shape[1]
    return lambda_var * var_loss + lambda_cov * cov_loss


# ============================================================
# LEARNING RATE SCHEDULES
# ============================================================
def fc_lr_multiplier(step):
    """
    F_c harmonic learning rate: 3-6-9 structure modulates the LR.
    Oscillates between 0.5x and 1.5x base LR with harmonic structure.
    The network experiences chaos (high LR, exploration),
    equilibrium (base LR), and resonance (low LR, consolidation)
    in a pattern determined by the 3-6-9 recurrence.
    """
    t = float(step) * 0.1  # scale for visible oscillation
    h = (np.sin(3 * OMEGA * t) + np.sin(6 * OMEGA * t) + np.sin(9 * OMEGA * t)) / 3.0
    return 1.0 + 0.5 * h  # range [0.5, 1.5]


def cosine_lr_multiplier(step, period=200):
    """Standard cosine annealing (control for F_c schedule)."""
    return 0.5 * (1.0 + np.cos(np.pi * (step % period) / period)) * 0.5 + 0.5  # range [0.5, 1.0]


# ============================================================
# SPECTRAL METRICS
# ============================================================
def compute_spectral_metrics(model, data_loader, device):
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for batch_data, in data_loader:
            batch_data = batch_data.to(device)
            h = model.encode(batch_data)
            all_hidden.append(h.cpu())
    all_hidden = torch.cat(all_hidden, dim=0).numpy()

    mean_amplitude = np.mean(np.abs(all_hidden))
    variance = np.var(all_hidden)

    cov_matrix = np.cov(all_hidden.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    eigenvalues = eigenvalues / eigenvalues.sum()
    spectral_entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))
    max_entropy = np.log(len(eigenvalues)) if len(eigenvalues) > 0 else 1.0
    spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0

    return mean_amplitude, variance, spectral_entropy


# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, data_loader, optimizer, criterion, device,
                use_se_reg=False, lr_schedule=None, step_counter=0):
    model.train()
    total_loss = 0
    step = step_counter

    for batch_data, in data_loader:
        # Apply LR schedule if provided
        if lr_schedule is not None:
            mult = lr_schedule(step)
            for pg in optimizer.param_groups:
                pg['lr'] = LEARNING_RATE * mult

        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        reconstruction, h = model(batch_data)
        loss = criterion(reconstruction, batch_data)

        if use_se_reg:
            loss = loss + se_regularization_loss(h)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step += 1

    return total_loss / len(data_loader), step


def generate_reconstructions(model, data_loader, device):
    model.eval()
    all_reconstructions = []
    with torch.no_grad():
        for batch_data, in data_loader:
            batch_data = batch_data.to(device)
            reconstruction, _ = model(batch_data)
            all_reconstructions.append(reconstruction.cpu())
    return torch.cat(all_reconstructions, dim=0)


# ============================================================
# RUN ONE CONDITION
# ============================================================
def run_condition(cond_name, cond_config, real_data, real_loader, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Autoencoder(input_dim=784, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"  Phase 1: Training on real data (seed={seed})...")
    for epoch in range(INITIAL_EPOCHS):
        loss, _ = train_epoch(model, real_loader, optimizer, criterion, DEVICE,
                              use_se_reg=cond_config.get('se_reg', False))
    print(f"  Initial training complete. Final loss: {loss:.6f}")

    metrics = {'amplitude': [], 'variance': [], 'spectral_entropy': []}
    amp, var, se = compute_spectral_metrics(model, real_loader, DEVICE)
    metrics['amplitude'].append(amp)
    metrics['variance'].append(var)
    metrics['spectral_entropy'].append(se)
    print(f"  Gen 0: Amp={amp:.4f}, Var={var:.4f}, SE={se:.4f}")

    current_data = real_data.clone()
    step_counter = 0

    for gen in range(1, NUM_GENERATIONS + 1):
        current_loader = DataLoader(TensorDataset(current_data), batch_size=BATCH_SIZE, shuffle=True)
        reconstructions = generate_reconstructions(model, current_loader, DEVICE)
        current_data = reconstructions.detach()

        recursive_loader = DataLoader(TensorDataset(current_data), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(RECURSIVE_EPOCHS):
            loss, step_counter = train_epoch(
                model, recursive_loader, optimizer, criterion, DEVICE,
                use_se_reg=cond_config.get('se_reg', False),
                lr_schedule=cond_config.get('lr_schedule', None),
                step_counter=step_counter
            )

        amp, var, se = compute_spectral_metrics(model, recursive_loader, DEVICE)
        metrics['amplitude'].append(amp)
        metrics['variance'].append(var)
        metrics['spectral_entropy'].append(se)

        if gen % 5 == 0 or gen == 1:
            print(f"  Gen {gen}: Amp={amp:.4f}, Var={var:.4f}, SE={se:.4f}, Loss={loss:.6f}")

    return metrics


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment():
    os.makedirs('results', exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading MNIST...")
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)

    subset_size = 10000
    real_data = torch.stack([mnist_train[i][0] for i in range(subset_size)])
    real_loader = DataLoader(TensorDataset(real_data), batch_size=BATCH_SIZE, shuffle=True)
    print(f"Using {subset_size} images, {LATENT_DIM} latent units")
    print()

    # ── PART 1: Five conditions ──
    conditions = {
        '(a) Baseline': {
            'se_reg': False, 'lr_schedule': None
        },
        '(b) SE Regularization': {
            'se_reg': True, 'lr_schedule': None
        },
        '(c) F_c LR Schedule': {
            'se_reg': False, 'lr_schedule': fc_lr_multiplier
        },
        '(d) Cosine LR (control)': {
            'se_reg': False, 'lr_schedule': cosine_lr_multiplier
        },
        '(e) SE Reg + F_c LR': {
            'se_reg': True, 'lr_schedule': fc_lr_multiplier
        },
    }

    all_results = {}
    for cond_name, cond_config in conditions.items():
        print(f"{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*60}")
        all_results[cond_name] = run_condition(cond_name, cond_config, real_data, real_loader)
        print()

    # ── PART 2: Multi-seed robustness for baseline vs SE reg ──
    print(f"{'='*60}")
    print("MULTI-SEED ROBUSTNESS: Baseline vs SE Regularization")
    print(f"{'='*60}")

    seeds = [42, 137, 369]
    multiseed_results = {'baseline': [], 'se_reg': []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        print("  Running baseline...")
        bl = run_condition('Baseline', {'se_reg': False}, real_data, real_loader, seed=seed)
        multiseed_results['baseline'].append(bl)

        print("  Running SE Regularization...")
        sr = run_condition('SE Reg', {'se_reg': True}, real_data, real_loader, seed=seed)
        multiseed_results['se_reg'].append(sr)

    # Print multi-seed summary
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    for label, runs in multiseed_results.items():
        final_ses = [r['spectral_entropy'][-1] for r in runs]
        final_vars = [r['variance'][-1] for r in runs]
        final_amps = [r['amplitude'][-1] for r in runs]
        print(f"\n  {label}:")
        print(f"    SE  final: {np.mean(final_ses):.4f} +/- {np.std(final_ses):.4f}")
        print(f"    Var final: {np.mean(final_vars):.4f} +/- {np.std(final_vars):.4f}")
        print(f"    Amp final: {np.mean(final_amps):.4f} +/- {np.std(final_amps):.4f}")

    # ── SAVE RESULTS ──
    print("\nSaving results...")

    colors = {
        '(a) Baseline': '#CC0000',
        '(b) SE Regularization': '#0066CC',
        '(c) F_c LR Schedule': '#00AA44',
        '(d) Cosine LR (control)': '#FF8800',
        '(e) SE Reg + F_c LR': '#9944CC',
    }

    generations = list(range(NUM_GENERATIONS + 1))

    # Individual plots
    for metric_name in ['amplitude', 'variance', 'spectral_entropy']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        for cond_name, mets in all_results.items():
            ax.plot(generations, mets[metric_name],
                    label=cond_name, color=colors[cond_name],
                    linewidth=2, marker='o', markersize=3)
        title_map = {
            'amplitude': 'Mean Amplitude of Hidden State',
            'variance': 'Variance of Hidden State',
            'spectral_entropy': 'Spectral Entropy of Hidden State'
        }
        ax.set_title(f"{title_map[metric_name]} - v4", fontsize=16, fontweight='bold')
        ax.set_xlabel('Recursive Generation', fontsize=13)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, NUM_GENERATIONS)
        plt.tight_layout()
        plt.savefig(f'results/{metric_name}_v4.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Combined dashboard
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle('SE Floor Experiment v4\nTraining Dynamics + Multi-Seed Robustness',
                 fontsize=18, fontweight='bold', y=0.98)
    metric_names = ['spectral_entropy', 'variance', 'amplitude']
    y_labels = ['Spectral Entropy', 'Variance', 'Mean Amplitude']
    for ax, mn, yl in zip(axes, metric_names, y_labels):
        for cond_name, mets in all_results.items():
            ax.plot(generations, mets[mn], label=cond_name, color=colors[cond_name],
                    linewidth=2, marker='o', markersize=3)
        ax.set_ylabel(yl, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, NUM_GENERATIONS)
        if mn == 'spectral_entropy':
            ax.legend(fontsize=9, loc='best')
    axes[-1].set_xlabel('Recursive Generation', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/combined_dashboard_v4.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Multi-seed plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for i, seed in enumerate(seeds):
        alpha = 0.4 if i > 0 else 1.0
        lbl_bl = f'Baseline (seed {seed})' if i == 0 else None
        lbl_sr = f'SE Reg (seed {seed})' if i == 0 else None
        ax.plot(generations, multiseed_results['baseline'][i]['spectral_entropy'],
                color='#CC0000', linewidth=1.5, alpha=alpha, label=lbl_bl)
        ax.plot(generations, multiseed_results['se_reg'][i]['spectral_entropy'],
                color='#0066CC', linewidth=1.5, alpha=alpha, label=lbl_sr)
    ax.set_title('Multi-Seed Robustness: Spectral Entropy', fontsize=16, fontweight='bold')
    ax.set_xlabel('Recursive Generation', fontsize=13)
    ax.set_ylabel('Spectral Entropy', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, NUM_GENERATIONS)
    plt.tight_layout()
    plt.savefig('results/multiseed_se_v4.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save raw data
    save_dict = {}
    for cond_name, mets in all_results.items():
        key = cond_name.split(')')[0].strip('(')
        for mn, values in mets.items():
            save_dict[f'{key}_{mn}'] = np.array(values)
    save_dict['generations'] = np.array(generations)
    for i, seed in enumerate(seeds):
        for label in ['baseline', 'se_reg']:
            for mn in ['amplitude', 'variance', 'spectral_entropy']:
                save_dict[f'multiseed_{label}_s{seed}_{mn}'] = np.array(multiseed_results[label][i][mn])
    np.savez('results/metrics_data_v4.npz', **save_dict)

    print()
    print(f"{'='*60}")
    print("EXPERIMENT COMPLETE - v4")
    print(f"{'='*60}")
    print("Results saved to results/ folder:")
    print("  - spectral_entropy_v4.png")
    print("  - variance_v4.png")
    print("  - amplitude_v4.png")
    print("  - combined_dashboard_v4.png")
    print("  - multiseed_se_v4.png")
    print("  - metrics_data_v4.npz")
    print()
    print("Four pathways tested. The arc is complete.")
    print("What remains is honest documentation.")


if __name__ == '__main__':
    run_experiment()
