"""
SE Floor Experiment — v5
========================
(Renamed 2026-07-24 at the author's direction; apart from this header
and the figure banner, byte-identical to the version that produced
the published results.)
Extended Durability + Sparsity Profile (Form from Void)

150 generations to test:
  - SE Reg durability beyond 30 generations
  - Whether F_c data enrichment separates from random noise at scale
  - Sparsity profile: does SE Reg develop increasing activation
    sparsity alongside maintained entropy? (form from void bridge)

  (a) Baseline: no intervention
  (b) SE Regularization: variance + covariance penalty
  (c) F_c Data Enrichment: 3-6-9 harmonics on synthetic data
  (d) Random Noise (control): matched-magnitude perturbation

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
NUM_GENERATIONS = 150       # Extended from 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PHI = (1 + np.sqrt(5)) / 2
OMEGA = 2.0 * np.pi
SEED = 42

LAMBDA_VAR = 1.0
LAMBDA_COV = 0.04
ENRICH_AMP = 0.01
SPARSITY_THRESHOLD = 0.01  # |h| below this counts as "near-zero"

print(f"Device: {DEVICE}")
print(f"v5: Extended Durability (150 gen) + Sparsity Profile")
print(f"Sparsity threshold: {SPARSITY_THRESHOLD}")
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
# DATA ENRICHMENT
# ============================================================
def enrich_data_fc(data, step, amplitude=ENRICH_AMP):
    dim = data.shape[-1]
    phases = torch.linspace(0, 2 * np.pi, dim, device=data.device)
    t = float(step)
    signal = amplitude * (
        torch.sin(3 * OMEGA * t + phases) +
        torch.sin(6 * OMEGA * t + phases) +
        torch.sin(9 * OMEGA * t + phases)
    )
    return torch.clamp(data + signal.unsqueeze(0), 0, 1)


def enrich_data_random(data, amplitude=ENRICH_AMP):
    noise = torch.randn_like(data) * amplitude * 3
    return torch.clamp(data + noise, 0, 1)


# ============================================================
# METRICS — now includes sparsity
# ============================================================
def compute_metrics(model, data_loader, device, sparsity_threshold=SPARSITY_THRESHOLD):
    model.eval()
    all_hidden = []
    total_activations = 0
    near_zero_count = 0

    with torch.no_grad():
        for batch_data, in data_loader:
            batch_data = batch_data.to(device)
            h = model.encode(batch_data)
            all_hidden.append(h.cpu())
            # Sparsity: count near-zero activations
            total_activations += h.numel()
            near_zero_count += (h.abs() < sparsity_threshold).sum().item()

    all_hidden = torch.cat(all_hidden, dim=0).numpy()
    sparsity = near_zero_count / total_activations if total_activations > 0 else 0.0

    mean_amplitude = np.mean(np.abs(all_hidden))
    variance = np.var(all_hidden)

    cov_matrix = np.cov(all_hidden.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    eigenvalues = eigenvalues / eigenvalues.sum()
    spectral_entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))
    max_entropy = np.log(len(eigenvalues)) if len(eigenvalues) > 0 else 1.0
    spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0

    return mean_amplitude, variance, spectral_entropy, sparsity


# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, data_loader, optimizer, criterion, device, use_se_reg=False):
    model.train()
    total_loss = 0
    for batch_data, in data_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        reconstruction, h = model(batch_data)
        loss = criterion(reconstruction, batch_data)
        if use_se_reg:
            loss = loss + se_regularization_loss(h)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def generate_reconstructions(model, data_loader, device):
    model.eval()
    all_recs = []
    with torch.no_grad():
        for batch_data, in data_loader:
            batch_data = batch_data.to(device)
            rec, _ = model(batch_data)
            all_recs.append(rec.cpu())
    return torch.cat(all_recs, dim=0)


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
    print(f"Using {subset_size} images, {LATENT_DIM} latent units, {NUM_GENERATIONS} generations")
    print()

    conditions = {
        '(a) Baseline': {
            'se_reg': False, 'data_enrich': None
        },
        '(b) SE Regularization': {
            'se_reg': True, 'data_enrich': None
        },
        '(c) F_c Data Enrichment': {
            'se_reg': False, 'data_enrich': 'fc'
        },
        '(d) Random Noise (control)': {
            'se_reg': False, 'data_enrich': 'random'
        },
    }

    all_results = {}

    for cond_name, cond_config in conditions.items():
        print(f"{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*60}")

        torch.manual_seed(SEED)
        model = Autoencoder(input_dim=784, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print("  Phase 1: Training on real data...")
        for epoch in range(INITIAL_EPOCHS):
            loss = train_epoch(model, real_loader, optimizer, criterion, DEVICE,
                               use_se_reg=cond_config['se_reg'])
        print(f"  Initial training complete. Final loss: {loss:.6f}")

        metrics = {'amplitude': [], 'variance': [], 'spectral_entropy': [], 'sparsity': []}

        amp, var, se, spar = compute_metrics(model, real_loader, DEVICE)
        metrics['amplitude'].append(amp)
        metrics['variance'].append(var)
        metrics['spectral_entropy'].append(se)
        metrics['sparsity'].append(spar)
        print(f"  Gen 0: Amp={amp:.4f}, Var={var:.4f}, SE={se:.4f}, Sparsity={spar:.4f}")

        current_data = real_data.clone()

        for gen in range(1, NUM_GENERATIONS + 1):
            current_loader = DataLoader(TensorDataset(current_data), batch_size=BATCH_SIZE, shuffle=True)
            reconstructions = generate_reconstructions(model, current_loader, DEVICE)

            current_data = reconstructions.detach()
            if cond_config['data_enrich'] == 'fc':
                current_data = enrich_data_fc(current_data, step=gen)
            elif cond_config['data_enrich'] == 'random':
                current_data = enrich_data_random(current_data)

            recursive_loader = DataLoader(TensorDataset(current_data), batch_size=BATCH_SIZE, shuffle=True)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            for epoch in range(RECURSIVE_EPOCHS):
                loss = train_epoch(model, recursive_loader, optimizer, criterion, DEVICE,
                                   use_se_reg=cond_config['se_reg'])

            amp, var, se, spar = compute_metrics(model, recursive_loader, DEVICE)
            metrics['amplitude'].append(amp)
            metrics['variance'].append(var)
            metrics['spectral_entropy'].append(se)
            metrics['sparsity'].append(spar)

            if gen % 10 == 0 or gen == 1 or gen == 5:
                print(f"  Gen {gen}: Amp={amp:.4f}, Var={var:.4f}, SE={se:.4f}, Sparsity={spar:.4f}, Loss={loss:.6f}")

        all_results[cond_name] = metrics
        print()

    # ── SAVE RESULTS ──
    print("Saving results...")

    colors = {
        '(a) Baseline': '#CC0000',
        '(b) SE Regularization': '#0066CC',
        '(c) F_c Data Enrichment': '#00AA44',
        '(d) Random Noise (control)': '#FF8800',
    }

    generations = list(range(NUM_GENERATIONS + 1))

    # Individual metric plots
    for metric_name in ['amplitude', 'variance', 'spectral_entropy', 'sparsity']:
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        for cond_name, mets in all_results.items():
            ax.plot(generations, mets[metric_name],
                    label=cond_name, color=colors[cond_name],
                    linewidth=2, alpha=0.9)
        title_map = {
            'amplitude': 'Mean Amplitude of Hidden State',
            'variance': 'Variance of Hidden State',
            'spectral_entropy': 'Spectral Entropy of Hidden State',
            'sparsity': f'Activation Sparsity (|h| < {SPARSITY_THRESHOLD})'
        }
        ax.set_title(f"{title_map[metric_name]} — v5 (150 gen)", fontsize=16, fontweight='bold')
        ax.set_xlabel('Recursive Generation', fontsize=13)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, NUM_GENERATIONS)
        plt.tight_layout()
        plt.savefig(f'results/{metric_name}_v5.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Combined dashboard (4 panels)
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    fig.suptitle('SE Floor Experiment v5\n150 Generations — Durability + Sparsity',
                 fontsize=18, fontweight='bold', y=0.98)
    metric_names = ['spectral_entropy', 'sparsity', 'variance', 'amplitude']
    y_labels = ['Spectral Entropy', 'Activation Sparsity', 'Variance', 'Mean Amplitude']
    for ax, mn, yl in zip(axes, metric_names, y_labels):
        for cond_name, mets in all_results.items():
            ax.plot(generations, mets[mn], label=cond_name, color=colors[cond_name],
                    linewidth=2, alpha=0.9)
        ax.set_ylabel(yl, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, NUM_GENERATIONS)
        if mn == 'spectral_entropy':
            ax.legend(fontsize=9, loc='best')
    axes[-1].set_xlabel('Recursive Generation', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/combined_dashboard_v5.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Form from Void: SE vs Sparsity scatter for SE Reg
    if '(b) SE Regularization' in all_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        se_data = all_results['(b) SE Regularization']['spectral_entropy']
        sp_data = all_results['(b) SE Regularization']['sparsity']
        scatter = ax.scatter(se_data, sp_data, c=generations, cmap='viridis',
                            s=20, alpha=0.8, edgecolors='none')
        ax.set_xlabel('Spectral Entropy', fontsize=13)
        ax.set_ylabel('Activation Sparsity', fontsize=13)
        ax.set_title('Form from Void: SE vs Sparsity under Regularization\n(color = generation)',
                     fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Generation')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/form_from_void_v5.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Final summary
    print()
    print(f"{'='*60}")
    print("FINAL SUMMARY — Generation 150")
    print(f"{'='*60}")
    for cond_name, mets in all_results.items():
        se_final = mets['spectral_entropy'][-1]
        var_final = mets['variance'][-1]
        amp_final = mets['amplitude'][-1]
        spar_final = mets['sparsity'][-1]
        spar_init = mets['sparsity'][0]
        print(f"\n  {cond_name}:")
        print(f"    SE:  {se_final:.4f}")
        print(f"    Var: {var_final:.4f}")
        print(f"    Amp: {amp_final:.4f}")
        print(f"    Sparsity: {spar_init:.4f} -> {spar_final:.4f} (delta: {spar_final-spar_init:+.4f})")

    # Save raw data
    save_dict = {}
    for cond_name, mets in all_results.items():
        key = cond_name.split(')')[0].strip('(')
        for mn, values in mets.items():
            save_dict[f'{key}_{mn}'] = np.array(values)
    save_dict['generations'] = np.array(generations)
    np.savez('results/metrics_data_v5.npz', **save_dict)

    print()
    print(f"{'='*60}")
    print("EXPERIMENT COMPLETE — v5")
    print(f"{'='*60}")
    print("Results saved to results/ folder:")
    print("  - spectral_entropy_v5.png")
    print("  - variance_v5.png")
    print("  - amplitude_v5.png")
    print("  - sparsity_v5.png")
    print("  - combined_dashboard_v5.png")
    print("  - form_from_void_v5.png")
    print("  - metrics_data_v5.npz")
    print()
    print("Form emerges from void.")
    print("The question is whether the network knows it.")


if __name__ == '__main__':
    run_experiment()
