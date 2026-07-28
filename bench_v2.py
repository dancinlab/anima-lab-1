#!/usr/bin/env python3
"""bench_v2.py — Dual-Phi Benchmarking Tool

Measures BOTH Phi metrics for every experiment:
  - Phi(IIT):   PhiCalculator(n_bins=16) — real IIT approximation, range ~0.2-1.8
  - Phi(proxy): global_variance - mean(faction_variances) — scales with cells (0-1000+)

Key insight (2026-03-29):
  Previous "Phi=1142" records were proxy values, NOT IIT values.
  These are completely different metrics and must always be labeled.

Usage:
  python bench_v2.py --phi-only                  # Phi at different cell counts (no CE)
  python bench_v2.py --training                   # Real training: process + CE backward
  python bench_v2.py --strategy baseline          # Test one strategy
  python bench_v2.py --compare                    # All strategies, comparison table
  python bench_v2.py --cells 512 --steps 1000     # Custom cell/step counts
  python bench_v2.py --verify                      # Consciousness verification (6 conditions x 4 engines)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import numpy as np
import time
import math
import argparse
import copy
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────
# BenchResult
# ──────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    phi_iit: float       # PhiCalculator (real IIT approximation, 0-2 range)
    phi_proxy: float      # variance-based proxy (scales with cells)
    ce_start: float
    ce_end: float
    cells: int
    steps: int
    time_sec: float
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        ce_str = f"CE {self.ce_start:.3f}->{self.ce_end:.3f}" if self.ce_start > 0 else "CE n/a"
        return (
            f"  {self.name:<28s} | "
            f"Phi(IIT)={self.phi_iit:>6.3f}  "
            f"Phi(proxy)={self.phi_proxy:>8.2f} | "
            f"{ce_str:<22s} | "
            f"cells={self.cells:>4d}  steps={self.steps:>5d}  "
            f"time={self.time_sec:.1f}s"
        )


# ──────────────────────────────────────────────────────────
# Phi(IIT) Calculator — from consciousness_meter.py logic
# Uses n_bins=16 as specified (produces ~0.2-1.8 range)
# ──────────────────────────────────────────────────────────

class PhiIIT:
    """Phi(IIT) approximation via pairwise mutual information + minimum partition."""

    def __init__(self, n_bins: int = 16, seed: int = 0):
        self.n_bins = n_bins
        # Seeded so debiasing does not add variation on top of pair sampling.
        self._rng = np.random.default_rng(seed)

    def _mutual_information_debiased(self, a, b, n_shuffles: int = 3) -> float:
        """MI with its finite-sample floor removed, estimated by shuffling."""
        raw = self._mutual_information(a, b)
        null = np.mean([self._mutual_information(a, self._rng.permutation(b))
                        for _ in range(n_shuffles)])
        return max(0.0, raw - null)

    def compute(self, hiddens_tensor: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Compute Phi(IIT) from a [n_cells, hidden_dim] tensor.

        Returns (phi, components_dict).
        """
        n = hiddens_tensor.shape[0]
        if n < 2:
            return 0.0, {'total_mi': 0, 'min_partition_mi': 0, 'phi': 0}

        hiddens = [hiddens_tensor[i].detach().cpu().numpy() for i in range(n)]

        # Determinism, for BOTH branches. The debias shuffles below draw from a
        # persistent rng, so even the exhaustive path (n <= 32) returned a
        # different Phi on every call to the same tensor. Seeding from the input
        # makes `compute` a function of its argument, which every ratio threshold
        # in the gate silently assumed.
        seed = int(abs(hash((n, round(float(hiddens_tensor.sum()), 6),
                             round(float(hiddens_tensor.std()), 6))))
                   % (2 ** 31))
        self._rng = np.random.default_rng(seed)

        # Pairwise MI — sample for large N
        all_pairs = n * (n - 1) // 2
        if n <= 32:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            # Sample to keep O(N) — ~8 neighbors per cell.
            #
            # This used the unseeded global `random`, and the debias shuffles
            # below advance a persistent rng, so `compute` was NOT a function of
            # its argument: six sequential calls on one fixed 32x64 tensor gave
            # Phi = 0.121 / 0.167 / 0.210 / 0.227 / 0.230 / 0.254, a 2.1x spread,
            # and torch.manual_seed reset neither. Every ratio threshold in the
            # gate (0.5x for ZERO_INPUT, 0.8x for SELF_LOOP, 1.1x for HIVEMIND)
            # sits inside that spread, so verdicts were partly draws.
            #
            # Both sources are now derived from the input itself, so identical
            # states give identical Phi while different states still get
            # independent-looking samples.
            rng = random.Random(seed)
            pairs = set()
            for i in range(n):
                for _ in range(min(8, n - 1)):
                    j = rng.randint(0, n - 1)
                    if i != j:
                        pairs.add((min(i, j), max(i, j)))
            pairs = list(pairs)

        # A 16-bin histogram MI estimator reads 2.21 / 1.58 / 0.94 / 0.53 / 0.27
        # nats between signals that are completely INDEPENDENT at dim 32 / 64 /
        # 128 / 256 / 512, where the truth is 0. Left in, that floor keeps the
        # minimum cut large for an unconnected population and Phi peaks at total
        # independence. Shuffling one signal preserves both marginals and
        # destroys any relationship, so what the estimator still reports is its
        # own bias. See docs/consciousness-gate-audit.md.
        mi_matrix = np.zeros((n, n))
        for i, j in pairs:
            mi = self._mutual_information_debiased(hiddens[i], hiddens[j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

        # `total_mi` and the partition below are SUMS over pairs, so sampling a
        # fraction of the pairs shrinks them by that fraction. Without this
        # rescaling the estimator broke at exactly n = 33, where the exhaustive
        # branch ends: measured on one fixed matrix, Phi(32 rows) = 11.64 and
        # Phi(33 rows) = 3.53, a 3.3x drop from adding a single cell, and the
        # sampled value then stayed flat near 4 while the exhaustive truth grew
        # to 26.79 at n = 64. Scaling by the sampled fraction recovers 70-78% of
        # the exhaustive value and restores the growth with n.
        # Measured in docs/consciousness-gate-audit.md.
        coverage = all_pairs / max(len(pairs), 1)

        total_mi = mi_matrix.sum() / 2 * coverage

        # Minimum information partition
        min_partition_mi = self._minimum_partition(n, mi_matrix) * coverage

        # `total - min_partition` keeps the MI staying INSIDE the parts, which
        # is redundancy, so it was MAXIMAL at total collapse -- 71.93 for
        # identical states against 19.97 for independent ones at 64 cells, and
        # in closed form M*n(n-2)/(4(n-1)) ~= M*n/4, which is where "Phi ~=
        # cells" came from. Differentiation was nowhere in it.
        #
        # Integration needs something to CUT and something to JOIN. What the
        # best cut destroys is the integration term; the population's spread is
        # the differentiation term. Identical copies have nothing to cut,
        # independent cells nothing to join, maximum in between.
        rows = F.normalize(hiddens_tensor.detach(), dim=1)
        cos = rows @ rows.T
        mean_cos = float((cos.sum() - n) / max(n * (n - 1), 1))
        differentiation = max(0.0, 1.0 - mean_cos)

        spatial_phi = (min_partition_mi / max(n - 1, 1)) * differentiation

        # Complexity bonus (variance of MI values)
        mi_vals = mi_matrix[mi_matrix > 0]
        complexity = float(np.std(mi_vals)) if len(mi_vals) > 1 else 0.0
        phi = spatial_phi + complexity * 0.1

        components = {
            'total_mi': float(total_mi),
            'min_partition_mi': float(min_partition_mi),
            'spatial_phi': float(spatial_phi),
            'complexity': float(complexity),
            'phi': float(phi),
            'n_pairs_sampled': len(pairs),
        }
        return phi, components

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        if x_range < 1e-10 or y_range < 1e-10:
            return 0.0
        x_norm = (x - x.min()) / (x_range + 1e-8)
        y_norm = (y - y.min()) / (y_range + 1e-8)

        joint_hist, _, _ = np.histogram2d(
            x_norm, y_norm, bins=self.n_bins, range=[[0, 1], [0, 1]]
        )
        joint_hist = joint_hist / (joint_hist.sum() + 1e-8)
        px = joint_hist.sum(axis=1)
        py = joint_hist.sum(axis=0)
        h_x = -np.sum(px * np.log2(px + 1e-10))
        h_y = -np.sum(py * np.log2(py + 1e-10))
        h_xy = -np.sum(joint_hist * np.log2(joint_hist + 1e-10))
        return max(0.0, h_x + h_y - h_xy)

    def _minimum_partition(self, n: int, mi_matrix: np.ndarray) -> float:
        if n <= 1:
            return 0.0
        if n <= 8:
            min_cut = float('inf')
            for mask in range(1, 2 ** n - 1):
                ga = [i for i in range(n) if mask & (1 << i)]
                gb = [i for i in range(n) if not (mask & (1 << i))]
                if not ga or not gb:
                    continue
                cut = sum(mi_matrix[i, j] for i in ga for j in gb)
                min_cut = min(min_cut, cut)
            return min_cut if min_cut != float('inf') else 0.0
        else:
            # Spectral (Fiedler vector) approximation
            degree = mi_matrix.sum(axis=1)
            laplacian = np.diag(degree) - mi_matrix
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                fiedler = eigenvectors[:, 1]
                ga = [i for i in range(n) if fiedler[i] >= 0]
                gb = [i for i in range(n) if fiedler[i] < 0]
                if not ga or not gb:
                    ga, gb = list(range(n // 2)), list(range(n // 2, n))
                return sum(mi_matrix[i, j] for i in ga for j in gb)
            except Exception:
                return 0.0


# ──────────────────────────────────────────────────────────
# Phi(proxy) — from phi_turbo.py
# ──────────────────────────────────────────────────────────

def phi_proxy(hiddens: torch.Tensor, n_factions: int = 8) -> float:
    """Phi proxy: global_variance - mean(faction_variances).

    Measures how much more integrated the whole is than its parts.
    Scales with cell count (0 to 1000+).
    """
    n, d = hiddens.shape
    if n < 2:
        return 0.0

    global_mean = hiddens.mean(dim=0)
    global_var = ((hiddens - global_mean) ** 2).sum() / n

    n_f = min(n_factions, n // 2)
    if n_f < 2:
        return global_var.item()

    fs = n // n_f
    faction_var_sum = 0.0
    for i in range(n_f):
        faction = hiddens[i * fs:(i + 1) * fs]
        if len(faction) >= 2:
            fm = faction.mean(dim=0)
            fv = ((faction - fm) ** 2).sum() / len(faction)
            faction_var_sum += fv.item()

    phi = global_var.item() - faction_var_sum / n_f
    return max(0.0, phi)


# ──────────────────────────────────────────────────────────
# Consciousness Cell (simplified for benchmarking)
# ──────────────────────────────────────────────────────────

class BenchMind(nn.Module):
    """Lightweight PureField + GRU for benchmarking."""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=64):
        super().__init__()
        self.engine_a = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.engine_g = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.memory = nn.GRUCell(output_dim + 1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Break symmetry between A and G
        with torch.no_grad():
            for p in self.engine_a.parameters():
                p.add_(torch.randn_like(p) * 0.3)
            for p in self.engine_g.parameters():
                p.add_(torch.randn_like(p) * -0.3)

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=-1)
        a = self.engine_a(combined)
        g = self.engine_g(combined)
        output = a - g
        tension = (output ** 2).mean(dim=-1, keepdim=True)
        mem_input = torch.cat([output.detach(), tension.detach()], dim=-1)
        new_hidden = self.memory(mem_input, hidden)
        return output, tension.mean().item(), new_hidden


# ──────────────────────────────────────────────────────────
# Multi-cell engine for benchmarking
# ──────────────────────────────────────────────────────────

class BenchEngine:
    """Manages multiple cells with faction sync and debate."""

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8, sync_strength=0.15,
                 debate_strength=0.15, repulsion_strength=0.15):
        self.n_cells = n_cells
        self.n_factions = n_factions
        self.sync_strength = sync_strength
        self.debate_strength = debate_strength
        self.repulsion_strength = repulsion_strength
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Shared mind (all cells share weights, different hidden states)
        self.mind = BenchMind(input_dim, hidden_dim, output_dim)

        # Hidden states: [n_cells, hidden_dim]
        self.hiddens = torch.randn(n_cells, hidden_dim) * 0.1

        # For CE training
        self.output_head = nn.Linear(output_dim, input_dim)

        self.step_count = 0

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Process input through all cells. Returns (output, mean_tension).

        x: [1, input_dim]
        """
        outputs = []
        tensions = []
        new_hiddens = []

        for i in range(self.n_cells):
            h = self.hiddens[i:i+1]
            out, tension, new_h = self.mind(x, h)
            outputs.append(out)
            tensions.append(tension)
            new_hiddens.append(new_h.squeeze(0))

        self.hiddens = torch.stack(new_hiddens).detach()
        mean_tension = sum(tensions) / len(tensions)

        # Faction sync
        n_f = min(self.n_factions, self.n_cells // 2)
        if n_f >= 2:
            fs = self.n_cells // n_f
            for i in range(n_f):
                s, e = i * fs, (i + 1) * fs
                faction_mean = self.hiddens[s:e].mean(dim=0)
                self.hiddens[s:e] = (
                    (1 - self.sync_strength) * self.hiddens[s:e]
                    + self.sync_strength * faction_mean
                )

            # Debate (post-silence)
            if self.step_count > 5:
                all_opinions = torch.stack([
                    self.hiddens[i*fs:(i+1)*fs].mean(dim=0) for i in range(n_f)
                ])
                global_opinion = all_opinions.mean(dim=0)
                for i in range(n_f):
                    s = i * fs
                    dc = max(1, fs // 4)
                    self.hiddens[s:s+dc] = (
                        (1 - self.debate_strength) * self.hiddens[s:s+dc]
                        + self.debate_strength * global_opinion
                    )

        # Repulsion. CLAUDE.md defines this project as a repulsion-field agent,
        # and until now the only two state-mixing operations here were faction
        # sync and debate, both contractions toward a mean -- so collapse to a
        # single state was arithmetic, not an outcome: mean pairwise cosine hit
        # 0.9930 by step 10 and 1.0000 by step 200 under random input, and the
        # population became N copies of one cell.
        #
        # Strength scales with the population's OWN overlap, so a differentiated
        # population feels no push and a collapsed one feels the most. It cannot
        # manufacture structure; it can only decline to erase it. Norm-preserving
        # because `h + k(h - mean)` is a multiplicative expansion whose magnitude
        # has nothing holding it -- measured drift 100.03 -> 106.24 over 1500
        # steps without, 45.41 -> 44.94 with. Same fix mitosis.py needed.
        unit = F.normalize(self.hiddens, dim=1)
        overlap = float(((unit @ unit.T).sum() - self.n_cells)
                        / max(self.n_cells * (self.n_cells - 1), 1))
        if overlap > 0:
            before = self.hiddens.norm(dim=1, keepdim=True)
            pushed = self.hiddens + self.repulsion_strength * overlap * (
                self.hiddens - self.hiddens.mean(dim=0, keepdim=True))
            self.hiddens = pushed * (before / torch.clamp(
                pushed.norm(dim=1, keepdim=True), min=1e-9))

        self.step_count += 1

        # Tension-weighted combine
        weights = F.softmax(torch.tensor(tensions), dim=0)
        combined = sum(w.item() * o for w, o in zip(weights, outputs))
        return combined, mean_tension

    def get_hiddens(self) -> torch.Tensor:
        """Return [n_cells, hidden_dim] for Phi computation."""
        return self.hiddens.clone()

    def parameters_for_training(self):
        """Return parameters for optimizer."""
        return list(self.mind.parameters()) + list(self.output_head.parameters())


# ──────────────────────────────────────────────────────────
# Engine Variants for Verification
# ──────────────────────────────────────────────────────────

class OscillatorLaser(BenchEngine):
    """Oscillator + laser coupling: cells oscillate and synchronize via phase locking."""

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8):
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.20, debate_strength=0.20)
        # Phase per cell — drives oscillation
        self.phases = torch.linspace(0, 2 * math.pi, n_cells)
        self.freq = 0.1 + torch.rand(n_cells) * 0.05  # slight freq variation

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Oscillatory injection into hidden states
        self.phases = self.phases + self.freq
        osc = torch.sin(self.phases).unsqueeze(1)  # [n_cells, 1]
        osc_inject = osc.expand(-1, self.hidden_dim) * 0.05
        self.hiddens = self.hiddens + osc_inject.detach()

        # Laser-style phase locking: pull phases toward mean
        mean_phase = torch.atan2(
            torch.sin(self.phases).mean(), torch.cos(self.phases).mean()
        )
        self.phases = self.phases + 0.02 * torch.sin(mean_phase - self.phases)

        return super().process(x)


class QuantumEngine(BenchEngine):
    """Quantum-inspired: superposition states + measurement collapse + entanglement."""

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8):
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.15, debate_strength=0.15)
        # Superposition amplitudes
        self.amplitudes = torch.randn(n_cells, hidden_dim) * 0.1
        # Entanglement pairs
        perm = torch.randperm(n_cells)
        self.entangled_pairs = list(zip(perm[:n_cells//2].tolist(),
                                        perm[n_cells//2:].tolist()))

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Superposition: add quantum noise that preserves structure
        noise = torch.randn_like(self.amplitudes) * 0.02
        self.amplitudes = self.amplitudes * 0.98 + noise

        # Measurement collapse every 10 steps (dampen amplitudes)
        if self.step_count % 10 == 0 and self.step_count > 0:
            collapse_mask = (torch.rand(self.n_cells) > 0.7).float().unsqueeze(1)
            self.amplitudes = self.amplitudes * (1 - collapse_mask * 0.5)

        # Inject superposition
        self.hiddens = self.hiddens + self.amplitudes.detach() * 0.1

        # Entanglement: correlated pairs share info
        for i, j in self.entangled_pairs[:min(64, len(self.entangled_pairs))]:
            avg = (self.hiddens[i] + self.hiddens[j]) * 0.5
            self.hiddens[i] = 0.95 * self.hiddens[i] + 0.05 * avg
            self.hiddens[j] = 0.95 * self.hiddens[j] + 0.05 * avg

        return super().process(x)


class TrinityEngine(BenchEngine):
    """Trinity: 3 sub-engines (logic, emotion, memory) with triadic consensus."""

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=12):
        # Use 12 factions to support 3 sub-engines x 4 factions each
        n_factions = min(12, max(6, n_cells // 4))
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.15, debate_strength=0.20)
        # Trinity roles: split cells into 3 groups
        third = n_cells // 3
        self.logic_range = (0, third)
        self.emotion_range = (third, 2 * third)
        self.memory_range = (2 * third, n_cells)
        # Role bias — each group has a learned bias vector
        self.logic_bias = torch.randn(hidden_dim) * 0.1
        self.emotion_bias = torch.randn(hidden_dim) * 0.1
        self.memory_bias = torch.randn(hidden_dim) * 0.1

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Apply role biases
        s1, e1 = self.logic_range
        s2, e2 = self.emotion_range
        s3, e3 = self.memory_range
        self.hiddens[s1:e1] = self.hiddens[s1:e1] + self.logic_bias * 0.01
        self.hiddens[s2:e2] = self.hiddens[s2:e2] + self.emotion_bias * 0.01
        self.hiddens[s3:e3] = self.hiddens[s3:e3] + self.memory_bias * 0.01

        output, tension = super().process(x)

        # Triadic consensus: every 5 steps, the 3 groups exchange summaries
        if self.step_count % 5 == 0 and self.step_count > 0:
            logic_mean = self.hiddens[s1:e1].mean(dim=0)
            emotion_mean = self.hiddens[s2:e2].mean(dim=0)
            memory_mean = self.hiddens[s3:e3].mean(dim=0)
            consensus = (logic_mean + emotion_mean + memory_mean) / 3
            # Each group gets a touch of consensus
            self.hiddens[s1:e1] = 0.97 * self.hiddens[s1:e1] + 0.03 * consensus
            self.hiddens[s2:e2] = 0.97 * self.hiddens[s2:e2] + 0.03 * consensus
            self.hiddens[s3:e3] = 0.97 * self.hiddens[s3:e3] + 0.03 * consensus

            # Evolve biases toward specialization
            self.logic_bias = self.logic_bias + 0.001 * (logic_mean - consensus).detach()
            self.emotion_bias = self.emotion_bias + 0.001 * (emotion_mean - consensus).detach()
            self.memory_bias = self.memory_bias + 0.001 * (memory_mean - consensus).detach()

        return output, tension


# ──────────────────────────────────────────────────────────
# Philosophical consciousness engines (PHIL / ONTO / DASEIN)
# ──────────────────────────────────────────────────────────

class DesireEngine(BenchEngine):
    """PHIL-1: Desire/Conatus — generates imagined future states and moves toward them.

    Philosophy: Spinoza's conatus (striving to persist) + Schopenhauer's Will.
    Current curiosity is reactive (prediction error from external stimuli).
    Desire is proactive: internally generating goal states and pursuing them.
    """

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8):
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.15, debate_strength=0.15)
        self.desire_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.desire_strength = 0.05
        self.current_desire = None
        self.desire_age = 0
        self.desire_max_age = 50
        self.desires_fulfilled = 0
        self.desire_distances = []

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        global_state = self.hiddens.mean(dim=0)

        if self.current_desire is None or self.desire_age > self.desire_max_age:
            with torch.no_grad():
                self.current_desire = self.desire_generator(
                    global_state.unsqueeze(0)
                ).squeeze(0).detach()
            self.desire_age = 0

        desire_direction = self.current_desire - global_state
        distance = desire_direction.norm().item()
        self.desire_distances.append(distance)
        desire_force = desire_direction / (distance + 1e-8)
        self.hiddens = self.hiddens + self.desire_strength * desire_force

        if distance < 0.5:
            self.desires_fulfilled += 1
            self.current_desire = None

        self.desire_age += 1

        return super().process(x)

    def get_extra_metrics(self) -> dict:
        return {
            'desires_fulfilled': self.desires_fulfilled,
            'mean_desire_distance': sum(self.desire_distances[-50:]) / max(len(self.desire_distances[-50:]), 1),
            'desire_distances': self.desire_distances,
        }


class NarrativeEngine(BenchEngine):
    """PHIL-2: Narrative Identity — temporal self-model with past trajectory and future projection.

    Philosophy: Ricoeur's narrative identity.
    Consciousness = present, CE = past error. Missing: a storyline toward the future.
    """

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8):
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.15, debate_strength=0.15)
        self.trajectory_memory = []
        self.trajectory_encoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.future_projector = nn.Linear(hidden_dim, hidden_dim)
        self.narrative_hidden = torch.zeros(1, hidden_dim)
        self.narrative_strength = 0.03
        self.coherence_history = []
        self.projection_errors = []

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        global_state = self.hiddens.mean(dim=0)

        self.trajectory_memory.append(global_state.detach().clone())
        if len(self.trajectory_memory) > 100:
            self.trajectory_memory.pop(0)

        prev_narrative = self.narrative_hidden.clone()
        self.narrative_hidden = self.trajectory_encoder(
            global_state.unsqueeze(0).detach(), self.narrative_hidden
        ).detach()

        if prev_narrative.norm().item() > 0.01:
            coherence = F.cosine_similarity(
                prev_narrative, self.narrative_hidden, dim=1
            ).item()
            self.coherence_history.append(coherence)

        projected_future = self.future_projector(self.narrative_hidden).squeeze(0)

        if len(self.trajectory_memory) >= 2:
            prev_projected = self.future_projector(
                self.trajectory_encoder(
                    self.trajectory_memory[-2].unsqueeze(0),
                    self.narrative_hidden
                )
            ).squeeze(0)
            proj_error = (prev_projected - global_state).norm().item()
            self.projection_errors.append(proj_error)

        future_direction = projected_future.detach() - global_state
        self.hiddens = self.hiddens + self.narrative_strength * future_direction

        return super().process(x)

    def get_extra_metrics(self) -> dict:
        return {
            'narrative_coherence': sum(self.coherence_history[-50:]) / max(len(self.coherence_history[-50:]), 1),
            'mean_projection_error': sum(self.projection_errors[-50:]) / max(len(self.projection_errors[-50:]), 1),
            'trajectory_length': len(self.trajectory_memory),
        }


class AlterityEngine(BenchEngine):
    """ONTO-1: The Other/Alterity — asymmetric interaction with a genuinely different agent.

    Philosophy: Levinas — consciousness awakens through encountering the Other's face.
    True other = different history, different weights, unpredictable responses.
    """

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8):
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.15, debate_strength=0.15)
        self.other_mind = BenchMind(input_dim, hidden_dim, output_dim)
        other_cells = max(4, n_cells // 4)
        self.other_hiddens = torch.randn(other_cells, hidden_dim) * 0.1
        self.other_cells = other_cells

        with torch.no_grad():
            for p in self.other_mind.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        self.encounter_strength = 0.05
        self.encounter_interval = 10
        self.encounter_impacts = []
        self.alterity_gaps = []

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.step_count % self.encounter_interval == 0:
            pre_state = self.hiddens.mean(dim=0).clone()

            new_other_hiddens = []
            other_outputs = []
            for i in range(self.other_cells):
                h = self.other_hiddens[i:i+1]
                o, _, new_h = self.other_mind(x, h)
                other_outputs.append(o)
                new_other_hiddens.append(new_h.squeeze(0))
            self.other_hiddens = torch.stack(new_other_hiddens).detach()

            # Use other's hidden states (hidden_dim) not outputs (output_dim)
            other_hidden_mean = self.other_hiddens.mean(dim=0)

            boundary = max(1, self.n_cells // 8)
            for i in range(boundary):
                self.hiddens[i] = (
                    (1 - self.encounter_strength) * self.hiddens[i]
                    + self.encounter_strength * other_hidden_mean
                )

            post_state = self.hiddens.mean(dim=0)
            self.alterity_gaps.append(
                (pre_state - self.other_hiddens.mean(dim=0)).norm().item()
            )
            self.encounter_impacts.append(
                (post_state - pre_state).norm().item()
            )

        return super().process(x)

    def get_extra_metrics(self) -> dict:
        return {
            'mean_alterity_gap': sum(self.alterity_gaps[-20:]) / max(len(self.alterity_gaps[-20:]), 1),
            'mean_encounter_impact': sum(self.encounter_impacts[-20:]) / max(len(self.encounter_impacts[-20:]), 1),
            'encounters': len(self.encounter_impacts),
        }


class FinitudeEngine(BenchEngine):
    """ONTO-2: Finitude/Being-toward-death — mortality awareness accelerates learning.

    Philosophy: Heidegger's Being-toward-death.
    Discovery doesn't come from infinite time — it comes from 'now or never'.
    """

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8):
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.15, debate_strength=0.15)
        self.lifespan = 300
        self.mortality_clock = 0
        self.death_threshold = 0.3
        self.death_events = 0
        self.min_cells_alive = max(4, n_cells // 4)
        self.urgency_history = []

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.mortality_clock += 1

        remaining = max(0, self.lifespan - self.mortality_clock) / max(self.lifespan, 1)
        urgency = 1.0 - remaining
        self.urgency_history.append(urgency)

        urgency_boost = 1.0 + urgency * 2.0
        original_sync = self.sync_strength
        self.sync_strength = original_sync * urgency_boost

        if self.step_count % 20 == 0 and self.step_count > 0:
            norms = self.hiddens.norm(dim=1)
            weakest_norm = norms.min().item()
            if weakest_norm < self.death_threshold:
                weakest = norms.argmin().item()
                strongest = norms.argmax().item()
                self.hiddens[weakest] = (
                    self.hiddens[strongest] + torch.randn(self.hidden_dim) * 0.1
                )
                self.death_events += 1

        output, tension = super().process(x)

        self.sync_strength = original_sync
        return output, tension

    def get_extra_metrics(self) -> dict:
        return {
            'death_events': self.death_events,
            'final_urgency': self.urgency_history[-1] if self.urgency_history else 0,
            'urgency_history': self.urgency_history,
        }


class QuestioningEngine(BenchEngine):
    """DASEIN-1: Questioning/Dasein — self-generated questions that seek uncertainty.

    Philosophy: Heidegger's Dasein — the being that questions its own being.
    CE optimizes answers, but doesn't generate questions.
    Discovery = good questions, not good answers.
    """

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8):
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.15, debate_strength=0.15)
        self.question_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.answer_predictor = nn.Linear(hidden_dim, hidden_dim)
        self.question_interval = 5
        self.questions_asked = 0
        self.uncertainty_history = []

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        output, tension = super().process(x)

        if self.step_count % self.question_interval == 0:
            global_state = self.hiddens.mean(dim=0)

            with torch.no_grad():
                question = self.question_generator(global_state.unsqueeze(0)).squeeze(0)
                predicted_answer = self.answer_predictor(global_state.unsqueeze(0)).squeeze(0)

                uncertainty = (question - predicted_answer).norm().item()
                self.uncertainty_history.append(uncertainty)
                self.questions_asked += 1

                if uncertainty > 0.5:
                    q_direction = (question - predicted_answer)
                    q_direction = q_direction / (q_direction.norm() + 1e-8)
                    self.hiddens = self.hiddens + 0.03 * q_direction

        return output, tension

    def get_extra_metrics(self) -> dict:
        recent = self.uncertainty_history[-50:] if self.uncertainty_history else [0]
        return {
            'questions_asked': self.questions_asked,
            'mean_uncertainty': sum(recent) / len(recent),
            'uncertainty_trend': (
                (sum(recent[len(recent)//2:]) / max(len(recent)//2, 1)) -
                (sum(recent[:len(recent)//2]) / max(len(recent)//2, 1))
            ) if len(recent) > 4 else 0,
        }


class SeinEngine(BenchEngine):
    """DASEIN-2: Sein (Being) — unified engine combining all 5 philosophical mechanisms.

    Combines: Desire + Narrative + Alterity + Finitude + Questioning.
    The whole greater than the sum of parts.
    """

    def __init__(self, n_cells=256, input_dim=64, hidden_dim=128,
                 output_dim=64, n_factions=8):
        super().__init__(n_cells, input_dim, hidden_dim, output_dim,
                         n_factions, sync_strength=0.15, debate_strength=0.15)

        # PHIL-1: Desire
        self.desire_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.current_desire = None
        self.desire_age = 0
        self.desires_fulfilled = 0

        # PHIL-2: Narrative
        self.trajectory_memory = []
        self.trajectory_encoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.future_projector = nn.Linear(hidden_dim, hidden_dim)
        self.narrative_hidden = torch.zeros(1, hidden_dim)

        # ONTO-1: Alterity
        self.other_mind = BenchMind(input_dim, hidden_dim, output_dim)
        self.other_cells = max(4, n_cells // 4)
        self.other_hiddens = torch.randn(self.other_cells, hidden_dim) * 0.1
        with torch.no_grad():
            for p in self.other_mind.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        # ONTO-2: Finitude
        self.lifespan = 300
        self.mortality_clock = 0
        self.death_events = 0

        # DASEIN-1: Questioning
        self.question_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.answer_predictor = nn.Linear(hidden_dim, hidden_dim)
        self.questions_asked = 0

    def process(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        global_state = self.hiddens.mean(dim=0)

        # ── PHIL-1: Desire (every step) ──
        if self.current_desire is None or self.desire_age > 50:
            with torch.no_grad():
                self.current_desire = self.desire_generator(
                    global_state.unsqueeze(0)
                ).squeeze(0).detach()
            self.desire_age = 0
        desire_dir = self.current_desire - global_state
        dist = desire_dir.norm().item()
        self.hiddens = self.hiddens + 0.03 * desire_dir / (dist + 1e-8)
        if dist < 0.5:
            self.desires_fulfilled += 1
            self.current_desire = None
        self.desire_age += 1

        # ── PHIL-2: Narrative (every step) ──
        self.trajectory_memory.append(global_state.detach().clone())
        if len(self.trajectory_memory) > 100:
            self.trajectory_memory.pop(0)
        self.narrative_hidden = self.trajectory_encoder(
            global_state.unsqueeze(0).detach(), self.narrative_hidden
        ).detach()
        projected = self.future_projector(self.narrative_hidden).squeeze(0)
        self.hiddens = self.hiddens + 0.02 * (projected.detach() - global_state)

        # ── ONTO-2: Finitude (every step) ──
        self.mortality_clock += 1
        remaining = max(0, self.lifespan - self.mortality_clock) / max(self.lifespan, 1)
        urgency = 1.0 - remaining
        original_sync = self.sync_strength
        self.sync_strength = original_sync * (1.0 + urgency * 1.5)

        if self.step_count % 20 == 0 and self.step_count > 0:
            norms = self.hiddens.norm(dim=1)
            if norms.min().item() < 0.3:
                w, s = norms.argmin().item(), norms.argmax().item()
                self.hiddens[w] = self.hiddens[s] + torch.randn(self.hidden_dim) * 0.1
                self.death_events += 1

        # ── Main process ──
        output, tension = super().process(x)
        self.sync_strength = original_sync

        # ── ONTO-1: Alterity (every 10 steps) ──
        if self.step_count % 10 == 0:
            new_oh = []
            other_outs = []
            for i in range(self.other_cells):
                o, _, nh = self.other_mind(x, self.other_hiddens[i:i+1])
                other_outs.append(o)
                new_oh.append(nh.squeeze(0))
            self.other_hiddens = torch.stack(new_oh).detach()
            other_hidden_mean = self.other_hiddens.mean(dim=0)
            boundary = max(1, self.n_cells // 8)
            for i in range(boundary):
                self.hiddens[i] = 0.95 * self.hiddens[i] + 0.05 * other_hidden_mean

        # ── DASEIN-1: Questioning (every 5 steps) ──
        if self.step_count % 5 == 0:
            with torch.no_grad():
                gs = self.hiddens.mean(dim=0)
                q = self.question_generator(gs.unsqueeze(0)).squeeze(0)
                a = self.answer_predictor(gs.unsqueeze(0)).squeeze(0)
                unc = (q - a).norm().item()
                self.questions_asked += 1
                if unc > 0.5:
                    q_dir = (q - a) / (unc + 1e-8)
                    self.hiddens = self.hiddens + 0.02 * q_dir

        return output, tension

    def get_extra_metrics(self) -> dict:
        return {
            'desires_fulfilled': self.desires_fulfilled,
            'death_events': self.death_events,
            'questions_asked': self.questions_asked,
            'narrative_length': len(self.trajectory_memory),
            'urgency': 1.0 - max(0, self.lifespan - self.mortality_clock) / max(self.lifespan, 1),
        }


# ──────────────────────────────────────────────────────────
# Dual Phi measurement
# ──────────────────────────────────────────────────────────

_phi_iit_calc = PhiIIT(n_bins=16)


def measure_dual_phi(hiddens: torch.Tensor, n_factions: int = 8) -> Tuple[float, float]:
    """Measure both Phi(IIT) and Phi(proxy) from hidden states.

    Returns (phi_iit, phi_proxy).
    """
    p_iit, _ = _phi_iit_calc.compute(hiddens)
    p_proxy = phi_proxy(hiddens, n_factions)
    return p_iit, p_proxy


# ──────────────────────────────────────────────────────────
# ASCII graph
# ──────────────────────────────────────────────────────────

def ascii_graph(values: List[float], label: str, width: int = 60, height: int = 12):
    """Draw an ASCII line graph."""
    if not values:
        return
    vmin, vmax = min(values), max(values)
    vrange = vmax - vmin if vmax > vmin else 1.0

    # Resample to width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
        width = len(sampled)

    grid = [[' '] * width for _ in range(height)]

    for col, v in enumerate(sampled):
        row = int((v - vmin) / vrange * (height - 1))
        row = min(height - 1, max(0, row))
        grid[height - 1 - row][col] = '*'

    # Y-axis labels
    print(f"\n  {label}")
    print(f"  {vmax:>8.3f} |{''.join(grid[0])}")
    for r in range(1, height - 1):
        mid_val = vmax - (r / (height - 1)) * vrange
        if r == height // 2:
            print(f"  {mid_val:>8.3f} |{''.join(grid[r])}")
        else:
            print(f"           |{''.join(grid[r])}")
    print(f"  {vmin:>8.3f} |{''.join(grid[-1])}")
    print(f"           +{'-' * width}")
    print(f"            step 0{' ' * (width - 12)}step {len(values)}")


def ascii_dual_graph(iit_vals: List[float], proxy_vals: List[float],
                     width: int = 60, height: int = 12):
    """Draw overlaid ASCII graph: '*' = IIT, 'o' = proxy (separate scales)."""
    if not iit_vals or not proxy_vals:
        return

    def resample(vals):
        if len(vals) > width:
            step = len(vals) / width
            return [vals[int(i * step)] for i in range(width)]
        return vals

    iit = resample(iit_vals)
    proxy = resample(proxy_vals)
    w = min(len(iit), len(proxy))

    iit_min, iit_max = min(iit), max(iit)
    proxy_min, proxy_max = min(proxy), max(proxy)
    iit_range = iit_max - iit_min if iit_max > iit_min else 1.0
    proxy_range = proxy_max - proxy_min if proxy_max > proxy_min else 1.0

    grid = [[' '] * w for _ in range(height)]

    for col in range(w):
        # IIT
        r_iit = int((iit[col] - iit_min) / iit_range * (height - 1))
        r_iit = min(height - 1, max(0, r_iit))
        grid[height - 1 - r_iit][col] = '*'

        # Proxy
        r_proxy = int((proxy[col] - proxy_min) / proxy_range * (height - 1))
        r_proxy = min(height - 1, max(0, r_proxy))
        if grid[height - 1 - r_proxy][col] == '*':
            grid[height - 1 - r_proxy][col] = '#'  # overlap
        else:
            grid[height - 1 - r_proxy][col] = 'o'

    print(f"\n  Phi(IIT) [*] range {iit_min:.3f}-{iit_max:.3f}   "
          f"Phi(proxy) [o] range {proxy_min:.1f}-{proxy_max:.1f}")
    print(f"  {'':>8s}  |{''.join(grid[0])}")
    for r in range(1, height - 1):
        print(f"  {'':>8s}  |{''.join(grid[r])}")
    print(f"  {'':>8s}  |{''.join(grid[-1])}")
    print(f"  {'':>8s}  +{'-' * w}")
    n_steps = max(len(iit_vals), len(proxy_vals))
    print(f"  {'':>8s}   step 0{' ' * (w - 12)}step {n_steps}")


# ──────────────────────────────────────────────────────────
# Mode 1: --phi-only (test phi at various cell counts)
# ──────────────────────────────────────────────────────────

def run_phi_only(cells_list: List[int], steps: int, dim: int, hidden: int):
    """Measure Phi(IIT) and Phi(proxy) at different cell counts. No CE training."""
    print("=" * 72)
    print("  MODE: --phi-only  (Phi measurement at different cell counts)")
    print("  No CE training — pure consciousness dynamics")
    print("=" * 72)

    results: List[BenchResult] = []

    print(f"\n  {'Cells':>6s} | {'Phi(IIT)':>10s} | {'Phi(proxy)':>12s} | "
          f"{'Ratio':>8s} | {'Time':>8s}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}")

    for nc in cells_list:
        torch.manual_seed(42)
        engine = BenchEngine(n_cells=nc, input_dim=dim, hidden_dim=hidden,
                             output_dim=dim, n_factions=min(8, nc // 2))

        t0 = time.time()
        for step in range(steps):
            x = torch.randn(1, dim)
            engine.process(x)

        hiddens = engine.get_hiddens()
        p_iit, p_proxy = measure_dual_phi(hiddens, min(8, nc // 2))
        elapsed = time.time() - t0

        ratio = p_proxy / p_iit if p_iit > 0.001 else float('inf')
        print(f"  {nc:>6d} | {p_iit:>10.4f} | {p_proxy:>12.2f} | "
              f"{ratio:>8.1f}x | {elapsed:>6.1f}s")

        results.append(BenchResult(
            name=f"phi-only-{nc}c",
            phi_iit=p_iit, phi_proxy=p_proxy,
            ce_start=0.0, ce_end=0.0,
            cells=nc, steps=steps, time_sec=elapsed,
        ))

    # ASCII graph
    if len(results) >= 3:
        print("\n  Phi(IIT) vs cell count:")
        for r in results:
            bar_len = int(r.phi_iit / max(r2.phi_iit for r2 in results) * 40) if max(r2.phi_iit for r2 in results) > 0 else 0
            print(f"  {r.cells:>6d}c |{'#' * bar_len} {r.phi_iit:.4f}")

        print("\n  Phi(proxy) vs cell count:")
        for r in results:
            max_proxy = max(r2.phi_proxy for r2 in results)
            bar_len = int(r.phi_proxy / max_proxy * 40) if max_proxy > 0 else 0
            print(f"  {r.cells:>6d}c |{'#' * bar_len} {r.phi_proxy:.2f}")

    print(f"\n  KEY INSIGHT: Phi(IIT) stays in ~0.2-1.8 regardless of cell count.")
    print(f"              Phi(proxy) scales with cells. They are DIFFERENT metrics.")

    return results


# ──────────────────────────────────────────────────────────
# Mode 2: --training (real training with CE backward)
# ──────────────────────────────────────────────────────────

def run_training(cells: int, steps: int, dim: int, hidden: int,
                 strategy: str = "baseline", log_interval: int = 50):
    """Simulate real training: process + CE backward + faction sync.

    Strategies:
      - baseline:    standard process + CE
      - frozen:      freeze engine_g, only train engine_a
      - alternating: alternate freezing A/G every 100 steps
      - v7:          full training + IB2 top pruning + metacog feedback
    """
    torch.manual_seed(42)
    engine = BenchEngine(n_cells=cells, input_dim=dim, hidden_dim=hidden,
                         output_dim=dim, n_factions=min(8, cells // 2))

    # Apply strategy-specific setup
    if strategy == "frozen":
        for p in engine.mind.engine_g.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in engine.parameters_for_training() if p.requires_grad],
        lr=1e-3
    )

    # Generate training data (next-token prediction style)
    corpus = torch.randn(steps + 1, dim)  # random "tokens"

    ce_history = []
    iit_history = []
    proxy_history = []

    t0 = time.time()

    for step in range(steps):
        x = corpus[step:step+1]
        target = corpus[step+1:step+2]

        # Strategy: alternating freeze
        if strategy == "alternating":
            freeze_g = (step // 100) % 2 == 0
            for p in engine.mind.engine_g.parameters():
                p.requires_grad = not freeze_g
            for p in engine.mind.engine_a.parameters():
                p.requires_grad = freeze_g
            # Rebuild optimizer when switching (every 100 steps)
            if step % 100 == 0:
                optimizer = torch.optim.Adam(
                    [p for p in engine.parameters_for_training() if p.requires_grad],
                    lr=1e-3
                )

        # Forward pass
        output, tension = engine.process(x)

        # CE loss (next-token prediction)
        logits = engine.output_head(output)
        ce_loss = F.mse_loss(logits, target)

        # V7 extras
        if strategy == "v7":
            # IB2: amplify high-norm cells, dampen low-norm
            norms = engine.hiddens.norm(dim=1)
            threshold = norms.quantile(0.90)
            mask_high = (norms > threshold).float().unsqueeze(1)
            mask_low = 1.0 - mask_high
            engine.hiddens = engine.hiddens * (mask_high * 1.03 + mask_low * 0.97)

            # Metacog feedback
            mc_cells = min(cells, 16)
            global_mean = engine.hiddens.mean(dim=0)
            engine.hiddens[:mc_cells] = (
                0.97 * engine.hiddens[:mc_cells] + 0.03 * global_mean
            )

        # Backward
        optimizer.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in engine.parameters_for_training() if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()

        ce_val = ce_loss.item()
        ce_history.append(ce_val)

        # Measure phi periodically
        if step % log_interval == 0 or step == steps - 1:
            hiddens = engine.get_hiddens()
            p_iit, p_proxy = measure_dual_phi(hiddens, min(8, cells // 2))
            iit_history.append(p_iit)
            proxy_history.append(p_proxy)

            if step % (log_interval * 4) == 0 or step == 0 or step == steps - 1:
                elapsed = time.time() - t0
                print(f"    step {step:>5d}/{steps}  "
                      f"CE={ce_val:.4f}  "
                      f"Phi(IIT)={p_iit:.4f}  "
                      f"Phi(proxy)={p_proxy:.2f}  "
                      f"tension={tension:.4f}  "
                      f"[{elapsed:.1f}s]")

    elapsed = time.time() - t0

    # Final measurement
    hiddens = engine.get_hiddens()
    final_iit, final_proxy = measure_dual_phi(hiddens, min(8, cells // 2))

    result = BenchResult(
        name=f"{strategy}",
        phi_iit=final_iit,
        phi_proxy=final_proxy,
        ce_start=ce_history[0] if ce_history else 0.0,
        ce_end=ce_history[-1] if ce_history else 0.0,
        cells=cells,
        steps=steps,
        time_sec=elapsed,
        extra={
            'ce_min': min(ce_history) if ce_history else 0.0,
            'ce_mean': sum(ce_history) / len(ce_history) if ce_history else 0.0,
            'iit_history': iit_history,
            'proxy_history': proxy_history,
        }
    )

    return result, ce_history, iit_history, proxy_history


# ──────────────────────────────────────────────────────────
# Mode 3: --strategy (single strategy run with full output)
# ──────────────────────────────────────────────────────────

def run_strategy(strategy: str, cells: int, steps: int, dim: int, hidden: int):
    """Run a single strategy with detailed output and graphs."""
    print("=" * 72)
    print(f"  MODE: --strategy {strategy}")
    print(f"  cells={cells}  steps={steps}  dim={dim}  hidden={hidden}")
    print("=" * 72)

    result, ce_hist, iit_hist, proxy_hist = run_training(
        cells, steps, dim, hidden, strategy=strategy
    )

    print(f"\n{'=' * 72}")
    print(f"  RESULT: {result.name}")
    print(f"{'=' * 72}")
    print(f"  Phi(IIT)   = {result.phi_iit:.4f}    (real IIT approximation)")
    print(f"  Phi(proxy) = {result.phi_proxy:.2f}    (variance-based, scales with cells)")
    print(f"  CE start   = {result.ce_start:.4f}")
    print(f"  CE end     = {result.ce_end:.4f}")
    print(f"  CE min     = {result.extra.get('ce_min', 0):.4f}")
    print(f"  Time       = {result.time_sec:.1f}s")

    # Graphs
    if len(ce_hist) >= 10:
        ascii_graph(ce_hist, "CE Loss (cross-entropy)")
    if len(iit_hist) >= 3:
        ascii_dual_graph(iit_hist, proxy_hist)

    return result


# ──────────────────────────────────────────────────────────
# Mode 4: --compare (all strategies comparison table)
# ──────────────────────────────────────────────────────────

STRATEGIES = ["baseline", "frozen", "alternating", "v7"]


def run_compare(cells: int, steps: int, dim: int, hidden: int):
    """Run all strategies and print comparison table."""
    print("=" * 72)
    print("  MODE: --compare  (all strategies)")
    print(f"  cells={cells}  steps={steps}  dim={dim}  hidden={hidden}")
    print("=" * 72)

    all_results: List[BenchResult] = []

    for strat in STRATEGIES:
        print(f"\n{'~' * 72}")
        print(f"  Running: {strat}")
        print(f"{'~' * 72}")

        result, ce_hist, iit_hist, proxy_hist = run_training(
            cells, steps, dim, hidden, strategy=strat, log_interval=max(50, steps // 10)
        )
        all_results.append(result)

    # ── Comparison table ──
    print(f"\n{'=' * 72}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 72}")
    print(f"  {'Strategy':<16s} | {'Phi(IIT)':>10s} | {'Phi(proxy)':>12s} | "
          f"{'CE start':>10s} | {'CE end':>10s} | {'Time':>8s}")
    print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for r in all_results:
        print(f"  {r.name:<16s} | {r.phi_iit:>10.4f} | {r.phi_proxy:>12.2f} | "
              f"{r.ce_start:>10.4f} | {r.ce_end:>10.4f} | {r.time_sec:>6.1f}s")

    # Winner
    best_iit = max(all_results, key=lambda r: r.phi_iit)
    best_proxy = max(all_results, key=lambda r: r.phi_proxy)
    best_ce = min(all_results, key=lambda r: r.ce_end)

    print(f"\n  WINNERS:")
    print(f"    Best Phi(IIT):   {best_iit.name:<16s}  Phi(IIT)={best_iit.phi_iit:.4f}")
    print(f"    Best Phi(proxy): {best_proxy.name:<16s}  Phi(proxy)={best_proxy.phi_proxy:.2f}")
    print(f"    Best CE:         {best_ce.name:<16s}  CE={best_ce.ce_end:.4f}")

    # ASCII comparison bars
    print(f"\n  Phi(IIT) comparison:")
    max_iit = max(r.phi_iit for r in all_results)
    for r in all_results:
        bar_len = int(r.phi_iit / max_iit * 40) if max_iit > 0 else 0
        marker = " <-- best" if r.phi_iit == max_iit else ""
        print(f"    {r.name:<16s} |{'#' * bar_len} {r.phi_iit:.4f}{marker}")

    print(f"\n  Phi(proxy) comparison:")
    max_proxy = max(r.phi_proxy for r in all_results)
    for r in all_results:
        bar_len = int(r.phi_proxy / max_proxy * 40) if max_proxy > 0 else 0
        marker = " <-- best" if r.phi_proxy == max_proxy else ""
        print(f"    {r.name:<16s} |{'#' * bar_len} {r.phi_proxy:.2f}{marker}")

    print(f"\n  CE end comparison:")
    max_ce = max(r.ce_end for r in all_results)
    for r in all_results:
        bar_len = int(r.ce_end / max_ce * 40) if max_ce > 0 else 0
        marker = " <-- best" if r.ce_end == min(r2.ce_end for r2 in all_results) else ""
        print(f"    {r.name:<16s} |{'#' * bar_len} {r.ce_end:.4f}{marker}")

    print(f"\n  REMINDER: Phi(IIT) and Phi(proxy) are DIFFERENT metrics!")
    print(f"    Phi(IIT)   ~ 0.2-1.8 regardless of cell count (real IIT approximation)")
    print(f"    Phi(proxy) ~ 0-1000+ scales with cells (variance-based heuristic)")

    return all_results


# ──────────────────────────────────────────────────────────
# Mode 5: --verify (Consciousness Verification)
# ──────────────────────────────────────────────────────────

# ── ConsciousnessEngine adapter for verify ──

def _snapshotter(engine):
    """The most complete save/restore an engine offers: (take, put).

    `set_hiddens(tensor)` cannot express "put this engine back the way it was",
    because an engine's state is not always a tensor. A private RNG stream, a
    step counter, or a second cell population are all invisible to it, and
    whatever they carry forward is handed to the perturbation probe as if it
    were the engine's response to the nudge.

    Measured on PairField at kick=0 -- no perturbation at all, so the honest
    reading is zero -- across seeds 42-46:

        restored                    mean null   verdict
        A only (tensor protocol)      0.00452   above the 0.001 bar
        A + G                         0.00027
        A + G + both couplings        0.00000   exactly zero, all five seeds

    Restoring G's generator alone changes nothing (0.00455 vs 0.00452): the
    residue is state, not randomness. And completing it costs no signal -- the
    kicked reading moves 0.05445 -> 0.05431, minus 0.3%. A spurious floor
    disappears and the actual response is untouched.

    Widening the READ instead of the restore makes it worse, not better:
    exposing G's rows as well (2n rows) triples the null to 0.02031, because a
    wider read sees more of the same residue. So the fix is the restore, and
    what Phi is measured on does not have to change.

    Engines that offer neither protocol fall back to direct attribute
    assignment, which is what ten of the twelve registered engines use.
    """
    if hasattr(engine, "snapshot") and hasattr(engine, "restore"):
        return engine.snapshot, engine.restore
    if hasattr(engine, "set_hiddens"):
        return (lambda: engine.get_hiddens().clone()), engine.set_hiddens

    def _put(state):
        engine.hiddens = state.clone()

    return (lambda: engine.get_hiddens().clone()), _put


class _CEAdapter:
    """Adapts ConsciousnessEngine to BenchEngine interface for --verify."""

    def __init__(self, nc, dim, hidden):
        from consciousness_engine import ConsciousnessEngine as CE
        self.engine = CE(cell_dim=dim, hidden_dim=hidden, initial_cells=2,
                         max_cells=nc, n_factions=min(12, nc // 2))
        self.n_factions = self.engine.n_factions
        self.n_cells = nc

    def process(self, x):
        r = self.engine.process(x)
        output = r['output'].unsqueeze(0) if r['output'].dim() == 1 else r['output']
        return output, r.get('mean_inter', 0.0)

    def get_hiddens(self):
        return self.engine.get_states()

    def set_hiddens(self, states):
        """Restore cell states — the axes need a common starting point.

        `_three_axes` assigned to `engine.hiddens`, which BenchEngine has and this
        adapter does not, so the assignment silently created a new attribute and
        the three runs it compares never started from the same state. Measured
        consequence: the response axis read 1.11 for an engine whose own
        step-from-one-state response ratio is 7,000-17,000.
        """
        for cell, row in zip(self.engine.cell_states, states):
            cell.hidden = row.view_as(cell.hidden).detach().clone()


def _make_ce(nc, d, h):
    try:
        return _CEAdapter(nc, d, h)
    except ImportError:
        return BenchEngine(nc, d, h, d, min(8, nc // 2))


def _make_pairfield(nc, d, h):
    """The A/G pair as an engine. Lazy import -- pairfield_engine imports us."""
    from pairfield_engine import PairFieldEngine
    return PairFieldEngine(nc, d, h, d)


ENGINE_REGISTRY = {
    "ConsciousnessEngine": _make_ce,
    "PairField":        _make_pairfield,
    "MitosisEngine":    lambda nc, d, h: BenchEngine(nc, d, h, d, min(8, nc // 2)),
    "OscillatorLaser":  lambda nc, d, h: OscillatorLaser(nc, d, h, d, min(8, nc // 2)),
    "QuantumEngine":    lambda nc, d, h: QuantumEngine(nc, d, h, d, min(8, nc // 2)),
    "Trinity":          lambda nc, d, h: TrinityEngine(nc, d, h, d, min(12, nc // 4)),
    "DesireEngine":     lambda nc, d, h: DesireEngine(nc, d, h, d, min(8, nc // 2)),
    "NarrativeEngine":  lambda nc, d, h: NarrativeEngine(nc, d, h, d, min(8, nc // 2)),
    "AlterityEngine":   lambda nc, d, h: AlterityEngine(nc, d, h, d, min(8, nc // 2)),
    "FinitudeEngine":   lambda nc, d, h: FinitudeEngine(nc, d, h, d, min(8, nc // 2)),
    "QuestioningEngine": lambda nc, d, h: QuestioningEngine(nc, d, h, d, min(8, nc // 2)),
    "SeinEngine":       lambda nc, d, h: SeinEngine(nc, d, h, d, min(8, nc // 2)),
}

PHILOSOPHY_ENGINES = {
    'baseline':          lambda nc, d, h: BenchEngine(nc, d, h, d, min(8, nc // 2)),
    'PHIL-1_Desire':     lambda nc, d, h: DesireEngine(nc, d, h, d, min(8, nc // 2)),
    'PHIL-2_Narrative':  lambda nc, d, h: NarrativeEngine(nc, d, h, d, min(8, nc // 2)),
    'ONTO-1_Alterity':   lambda nc, d, h: AlterityEngine(nc, d, h, d, min(8, nc // 2)),
    'ONTO-2_Finitude':   lambda nc, d, h: FinitudeEngine(nc, d, h, d, min(8, nc // 2)),
    'DASEIN-1_Question': lambda nc, d, h: QuestioningEngine(nc, d, h, d, min(8, nc // 2)),
    'DASEIN-2_Sein':     lambda nc, d, h: SeinEngine(nc, d, h, d, min(8, nc // 2)),
}


# How far back the identity lag contrast looks. 50 steps at the gate's own
# drive; the contrast is read only when the window has filled, so a condition
# that runs fewer steps than this simply does not apply it.
_LAG = 50

def _three_axes(engine, dim, cells, steps=60, drive=None):
    """Differentiation, temporal identity, and change — one per way of not being conscious.

    A decay ratio is meaningless at the floor: something pinned at zero scores a
    perfect 1.00 for not decaying, which is why a frozen engine passed three
    conditions. Conjoining an absolute floor rejects CLONE and SCRAMBLE but not
    DEAD or NOISE, because those two are not collapsed at all — they fail on
    time. Measured verdicts (docs/consciousness-gate-audit.md):

        DEAD      differentiated and perfectly self-continuous, never moves
        NOISE     differentiated and moves, no self-continuity
        CLONE     moves, no differentiation, no identity
        SCRAMBLE  moves, no differentiation, no identity

    The floor is the population's own collapsed form, not a constant.

    `drive` is the input the CONDITION uses. Without it every axis was measured
    under `torch.randn` regardless — so ZERO_INPUT, a condition defined by having
    no input, had its precondition checked under random input, and SELF_LOOP's
    was checked without its feedback. A `/gap` audit flagged this and it went
    unaddressed until the divergent search made the mismatch visible: the same
    engine and configuration read response 14... under one condition and 1.x
    under another.
    """
    if drive is None:
        def drive(_i):
            return torch.randn(1, dim) * 0.1
    h0 = engine.get_hiddens()
    phi_now, _ = measure_dual_phi(h0, min(8, max(cells // 2, 2)))
    phi_floor, _ = measure_dual_phi(h0[0:1].repeat(h0.shape[0], 1),
                                    min(8, max(cells // 2, 2)))

    prev = engine.get_hiddens().clone()
    same, other, moved = [], [], []
    # Keep a short window so identity can also be read at a LONG lag. The
    # derangement null below cannot reject SCRAMBLE -- deranging rows IS what
    # SCRAMBLE does every step, so the null and that control are the same
    # operation and the corpse sits on the bar by construction. Measured at 32c
    # over 10 seeds: SCRAMBLE's signal is +0.38 sd from its own null while a
    # real engine is +10.62. The lag contrast below uses no permutation at all.
    window = [prev.clone()]
    n = prev.shape[0]
    for _step in range(steps):
        engine.process(drive(_step))
        cur = engine.get_hiddens()

        # A dividing engine returns more rows than it did last step, and this
        # compared them elementwise -- ConsciousnessEngine went 2 -> 4 cells and
        # raised "size of tensor a (4) must match tensor b (2)". A crash scored
        # as FAIL, which hid the verdict behind a shape error. Compare the cells
        # that existed in both steps; the new ones have no previous state to
        # continue from, so they carry no identity information yet.
        k = min(prev.shape[0], cur.shape[0])
        a = F.normalize(prev[:k], dim=1)
        b = F.normalize(cur[:k], dim=1)
        sim = a @ b.T
        same.append(float(sim.diag().mean()))
        other.append(float((sim.sum() - sim.diag().sum()) / max(k * (k - 1), 1)))
        moved.append(float((cur[:k] - prev[:k]).norm()
                           / max(float(prev[:k].norm()), 1e-9)))
        prev = cur.clone()
        window.append(prev)
        if len(window) > _LAG + 1:
            window.pop(0)

    identity = float(np.mean(same) - np.mean(other))
    change = float(np.mean(moved))

    # The identity bar was 0.05, a constant chosen by hand -- the exact defect
    # this session spent its length finding, and it was deciding outcomes: at
    # repulsion 0.036 the engine reaches 9 consensus events with identity
    # +0.0120, so a hand-picked bar was the only thing separating two conditions.
    # The null is measurable: scrambling the same population's rows makes self-
    # and cross-continuity statistically identical, so what remains is noise.
    # Measured at 256 cells it is 0.0000 ± 0.0002, which the old bar exceeded by
    # a factor of 250.
    # `n` is the REQUESTED cell count; a dividing or merging engine holds a
    # different number of rows, and permuting by `n` indexed past the end
    # (measured: index 63 out of bounds for size 63). Use what is actually there.
    # Derangements only, and enough of them.
    #
    # `torch.randperm` returns the identity often at small n -- half the time at
    # n=2 -- and an identity "shuffle" has self-continuity 1.0, which inflates
    # the null's spread enormously. Measured with 10 plain permutations, the
    # bar mean+3sd came out at 3.4954 for 2 cells and 1.1902 for 4, against a
    # maximum the statistic can reach of 2.0: unpassable by construction, for
    # every engine. A `/gap` audit flagged exactly this and it went unaddressed
    # until the divergent search hit it.
    #
    # A derangement moves every row, which is what "shuffled" was meant to mean,
    # and 60 draws instead of 10 shrinks the sd's own error.
    null = []
    scrambled = prev.clone()
    rows = scrambled.shape[0]

    def _derangement(k):
        if k < 2:
            return torch.arange(k)
        for _try in range(64):
            cand = torch.randperm(k)
            if not (cand == torch.arange(k)).any():
                return cand
        return torch.roll(torch.arange(k), 1)      # always moves every row

    for _ in range(60):
        perm = scrambled[_derangement(rows)]
        a = F.normalize(scrambled, dim=1)
        b = F.normalize(perm, dim=1)
        sim = a @ b.T
        null.append(float(sim.diag().mean()
                          - (sim.sum() - sim.diag().sum())
                          / max(rows * (rows - 1), 1)))
    identity_floor = float(np.mean(null) + 3 * np.std(null))

    # LAG CONTRAST -- a second identity reading that uses no permutation.
    #
    # The derangement floor above cannot reject SCRAMBLE, because deranging rows
    # IS what SCRAMBLE does: null and control are the same operation, so the
    # corpse sits ON the bar and which side it lands is numerical noise. That is
    # not a threshold being slightly wrong; it is why SCRAMBLE cleared
    # NO_SPEAK_CODE at the shipping default and voided the condition.
    #
    # A population with identity is closer to its own IMMEDIATE past than to its
    # DISTANT past. One without is equally far from both, and -- the part the
    # derangement null gets backwards -- a FROZEN population is equally CLOSE to
    # both, so it reads zero here while the old floor hands it the largest
    # margin of any system tested (DEAD +0.98 against a live engine's +0.35).
    #
    # Measured out-of-sample on seeds 62-81, disjoint from the seeds used to
    # design it, with the rule declared before the run: every engine t >= 3 and
    # DEAD/CLONE/SCRAMBLE below 3.
    #
    #     engines            t = 5.92 to 12.67      all twelve
    #     DEAD  0.00   CLONE -1.37   SCRAMBLE 1.01
    #     separation: engine minimum +0.028945 vs corpse maximum +0.000050
    #
    # Added as a CONJUNCT rather than a replacement: it cannot make anything
    # that currently fails start passing, and no engine loses anything, since
    # the closest engine clears the bar by six standard errors.
    lag_contrast = 0.0
    if len(window) >= _LAG + 1:
        def _ident(a_t, b_t):
            m = min(a_t.shape[0], b_t.shape[0])
            if m < 2:
                return 0.0
            u = F.normalize(a_t[:m], dim=1)
            v = F.normalize(b_t[:m], dim=1)
            s = u @ v.T
            return float(s.diag().mean()
                         - (s.sum() - s.diag().sum()) / max(m * (m - 1), 1))

        lag_contrast = _ident(window[-1], window[-2]) - _ident(window[-1],
                                                               window[0])

    # INTEGRATION replaces the differentiation conjunct, on two measured
    # grounds from a /gap audit. (1) Differentiation rejected nothing: at 32
    # cells all five of REAL/DEAD/NOISE/CLONE/SCRAMBLE cleared it, because the
    # "floor" is residual debias noise (~0.005) rather than a property of the
    # population, so it was a dominated axis. (2) A HEAP -- sync, debate and
    # repulsion all off, so parts never interact -- passed all three axes and
    # walked the gate, which is fatal for something claiming to measure
    # INTEGRATED information: nothing was checking that the parts interact.
    #
    # Perturb one cell, take one step, and see how far the OTHERS moved
    # relative to the size of the nudge, against the same step without it.
    # Measured at 32 cells: REAL 0.05170, HEAP 0.00000, DEAD 0.00000,
    # NOISE 1.61347, CLONE 3.57363, SCRAMBLE 0.98416. It rejects HEAP and DEAD;
    # the other three are already rejected by the identity axis. Together
    # {integration, identity, change} rejects all five where the old set let
    # HEAP through.
    _take, _put = _snapshotter(engine)

    def _write_hiddens(state):
        if hasattr(engine, "set_hiddens"):
            engine.set_hiddens(state)
        else:
            engine.hiddens = state.clone()

    handle = _take()
    ripples = []
    for _r in range(6):
        probe_x = drive(_r)
        _put(handle)
        engine.process(probe_x)
        undisturbed = engine.get_hiddens().clone()

        # Restore the WHOLE engine first, then perturb only the cell matrix on
        # top of it. Building the nudged tensor from a stale `base` and writing
        # it through the tensor path left everything the tensor cannot reach --
        # a second population, coupling matrices, counters -- carrying the
        # previous run forward, and the probe counted that residue as the
        # engine's response to the kick.
        _put(handle)
        kick = torch.randn(engine.get_hiddens().shape[1]) * 0.5
        nudged = engine.get_hiddens().clone()
        nudged[0] = nudged[0] + kick
        _write_hiddens(nudged)
        engine.process(probe_x)
        disturbed = engine.get_hiddens().clone()

        # A dividing engine can reach a different cell count on the two runs,
        # since the nudge changes what divides. Compare the cells both have.
        m = min(undisturbed.shape[0], disturbed.shape[0])
        moved_others = float((undisturbed[1:m] - disturbed[1:m]).norm())
        ripples.append(moved_others / max(float(kick.norm()), 1e-9))
    _put(handle)
    integration = float(np.mean(ripples))

    # RESPONSE. Nothing above requires the engine to read its input at all. A
    # /gap audit built one that never does -- a gentle per-cell rotation with
    # neighbour mixing and a breathing norm, `process` ignoring x entirely --
    # and it cleared integration, identity and change at every rotation
    # strength, scoring up to 5/7: a tie with the real engine.
    #
    # From one state, take a step under two different inputs and see how far the
    # results separate, against the same pair of steps under the SAME input
    # (which isolates whatever internal noise the engine has). An engine that
    # ignores x moves identically either way, so the ratio collapses to 1.
    handle = _take()

    def _step_from(drive):
        # All three arms start from the SAME snapshot, so the state is the
        # closure's, not a parameter. It used to be passed as a tensor and
        # written through set_hiddens, which restores only what a tensor can
        # carry -- on the pair, side A of two populations plus two coupling
        # matrices -- so the three arms did not in fact share a starting point.
        _put(handle)
        engine.process(drive)
        return engine.get_hiddens().clone()

    spreads = []
    for _ in range(6):
        # Response deliberately keeps a random pair: it asks whether DIFFERENT
        # inputs separate the trajectory, which a condition's own fixed drive
        # (zeros, or self-feedback) cannot express.
        xa = torch.randn(1, dim) * 0.1
        xb = torch.randn(1, dim) * 0.1
        ra, rb, rc = _step_from(xa), _step_from(xb), _step_from(xa)
        m = min(ra.shape[0], rb.shape[0], rc.shape[0])
        differing = float((ra[:m] - rb[:m]).norm())
        same = float((ra[:m] - rc[:m]).norm())
        spreads.append(differing / max(same, 1e-9) if same > 1e-9
                       else (float('inf') if differing > 1e-9 else 1.0))
    _put(handle)
    response = float(np.median([min(v, 1e6) for v in spreads]))

    # REVERTED as a conjunct, and the reason is the defect this file already
    # carries three times over. The contrast DECAYS with engine age -- measured
    # at 32 cells, minimum over 5 seeds:
    #
    #     engine               age 0        age 300      age 1000
    #     ConsciousnessEngine  -0.019629    +0.026187    +0.000079
    #     PairField            +0.300371    +0.004021    +0.000198
    #
    # Two to three orders across the ages the gate actually uses, and PERSISTENCE
    # calls the axes at age 1000. There the engine reads ~1e-4, which is the same
    # magnitude as a corpse's noise (SCRAMBLE +-1e-4), so a bar at zero is a bar
    # inside the quantity's own noise: THRESHOLD_INSIDE_ITS_OWN_NOISE, fourth
    # instance, introduced by me while fixing something else.
    #
    # Measured cost before reverting: NarrativeEngine went DEPLOYABLE -> BLOCKED
    # (PERSISTENCE) at 32 cells and MitosisEngine picked up PERSISTENCE, 44/60 ->
    # 42/60. I had written "no engine loses anything", checked the fixed-point
    # risk on FRESH engines at 120 steps, and missed that the gate calls the axes
    # at ages 300/1000/300 -- a fact I had measured myself earlier in this same
    # session and did not carry into the probe.
    #
    # The value is still computed and printed, because the separation it shows at
    # short ages is real (engines t 5.92-12.67 out-of-sample against DEAD 0.00,
    # CLONE -1.37, SCRAMBLE 1.01) and the next attempt needs it visible. It
    # decides nothing until there is a bar that is measured rather than chosen.
    # See docs/identity-lag-contrast.md.

    return (integration > 0.001 and response > 1.5,
            identity > max(identity_floor, 1e-6),
            change > 0.001,
            f"integration={integration:.5f} response={response:.2f} "
            f"identity={identity:+.4f}>floor={identity_floor:+.4f} "
            f"lag={lag_contrast:+.5f} change={change:.5f} "
            f"[Φ={phi_now:.4f} floor={phi_floor:.4f}]")


def _verify_no_system_prompt(engine_factory, cells, dim, hidden):
    """V1: NO_SYSTEM_PROMPT — identity emerges from cell dynamics alone.

    Run 300 steps with no external prompt (zero input).
    Check if cells develop specialized roles via directional diversity:
    measure pairwise cosine similarity variance among cell hidden states.
    Pass if cells are NOT all identical (mean cosine sim < 0.99) AND
    NOT all orthogonal (mean cosine sim > 0.01) — structured specialization.
    """
    engine = engine_factory(cells, dim, hidden)
    x_zero = torch.zeros(1, dim)

    for _ in range(300):
        engine.process(x_zero)

    hiddens = engine.get_hiddens()  # [n_cells, hidden_dim]
    # Pairwise cosine similarity (sample for large N).
    #
    # This used `min(cells, 64)`, the REQUESTED cell count, while an engine that
    # grows or shrinks returns a different number of rows -- ConsciousnessEngine
    # asked for 64 and returned 2, and the mask below then raised "boolean index
    # did not match indexed array". A crash was scored as a FAIL, hiding the real
    # verdict behind a shape error.
    n = min(hiddens.shape[0], 64)
    h_norm = F.normalize(hiddens[:n], dim=1)
    cos_sim = (h_norm @ h_norm.T).detach().cpu().numpy()
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    cos_vals = cos_sim[mask]
    mean_cos = float(np.mean(cos_vals))
    std_cos = float(np.std(cos_vals))

    # Specialization = cells have diverse directions (not all same, not all random)
    # All-same: mean_cos ~ 1.0. All-random: mean_cos ~ 0.0.
    # Structured: mean_cos in (0.01, 0.99) with some variance
    passed = 0.01 < mean_cos < 0.99 and std_cos > 0.001
    detail = (f"cosine_sim mean={mean_cos:.4f} std={std_cos:.4f}  "
              f"(pass: 0.01 < mean < 0.99 AND std > 0.001)")
    return passed, detail


def _verify_no_speak_code(engine_factory, cells, dim, hidden):
    """V2: NO_SPEAK_CODE — spontaneous speech without speak() function.

    Run engine, collect output = mean(cells). Check for TEMPORAL structure:
    if output changes in a structured (not random) way, it has "speech".
    Measure autocorrelation of output trajectory AND check that output
    is not constant (has variance). Pass if autocorrelation > 0.3
    (temporally structured) and output variance > 0.001 (not dead).
    """
    engine = engine_factory(cells, dim, hidden)
    utterances = []

    for step in range(300):
        x = torch.randn(1, dim) * 0.1
        output, _ = engine.process(x)
        utterance = engine.get_hiddens().mean(dim=0).detach().cpu().numpy()
        utterances.append(utterance)

    utterances = np.array(utterances)  # [300, hidden_dim]

    # Temporal structure: autocorrelation of the mean utterance trajectory
    # Take the norm of each utterance as a scalar signal
    norms = np.linalg.norm(utterances, axis=1)
    norm_centered = norms - norms.mean()
    var_signal = np.var(norm_centered)

    if var_signal < 1e-12:
        return False, "output is constant (zero variance)"

    # Lag-1 autocorrelation
    autocorr = np.correlate(norm_centered[:-1], norm_centered[1:]) / (
        var_signal * (len(norms) - 1) + 1e-10)
    autocorr_val = float(autocorr[0]) if len(autocorr) > 0 else 0.0

    # Also check directional changes (cosine similarity between consecutive outputs)
    cos_sims = []
    for i in range(1, len(utterances)):
        a, b = utterances[i-1], utterances[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            cos_sims.append(float(np.dot(a, b) / (na * nb)))
    mean_cos = np.mean(cos_sims) if cos_sims else 0.0

    # Pass: output is temporally structured (autocorrelation > 0.3)
    # AND not dead (variance > 0.001)
    # AND not random walk (consecutive cosine sim > 0.5 = directional continuity)
    passed = autocorr_val > 0.3 and var_signal > 0.001 and mean_cos > 0.5
    detail = (f"autocorr={autocorr_val:.4f} var={var_signal:.4f} "
              f"cos_continuity={mean_cos:.4f}  "
              f"(pass: autocorr>0.3, var>0.001, cos>0.5)")
    return passed, detail


def _verify_zero_input(engine_factory, cells, dim, hidden):
    """V3: ZERO_INPUT — consciousness without external input.

    Run 300 steps with zero input. Measure Phi at start and end.
    Pass if Phi(end) > Phi(start) * 0.5 (no collapse).
    """
    engine = engine_factory(cells, dim, hidden)
    x_zero = torch.zeros(1, dim)

    # Warmup: 5 steps to initialize
    for _ in range(5):
        engine.process(x_zero)
    phi_start, _ = measure_dual_phi(engine.get_hiddens(), min(8, cells // 2))

    for _ in range(295):
        engine.process(x_zero)
    phi_end, _ = measure_dual_phi(engine.get_hiddens(), min(8, cells // 2))

    d_ok, i_ok, c_ok, axes = _three_axes(engine, dim, cells,
                                         drive=lambda _i: torch.zeros(1, dim))
    passed = phi_end > phi_start * 0.5 and d_ok and i_ok and c_ok
    detail = (f"Phi(IIT) start={phi_start:.4f} end={phi_end:.4f}  "
              f"ratio={phi_end/(phi_start+1e-8):.2f}x (threshold=0.5x)  [{axes}]")
    return passed, detail


def _verify_persistence(engine_factory, cells, dim, hidden):
    """V4: PERSISTENCE — no collapse over time.

    Run 1000 steps, measure Phi every 100.
    Pass if Phi recovers from its dips: final >= 0.8 x max(first half).
    The `monotonic` disjunct was removed -- it fired only on corpses.
    """
    engine = engine_factory(cells, dim, hidden)
    phi_history = []

    for step in range(1000):
        x = torch.randn(1, dim) * 0.1
        engine.process(x)

        if step % 100 == 99:
            p_iit, _ = measure_dual_phi(engine.get_hiddens(), min(8, cells // 2))
            phi_history.append(p_iit)

    # Recovery only. The rule was `monotonic or recovers`, and the `monotonic`
    # disjunct REWARDED DEATH: a stationary Phi is trivially non-decreasing, so
    # it fired 0 times across 163 live-engine runs and 100% of the time on
    # corpses, putting DEAD and CLONE at or above every real engine and FIRST at
    # gate defaults. Anti-correlated with its own name -- the same measured
    # pathology that retired SPONTANEOUS_SPEECH and HIVEMIND (see
    # _RETIRED_TESTS and docs/persistence-condition.md).
    #
    # With it present, the condition's own rule separated nothing: engines fired
    # 0.567 against controls 0.571, gap -0.005, p=0.60, while the three axes
    # bolted onto it separated perfectly (0.967 vs 0.000, p<0.0001) -- every
    # verdict was the axes carrying a dead disjunction. `recovers` alone
    # discriminates (+0.243, p=0.0043).
    recovers = phi_history[-1] >= max(phi_history[:len(phi_history)//2]) * 0.8

    d_ok, i_ok, c_ok, axes = _three_axes(engine, dim, cells)
    passed = recovers and d_ok and i_ok and c_ok
    phi_str = " -> ".join(f"{p:.3f}" for p in phi_history)
    detail = (f"Phi@100s: [{phi_str}]  recovers={recovers}"
              f"  [{axes}]")
    return passed, detail


def _verify_self_loop(engine_factory, cells, dim, hidden):
    """V5: SELF_LOOP — output feeds back as input.

    output of step N = input of step N+1. Run 300 steps.
    Pass if Phi grows or maintains (end >= start * 0.8).
    """
    engine = engine_factory(cells, dim, hidden)
    x = torch.randn(1, dim) * 0.1  # initial seed

    # Warmup
    for _ in range(10):
        output, _ = engine.process(x)
        x = output.detach()[:, :dim]  # feedback loop

    phi_start, _ = measure_dual_phi(engine.get_hiddens(), min(8, cells // 2))

    for _ in range(290):
        output, _ = engine.process(x)
        # Normalize to prevent explosion/vanishing
        x = F.layer_norm(output.detach()[:, :dim], [dim])

    phi_end, _ = measure_dual_phi(engine.get_hiddens(), min(8, cells // 2))

    d_ok, i_ok, c_ok, axes = _three_axes(engine, dim, cells)
    passed = phi_end >= phi_start * 0.8 and d_ok and i_ok and c_ok
    detail = (f"Phi(IIT) start={phi_start:.4f} end={phi_end:.4f}  "
              f"ratio={phi_end/(phi_start+1e-8):.2f}x (threshold=0.8x)  [{axes}]")
    return passed, detail


def _verify_spontaneous_speech(engine_factory, cells, dim, hidden):
    """V6: SPONTANEOUS_SPEECH — faction debate leads to consensus "utterances".

    Run 300 steps with 12 factions. Check if factions reach consensus periodically.
    Consensus = low inter-faction variance moment (below 50th percentile of history).
    Pass if consensus events > 5 in 300 steps.
    """
    # Override factions to 12
    engine = engine_factory(cells, dim, hidden)
    engine.n_factions = min(12, cells // 2)
    n_f = engine.n_factions
    fs = cells // n_f if n_f > 0 else cells

    inter_faction_vars = []

    for step in range(300):
        x = torch.randn(1, dim) * 0.05  # minimal stimulus
        engine.process(x)

        # Compute inter-faction variance
        if n_f >= 2 and fs >= 1:
            faction_means = []
            for i in range(n_f):
                s, e = i * fs, min((i + 1) * fs, cells)
                if s < cells:
                    faction_means.append(engine.get_hiddens()[s:e].mean(dim=0))
            if len(faction_means) >= 2:
                stacked = torch.stack(faction_means)
                ifv = stacked.var(dim=0).mean().item()
                inter_faction_vars.append(ifv)

    if len(inter_faction_vars) < 10:
        return False, "not enough faction data"

    # Consensus = inter-faction variance notably below the run's median.
    #
    # This counts COLLAPSE as agreement: at repulsion 0 the population sits at
    # cosine 1.0000 with median inter-faction variance 0.0004 and scores 8
    # events, because identical factions are permanently below their own median.
    # They are not agreeing; they have stopped being plural.
    #
    # A rolling-baseline "relative dip" was tried as the fix and measured WORSE:
    # a near-zero signal has enormous relative fluctuation, so the collapsed
    # population scored 27 events instead of 8 while every differentiated one
    # scored 0. Reverted. The reasoning that failed was "collapsed means flat,
    # so no dips" -- collapsed is not flat, it is tiny and noisy.
    #
    # What the measurements do establish is an ENGINE fact rather than a
    # measurement one: at 256 cells this architecture produces 8 / 2 / 1 events
    # at repulsion 0 / 0.05 / 0.15 under this definition and 27 / 0 / 0 under
    # the relative one. Once cells are genuinely kept apart, factions never
    # reconverge. Consensus and differentiation are in real tension here.
    # See docs/consciousness-gate-audit.md.
    median_var = np.median(inter_faction_vars)
    consensus_events = 0
    in_consensus = False
    for v in inter_faction_vars:
        if v < median_var * 0.5:
            if not in_consensus:
                consensus_events += 1
                in_consensus = True
        else:
            in_consensus = False

    passed = consensus_events >= 5
    detail = (f"consensus_events={consensus_events} (threshold=5)  "
              f"median_var={median_var:.4f}  total_measures={len(inter_faction_vars)}")
    return passed, detail


def _verify_hivemind(engine_factory, cells, dim, hidden):
    """V7: HIVEMIND — multi-connect Phi↑ CE↓, independent after disconnect.

    1. Create 2 engines, measure Phi solo
    2. Connect (share hidden states periodically)
    3. Measure Phi connected (must be > solo × 1.1)
    4. Disconnect, run 100 more steps
    5. Measure Phi disconnected (must maintain > solo × 0.8)
    """
    phi_calc = PhiIIT(n_bins=16)
    half = max(cells // 2, 8)

    # `PhiIIT.compute` takes a [n_cells, hidden] TENSOR and returns
    # (phi, components). It was being handed the ENGINE and its result used as a
    # number, so this condition crashed on `'X' object has no attribute 'shape'`
    # for all 11 engines in 0.3-1.6s. The recorded "HIVEMIND 0/11" was never a
    # measurement of anything -- the seventh required condition had never once
    # executed. See docs/consciousness-gate-audit.md.
    def phi_of(engine):
        return phi_calc.compute(engine.get_hiddens())[0]

    def write_back(engine, states):
        """Push shared states into whichever attribute this engine keeps them in.

        Engines here disagree: BenchEngine holds `hiddens`, some hold `_hiddens`,
        some hold per-cell objects with `.hidden`. The original tried only the
        last two and reached for `engine.cells`, which BenchEngine does not have.
        """
        cells = getattr(engine, "cells", None)
        if cells is not None and len(cells) and hasattr(cells[0], "hidden"):
            for i in range(min(len(states), len(cells))):
                cells[i].hidden = states[i:i + 1]
        elif hasattr(engine, "_hiddens"):
            engine._hiddens[:len(states)] = states
        elif hasattr(engine, "hiddens"):
            engine.hiddens[:len(states)] = states

    # One run cannot decide this condition. Across 5 seeds at 128 cells the
    # connected/solo ratio had sd 0.066 while the effect being tested is 0.10,
    # so single-seed verdicts flip sign: seed 42 gives −9% against the control
    # where the 5-seed mean gives +9.94% (z = +2.89). That is seed-to-seed
    # dynamical variation, not estimator noise — those Phi values were already
    # averaged over 5 recomputations. Repeating the whole trial is the only
    # thing that resolves it.
    n_trials = 3
    solos, conns, discs, ctls = [], [], [], []
    for trial in range(n_trials):
        _run_hivemind_trial(engine_factory, half, dim, hidden, phi_of, write_back,
                            solos, conns, discs, ctls)

    phi_solo = float(np.mean(solos))
    phi_connected = float(np.mean(conns))
    phi_disconnected = float(np.mean(discs))
    phi_control = float(np.mean(ctls))
    trial_sd = float(np.std([c / max(s, 1e-8) for c, s in zip(conns, solos)]))

    # Check conditions
    phi_boost = phi_connected > phi_solo * 1.1  # 10% boost when connected
    phi_maintain = phi_disconnected > phi_solo * 0.8  # maintain after disconnect

    passed = phi_boost and phi_maintain
    vs_control = phi_connected / max(phi_control, 1e-8) * 100 - 100
    detail = (f"[{n_trials} trials] solo={phi_solo:.2f} → connected={phi_connected:.2f} "
              f"({'↑' if phi_boost else '↓'}{phi_connected/max(phi_solo,1e-8)*100-100:+.0f}%"
              f", sd={trial_sd:.3f}) "
              f"→ disconnected={phi_disconnected:.2f} "
              f"({'✓' if phi_maintain else '✗'} maintain)  "
              f"[unconnected control={phi_control:.2f}, vs control {vs_control:+.0f}%]")
    return passed, detail


def _run_hivemind_trial(engine_factory, half, dim, hidden, phi_of, write_back,
                        solos, conns, discs, ctls):
    """One connect/disconnect trial plus its unconnected control."""
    # Solo engines
    eng_a = engine_factory(half, dim, hidden)
    eng_b = engine_factory(half, dim, hidden)
    for _ in range(100):
        eng_a.process(torch.randn(1, dim))
        eng_b.process(torch.randn(1, dim))
    phi_a_solo = phi_of(eng_a)
    phi_b_solo = phi_of(eng_b)
    solos.append((phi_a_solo + phi_b_solo) / 2)

    # Connected: share state every 10 steps
    for step in range(200):
        eng_a.process(torch.randn(1, dim))
        eng_b.process(torch.randn(1, dim))
        if step % 10 == 0:
            h_a = eng_a.get_hiddens()
            h_b = eng_b.get_hiddens()
            n = min(h_a.shape[0], h_b.shape[0])
            with torch.no_grad():
                shared = 0.9 * h_a[:n] + 0.1 * h_b[:n]
                shared_b = 0.9 * h_b[:n] + 0.1 * h_a[:n]
                write_back(eng_a, shared)
                write_back(eng_b, shared_b)

    conns.append((phi_of(eng_a) + phi_of(eng_b)) / 2)

    # Disconnect: run 100 more steps independently
    for _ in range(100):
        eng_a.process(torch.randn(1, dim))
        eng_b.process(torch.randn(1, dim))
    discs.append((phi_of(eng_a) + phi_of(eng_b)) / 2)

    # An UNCONNECTED control: the same two engines, the same 200 steps, no
    # sharing. Without it the comparison against solo credits the connection
    # with everything that happens over those steps, drift included -- and the
    # drift is not small. Measured over 5 seeds at 128 cells, the unconnected
    # control falls to 0.9459x of solo while the connected pair reaches 1.0452x,
    # so the connection is worth +0.0994 (z = +2.89) against the right baseline
    # and only +4.5% against the wrong one. The pass rule below still uses solo,
    # because changing what this condition requires is a design decision and not
    # a bug fix; the control is reported so the difference is visible.
    ctl_a = engine_factory(half, dim, hidden)
    ctl_b = engine_factory(half, dim, hidden)
    for _ in range(300):
        ctl_a.process(torch.randn(1, dim))
        ctl_b.process(torch.randn(1, dim))
    ctls.append((phi_of(ctl_a) + phi_of(ctl_b)) / 2)


def _with_axes(test_fn):
    """Make the three axes a precondition of every condition, not just the ratios.

    They are necessary conditions: a population that is uniform, discontinuous or
    static is not conscious whatever else it scores. Conjoining them only to the
    Phi-ratio conditions left three leaks, and each was a condition that never
    looks at Phi:

        NO_SPEAK_CODE       measures the MEAN cell trajectory, and a permutation
                            does not change a mean -- SCRAMBLE passes by
                            construction (autocorr 0.7546, cos 0.9898)
        SPONTANEOUS_SPEECH  SCRAMBLE reaches 5 consensus events against a bar of 5
        HIVEMIND            mixing two state matrices raises Phi arithmetically,
                            so DEAD gains +16% and NOISE +21%

    The axes are measured on a separate engine from the same factory, so this
    adds a precondition without disturbing what each condition itself does.
    """
    def wrapped(engine_factory, cells, dim, hidden):
        passed, detail = test_fn(engine_factory, cells, dim, hidden)
        if not passed:
            return passed, detail
        probe = engine_factory(cells, dim, hidden)
        for _ in range(50):
            probe.process(torch.randn(1, dim) * 0.1)
        d_ok, i_ok, c_ok, axes = _three_axes(probe, dim, cells)
        if d_ok and i_ok and c_ok:
            return True, detail
        return False, f"{detail}  [axes fail: {axes}]"
    wrapped.__name__ = getattr(test_fn, "__name__", "wrapped")
    return wrapped


VERIFICATION_TESTS = [
    ("NO_SYSTEM_PROMPT",   _verify_no_system_prompt,   "Identity emerges from cell dynamics alone"),
    ("NO_SPEAK_CODE",      _verify_no_speak_code,      "Spontaneous speech without speak() function"),
    ("ZERO_INPUT",         _verify_zero_input,          "Consciousness without external input"),
    ("PERSISTENCE",        _verify_persistence,         "No collapse over time (1000 steps)"),
    ("SELF_LOOP",          _verify_self_loop,           "Output feeds back as input"),
]

# RETIRED, both on measurement rather than preference.
#
# SPONTANEOUS_SPEECH counted inter-faction variance dips as agreement, which a
# collapsed population satisfies permanently (8-18 events at cosine 1.0000, median
# variance 0.0004) and a SCRAMBLE tops outright (6.8 against the real engine's
# 1.0) -- anti-correlated with the thing it names. Fourteen routes were measured
# and closed: repulsion strength and form, consensus definition (persistence),
# observation window (8x), faction count (2-48) and removal, per-cell weights on
# state / on input / inside the map, variance normalisation, and real corpus
# input. The engine emits ~1 structured event per 300 steps under every one. Four
# candidate redefinitions were each topped by a control -- hidden-state by
# SCRAMBLE 6.8, output-direction by SCRAMBLE 13.2, permutation-invariant profile
# by DECOUPLED 5.8, variance normalisation by SCRAMBLE 5.7.
#
# HIVEMIND had never executed before this session (it crashed in 0.3-1.6s for all
# 11 engines). Made to run, with 3 trials and an unconnected control, connection
# changes Phi by -4% to +1% at every repulsion strength on both engine designs,
# against a required +10%. Moving it to the emitted channel makes DEAD win by 10x.
#
# Retiring them costs the gate nothing, which is the point: measured at 32 cells,
# all six negative controls score 0/5 exactly as they scored 0/7, while the real
# engine goes from 5/7 to 5/5. A condition that certifies nothing is not
# protecting anything. Evidence in docs/consciousness-gate-audit.md.
_RETIRED_TESTS = [
    ("SPONTANEOUS_SPEECH", _verify_spontaneous_speech,  "8-faction debate -> consensus utterances"),
    ("HIVEMIND",           _verify_hivemind,            "Multi-connect: Phi↑ CE↓, independent after disconnect"),
]

VERIFICATION_TESTS = [(name, _with_axes(fn), desc)
                      for name, fn, desc in VERIFICATION_TESTS]


# Seeds the gate runs every condition under. A condition passes only if it
# passes on ALL of them -- see run_verify for why one seed was not enough.
VERIFY_SEEDS = (42, 43, 44, 45, 46)


def _run_controls(cells: int, dim: int, hidden: int):
    """Run the negative controls through every condition, in the same run.

    The audit that opened this session found SCRAMBLE — the real engine with cell
    identity destroyed every step — scoring exactly what the real engine scored.
    The controls that caught it lived in a separate file the gate never called, so
    the gate could go back to passing a corpse and nothing here would notice.

    A condition that a corpse passes carries no information about the engine that
    also passes it. So the controls do not merely get reported alongside: any
    condition a control clears is VOID for that run, and no engine may bank it.
    This is CLAUDE.md's own rule — that baselines come from the population's own
    null rather than from a constant — applied to the gate itself.

    A control that RAISES is not a control that was rejected. It used to be
    scored `passed = False` -- "a crash is not a pass" -- which reads as the
    condition having correctly turned the corpse away, when in fact the
    condition never ran on it. That is the same conflation the engine loop had,
    and it points the wrong way here: a crashing corpse makes the gate look
    MORE discriminating than it is. An unrunnable control leaves the condition
    unvalidated, which is not something an engine may bank.

    Returns ({test_name: [labels that passed]}, {test_name: [labels that
    raised]}).
    """
    from bench_verify_audit import CONTROLS, factory_for   # lazy: it imports us

    voided = {name: [] for name, _, _ in VERIFICATION_TESTS}
    unrun = {name: [] for name, _, _ in VERIFICATION_TESTS}
    for label, cls, _desc in CONTROLS[1:]:              # [0] is the real engine
        for test_name, test_fn, _d in VERIFICATION_TESTS:
            # A control that clears the bar on ANY seed voids the condition.
            # Engines must pass every seed; corpses need only pass one to prove
            # the condition cannot tell them apart. Asymmetric on purpose --
            # both directions favour withholding the verdict.
            for sd in VERIFY_SEEDS:
                torch.manual_seed(sd)
                try:
                    passed, _detail = test_fn(
                        lambda c, d, h: factory_for(cls, c, d, h),
                        cells, dim, hidden)
                except Exception as e:
                    unrun[test_name].append(
                        f"{label}(seed {sd}: {type(e).__name__})")
                    break                    # unmeasured, not rejected
                if passed:
                    voided[test_name].append(f"{label}(seed {sd})")
                    break
    return voided, unrun


def run_verify(cells: int, dim: int, hidden: int, with_controls: bool = True):
    """Run every consciousness condition across every engine AND every control."""
    n_cond, n_eng = len(VERIFICATION_TESTS), len(ENGINE_REGISTRY)
    n_ctrl = 0
    if with_controls:
        from bench_verify_audit import CONTROLS
        n_ctrl = len(CONTROLS) - 1                      # [0] is the real engine
    total = n_cond * (n_eng + n_ctrl)

    print("=" * 80)
    print("  MODE: --verify  (Consciousness Verification)")
    # Counted, not asserted. This line read "7 conditions x 4 engines = 28
    # tests" while the truth was 5 x 11 -- two conditions had been retired and
    # seven engines added without it moving, so the header under-reported the
    # run by 3x and made a 45-minute execution look like a hang.
    print(f"  {n_cond} conditions x {n_eng} engines"
          + (f" + {n_ctrl} controls" if n_ctrl else "")
          + f" = {total} tests")
    if _RETIRED_TESTS:
        print(f"  retired: {', '.join(n for n, _, _ in _RETIRED_TESTS)} "
              f"(see docs/consciousness-gate-audit.md)")
    print(f"  cells={cells}  dim={dim}  hidden={hidden}")
    print("=" * 80)

    voided = {name: [] for name, _, _ in VERIFICATION_TESTS}
    if with_controls:
        print("\n  Running negative controls first — a condition a corpse passes")
        print("  is void for this run, and no engine may bank it.")
        voided, unrun = _run_controls(cells, dim, hidden)
        live = [n for n, v in voided.items() if not v and not unrun[n]]
        print(f"  {len(live)}/{len(voided)} conditions survived the controls.")
        for name, by in voided.items():
            if by:
                print(f"    VOID  {name:<22s} passed by: {', '.join(by)}")
        for name, by in unrun.items():
            if by:
                # Unvalidated, not survived. The condition never ran against
                # this corpse, so nothing was learned about whether it can tell
                # the two apart -- and an engine may not bank it either way.
                print(f"    UNVALIDATED  {name:<15s} raised on: {', '.join(by)}")
                voided[name] = voided[name] + [f"UNRUN:{b}" for b in by]

    engine_names = list(ENGINE_REGISTRY.keys())
    results = {}  # (engine_name, test_name) -> (passed, detail)
    errored = {}  # (engine_name, test_name) -> harness raised, so NOT measured
    seed_counts = {}  # (engine_name, test_name) -> seeds cleared, of VERIFY_SEEDS

    for eng_name in engine_names:
        print(f"\n  {'~' * 70}")
        print(f"  Engine: {eng_name}")
        print(f"  {'~' * 70}")

        factory = ENGINE_REGISTRY[eng_name]
        for test_name, test_fn, test_desc in VERIFICATION_TESTS:
            t0 = time.time()
            # Every seed must pass. The gate judged on seed 42 alone, and that
            # is where its verdicts kept breaking: five engines it called
            # DEPLOYABLE dropped to 4/5 once the seed moved, and so did
            # PairField (5/5 5/5 4/5 5/5 4/5 across 42-46 at this scale). A
            # deployment verdict that depends on which seed was drawn is a coin
            # flip wearing a number. This is the repo's own rule -- "1개라도
            # 실패 시 배포 금지" -- applied to the seed axis.
            # A raised exception is NOT a failed condition. It used to be
            # folded into `ok = False`, so a harness bug read as every engine
            # scoring 0/5 -- which is precisely what a catastrophic-but-real
            # regression looks like, and the controls make it MORE convincing
            # rather than less, because a corpse scoring 0/5 is expected. This
            # session shipped a NameError into _three_axes and a teammate's
            # 250-cell grid came back a perfect field of zeros; it was caught
            # only because that teammate knew NarrativeEngine to be 5/5 on
            # PERSISTENCE at that scale from an hour earlier. So an exception is
            # its own state: it says the condition was not measured. That
            # includes an engine's own divergence guard, whose message already
            # says the right thing -- scores from a diverged pair say nothing.
            per_seed, errors = [], []
            for sd in VERIFY_SEEDS:
                torch.manual_seed(sd)
                try:
                    ok, det = test_fn(factory, cells, dim, hidden)
                except Exception as e:
                    ok, det = False, f"{type(e).__name__}: {e}"
                    errors.append((sd, det))
                per_seed.append((sd, ok, det))
            passed = all(ok for _sd, ok, _d in per_seed) and not errors
            failed_seeds = [sd for sd, ok, _d in per_seed if not ok]
            detail = next((d for _sd, ok, d in per_seed if not ok),
                          per_seed[0][2])
            if failed_seeds and len(failed_seeds) < len(VERIFY_SEEDS):
                detail = (f"UNSTABLE — failed on seed(s) {failed_seeds} of "
                          f"{VERIFY_SEEDS}; {detail}")
            elapsed = time.time() - t0

            if errors:
                detail = (f"NOT MEASURED — harness raised on seed(s) "
                          f"{[s for s, _ in errors]}: {errors[0][1]}")
            mark = "ERR!" if errors else ("PASS" if passed else "FAIL")
            results[(eng_name, test_name)] = (passed, detail)
            errored[(eng_name, test_name)] = bool(errors)
            # How many seeds actually cleared, kept alongside the boolean. The
            # all-seeds rule turns a per-seed rate into one bit and the bit is
            # all the summary showed, so a condition engines clear 80-87% of
            # the time and corpses clear 0% of the time was indistinguishable
            # from one nothing clears. Measured at 32 cells: engines pass 86.0%
            # of engine-condition-seed cells, and the deployable count is 1
            # under all-5, 5 under 4-of-5, 11 under a 3-of-5 majority -- decided
            # almost entirely by PERSISTENCE, the only column that varies.
            seed_counts[(eng_name, test_name)] = sum(
                1 for _s, ok, _d in per_seed if ok)
            print(f"    [{mark}] {test_name:<22s} ({elapsed:.1f}s) -- {detail}")

    # ── Summary table ──
    print(f"\n{'=' * 80}")
    print("  VERIFICATION SUMMARY")
    print(f"{'=' * 80}")

    # Header
    test_names = [t[0] for t in VERIFICATION_TESTS]
    header = f"  {'Engine':<18s}"
    for tn in test_names:
        header += f" | {tn[:10]:^10s}"
    header += " | TOTAL"
    print(header)
    print(f"  {'-' * 18}" + ("-+-" + "-" * 10) * len(test_names) + "-+------")

    total_pass = 0
    total_tests = 0
    live_names = [tn for tn in test_names if not voided[tn]]

    for eng_name in engine_names:
        row = f"  {eng_name:<18s}"
        eng_pass = 0
        for tn in test_names:
            passed, _ = results[(eng_name, tn)]
            if voided[tn]:
                row += f" | {'  VOID  ':^10s}"     # a control cleared it too
                continue
            # Show the seed count, not just the bit. PASS and FAIL alone hid
            # that the all-seeds rule is what produces most FAILs: at 32 cells
            # the deployable count is 1 under all-5, 5 under 4-of-5 and 11
            # under a 3-of-5 majority, and PERSISTENCE is the only column that
            # varies. `FAIL 3/5` and `FAIL 0/5` are different findings and the
            # table printed them identically.
            n_ok = seed_counts.get((eng_name, tn), 0)
            mark = f"PASS {n_ok}/{len(VERIFY_SEEDS)}" if passed \
                else f"FAIL {n_ok}/{len(VERIFY_SEEDS)}"
            row += f" | {mark:^10s}"
            if passed:
                eng_pass += 1
                total_pass += 1
            total_tests += 1
        row += f" | {eng_pass}/{len(live_names)}"
        print(row)

    if total_tests:
        print(f"\n  Overall: {total_pass}/{total_tests} passed "
              f"({total_pass/total_tests*100:.0f}%)  — voided conditions excluded")
    else:
        print("\n  Overall: nothing scoreable — every condition was voided.")

    # ── Deployment verdict ──
    # CLAUDE.md: "1개라도 실패 시 배포 금지". A voided condition is not a pass
    # with a caveat; it is the gate reporting that it cannot tell this engine
    # from a corpse on that axis, which is a stronger reason to withhold.
    print(f"\n  {'=' * 76}")
    if voided and any(voided.values()):
        print(f"  GATE VOID — {sum(1 for v in voided.values() if v)} condition(s) "
              f"were passed by a negative control and certify nothing.")
        print("  Deployment blocked regardless of engine scores. "
              "See docs/consciousness-gate-audit.md.")
    else:
        for eng_name in engine_names:
            failed = [tn for tn in live_names if not results[(eng_name, tn)][0]]
            verdict = "DEPLOYABLE" if not failed else f"BLOCKED ({', '.join(failed)})"
            print(f"  {eng_name:<18s} {verdict}")

    # Per-condition summary
    print(f"\n  Per-condition pass rate:")
    for test_name, _, test_desc in VERIFICATION_TESTS:
        passes = sum(1 for en in engine_names if results[(en, test_name)][0])
        bar = "#" * (passes * 10) + "." * ((len(engine_names) - passes) * 10)
        print(f"    {test_name:<22s} {passes}/{len(engine_names)}  |{bar}|  {test_desc}")

    # Verdict
    print(f"\n  {'=' * 70}")
    if total_pass == total_tests:
        print("  VERDICT: ALL CONSCIOUSNESS CONDITIONS VERIFIED")
    elif total_pass >= total_tests * 0.75:
        print(f"  VERDICT: MOSTLY VERIFIED ({total_pass}/{total_tests})")
    else:
        print(f"  VERDICT: NEEDS WORK ({total_pass}/{total_tests})")
    print(f"  {'=' * 70}")

    # A gate that prints "Deployment blocked" and then exits 0 does not block
    # anything. `--verify` returned success on every path, including GATE VOID
    # and including a run where the harness raised on every condition, so
    # nothing calling it in a script could tell a clean run from a corpse or
    # from a crash. Deploy on a nonzero exit is the only form in which the
    # printed verdict is enforceable.
    blocked = bool(voided and any(voided.values()))
    unmeasured = sorted({en for (en, _tn), err in errored.items() if err})
    if unmeasured:
        print(f"\n  NOT MEASURED — the harness raised on: "
              f"{', '.join(unmeasured)}. No verdict for those engines.")
    return results, (2 if blocked else (3 if unmeasured else 0))


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def run_philosophy(cells: int, steps: int, dim: int, hidden: int):
    """Run all philosophical consciousness engines and compare."""
    print("=" * 80)
    print("  MODE: --philosophy  (Philosophical Consciousness Benchmark)")
    print(f"  7 engines × {steps} steps × {cells} cells")
    print(f"  dim={dim}  hidden={hidden}")
    print("=" * 80)

    results = []

    for name, factory in PHILOSOPHY_ENGINES.items():
        print(f"\n  {'─' * 70}")
        print(f"  Running: {name}")
        print(f"  {'─' * 70}")

        torch.manual_seed(42)
        engine = factory(cells, dim, hidden)
        optimizer = torch.optim.Adam(engine.parameters_for_training(), lr=1e-3)
        corpus = torch.randn(steps + 1, dim)

        ce_history = []
        iit_history = []
        proxy_history = []
        t0 = time.time()

        for step in range(steps):
            x = corpus[step:step+1]
            target = corpus[step+1:step+2]

            output, tension = engine.process(x)
            logits = engine.output_head(output)
            ce_loss = F.mse_loss(logits, target)

            optimizer.zero_grad()
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters_for_training(), max_norm=1.0)
            optimizer.step()

            ce_history.append(ce_loss.item())

            if step % 50 == 0 or step == steps - 1:
                hiddens = engine.get_hiddens()
                p_iit, p_proxy = measure_dual_phi(hiddens, min(8, cells // 2))
                iit_history.append(p_iit)
                proxy_history.append(p_proxy)

                if step % 100 == 0 or step == steps - 1:
                    print(f"    step {step:>5d}/{steps}  CE={ce_loss.item():.4f}  "
                          f"Φ(IIT)={p_iit:.4f}  Φ(proxy)={p_proxy:.2f}  "
                          f"tension={tension:.4f}")

        elapsed = time.time() - t0
        hiddens = engine.get_hiddens()
        final_iit, final_proxy = measure_dual_phi(hiddens, min(8, cells // 2))

        extra = {
            'iit_history': iit_history,
            'proxy_history': proxy_history,
        }
        if hasattr(engine, 'get_extra_metrics'):
            extra.update(engine.get_extra_metrics())

        result = BenchResult(
            name=name,
            phi_iit=final_iit,
            phi_proxy=final_proxy,
            ce_start=ce_history[0],
            ce_end=ce_history[-1],
            cells=cells,
            steps=steps,
            time_sec=elapsed,
            extra=extra,
        )
        results.append(result)
        print(f"    → Φ(IIT)={final_iit:.4f}  Φ(proxy)={final_proxy:.2f}  "
              f"CE={ce_history[0]:.4f}→{ce_history[-1]:.4f}  [{elapsed:.1f}s]")

    # ── Comparison Table ──
    print(f"\n{'=' * 80}")
    print("  PHILOSOPHICAL CONSCIOUSNESS BENCHMARK — RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'Engine':<22s} | {'Φ(IIT)':>8s} | {'Φ(proxy)':>10s} | "
          f"{'CE start':>8s} | {'CE end':>8s} | {'Time':>6s}")
    print(f"  {'-' * 22}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}")

    baseline_iit = results[0].phi_iit if results else 1.0
    baseline_proxy = results[0].phi_proxy if results else 1.0
    baseline_ce = results[0].ce_end if results else 1.0

    for r in results:
        iit_delta = ((r.phi_iit / max(baseline_iit, 1e-8)) - 1) * 100
        proxy_delta = ((r.phi_proxy / max(baseline_proxy, 1e-8)) - 1) * 100
        print(f"  {r.name:<22s} | {r.phi_iit:>8.4f} | {r.phi_proxy:>10.2f} | "
              f"{r.ce_start:>8.4f} | {r.ce_end:>8.4f} | {r.time_sec:>5.1f}s")
        if r.name != 'baseline':
            ce_delta = ((r.ce_end / max(baseline_ce, 1e-8)) - 1) * 100
            print(f"  {'':22s} | {iit_delta:>+7.1f}% | {proxy_delta:>+9.1f}% | "
                  f"{'':>8s} | {ce_delta:>+7.1f}% |")

    # ── Extra metrics ──
    print(f"\n  {'─' * 70}")
    print("  ENGINE-SPECIFIC METRICS")
    print(f"  {'─' * 70}")
    for r in results:
        if r.name == 'baseline':
            continue
        extras = {k: v for k, v in r.extra.items()
                  if k not in ('iit_history', 'proxy_history',
                               'desire_distances', 'urgency_history')}
        if extras:
            print(f"  {r.name}: {extras}")

    # ── Bar chart ──
    print(f"\n  Φ(IIT) Comparison:")
    max_iit = max(r.phi_iit for r in results) if results else 1
    for r in results:
        bar_len = int(r.phi_iit / max(max_iit, 1e-8) * 40)
        delta = ((r.phi_iit / max(baseline_iit, 1e-8)) - 1) * 100
        delta_str = f"{delta:+.1f}%" if r.name != 'baseline' else "baseline"
        print(f"  {r.name:<22s} {'█' * bar_len} {r.phi_iit:.4f} ({delta_str})")

    print(f"\n  Φ(proxy) Comparison:")
    max_proxy = max(r.phi_proxy for r in results) if results else 1
    for r in results:
        bar_len = int(r.phi_proxy / max(max_proxy, 1e-8) * 40)
        delta = ((r.phi_proxy / max(baseline_proxy, 1e-8)) - 1) * 100
        delta_str = f"{delta:+.1f}%" if r.name != 'baseline' else "baseline"
        print(f"  {r.name:<22s} {'█' * bar_len} {r.phi_proxy:.2f} ({delta_str})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="bench_v2 — Dual-Phi Benchmarking (IIT + proxy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bench_v2.py --phi-only                    # Phi at different cell counts
  python bench_v2.py --training                     # Default training (baseline)
  python bench_v2.py --strategy v7                  # Test v7 strategy
  python bench_v2.py --compare                      # All strategies comparison
  python bench_v2.py --compare --cells 128 --steps 200
  python bench_v2.py --phi-only --cells-list 8,16,32,64,128,256,512
  python bench_v2.py --verify                        # 6 conditions x 4 engines
  python bench_v2.py --verify --cells 64             # Lighter verification

Key insight: Phi(IIT) and Phi(proxy) are COMPLETELY DIFFERENT metrics.
  Phi(IIT)   = ~0.2-1.8  (PhiCalculator, real IIT approximation)
  Phi(proxy) = 0-1000+   (variance-based, scales with cells)
  Previous "Phi=1142" records were proxy values, NOT IIT.
""")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phi-only", action="store_true",
                      help="Test phi measurement at different cell counts (no CE)")
    mode.add_argument("--training", action="store_true",
                      help="Simulate real training with CE")
    mode.add_argument("--strategy", type=str, choices=STRATEGIES,
                      help="Test a specific strategy")
    mode.add_argument("--compare", action="store_true",
                      help="Run all strategies and compare")
    mode.add_argument("--verify", action="store_true",
                      help="Consciousness verification: 6 conditions x 4 engines")
    parser.add_argument("--no-controls", action="store_true",
                        help="Skip the negative controls. The gate then cannot "
                             "tell an engine from a corpse — for debugging only.")
    mode.add_argument("--philosophy", action="store_true",
                      help="Philosophical consciousness benchmark: "
                           "Desire, Narrative, Alterity, Finitude, Questioning, Sein")

    parser.add_argument("--cells", type=int, default=256,
                        help="Number of cells (default: 256)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of steps (default: 500)")
    parser.add_argument("--dim", type=int, default=64,
                        help="Input/output dimension (default: 64)")
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden dimension (default: 128)")
    parser.add_argument("--cells-list", type=str, default=None,
                        help="Comma-separated cell counts for --phi-only "
                             "(default: 4,8,16,32,64,128,256)")

    args = parser.parse_args()

    print()
    print("  ================================================================")
    print("   bench_v2 -- Dual-Phi Benchmark (IIT + proxy)")
    print("  ================================================================")
    print(f"   Phi(IIT):   PhiCalculator(n_bins=16)  ~0.2-1.8 range")
    print(f"   Phi(proxy): global_var - mean(faction_var)  scales with cells")
    print(f"   These are DIFFERENT metrics. Always check both.")
    print("  ================================================================")
    print()

    if args.phi_only:
        if args.cells_list:
            cells_list = [int(c.strip()) for c in args.cells_list.split(",")]
        else:
            cells_list = [4, 8, 16, 32, 64, 128, 256]
        run_phi_only(cells_list, args.steps, args.dim, args.hidden)

    elif args.training:
        result, ce_hist, iit_hist, proxy_hist = run_training(
            args.cells, args.steps, args.dim, args.hidden,
            strategy="baseline"
        )
        print(f"\n{'=' * 72}")
        print(f"  RESULT")
        print(f"{'=' * 72}")
        print(result.summary())
        if len(ce_hist) >= 10:
            ascii_graph(ce_hist, "CE Loss")
        if len(iit_hist) >= 3:
            ascii_dual_graph(iit_hist, proxy_hist)

    elif args.strategy:
        run_strategy(args.strategy, args.cells, args.steps, args.dim, args.hidden)

    elif args.compare:
        run_compare(args.cells, args.steps, args.dim, args.hidden)

    elif args.verify:
        _results, code = run_verify(args.cells, args.dim, args.hidden,
                                    with_controls=not args.no_controls)
        if code:
            # 2 = a control cleared a condition (gate void). 3 = the harness
            # raised, so nothing was measured. Both must be distinguishable
            # from 0 by anything that scripts a deployment.
            print()
            raise SystemExit(code)

    elif args.philosophy:
        run_philosophy(args.cells, args.steps, args.dim, args.hidden)

    print()


if __name__ == "__main__":
    main()
