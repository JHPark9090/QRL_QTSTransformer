"""
Quantum Time-Series Transformer for Simple RL Environments -- v6
=================================================================
v6 = v1 base architecture + ANO/DiffQAS ablation + anti-forgetting

v1 base (validated: CartPole peak 500/500):
  - Sigmoid angle scaling [0, 1] (no 2pi centering)
  - MSELoss (not SmoothL1Loss)
  - No positional encoding, no gradient clipping

v5 innovations carried forward:
  1. ANO (Adaptive Non-Local Observables): Learnable k-local Hermitian measurements
     replace fixed PauliX/Y/Z (Lin et al. 2025)
  2. DiffQAS (Differentiable Quantum Architecture Search): Gate-level search using
     parametric Rot/CRot gates (Sun et al. 2023)

Anti-forgetting mechanisms (new in v6):
  - Best-model checkpointing: saves model at peak AvgR separately
  - Early stopping: halts when AvgR exceeds threshold for N consecutive episodes
  - LR reduction: halves LR when AvgR plateaus (patience-based)
  - Slow target sync: sync_every=500 to reduce policy churn

Two-phase training (when DiffQAS enabled):
  Phase 1 (search): VQC + ANO + arch params train jointly for --search-episodes
  Phase 2 (production): Arch params discretized and frozen, VQC + ANO continue

Ablation flags:
  --no-ano --no-dqas  : v1 baseline (fixed circuit + fixed Paulis)
  --no-dqas           : ANO only (fixed circuit + learnable Hermitians)
  --no-ano            : DiffQAS only (architecture search + fixed Paulis)
  (default)           : Full v6 (architecture search + learnable Hermitians)

Supported Environments:
- CartPole-v1: 4-dimensional continuous state
- FrozenLake-v1: 16-dimensional one-hot encoded state
- MountainCar-v0: 2-dimensional continuous state
- Acrobot-v1: 6-dimensional continuous state
"""

import math
import torch
from torch import nn
import numpy as np
from pathlib import Path
from collections import deque
import random, time, datetime, os, signal, sys
import argparse
from math import log2

import pennylane as qml
import matplotlib.pyplot as plt
import gymnasium as gym


# ================================================================================
# ARGUMENT PARSER
# ================================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description='Quantum Transformer v6 (v1 base + ANO/DiffQAS + anti-forgetting)'
    )

    # Environment selection
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        choices=["CartPole-v1", "FrozenLake-v1",
                                "MountainCar-v0", "Acrobot-v1"])

    # Quantum parameters
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--n-timesteps", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Ablation flags
    parser.add_argument("--no-ano", action="store_true",
                        help="Disable ANO: use fixed PauliX/Y/Z measurements")
    parser.add_argument("--no-dqas", action="store_true",
                        help="Disable DiffQAS: use fixed RY-CRX sim14 circuit")

    # ANO parameters
    parser.add_argument("--ano-k-local", type=int, default=2)
    parser.add_argument("--ano-lr", type=float, default=0.01)

    # DiffQAS parameters
    parser.add_argument("--arch-lr", type=float, default=0.001)
    parser.add_argument("--search-episodes", type=int, default=50)

    # RL parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--exploration-rate-start', type=float, default=1.0)
    parser.add_argument('--exploration-rate-decay', type=float, default=0.995)
    parser.add_argument('--exploration-rate-min', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learn-step', type=int, default=1)
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--memory-size', type=int, default=10000)

    # Anti-forgetting parameters
    parser.add_argument('--early-stop-reward', type=float, default=490.0,
                        help="AvgR threshold for early stopping (default: 490 for CartPole)")
    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help="Consecutive episodes above threshold to trigger early stop")
    parser.add_argument('--lr-reduce-patience', type=int, default=50,
                        help="Episodes without AvgR improvement before halving LR")
    parser.add_argument('--lr-reduce-factor', type=float, default=0.5,
                        help="Factor to multiply LR by when reducing")
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help="Minimum learning rate after reductions")
    parser.add_argument('--sync-every', type=int, default=500,
                        help="Sync target network every N learn steps (default: 500)")

    # Training parameters
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log-index", type=int, default=1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=50)

    return parser.parse_args()

args = get_args()


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================
def set_global_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seeds set to: {seed_value}")


# ================================================================================
# CHECKPOINT DIRECTORIES
# ================================================================================
env_name_clean = args.env.replace("-", "").replace("v", "")
use_ano = not args.no_ano
use_dqas = not args.no_dqas
if use_ano and use_dqas:
    _ablation_tag = "full"
elif use_ano:
    _ablation_tag = "ANOonly"
elif use_dqas:
    _ablation_tag = "DQASonly"
else:
    _ablation_tag = "baseline"
RUN_ID = (f"QTransformerV6_{env_name_clean}_Q{args.n_qubits}_L{args.n_layers}"
          f"_D{args.degree}_K{args.ano_k_local}_{_ablation_tag}_Run{args.log_index}")
if args.save_dir:
    CHECKPOINT_BASE_DIR = Path(args.save_dir)
else:
    CHECKPOINT_BASE_DIR = Path("SimpleRLCheckpoints")
SAVE_DIR = CHECKPOINT_BASE_DIR / RUN_ID

SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE_PATH = SAVE_DIR / "latest_checkpoint.chkpt"
BEST_MODEL_PATH = SAVE_DIR / "best_model.chkpt"


# ================================================================================
# ENVIRONMENT SETUP
# ================================================================================
env = gym.make(args.env)


# ================================================================================
# STATE PREPROCESSING
# ================================================================================
class StateProcessor:
    def __init__(self, env, n_timesteps):
        self.env_name = args.env
        self.n_timesteps = n_timesteps

        if hasattr(env.observation_space, 'n'):
            self.state_dim = env.observation_space.n
            self.discrete_state = True
        else:
            self.state_dim = env.observation_space.shape[0]
            self.discrete_state = False

        self.state_history = deque(maxlen=n_timesteps)
        self.reset()

    def reset(self):
        self.state_history.clear()
        zero_state = np.zeros(self.state_dim)
        for _ in range(self.n_timesteps):
            self.state_history.append(zero_state)

    def process_state(self, state):
        if self.discrete_state:
            one_hot = np.zeros(self.state_dim)
            one_hot[state] = 1.0
            return one_hot
        else:
            return np.array(state, dtype=np.float32)

    def add_state(self, state):
        processed = self.process_state(state)
        self.state_history.append(processed)
        history_array = np.array(list(self.state_history))
        return torch.tensor(history_array, dtype=torch.float32)


# ================================================================================
# ANO HELPER -- Adaptive Non-Local Observables (Lin et al. 2025)
# ================================================================================
def create_Hermitian(N, A, B, D):
    h = torch.zeros((N, N), dtype=torch.complex128)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()
        count += i
    H = h.clone() + h.clone().conj().T
    return H


# ================================================================================
# FIXED SIM14 CIRCUIT (v1 baseline -- used when --no-dqas)
# ================================================================================
def sim14_circuit(params, wires, layers=1):
    is_batched = params.ndim == 2
    param_idx = 0

    for _ in range(layers):
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


# ================================================================================
# DIFFQAS CIRCUIT COMPONENTS -- Gate-Level Parametric Search
# ================================================================================
SINGLE_GATES = {'RY': qml.RY, 'RX': qml.RX, 'RZ': qml.RZ}
CTRL_GATES = {'CRX': qml.CRX, 'CRY': qml.CRY, 'CRZ': qml.CRZ}

_SINGLE_REFS = {
    'RY': (0.0, 0.0),
    'RX': (1.0, 0.0),
    'RZ': (0.0, 1.0),
}
_CTRL_REFS = {
    'CRX': (0.0, 0.0),
    'CRY': (1.0, 0.0),
    'CRZ': (0.0, 1.0),
}


def _softmax_weights(arch_vec, refs):
    dists = torch.stack([
        (arch_vec[0] - r0) ** 2 + (arch_vec[1] - r1) ** 2
        for (r0, r1) in refs.values()
    ])
    return torch.softmax(-dists, dim=0)


def dqas_sim14_circuit(params, arch_params, wires, layers=1):
    is_batched = params.ndim == 2
    param_idx = 0

    for layer_idx in range(layers):
        # Slot 0: single-qubit (parametric)
        w0 = _softmax_weights(arch_params[layer_idx, 0], _SINGLE_REFS)
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            for gate_idx, gate_fn in enumerate(SINGLE_GATES.values()):
                qml.RY(w0[gate_idx] * angle, wires=i)
            param_idx += 1

        # Slot 1: controlled (reverse order)
        w1 = _softmax_weights(arch_params[layer_idx, 1], _CTRL_REFS)
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            for gate_idx, gate_fn in enumerate(CTRL_GATES.values()):
                qml.CRX(w1[gate_idx] * angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        # Slot 2: single-qubit
        w2 = _softmax_weights(arch_params[layer_idx, 2], _SINGLE_REFS)
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            for gate_idx, gate_fn in enumerate(SINGLE_GATES.values()):
                qml.RY(w2[gate_idx] * angle, wires=i)
            param_idx += 1

        # Slot 3: controlled (counter direction)
        w3 = _softmax_weights(arch_params[layer_idx, 3], _CTRL_REFS)
        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            for gate_idx, gate_fn in enumerate(CTRL_GATES.values()):
                qml.CRX(w3[gate_idx] * angle, wires=[i, (i - 1) % wires])
            param_idx += 1


def classify_single_qubit_gate(phi, omega):
    dists = {name: (phi - r0) ** 2 + (omega - r1) ** 2
             for name, (r0, r1) in _SINGLE_REFS.items()}
    return min(dists, key=dists.get)


def classify_controlled_gate(phi, omega):
    dists = {name: (phi - r0) ** 2 + (omega - r1) ** 2
             for name, (r0, r1) in _CTRL_REFS.items()}
    return min(dists, key=dists.get)


def discrete_sim14_circuit(params, gate_config, wires, layers=1):
    is_batched = params.ndim == 2
    param_idx = 0

    for layer_idx in range(layers):
        # Slot 0
        gate_name = gate_config[(layer_idx, 0)]
        gate_fn = SINGLE_GATES[gate_name]
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            gate_fn(angle, wires=i)
            param_idx += 1

        # Slot 1
        gate_name = gate_config[(layer_idx, 1)]
        gate_fn = CTRL_GATES[gate_name]
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            gate_fn(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        # Slot 2
        gate_name = gate_config[(layer_idx, 2)]
        gate_fn = SINGLE_GATES[gate_name]
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            gate_fn(angle, wires=i)
            param_idx += 1

        # Slot 3
        gate_name = gate_config[(layer_idx, 3)]
        gate_fn = CTRL_GATES[gate_name]
        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            gate_fn(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


# ================================================================================
# QSVT POLYNOMIAL STATE PREPARATION
# ================================================================================
def evaluate_polynomial_state_pl(base_states, timestep_params, qnode,
                                  n_qubits, mix_coeffs, poly_coeffs):
    bsz, n_timesteps, n_rots = timestep_params.shape
    degree = poly_coeffs.shape[0] - 1

    result = torch.zeros(bsz, 2 ** n_qubits, dtype=torch.complex64,
                         device=timestep_params.device)

    for d in range(degree + 1):
        mixed_state = torch.zeros(bsz, 2 ** n_qubits, dtype=torch.complex64,
                                   device=timestep_params.device)
        for t in range(n_timesteps):
            state_out = qnode(base_states, timestep_params[:, t, :])
            mixed_state += mix_coeffs[:, t:t+1] * state_out
        result += poly_coeffs[d] * mixed_state
        if d < degree:
            base_states = mixed_state / (
                torch.linalg.vector_norm(mixed_state, dim=1, keepdim=True) + 1e-9)

    return result


def evaluate_polynomial_state_pl_dqas(base_states, timestep_params, qnode,
                                       n_qubits, mix_coeffs, poly_coeffs,
                                       arch_params):
    bsz, n_timesteps, n_rots = timestep_params.shape
    degree = poly_coeffs.shape[0] - 1

    result = torch.zeros(bsz, 2 ** n_qubits, dtype=torch.complex64,
                         device=timestep_params.device)

    for d in range(degree + 1):
        mixed_state = torch.zeros(bsz, 2 ** n_qubits, dtype=torch.complex64,
                                   device=timestep_params.device)
        for t in range(n_timesteps):
            state_out = qnode(base_states, timestep_params[:, t, :], arch_params)
            mixed_state += mix_coeffs[:, t:t+1] * state_out
        result += poly_coeffs[d] * mixed_state
        if d < degree:
            base_states = mixed_state / (
                torch.linalg.vector_norm(mixed_state, dim=1, keepdim=True) + 1e-9)

    return result


# ================================================================================
# QUANTUM TRANSFORMER MODEL (v1 base + ANO/DiffQAS)
# ================================================================================
class QuantumTSTransformerRL_v6(nn.Module):
    """v1 base architecture + ANO/DiffQAS ablation support.

    Key differences from v5:
      - No positional encoding (v1 style)
      - Sigmoid [0,1] angle scaling (v1 style, not 2pi)
      - MSELoss used by agent (not SmoothL1Loss)
    """
    def __init__(self, state_dim: int, n_qubits: int, n_timesteps: int,
                 degree: int, n_ansatz_layers: int, output_dim: int,
                 dropout: float, device, ano_k_local: int = 2,
                 use_ano: bool = True, use_dqas: bool = True):
        super().__init__()

        self.state_dim = state_dim
        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.device = device
        self.ano_k_local = ano_k_local
        self.use_ano = use_ano
        self.use_dqas = use_dqas

        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1

        # Phase tracking
        self.phase = 1 if use_dqas else 2
        self.gate_config_qsvt = None
        self.gate_config_qff = None

        # Classical layers (v1 style: no PE, sigmoid scaling)
        self.feature_projection = nn.Linear(state_dim, self.n_rots)
        self.dropout = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()

        # --- ANO parameters ---
        if use_ano:
            self.n_windows = n_qubits
            self.N_ano = 2 ** ano_k_local

            n_off_diag = self.N_ano * (self.N_ano - 1) // 2

            self.A = nn.ParameterList([
                nn.Parameter(torch.empty(n_off_diag)) for _ in range(self.n_windows)
            ])
            self.B = nn.ParameterList([
                nn.Parameter(torch.empty(n_off_diag)) for _ in range(self.n_windows)
            ])
            self.D = nn.ParameterList([
                nn.Parameter(torch.empty(self.N_ano)) for _ in range(self.n_windows)
            ])

            for w in range(self.n_windows):
                nn.init.normal_(self.A[w], std=2.0)
                nn.init.normal_(self.B[w], std=2.0)
                nn.init.normal_(self.D[w], std=2.0)

            qff_output_dim = self.n_windows
        else:
            self.n_windows = None
            self.N_ano = None
            qff_output_dim = 3 * n_qubits

        # --- Architecture search parameters ---
        if use_dqas:
            self.arch_params_qsvt = nn.Parameter(torch.zeros(n_ansatz_layers, 4, 2))
            self.arch_params_qff = nn.Parameter(torch.zeros(1, 4, 2))

        # Output head (v1 style)
        self.output_ff = nn.Sequential(
            nn.Linear(qff_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        # Trainable quantum parameters
        self.n_poly_coeffs = self.degree + 1
        self.poly_coeffs = nn.Parameter(torch.rand(self.n_poly_coeffs))
        self.mix_coeffs = nn.Parameter(
            torch.rand(self.n_timesteps, dtype=torch.complex64)
        )
        self.qff_params = nn.Parameter(torch.rand(self.qff_n_rots))

        # Build QNodes
        self._build_qnodes()

    def _build_qnodes(self):
        _n_qubits = self.n_qubits
        _n_ansatz_layers = self.n_ansatz_layers
        _use_ano = self.use_ano
        _use_dqas = self.use_dqas
        _n_windows = self.n_windows
        _ano_k_local = self.ano_k_local

        self.dev = qml.device("default.qubit", wires=_n_qubits)

        # ---- QSVT state QNode ----
        if _use_dqas and self.phase == 1:
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _timestep_state_qnode(initial_state, params, arch_params):
                qml.StatePrep(initial_state, wires=range(_n_qubits))
                dqas_sim14_circuit(params, arch_params, wires=_n_qubits,
                                   layers=_n_ansatz_layers)
                return qml.state()
            self.timestep_state_qnode = _timestep_state_qnode

        elif _use_dqas and self.phase == 2:
            _gate_config_qsvt = self.gate_config_qsvt

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _timestep_state_qnode(initial_state, params):
                qml.StatePrep(initial_state, wires=range(_n_qubits))
                discrete_sim14_circuit(params, _gate_config_qsvt, wires=_n_qubits,
                                       layers=_n_ansatz_layers)
                return qml.state()
            self.timestep_state_qnode = _timestep_state_qnode

        else:
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _timestep_state_qnode(initial_state, params):
                qml.StatePrep(initial_state, wires=range(_n_qubits))
                sim14_circuit(params, wires=_n_qubits, layers=_n_ansatz_layers)
                return qml.state()
            self.timestep_state_qnode = _timestep_state_qnode

        # ---- QFF expval QNode ----
        if _use_dqas and self.phase == 1:
            if _use_ano:
                @qml.qnode(self.dev, interface="torch", diff_method="backprop")
                def _qff_qnode_expval(initial_state, params, arch_params, *H_flat):
                    qml.StatePrep(initial_state, wires=range(_n_qubits))
                    dqas_sim14_circuit(params, arch_params, wires=_n_qubits, layers=1)
                    observables = []
                    for w in range(_n_windows):
                        window_wires = [(w + j) % _n_qubits for j in range(_ano_k_local)]
                        observables.append(qml.expval(qml.Hermitian(H_flat[w], wires=window_wires)))
                    return observables
            else:
                @qml.qnode(self.dev, interface="torch", diff_method="backprop")
                def _qff_qnode_expval(initial_state, params, arch_params):
                    qml.StatePrep(initial_state, wires=range(_n_qubits))
                    dqas_sim14_circuit(params, arch_params, wires=_n_qubits, layers=1)
                    observables = ([qml.PauliX(i) for i in range(_n_qubits)] +
                                  [qml.PauliY(i) for i in range(_n_qubits)] +
                                  [qml.PauliZ(i) for i in range(_n_qubits)])
                    return [qml.expval(op) for op in observables]
            self.qff_qnode_expval = _qff_qnode_expval

        elif _use_dqas and self.phase == 2:
            _gate_config_qff = self.gate_config_qff

            if _use_ano:
                @qml.qnode(self.dev, interface="torch", diff_method="backprop")
                def _qff_qnode_expval(initial_state, params, *H_flat):
                    qml.StatePrep(initial_state, wires=range(_n_qubits))
                    discrete_sim14_circuit(params, _gate_config_qff, wires=_n_qubits, layers=1)
                    observables = []
                    for w in range(_n_windows):
                        window_wires = [(w + j) % _n_qubits for j in range(_ano_k_local)]
                        observables.append(qml.expval(qml.Hermitian(H_flat[w], wires=window_wires)))
                    return observables
            else:
                @qml.qnode(self.dev, interface="torch", diff_method="backprop")
                def _qff_qnode_expval(initial_state, params):
                    qml.StatePrep(initial_state, wires=range(_n_qubits))
                    discrete_sim14_circuit(params, _gate_config_qff, wires=_n_qubits, layers=1)
                    observables = ([qml.PauliX(i) for i in range(_n_qubits)] +
                                  [qml.PauliY(i) for i in range(_n_qubits)] +
                                  [qml.PauliZ(i) for i in range(_n_qubits)])
                    return [qml.expval(op) for op in observables]
            self.qff_qnode_expval = _qff_qnode_expval

        else:
            if _use_ano:
                @qml.qnode(self.dev, interface="torch", diff_method="backprop")
                def _qff_qnode_expval(initial_state, params, *H_flat):
                    qml.StatePrep(initial_state, wires=range(_n_qubits))
                    sim14_circuit(params, wires=_n_qubits, layers=1)
                    observables = []
                    for w in range(_n_windows):
                        window_wires = [(w + j) % _n_qubits for j in range(_ano_k_local)]
                        observables.append(qml.expval(qml.Hermitian(H_flat[w], wires=window_wires)))
                    return observables
            else:
                @qml.qnode(self.dev, interface="torch", diff_method="backprop")
                def _qff_qnode_expval(initial_state, params):
                    qml.StatePrep(initial_state, wires=range(_n_qubits))
                    sim14_circuit(params, wires=_n_qubits, layers=1)
                    observables = ([qml.PauliX(i) for i in range(_n_qubits)] +
                                  [qml.PauliY(i) for i in range(_n_qubits)] +
                                  [qml.PauliZ(i) for i in range(_n_qubits)])
                    return [qml.expval(op) for op in observables]
            self.qff_qnode_expval = _qff_qnode_expval

    def forward(self, x):
        # x: (batch, n_timesteps, state_dim)
        bsz = x.shape[0]

        # No PE (v1 style)
        x = self.feature_projection(self.dropout(x))
        # Sigmoid [0,1] angle scaling (v1 style -- NOT 2pi centered)
        timestep_params = self.rot_sigm(x)

        base_states = torch.zeros(
            bsz, 2 ** self.n_qubits,
            dtype=torch.complex64,
            device=self.device
        )
        base_states[:, 0] = 1.0

        if self.use_ano:
            H_list = [create_Hermitian(self.N_ano, self.A[w], self.B[w], self.D[w])
                      for w in range(self.n_windows)]

        # QSVT
        if self.use_dqas and self.phase == 1:
            mixed_timestep = evaluate_polynomial_state_pl_dqas(
                base_states, timestep_params, self.timestep_state_qnode,
                self.n_qubits, self.mix_coeffs.repeat(bsz, 1),
                self.poly_coeffs, self.arch_params_qsvt)
        else:
            mixed_timestep = evaluate_polynomial_state_pl(
                base_states, timestep_params, self.timestep_state_qnode,
                self.n_qubits, self.mix_coeffs.repeat(bsz, 1),
                self.poly_coeffs)

        norm = torch.linalg.vector_norm(mixed_timestep, dim=1, keepdim=True)
        normalized_mixed_timestep = mixed_timestep / (norm + 1e-9)

        # QFF measurement
        if self.use_dqas and self.phase == 1:
            if self.use_ano:
                exps = self.qff_qnode_expval(
                    normalized_mixed_timestep, self.qff_params,
                    self.arch_params_qff, *H_list)
            else:
                exps = self.qff_qnode_expval(
                    normalized_mixed_timestep, self.qff_params,
                    self.arch_params_qff)
        else:
            if self.use_ano:
                exps = self.qff_qnode_expval(
                    normalized_mixed_timestep, self.qff_params, *H_list)
            else:
                exps = self.qff_qnode_expval(
                    normalized_mixed_timestep, self.qff_params)

        exps = torch.stack(exps, dim=1)
        exps = exps.float()
        op = self.output_ff(exps)
        return op


# ================================================================================
# REPLAY BUFFER
# ================================================================================
class ReplayBuffer:
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device)
        )

    def __len__(self):
        return self.size


# ================================================================================
# DQN AGENT (v6: v1 base + three-optimizer + anti-forgetting)
# ================================================================================
class QuantumSimpleRLAgent_v6:
    def __init__(self, state_processor, action_dim, quantum_params):
        self.state_processor = state_processor
        self.action_dim = action_dim
        self.device = args.device

        # Q-Networks
        self.online_net = QuantumTSTransformerRL_v6(
            state_dim=state_processor.state_dim,
            n_qubits=quantum_params['n_qubits'],
            n_timesteps=quantum_params['n_timesteps'],
            degree=quantum_params['degree'],
            n_ansatz_layers=quantum_params['n_ansatz_layers'],
            output_dim=action_dim,
            dropout=quantum_params['dropout'],
            device=self.device,
            ano_k_local=quantum_params['ano_k_local'],
            use_ano=quantum_params['use_ano'],
            use_dqas=quantum_params['use_dqas']
        ).to(self.device)

        self.target_net = QuantumTSTransformerRL_v6(
            state_dim=state_processor.state_dim,
            n_qubits=quantum_params['n_qubits'],
            n_timesteps=quantum_params['n_timesteps'],
            degree=quantum_params['degree'],
            n_ansatz_layers=quantum_params['n_ansatz_layers'],
            output_dim=action_dim,
            dropout=quantum_params['dropout'],
            device=self.device,
            ano_k_local=quantum_params['ano_k_local'],
            use_ano=quantum_params['use_ano'],
            use_dqas=quantum_params['use_dqas']
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        # Setup optimizers (three groups)
        self._setup_optimizers()

        # v1 loss function: MSELoss (not SmoothL1Loss)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        state_shape = (quantum_params['n_timesteps'], state_processor.state_dim)
        self.memory = ReplayBuffer(args.memory_size, state_shape, self.device)

        # Exploration
        self.exploration_rate = args.exploration_rate_start
        self.exploration_rate_decay = args.exploration_rate_decay
        self.exploration_rate_min = args.exploration_rate_min

        # Training counters
        self.curr_step = 0
        self.sync_every = args.sync_every  # v6: configurable (default 500)

        # Phase tracking
        self.phase = 1

        # Anti-forgetting state
        self.best_avg_reward = -float('inf')
        self.best_avg_reward_episode = 0
        self.episodes_above_threshold = 0
        self.episodes_since_improvement = 0

    def _setup_optimizers(self):
        use_ano_flag = not args.no_ano
        use_dqas_flag = not args.no_dqas

        vqc_params = []
        ano_params = []
        arch_params = []

        for name, param in self.online_net.named_parameters():
            if not param.requires_grad:
                continue
            if use_ano_flag and ('A.' in name or 'B.' in name or 'D.' in name):
                ano_params.append(param)
            elif use_dqas_flag and 'arch_params' in name:
                arch_params.append(param)
            else:
                vqc_params.append(param)

        self.vqc_optimizer = torch.optim.Adam(vqc_params, lr=args.lr)

        if ano_params:
            self.ano_optimizer = torch.optim.Adam(ano_params, lr=args.ano_lr)
        else:
            self.ano_optimizer = None

        if arch_params:
            self.arch_optimizer = torch.optim.Adam(arch_params, lr=args.arch_lr)
        else:
            self.arch_optimizer = None

        n_vqc = sum(p.numel() for p in vqc_params)
        n_ano = sum(p.numel() for p in ano_params)
        n_arch = sum(p.numel() for p in arch_params)
        print(f"\nParameter groups:")
        print(f"  VQC params:  {n_vqc:,}")
        print(f"  ANO params:  {n_ano:,}")
        print(f"  Arch params: {n_arch:,}")
        print(f"  Total:       {n_vqc + n_ano + n_arch:,}")

    def select_action(self, state_history):
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = state_history.unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < args.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(args.batch_size)

        current_q_values = self.online_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones.float()) * args.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.vqc_optimizer.zero_grad()
        if self.ano_optimizer is not None:
            self.ano_optimizer.zero_grad()
        if self.phase == 1 and self.arch_optimizer is not None:
            self.arch_optimizer.zero_grad()

        loss.backward()
        # v1 style: no gradient clipping

        self.vqc_optimizer.step()
        if self.ano_optimizer is not None:
            self.ano_optimizer.step()
        if self.phase == 1 and self.arch_optimizer is not None:
            self.arch_optimizer.step()

        # Decay exploration rate per step (v1 behavior)
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay
        )

        # Sync target network (v6: slower sync)
        self.curr_step += 1
        if self.curr_step % self.sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def reduce_lr(self):
        """Halve the learning rate for all active optimizers."""
        for opt in [self.vqc_optimizer, self.ano_optimizer, self.arch_optimizer]:
            if opt is None:
                continue
            for pg in opt.param_groups:
                old_lr = pg['lr']
                pg['lr'] = max(old_lr * args.lr_reduce_factor, args.lr_min)
        new_lr = self.vqc_optimizer.param_groups[0]['lr']
        print(f"  LR reduced -> {new_lr:.6f}")

    def phase_transition(self):
        print("\n" + "=" * 80)
        print("PHASE TRANSITION: Search -> Production")
        print("=" * 80)

        online_transformer = self.online_net
        target_transformer = self.target_net

        arch_qsvt = online_transformer.arch_params_qsvt.detach()
        gate_config_qsvt = {}
        print("\nDiscovered QSVT architecture:")
        for layer_idx in range(arch_qsvt.shape[0]):
            for slot in range(4):
                phi = arch_qsvt[layer_idx, slot, 0]
                omega = arch_qsvt[layer_idx, slot, 1]
                if slot in (0, 2):
                    gate_name = classify_single_qubit_gate(phi, omega)
                else:
                    gate_name = classify_controlled_gate(phi, omega)
                gate_config_qsvt[(layer_idx, slot)] = gate_name
                print(f"  Layer {layer_idx}, Slot {slot}: "
                      f"phi={phi.item():.4f}, omega={omega.item():.4f} -> {gate_name}")

        arch_qff = online_transformer.arch_params_qff.detach()
        gate_config_qff = {}
        print("\nDiscovered QFF architecture:")
        for slot in range(4):
            phi = arch_qff[0, slot, 0]
            omega = arch_qff[0, slot, 1]
            if slot in (0, 2):
                gate_name = classify_single_qubit_gate(phi, omega)
            else:
                gate_name = classify_controlled_gate(phi, omega)
            gate_config_qff[(0, slot)] = gate_name
            print(f"  Slot {slot}: "
                  f"phi={phi.item():.4f}, omega={omega.item():.4f} -> {gate_name}")

        for transformer in [online_transformer, target_transformer]:
            transformer.phase = 2
            transformer.gate_config_qsvt = gate_config_qsvt
            transformer.gate_config_qff = gate_config_qff
            transformer.arch_params_qsvt.requires_grad_(False)
            transformer.arch_params_qff.requires_grad_(False)
            transformer._build_qnodes()

        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.phase = 2
        self._setup_optimizers()

        print("\nPhase 2: Architecture frozen, VQC + ANO training continues")
        print("=" * 80 + "\n")

    def save_checkpoint(self, episode, metrics, checkpoint_path):
        checkpoint = {
            'episode': episode,
            'curr_step': self.curr_step,
            'online_net_state': self.online_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'vqc_optimizer_state': self.vqc_optimizer.state_dict(),
            'ano_optimizer_state': (self.ano_optimizer.state_dict()
                                    if self.ano_optimizer is not None else None),
            'arch_optimizer_state': (self.arch_optimizer.state_dict()
                                     if self.arch_optimizer is not None else None),
            'exploration_rate': self.exploration_rate,
            'phase': self.phase,
            'gate_config_qsvt': self.online_net.gate_config_qsvt,
            'gate_config_qff': self.online_net.gate_config_qff,
            'metrics': metrics,
            'best_avg_reward': self.best_avg_reward,
            'best_avg_reward_episode': self.best_avg_reward_episode,
            'episodes_since_improvement': self.episodes_since_improvement,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved (Episode {episode}, Phase {self.phase})", flush=True)

    def save_best_model(self, episode, metrics):
        checkpoint = {
            'episode': episode,
            'online_net_state': self.online_net.state_dict(),
            'best_avg_reward': self.best_avg_reward,
            'phase': self.phase,
            'gate_config_qsvt': self.online_net.gate_config_qsvt,
            'gate_config_qff': self.online_net.gate_config_qff,
        }
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"  Best model saved (AvgR={self.best_avg_reward:.1f} @ ep {episode})", flush=True)

    def load_checkpoint(self, checkpoint_path):
        if not checkpoint_path.exists():
            print(f"No checkpoint found. Starting from scratch.")
            return 0, {}

        print(f"Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        saved_phase = checkpoint.get('phase', 1)
        if saved_phase == 2:
            gate_config_qsvt = checkpoint['gate_config_qsvt']
            gate_config_qff = checkpoint['gate_config_qff']

            for transformer in [self.online_net, self.target_net]:
                transformer.phase = 2
                transformer.gate_config_qsvt = gate_config_qsvt
                transformer.gate_config_qff = gate_config_qff
                transformer.arch_params_qsvt.requires_grad_(False)
                transformer.arch_params_qff.requires_grad_(False)
                transformer._build_qnodes()

            self.phase = 2

        self.online_net.load_state_dict(checkpoint['online_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])

        self._setup_optimizers()

        self.vqc_optimizer.load_state_dict(checkpoint['vqc_optimizer_state'])
        if (self.ano_optimizer is not None
                and checkpoint.get('ano_optimizer_state') is not None):
            self.ano_optimizer.load_state_dict(checkpoint['ano_optimizer_state'])
        if (self.arch_optimizer is not None
                and checkpoint.get('arch_optimizer_state') is not None):
            self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer_state'])

        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['curr_step']
        self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
        self.best_avg_reward_episode = checkpoint.get('best_avg_reward_episode', 0)
        self.episodes_since_improvement = checkpoint.get('episodes_since_improvement', 0)

        torch_rng_state = checkpoint['torch_rng_state']
        if torch_rng_state.device != torch.device('cpu'):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        completed_episode = checkpoint['episode']
        metrics = checkpoint.get('metrics', {})
        print(f"Resumed at Phase {self.phase}, Episode {completed_episode + 1}")
        print(f"  Best AvgR so far: {self.best_avg_reward:.1f} @ ep {self.best_avg_reward_episode}")

        return completed_episode, metrics


# ================================================================================
# TRAINING LOOP
# ================================================================================
def train():
    set_global_seeds(args.seed)

    use_ano_flag = not args.no_ano
    use_dqas_flag = not args.no_dqas

    state_processor = StateProcessor(env, args.n_timesteps)

    quantum_params = {
        'n_qubits': args.n_qubits,
        'n_timesteps': args.n_timesteps,
        'degree': args.degree,
        'n_ansatz_layers': args.n_layers,
        'dropout': args.dropout,
        'ano_k_local': args.ano_k_local,
        'use_ano': use_ano_flag,
        'use_dqas': use_dqas_flag
    }

    agent = QuantumSimpleRLAgent_v6(
        state_processor,
        action_dim=env.action_space.n,
        quantum_params=quantum_params
    )

    start_episode = 0
    metrics = {'rewards': [], 'lengths': [], 'losses': []}
    if args.resume and CHECKPOINT_FILE_PATH.exists():
        start_episode, metrics = agent.load_checkpoint(CHECKPOINT_FILE_PATH)
        start_episode += 1

    # Ablation condition label
    if use_ano_flag and use_dqas_flag:
        condition_label = "Full v6 (ANO + DiffQAS)"
    elif use_ano_flag:
        condition_label = "ANO only (fixed RY-CRX circuit)"
    elif use_dqas_flag:
        condition_label = "DiffQAS only (fixed PauliX/Y/Z)"
    else:
        condition_label = "v1 Baseline (fixed circuit + fixed Paulis)"

    print("\n" + "=" * 80)
    print(f"QUANTUM TRANSFORMER V6 - {args.env}")
    print(f"Ablation: {condition_label}")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Environment:     {args.env}")
    print(f"  State Dim:       {state_processor.state_dim}")
    print(f"  Action Space:    {env.action_space.n} actions")
    print(f"  Qubits:          {args.n_qubits}")
    print(f"  Layers:          {args.n_layers}")
    print(f"  QSVT Degree:     {args.degree}")
    print(f"  Timesteps:       {args.n_timesteps}")
    print(f"  Device:          {args.device}")
    print(f"  ANO:             {'ON (k={}, windows={}, lr={})'.format(args.ano_k_local, args.n_qubits, args.ano_lr) if use_ano_flag else 'OFF (fixed PauliX/Y/Z)'}")
    print(f"  DiffQAS:         {'ON (arch_lr={}, search_ep={})'.format(args.arch_lr, args.search_episodes) if use_dqas_flag else 'OFF (fixed RY-CRX)'}")
    if use_dqas_flag:
        print(f"  Current phase:   {agent.phase}")
    print(f"  Base:            v1 (sigmoid [0,1], MSELoss, no PE, no grad clip)")
    print(f"\nAnti-forgetting:")
    print(f"  sync_every:      {args.sync_every}")
    print(f"  early_stop:      AvgR > {args.early_stop_reward} for {args.early_stop_patience} eps")
    print(f"  lr_reduce:       patience={args.lr_reduce_patience}, factor={args.lr_reduce_factor}, min={args.lr_min}")
    print(f"\nSave Directory: {SAVE_DIR}")
    print("=" * 80 + "\n")

    # Signal handling
    current_episode = [start_episode]

    def save_checkpoint_on_signal(signum, frame):
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n\n{'='*80}")
        print(f"{signal_name} received! Saving checkpoint before exit...")
        print(f"{'='*80}")
        try:
            agent.save_checkpoint(current_episode[0], metrics, CHECKPOINT_FILE_PATH)
            plot_training_curves(metrics, SAVE_DIR)
            print(f"Emergency checkpoint saved at episode {current_episode[0]}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        print(f"{'='*80}\n")
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_checkpoint_on_signal)
    signal.signal(signal.SIGINT, save_checkpoint_on_signal)
    print(f"Signal handlers registered (SIGTERM, SIGINT)\n")

    # Training loop
    early_stopped = False
    for episode in range(start_episode, args.num_episodes):
        current_episode[0] = episode

        # Phase transition check
        if use_dqas_flag and agent.phase == 1 and episode >= args.search_episodes:
            agent.phase_transition()

        state, _ = env.reset()
        state_processor.reset()
        state_history = state_processor.add_state(state)

        episode_reward = 0
        episode_loss = []

        for step in range(args.max_steps):
            if args.render:
                env.render()

            action = agent.select_action(state_history)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state_history = state_processor.add_state(next_state)

            agent.store_transition(
                state_history, action, reward, next_state_history, done
            )

            if step % args.learn_step == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)

            episode_reward += reward
            state_history = next_state_history

            if done:
                break

        # Record metrics
        metrics['rewards'].append(episode_reward)
        metrics['lengths'].append(step + 1)
        metrics['losses'].append(np.mean(episode_loss) if episode_loss else 0.0)

        # Compute running average reward
        avg_reward = np.mean(metrics['rewards'][-100:])

        # --- Anti-forgetting: best model checkpointing ---
        if avg_reward > agent.best_avg_reward:
            agent.best_avg_reward = avg_reward
            agent.best_avg_reward_episode = episode
            agent.episodes_since_improvement = 0
            agent.save_best_model(episode, metrics)
        else:
            agent.episodes_since_improvement += 1

        # --- Anti-forgetting: LR reduction on plateau ---
        if agent.episodes_since_improvement >= args.lr_reduce_patience:
            current_lr = agent.vqc_optimizer.param_groups[0]['lr']
            if current_lr > args.lr_min:
                print(f"  No improvement for {args.lr_reduce_patience} episodes.")
                agent.reduce_lr()
            agent.episodes_since_improvement = 0  # Reset counter after reduction

        # --- Anti-forgetting: early stopping ---
        if avg_reward >= args.early_stop_reward:
            agent.episodes_above_threshold += 1
            if agent.episodes_above_threshold >= args.early_stop_patience:
                print(f"\nEarly stopping: AvgR >= {args.early_stop_reward} "
                      f"for {args.early_stop_patience} consecutive episodes!")
                agent.save_checkpoint(episode, metrics, CHECKPOINT_FILE_PATH)
                agent.save_best_model(episode, metrics)
                plot_training_curves(metrics, SAVE_DIR)
                early_stopped = True
                break
        else:
            agent.episodes_above_threshold = 0

        # Print progress
        if episode % 10 == 0:
            avg_loss = np.mean(metrics['losses'][-100:])
            phase_str = f"P{agent.phase}"
            current_lr = agent.vqc_optimizer.param_groups[0]['lr']
            print(
                f"Ep {episode:>4} [{phase_str}] | R: {episode_reward:>6.1f} | "
                f"AvgR: {avg_reward:>6.1f} | BestAvgR: {agent.best_avg_reward:>6.1f} | "
                f"L: {step+1:>4} | Loss: {avg_loss:>6.4f} | "
                f"e: {agent.exploration_rate:.3f} | lr: {current_lr:.6f}",
                flush=True
            )

        # Save checkpoint
        if (episode + 1) % args.save_every == 0 or episode == args.num_episodes - 1:
            agent.save_checkpoint(episode, metrics, CHECKPOINT_FILE_PATH)
            plot_training_curves(metrics, SAVE_DIR)

    env.close()
    if early_stopped:
        print(f"\nTraining early-stopped at episode {episode}.")
    print(f"\nBest AvgR: {agent.best_avg_reward:.1f} at episode {agent.best_avg_reward_episode}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")
    print("\n" + "=" * 80)
    print("Training finished!")
    print("=" * 80 + "\n")


# ================================================================================
# PLOTTING
# ================================================================================
def plot_training_curves(metrics, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (key, color) in enumerate([('rewards', 'blue'), ('lengths', 'green'), ('losses', 'red')]):
        axes[idx].plot(metrics[key], alpha=0.3, color=color)
        if len(metrics[key]) > 0:
            window = min(100, len(metrics[key]))
            moving_avg = np.convolve(metrics[key], np.ones(window)/window, mode='valid')
            axes[idx].plot(moving_avg, color=color, linewidth=2)
        axes[idx].set_xlabel('Episode')
        axes[idx].set_ylabel(key.capitalize())
        axes[idx].set_title(f'Episode {key.capitalize()}')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=100, bbox_inches='tight')
    plt.close()


# ================================================================================
# MAIN
# ================================================================================
if __name__ == '__main__':
    train()
