"""
Quantum Time-Series Transformer for Atari Environments — v5
=============================================================
This script implements a quantum transformer-based RL agent for classic Atari games
using hybrid CNN + Quantum architecture with two key innovations:

v5 innovations over v3:
  1. ANO (Adaptive Non-Local Observables): Learnable k-local Hermitian measurements
     replace fixed PauliX/Y/Z — expands output expressivity without deeper circuits
     (Lin et al. 2025)
  2. DiffQAS (Differentiable Quantum Architecture Search): Gate-level search using
     parametric Rot/CRot gates that specialize to RY/RX/RZ during Phase 1 search,
     then discretize for Phase 2 production training (Sun et al. 2023)

Two-phase training (when DiffQAS enabled):
  Phase 1 (search): VQC + ANO + arch params train jointly for --search-episodes
  Phase 2 (production): Arch params discretized and frozen, VQC + ANO continue

Ablation flags:
  --no-ano   : Disable ANO, use fixed PauliX/Y/Z measurements (v3 behavior)
  --no-dqas  : Disable DiffQAS, use fixed RY-CRX sim14 circuit (v3 behavior)

  Four ablation conditions:
    --no-ano --no-dqas  : v3 baseline (fixed circuit + fixed Paulis)
    --no-dqas           : ANO only (fixed circuit + learnable Hermitians)
    --no-ano            : DiffQAS only (architecture search + fixed Paulis)
    (default)           : Full v5 (architecture search + learnable Hermitians)

Carries forward all v3 features:
  - Sinusoidal Positional Encoding (Vaswani et al., 2017)
  - 2pi angle scaling: (sigmoid - 0.5) * 2pi for full rotation range [-pi, pi]
  - Separate CNN instances for online/target networks

Supported Games:
- DonkeyKong-v5, Pacman-v5, MarioBros-v5, SpaceInvaders-v5,
  Tetris-v5, Breakout-v5, Pong-v5

Architecture:
  Atari Frame (210x160 RGB) -> Preprocessing -> 4 Stacked Frames (4x84x84)
  -> CNN Feature Extractor -> + Sinusoidal PE -> Quantum Transformer -> Q-values
"""

import math
import torch
from torch import nn
from torchvision import transforms as T
import numpy as np
from pathlib import Path
from collections import deque
import random, time, datetime, os
import argparse
from math import log2

import pennylane as qml
import matplotlib.pyplot as plt

# Use gymnasium (modern replacement for gym) with ale-py
try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    print("Using gymnasium with ale-py for Atari environments")
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install required packages:")
    print("  pip install gymnasium ale-py gymnasium[atari] gymnasium[accept-rom-license]")
    raise


# ================================================================================
# ARGUMENT PARSER
# ================================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description='Quantum Transformer v5 (ANO + DiffQAS) for Atari Games'
    )

    # Environment selection
    parser.add_argument("--env", type=str, default="ALE/DonkeyKong-v5",
                        choices=["ALE/DonkeyKong-v5", "ALE/Pacman-v5", "ALE/MarioBros-v5",
                                "ALE/SpaceInvaders-v5", "ALE/Tetris-v5",
                                "ALE/Breakout-v5", "ALE/Pong-v5"],
                        help="Atari game to train on (gymnasium v5 naming)")

    # Quantum parameters
    parser.add_argument("--n-qubits", type=int, default=6,
                        help="Number of qubits in quantum circuit")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of ansatz layers")
    parser.add_argument("--degree", type=int, default=2,
                        help="Degree of QSVT polynomial")
    parser.add_argument("--n-timesteps", type=int, default=4,
                        help="Number of timesteps (frame stack)")
    parser.add_argument("--feature-dim", type=int, default=128,
                        help="Feature dimension after CNN extraction")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Ablation flags (v5)
    parser.add_argument("--no-ano", action="store_true",
                        help="Disable ANO: use fixed PauliX/Y/Z measurements (v3 behavior)")
    parser.add_argument("--no-dqas", action="store_true",
                        help="Disable DiffQAS: use fixed RY-CRX sim14 circuit (v3 behavior)")

    # ANO parameters (v5)
    parser.add_argument("--ano-k-local", type=int, default=2,
                        help="Locality of ANO observables (ignored if --no-ano)")
    parser.add_argument("--ano-lr", type=float, default=0.01,
                        help="Learning rate for ANO params (ignored if --no-ano)")

    # DiffQAS parameters (v5)
    parser.add_argument("--arch-lr", type=float, default=0.001,
                        help="Learning rate for architecture search params (ignored if --no-dqas)")
    parser.add_argument("--search-episodes", type=int, default=100,
                        help="Episodes for architecture search phase (ignored if --no-dqas)")

    # RL parameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--exploration-rate-start', type=float, default=1.0)
    parser.add_argument('--exploration-rate-decay', type=float, default=0.9999)
    parser.add_argument('--exploration-rate-min', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument('--learn-step', type=int, default=4,
                        help="Learn every N steps")
    parser.add_argument('--num-episodes', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=10000,
                        help="Max steps per episode")
    parser.add_argument('--lr', type=float, default=0.00025,
                        help="Learning rate")
    parser.add_argument('--memory-size', type=int, default=100000,
                        help="Replay buffer size")

    # Training parameters
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log-index", type=int, default=1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during training")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--save-dir", type=str, default="QRL_QTSTransformer/checkpoints",
                        help="Base directory for checkpoints")

    return parser.parse_args()

args = get_args()


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================
def set_global_seeds(seed_value):
    """Sets global seeds for reproducibility."""
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
env_name_clean = args.env.replace("ALE/", "").replace("-", "").replace("v", "")
# Encode ablation condition in RUN_ID
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
RUN_ID = (f"QTransformerV5_{env_name_clean}_Q{args.n_qubits}_L{args.n_layers}"
          f"_D{args.degree}_K{args.ano_k_local}_{_ablation_tag}_Run{args.log_index}")
CHECKPOINT_BASE_DIR = Path(args.save_dir)
SAVE_DIR = CHECKPOINT_BASE_DIR / RUN_ID

SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE_PATH = SAVE_DIR / "latest_checkpoint.chkpt"


# ================================================================================
# ATARI PREPROCESSING
# ================================================================================
class AtariPreprocessing(gym.ObservationWrapper):
    """
    Atari preprocessing wrapper:
    - Converts to grayscale
    - Resizes to 84x84
    - Normalizes pixel values
    """
    def __init__(self, env, grayscale=True, resize_shape=(84, 84), normalize=True):
        super().__init__(env)
        self.grayscale = grayscale
        self.resize_shape = resize_shape
        self.normalize = normalize

        if grayscale:
            obs_shape = (1, *resize_shape)
        else:
            obs_shape = (3, *resize_shape)

        self.observation_space = gym.spaces.Box(
            low=0.0 if normalize else 0,
            high=1.0 if normalize else 255,
            shape=obs_shape,
            dtype=np.float32
        )

    def observation(self, obs):
        """Preprocess Atari frame."""
        obs = torch.from_numpy(obs).permute(2, 0, 1).float()
        if self.grayscale:
            obs = T.Grayscale()(obs)
        obs = T.Resize(self.resize_shape, antialias=True)(obs)
        if self.normalize:
            obs = obs / 255.0
        return obs.squeeze(0).numpy()


class FrameStack(gym.Wrapper):
    """Stack consecutive frames for temporal information."""
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        done = terminated or truncated
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.array(list(self.frames))


class SkipFrame(gym.Wrapper):
    """Skip frames to speed up training (action repeat)."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# ================================================================================
# ENVIRONMENT SETUP
# ================================================================================
def create_atari_env(env_name, render_mode=None):
    """Create and wrap Atari environment."""
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)

    env = SkipFrame(env, skip=4)
    env = AtariPreprocessing(env, grayscale=True, resize_shape=(84, 84), normalize=True)
    env = FrameStack(env, num_stack=4)

    return env


env = create_atari_env(args.env, render_mode='human' if args.render else None)


# ================================================================================
# ANO HELPER — Adaptive Non-Local Observables (Lin et al. 2025)
# ================================================================================
def create_Hermitian(N, A, B, D):
    """Build N x N Hermitian matrix from learnable off-diagonal and diagonal params.

    Args:
        N: matrix dimension (2^k_local)
        A: real off-diagonal params, size N*(N-1)/2
        B: imaginary off-diagonal params, size N*(N-1)/2
        D: diagonal params, size N
    """
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
# FIXED SIM14 CIRCUIT (v3 baseline — used when --no-dqas)
# ================================================================================
def sim14_circuit(params, wires, layers=1):
    """Sim14 quantum circuit ansatz with fixed RY-CRX gates (v3 behavior)."""
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
# DIFFQAS CIRCUIT COMPONENTS — Gate-Level Parametric Search
# ================================================================================

# Canonical gate reference points in (phi, omega) space for Rot(phi, theta, omega):
#   RY: (0, 0)        -> Rot(0, theta, 0)
#   RX: (-pi/2, pi/2) -> Rot(-pi/2, theta, pi/2)
#   RZ: (pi/2, -pi/2) -> Rot(pi/2, theta, -pi/2) [up to global phase]
# Same mapping applies for CRot -> {CRX, CRY, CRZ}

SINGLE_GATES = {'RY': qml.RY, 'RX': qml.RX, 'RZ': qml.RZ}
CTRL_GATES = {'CRX': qml.CRX, 'CRY': qml.CRY, 'CRZ': qml.CRZ}

# Reference (phi, omega) for each gate type
_SINGLE_REFS = {
    'RY': (0.0, 0.0),
    'RX': (-math.pi / 2, math.pi / 2),
    'RZ': (math.pi / 2, -math.pi / 2),
}
_CTRL_REFS = {
    'CRY': (0.0, 0.0),
    'CRX': (-math.pi / 2, math.pi / 2),
    'CRZ': (math.pi / 2, -math.pi / 2),
}


def classify_single_qubit_gate(phi, omega):
    """Classify Rot(phi, theta, omega) to closest canonical single-qubit gate."""
    phi_val = phi.item() if isinstance(phi, torch.Tensor) else phi
    omega_val = omega.item() if isinstance(omega, torch.Tensor) else omega
    dists = {}
    for name, (ref_p, ref_o) in _SINGLE_REFS.items():
        dists[name] = (phi_val - ref_p) ** 2 + (omega_val - ref_o) ** 2
    return min(dists, key=dists.get)


def classify_controlled_gate(phi, omega):
    """Classify CRot(phi, theta, omega) to closest canonical controlled gate."""
    phi_val = phi.item() if isinstance(phi, torch.Tensor) else phi
    omega_val = omega.item() if isinstance(omega, torch.Tensor) else omega
    dists = {}
    for name, (ref_p, ref_o) in _CTRL_REFS.items():
        dists[name] = (phi_val - ref_p) ** 2 + (omega_val - ref_o) ** 2
    return min(dists, key=dists.get)


def dqas_sim14_circuit(params, arch_params, wires, layers=1):
    """Sim14 circuit with parametric Rot/CRot for architecture search (Phase 1).

    arch_params shape: (layers, 4, 2) — (phi, omega) per slot per layer
    params: standard variational params, same indexing as v3 sim14_circuit
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for layer_idx in range(layers):
        # Slot 0: single-qubit (was RY)
        phi_0 = arch_params[layer_idx, 0, 0]
        omega_0 = arch_params[layer_idx, 0, 1]
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.Rot(phi_0, angle, omega_0, wires=i)
            param_idx += 1

        # Slot 1: controlled forward (was CRX)
        phi_1 = arch_params[layer_idx, 1, 0]
        omega_1 = arch_params[layer_idx, 1, 1]
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRot(phi_1, angle, omega_1, wires=[i, (i + 1) % wires])
            param_idx += 1

        # Slot 2: single-qubit (was RY)
        phi_2 = arch_params[layer_idx, 2, 0]
        omega_2 = arch_params[layer_idx, 2, 1]
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.Rot(phi_2, angle, omega_2, wires=i)
            param_idx += 1

        # Slot 3: controlled backward (was CRX)
        phi_3 = arch_params[layer_idx, 3, 0]
        omega_3 = arch_params[layer_idx, 3, 1]
        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRot(phi_3, angle, omega_3, wires=[i, (i - 1) % wires])
            param_idx += 1


def discrete_sim14_circuit(params, gate_config, wires, layers=1):
    """Sim14 circuit with discrete gate choices (Phase 2).

    gate_config: dict {(layer, slot): gate_name}
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for layer_idx in range(layers):
        # Slot 0: single-qubit
        gate_name_0 = gate_config[(layer_idx, 0)]
        gate_fn_0 = SINGLE_GATES[gate_name_0]
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            gate_fn_0(angle, wires=i)
            param_idx += 1

        # Slot 1: controlled forward
        gate_name_1 = gate_config[(layer_idx, 1)]
        gate_fn_1 = CTRL_GATES[gate_name_1]
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            gate_fn_1(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        # Slot 2: single-qubit
        gate_name_2 = gate_config[(layer_idx, 2)]
        gate_fn_2 = SINGLE_GATES[gate_name_2]
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            gate_fn_2(angle, wires=i)
            param_idx += 1

        # Slot 3: controlled backward
        gate_name_3 = gate_config[(layer_idx, 3)]
        gate_fn_3 = CTRL_GATES[gate_name_3]
        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            gate_fn_3(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


# ================================================================================
# QSVT HELPERS
# ================================================================================
def apply_unitaries_pl(base_states, unitary_params, qnode_state, coeffs):
    """Apply linear combination of unitaries (fully vectorized) — Phase 2."""
    bsz, n_timesteps, n_rots = unitary_params.shape
    n_qbs = int(log2(base_states.shape[1]))

    flat_params = unitary_params.reshape(bsz * n_timesteps, n_rots)
    repeated_base_states = base_states.repeat_interleave(n_timesteps, dim=0)

    evolved_states = qnode_state(
        initial_state=repeated_base_states,
        params=flat_params
    )

    evolved_states_reshaped = evolved_states.reshape(bsz, n_timesteps, 2**n_qbs)
    evolved_states_reshaped = evolved_states_reshaped.to(torch.complex64)
    coeffs = coeffs.to(torch.complex64)

    lcs = torch.einsum('bti,bt->bi', evolved_states_reshaped, coeffs)
    return lcs


def evaluate_polynomial_state_pl(base_states, unitary_params, qnode_state,
                                  n_qbs, lcu_coeffs, poly_coeffs):
    """QSVT polynomial state preparation — Phase 2."""
    acc = poly_coeffs[0] * base_states
    working_register = base_states

    for c in poly_coeffs[1:]:
        working_register = apply_unitaries_pl(
            working_register, unitary_params, qnode_state, lcu_coeffs
        )
        acc = acc + c * working_register

    return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)


def apply_unitaries_pl_dqas(base_states, unitary_params, qnode_state, coeffs, arch_params):
    """Apply linear combination of unitaries — Phase 1 (passes arch_params to QNode)."""
    bsz, n_timesteps, n_rots = unitary_params.shape
    n_qbs = int(log2(base_states.shape[1]))

    flat_params = unitary_params.reshape(bsz * n_timesteps, n_rots)
    repeated_base_states = base_states.repeat_interleave(n_timesteps, dim=0)

    evolved_states = qnode_state(
        initial_state=repeated_base_states,
        params=flat_params,
        arch_params=arch_params
    )

    evolved_states_reshaped = evolved_states.reshape(bsz, n_timesteps, 2**n_qbs)
    evolved_states_reshaped = evolved_states_reshaped.to(torch.complex64)
    coeffs = coeffs.to(torch.complex64)

    lcs = torch.einsum('bti,bt->bi', evolved_states_reshaped, coeffs)
    return lcs


def evaluate_polynomial_state_pl_dqas(base_states, unitary_params, qnode_state,
                                       n_qbs, lcu_coeffs, poly_coeffs, arch_params):
    """QSVT polynomial state preparation — Phase 1 (passes arch_params)."""
    acc = poly_coeffs[0] * base_states
    working_register = base_states

    for c in poly_coeffs[1:]:
        working_register = apply_unitaries_pl_dqas(
            working_register, unitary_params, qnode_state, lcu_coeffs, arch_params
        )
        acc = acc + c * working_register

    return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)


# ================================================================================
# CNN FEATURE EXTRACTOR
# ================================================================================
class AtariCNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for Atari frames.
    Architecture based on DQN Nature paper (Mnih et al., 2015).
    """
    def __init__(self, output_dim=128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_output_size = 64 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim * 4)
        )
        self.output_dim = output_dim

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = x.view(batch_size, 4, self.output_dim)
        return x


# ================================================================================
# QUANTUM TIME-SERIES TRANSFORMER v5 (ANO + DiffQAS)
# ================================================================================
class QuantumTSTransformerRL_v5(nn.Module):
    """Quantum Time-Series Transformer for Atari (v5).

    Supports four ablation conditions via use_ano / use_dqas flags:
      - use_ano=True:  Learnable k-local Hermitian measurements (ANO)
      - use_ano=False: Fixed PauliX/Y/Z measurements (v3 behavior)
      - use_dqas=True: Parametric Rot/CRot with two-phase architecture search
      - use_dqas=False: Fixed RY-CRX sim14 circuit (v3 behavior)
    """
    def __init__(self, n_qubits: int, n_timesteps: int, degree: int,
                 n_ansatz_layers: int, feature_dim: int, output_dim: int,
                 dropout: float, device, ano_k_local: int = 2,
                 use_ano: bool = True, use_dqas: bool = True):
        super().__init__()

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

        # Phase tracking (only meaningful when use_dqas=True)
        self.phase = 1 if use_dqas else 2  # no-dqas skips straight to "production"
        self.gate_config_qsvt = None
        self.gate_config_qff = None

        # Classical layers
        self.feature_projection = nn.Linear(feature_dim, self.n_rots)
        self.dropout = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()

        # --- ANO parameters (only when use_ano=True) ---
        if use_ano:
            self.n_windows = n_qubits  # circular sliding windows
            self.N_ano = 2 ** ano_k_local  # matrix dimension per window

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

            qff_output_dim = self.n_windows  # ANO: one expval per window
        else:
            self.n_windows = None
            self.N_ano = None
            qff_output_dim = 3 * n_qubits  # fixed PauliX/Y/Z per qubit

        # --- Architecture search parameters (only when use_dqas=True) ---
        if use_dqas:
            # Shape: (layers, 4_slots, 2_params[phi,omega])
            # Initialized to zeros -> Rot(0, theta, 0) = RY(theta), matching v3
            self.arch_params_qsvt = nn.Parameter(torch.zeros(n_ansatz_layers, 4, 2))
            self.arch_params_qff = nn.Parameter(torch.zeros(1, 4, 2))

        # Output head dimension depends on ANO vs fixed Pauli
        self.output_ff = nn.Sequential(
            nn.Linear(qff_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        # --- Sinusoidal Positional Encoding (Vaswani et al., 2017) ---
        pe = torch.zeros(n_timesteps, feature_dim)
        pos = torch.arange(n_timesteps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, feature_dim, 2).float()
                        * -(math.log(10000.0) / feature_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:feature_dim // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, feature_dim)

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
        """Build PennyLane QNodes for current phase and ablation config.

        Circuit selection (use_dqas):
          - Phase 1 + use_dqas: dqas_sim14_circuit with arch_params
          - Phase 2 + use_dqas: discrete_sim14_circuit with baked gate_config
          - !use_dqas:          sim14_circuit (fixed RY-CRX, same as v3)

        Measurement selection (use_ano):
          - use_ano:  qml.Hermitian on sliding k-local windows (*H_flat args)
          - !use_ano: fixed PauliX/Y/Z per qubit (no extra args)
        """
        _n_qubits = self.n_qubits
        _n_ansatz_layers = self.n_ansatz_layers
        _use_ano = self.use_ano
        _use_dqas = self.use_dqas
        _n_windows = self.n_windows
        _ano_k_local = self.ano_k_local

        self.dev = qml.device("default.qubit", wires=_n_qubits)

        # ---- QSVT state QNode (returns qml.state()) ----
        if _use_dqas and self.phase == 1:
            # DiffQAS Phase 1: parametric Rot/CRot with arch_params
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _timestep_state_qnode(initial_state, params, arch_params):
                qml.StatePrep(initial_state, wires=range(_n_qubits))
                dqas_sim14_circuit(params, arch_params, wires=_n_qubits,
                                   layers=_n_ansatz_layers)
                return qml.state()

            self.timestep_state_qnode = _timestep_state_qnode

        elif _use_dqas and self.phase == 2:
            # DiffQAS Phase 2: discrete gates baked in
            _gate_config_qsvt = self.gate_config_qsvt

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _timestep_state_qnode(initial_state, params):
                qml.StatePrep(initial_state, wires=range(_n_qubits))
                discrete_sim14_circuit(params, _gate_config_qsvt, wires=_n_qubits,
                                       layers=_n_ansatz_layers)
                return qml.state()

            self.timestep_state_qnode = _timestep_state_qnode

        else:
            # No DiffQAS: fixed RY-CRX sim14 (v3 behavior)
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _timestep_state_qnode(initial_state, params):
                qml.StatePrep(initial_state, wires=range(_n_qubits))
                sim14_circuit(params, wires=_n_qubits, layers=_n_ansatz_layers)
                return qml.state()

            self.timestep_state_qnode = _timestep_state_qnode

        # ---- QFF expval QNode (circuit + measurement) ----
        # Build the circuit part based on DiffQAS phase, then measurement based on ANO

        if _use_dqas and self.phase == 1:
            # DiffQAS Phase 1 circuit + ANO or fixed Pauli measurement
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
            # DiffQAS Phase 2 circuit + ANO or fixed Pauli
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
            # No DiffQAS: fixed sim14 circuit + ANO or fixed Pauli
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
                # Pure v3 baseline: fixed sim14 + fixed PauliX/Y/Z
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
        # x: (batch, n_timesteps, feature_dim)
        bsz = x.shape[0]

        # Positional encoding injection (v3)
        x = x + self.pe[:, :x.size(1)]

        x = self.feature_projection(self.dropout(x))
        # 2pi angle scaling (v3): full [-pi, pi] rotation range centered at 0
        timestep_params = (self.rot_sigm(x) - 0.5) * (2 * math.pi)

        base_states = torch.zeros(
            bsz, 2 ** self.n_qubits,
            dtype=torch.complex64,
            device=self.device
        )
        base_states[:, 0] = 1.0

        # Build Hermitian matrices for ANO (only when use_ano=True)
        if self.use_ano:
            H_list = [create_Hermitian(self.N_ano, self.A[w], self.B[w], self.D[w])
                      for w in range(self.n_windows)]

        # QSVT: polynomial state preparation
        # DiffQAS Phase 1 passes arch_params; Phase 2 and no-dqas do not
        if self.use_dqas and self.phase == 1:
            mixed_timestep = evaluate_polynomial_state_pl_dqas(
                base_states,
                timestep_params,
                self.timestep_state_qnode,
                self.n_qubits,
                self.mix_coeffs.repeat(bsz, 1),
                self.poly_coeffs,
                self.arch_params_qsvt
            )
        else:
            mixed_timestep = evaluate_polynomial_state_pl(
                base_states,
                timestep_params,
                self.timestep_state_qnode,
                self.n_qubits,
                self.mix_coeffs.repeat(bsz, 1),
                self.poly_coeffs
            )

        norm = torch.linalg.vector_norm(mixed_timestep, dim=1, keepdim=True)
        normalized_mixed_timestep = mixed_timestep / (norm + 1e-9)

        # QFF: measurement — all args passed positionally
        if self.use_dqas and self.phase == 1:
            # DiffQAS Phase 1: always pass arch_params
            if self.use_ano:
                exps = self.qff_qnode_expval(
                    normalized_mixed_timestep,
                    self.qff_params,
                    self.arch_params_qff,
                    *H_list
                )
            else:
                exps = self.qff_qnode_expval(
                    normalized_mixed_timestep,
                    self.qff_params,
                    self.arch_params_qff
                )
        else:
            # DiffQAS Phase 2 or no-dqas: no arch_params
            if self.use_ano:
                exps = self.qff_qnode_expval(
                    normalized_mixed_timestep,
                    self.qff_params,
                    *H_list
                )
            else:
                exps = self.qff_qnode_expval(
                    normalized_mixed_timestep,
                    self.qff_params
                )

        exps = torch.stack(exps, dim=1)
        exps = exps.float()
        op = self.output_ff(exps)
        return op


# ================================================================================
# HYBRID NETWORK
# ================================================================================
class QuantumTransformerAtariNet_v5(nn.Module):
    """Complete Q-Network: CNN + Quantum Transformer v5."""
    def __init__(self, action_dim, device, quantum_params):
        super().__init__()

        self.action_dim = action_dim
        self.device = device

        self.online = self._build_quantum_network(quantum_params)
        self.target = self._build_quantum_network(quantum_params)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        for p in self.target.parameters():
            p.requires_grad = False

    def _build_quantum_network(self, qparams):
        """Build complete quantum network pipeline with its own CNN."""
        return nn.Sequential(
            AtariCNNFeatureExtractor(
                output_dim=qparams['feature_dim']
            ),
            QuantumTSTransformerRL_v5(
                n_qubits=qparams['n_qubits'],
                n_timesteps=qparams['n_timesteps'],
                degree=qparams['degree'],
                n_ansatz_layers=qparams['n_ansatz_layers'],
                feature_dim=qparams['feature_dim'],
                output_dim=self.action_dim,
                dropout=qparams['dropout'],
                device=self.device,
                ano_k_local=qparams['ano_k_local'],
                use_ano=qparams['use_ano'],
                use_dqas=qparams['use_dqas']
            )
        )

    def forward(self, state_batch, model):
        """Forward pass through online or target network."""
        if model == "online":
            return self.online(state_batch)
        else:
            return self.target(state_batch)


# ================================================================================
# REPLAY BUFFER
# ================================================================================
class ReplayBuffer:
    """Experience replay buffer."""
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
        """Add experience."""
        self.states[self.position] = torch.from_numpy(state)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.from_numpy(next_state)
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch."""
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
# DQN AGENT (v5: Three-Optimizer with Phase Transition)
# ================================================================================
class QuantumAtariAgent_v5:
    """DQN agent with three optimizer groups for ANO + DiffQAS."""
    def __init__(self, action_dim, quantum_params):
        self.action_dim = action_dim
        self.device = args.device

        # Q-Networks
        self.net = QuantumTransformerAtariNet_v5(
            action_dim, self.device, quantum_params
        ).to(self.device)

        # Classify parameters into 3 groups
        self._setup_optimizers()

        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        state_shape = (4, 84, 84)
        self.memory = ReplayBuffer(args.memory_size, state_shape, self.device)

        # Exploration
        self.exploration_rate = args.exploration_rate_start
        self.exploration_rate_decay = args.exploration_rate_decay
        self.exploration_rate_min = args.exploration_rate_min

        # Training counters
        self.curr_step = 0
        self.sync_every = 1000
        self.burnin = 10000

        # Phase tracking
        self.phase = 1

    def _setup_optimizers(self):
        """Set up optimizer groups based on current phase and ablation flags."""
        use_ano = not args.no_ano
        use_dqas = not args.no_dqas

        vqc_params = []
        ano_params = []
        arch_params = []

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if use_ano and ('A.' in name or 'B.' in name or 'D.' in name):
                ano_params.append(param)
            elif use_dqas and 'arch_params' in name:
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

        # Print parameter counts
        n_vqc = sum(p.numel() for p in vqc_params)
        n_ano = sum(p.numel() for p in ano_params)
        n_arch = sum(p.numel() for p in arch_params)
        print(f"\nParameter groups:")
        print(f"  VQC params:  {n_vqc:,}")
        print(f"  ANO params:  {n_ano:,}")
        print(f"  Arch params: {n_arch:,}")
        print(f"  Total:       {n_vqc + n_ano + n_arch:,}")

    def select_action(self, state):
        """Select action using epsilon-greedy."""
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.net(state_tensor, model="online")
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """Perform one learning step with multi-optimizer updates."""
        self.curr_step += 1

        if len(self.memory) < self.burnin:
            return None

        if self.curr_step % args.learn_step != 0:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(args.batch_size)

        # Compute current Q-values
        current_q_values = self.net(states, model="online")
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            next_q_online = self.net(next_states, model="online")
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = self.net(next_states, model="target")
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones.float()) * args.gamma * next_q

        # Update — zero all active optimizers
        loss = self.loss_fn(current_q, target_q)
        self.vqc_optimizer.zero_grad()
        if self.ano_optimizer is not None:
            self.ano_optimizer.zero_grad()
        if self.phase == 1 and self.arch_optimizer is not None:
            self.arch_optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)

        self.vqc_optimizer.step()
        if self.ano_optimizer is not None:
            self.ano_optimizer.step()
        if self.phase == 1 and self.arch_optimizer is not None:
            self.arch_optimizer.step()

        # Decay exploration
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay
        )

        # Sync target network
        if self.curr_step % self.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        return loss.item()

    def phase_transition(self):
        """Transition from Phase 1 (search) to Phase 2 (production).

        1. Read arch_params and classify to discrete gates
        2. Rebuild QNodes with discrete gates
        3. Recreate optimizers (excluding arch_params)
        """
        print("\n" + "=" * 80)
        print("PHASE TRANSITION: Search -> Production")
        print("=" * 80)

        # Get the quantum transformer module (index 1 in Sequential)
        online_transformer = self.net.online[1]
        target_transformer = self.net.target[1]

        # Classify QSVT architecture
        arch_qsvt = online_transformer.arch_params_qsvt.detach()
        gate_config_qsvt = {}
        print("\nDiscovered QSVT architecture:")
        for layer_idx in range(arch_qsvt.shape[0]):
            for slot in range(4):
                phi = arch_qsvt[layer_idx, slot, 0]
                omega = arch_qsvt[layer_idx, slot, 1]
                if slot in (0, 2):  # single-qubit slots
                    gate_name = classify_single_qubit_gate(phi, omega)
                else:  # controlled slots
                    gate_name = classify_controlled_gate(phi, omega)
                gate_config_qsvt[(layer_idx, slot)] = gate_name
                print(f"  Layer {layer_idx}, Slot {slot}: "
                      f"phi={phi.item():.4f}, omega={omega.item():.4f} -> {gate_name}")

        # Classify QFF architecture
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

        # Update both online and target transformers
        for transformer in [online_transformer, target_transformer]:
            transformer.phase = 2
            transformer.gate_config_qsvt = gate_config_qsvt
            transformer.gate_config_qff = gate_config_qff
            # Freeze arch params
            transformer.arch_params_qsvt.requires_grad_(False)
            transformer.arch_params_qff.requires_grad_(False)
            # Rebuild QNodes with discrete gates
            transformer._build_qnodes()

        # Sync target after rebuild
        self.net.target.load_state_dict(self.net.online.state_dict())
        for p in self.net.target.parameters():
            p.requires_grad = False

        # Recreate optimizers (arch_params now excluded)
        self.phase = 2
        self._setup_optimizers()

        print("\nPhase 2: Architecture frozen, VQC + ANO training continues")
        print("=" * 80 + "\n")

    def save_checkpoint(self, episode, metrics, checkpoint_path):
        """Save checkpoint with phase info."""
        # Get gate configs from online transformer
        online_transformer = self.net.online[1]

        checkpoint = {
            'episode': episode,
            'curr_step': self.curr_step,
            'net_state': self.net.state_dict(),
            'vqc_optimizer_state': self.vqc_optimizer.state_dict(),
            'ano_optimizer_state': (self.ano_optimizer.state_dict()
                                    if self.ano_optimizer is not None else None),
            'arch_optimizer_state': (self.arch_optimizer.state_dict()
                                     if self.arch_optimizer is not None else None),
            'exploration_rate': self.exploration_rate,
            'phase': self.phase,
            'gate_config_qsvt': online_transformer.gate_config_qsvt,
            'gate_config_qff': online_transformer.gate_config_qff,
            'metrics': metrics,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved (Episode {episode}, Phase {self.phase})")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore phase."""
        if not checkpoint_path.exists():
            print(f"No checkpoint found. Starting from scratch.")
            return 0, {}

        print(f"Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore phase before loading state (need correct QNodes)
        saved_phase = checkpoint.get('phase', 1)
        if saved_phase == 2:
            gate_config_qsvt = checkpoint['gate_config_qsvt']
            gate_config_qff = checkpoint['gate_config_qff']

            for transformer in [self.net.online[1], self.net.target[1]]:
                transformer.phase = 2
                transformer.gate_config_qsvt = gate_config_qsvt
                transformer.gate_config_qff = gate_config_qff
                transformer.arch_params_qsvt.requires_grad_(False)
                transformer.arch_params_qff.requires_grad_(False)
                transformer._build_qnodes()

            self.phase = 2

        self.net.load_state_dict(checkpoint['net_state'])

        # Recreate optimizers for correct param groups
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

        torch_rng_state = checkpoint['torch_rng_state']
        if torch_rng_state.device != torch.device('cpu'):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        print(f"Resumed at Phase {self.phase}")

        return checkpoint['episode'], checkpoint.get('metrics', {})


# ================================================================================
# TRAINING
# ================================================================================
def train():
    """Main training loop with two-phase architecture search."""
    set_global_seeds(args.seed)

    use_ano = not args.no_ano
    use_dqas = not args.no_dqas

    quantum_params = {
        'n_qubits': args.n_qubits,
        'n_timesteps': args.n_timesteps,
        'degree': args.degree,
        'n_ansatz_layers': args.n_layers,
        'feature_dim': args.feature_dim,
        'dropout': args.dropout,
        'ano_k_local': args.ano_k_local,
        'use_ano': use_ano,
        'use_dqas': use_dqas
    }

    agent = QuantumAtariAgent_v5(env.action_space.n, quantum_params)

    start_episode = 0
    metrics = {'rewards': [], 'lengths': [], 'losses': []}
    if args.resume and CHECKPOINT_FILE_PATH.exists():
        start_episode, metrics = agent.load_checkpoint(CHECKPOINT_FILE_PATH)
        start_episode += 1

    # Ablation condition label
    if use_ano and use_dqas:
        condition_label = "Full v5 (ANO + DiffQAS)"
    elif use_ano:
        condition_label = "ANO only (fixed RY-CRX circuit)"
    elif use_dqas:
        condition_label = "DiffQAS only (fixed PauliX/Y/Z)"
    else:
        condition_label = "v3 Baseline (fixed circuit + fixed Paulis)"

    print("\n" + "=" * 80)
    print(f"QUANTUM TRANSFORMER V5 - {args.env}")
    print(f"Ablation: {condition_label}")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Game:            {args.env}")
    print(f"  Action Space:    {env.action_space.n} actions")
    print(f"  Qubits:          {args.n_qubits}")
    print(f"  Layers:          {args.n_layers}")
    print(f"  QSVT Degree:     {args.degree}")
    print(f"  Device:          {args.device}")
    print(f"  ANO:             {'ON (k={}, windows={}, lr={})'.format(args.ano_k_local, args.n_qubits, args.ano_lr) if use_ano else 'OFF (fixed PauliX/Y/Z)'}")
    print(f"  DiffQAS:         {'ON (arch_lr={}, search_ep={})'.format(args.arch_lr, args.search_episodes) if use_dqas else 'OFF (fixed RY-CRX)'}")
    if use_dqas:
        print(f"  Current phase:   {agent.phase}")
    print(f"  v3 Base:         Sinusoidal PE + 2pi angle scaling")
    print(f"\nSave Directory: {SAVE_DIR}")
    print("=" * 80 + "\n")

    for episode in range(start_episode, args.num_episodes):
        # Phase transition check (only when DiffQAS is enabled)
        if use_dqas and agent.phase == 1 and episode >= args.search_episodes:
            agent.phase_transition()

        state, info = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        metrics['rewards'].append(episode_reward)
        metrics['lengths'].append(step + 1)
        metrics['losses'].append(np.mean(episode_loss) if episode_loss else 0.0)

        if episode % 10 == 0:
            avg_reward = np.mean(metrics['rewards'][-100:])
            avg_length = np.mean(metrics['lengths'][-100:])
            avg_loss = np.mean(metrics['losses'][-100:])
            phase_str = f"P{agent.phase}"
            print(
                f"Ep {episode:>4} [{phase_str}] | R: {episode_reward:>7.1f} | "
                f"AvgR: {avg_reward:>7.1f} | L: {step+1:>4} | "
                f"Loss: {avg_loss:>6.4f} | e: {agent.exploration_rate:.3f}"
            )

        if (episode + 1) % args.save_every == 0 or episode == args.num_episodes - 1:
            agent.save_checkpoint(episode, metrics, CHECKPOINT_FILE_PATH)
            plot_training_curves(metrics, SAVE_DIR)

    env.close()
    print("\nTraining finished!\n")


def plot_training_curves(metrics, save_dir):
    """Plot training curves."""
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


if __name__ == '__main__':
    train()
