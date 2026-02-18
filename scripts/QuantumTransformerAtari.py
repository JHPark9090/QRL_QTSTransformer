"""
Quantum Time-Series Transformer for Atari Environments
=======================================================
This script implements a quantum transformer-based RL agent for classic Atari games
using hybrid CNN + Quantum architecture.

Supported Games:
- DonkeyKong-v5
- Pacman-v5 (NOT Ms. Pacman)
- MarioBros-v5
- SpaceInvaders-v5
- Tetris-v5

Architecture:
  Atari Frame (210×160 RGB) → Preprocessing → 4 Stacked Frames (4×84×84)
  → CNN Feature Extractor → Quantum Transformer → Q-values

Note: Uses gymnasium (modern gym) with ale-py for Atari environments.
"""

import torch
from torch import nn
from torchvision import transforms as T
import numpy as np
from pathlib import Path
from collections import deque
import random, time, datetime, os, signal, sys
import argparse
from math import log2

import pennylane as qml
import matplotlib.pyplot as plt

# Use gymnasium (modern replacement for gym) with ale-py
try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    print("✓ Using gymnasium with ale-py for Atari environments")
except ImportError as e:
    print(f"❌ ERROR: {e}")
    print("Install required packages:")
    print("  pip install gymnasium ale-py gymnasium[atari] gymnasium[accept-rom-license]")
    raise


# ================================================================================
# ARGUMENT PARSER
# ================================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description='Quantum Transformer for Atari Games'
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
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate (reduced from 0.00025 for stability)")
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
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Base directory for saving checkpoints and results")

    # === NEW: Advanced DQN Features ===
    # Prioritized Experience Replay
    parser.add_argument("--use-per", action="store_true",
                        help="Use Prioritized Experience Replay")
    parser.add_argument("--per-alpha", type=float, default=0.6,
                        help="PER priority exponent (0=uniform, 1=full priority)")
    parser.add_argument("--per-beta-start", type=float, default=0.4,
                        help="PER initial importance sampling weight")
    parser.add_argument("--per-beta-frames", type=int, default=100000,
                        help="Frames over which to anneal beta to 1.0")
    parser.add_argument("--per-epsilon", type=float, default=1e-6,
                        help="Small constant added to priorities")

    # Early Stopping
    parser.add_argument("--early-stopping", action="store_true",
                        help="Enable early stopping based on reward plateau")
    parser.add_argument("--patience", type=int, default=500,
                        help="Episodes without improvement before stopping")
    parser.add_argument("--min-episodes", type=int, default=1000,
                        help="Minimum episodes before early stopping can trigger")

    # Target Network Updates
    parser.add_argument("--target-update", type=str, default="hard",
                        choices=["hard", "soft"],
                        help="Target network update type")
    parser.add_argument("--tau", type=float, default=0.001,
                        help="Soft update coefficient (only used if --target-update=soft)")
    parser.add_argument("--sync-every", type=int, default=1000,
                        help="Steps between hard target updates")

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
    print(f"✓ Global seeds set to: {seed_value}")


# ================================================================================
# CHECKPOINT DIRECTORIES
# ================================================================================
env_name_clean = args.env.replace("-", "").replace("v", "").replace("ALE/", "")
RUN_ID = f"QTransformer_{env_name_clean}_Q{args.n_qubits}_L{args.n_layers}_D{args.degree}_Run{args.log_index}"
if args.save_dir:
    CHECKPOINT_BASE_DIR = Path(args.save_dir)
else:
    CHECKPOINT_BASE_DIR = Path("AtariCheckpoints")
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
    - Resizes to 84×84
    - Normalizes pixel values
    """
    def __init__(self, env, grayscale=True, resize_shape=(84, 84), normalize=True):
        super().__init__(env)
        self.grayscale = grayscale
        self.resize_shape = resize_shape
        self.normalize = normalize

        # Update observation space
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
        # Convert to tensor
        obs = torch.from_numpy(obs).permute(2, 0, 1).float()

        # Grayscale conversion
        if self.grayscale:
            obs = T.Grayscale()(obs)

        # Resize
        obs = T.Resize(self.resize_shape, antialias=True)(obs)

        # Normalize
        if self.normalize:
            obs = obs / 255.0

        return obs.squeeze(0).numpy()  # Return as numpy for gym compatibility


class FrameStack(gym.Wrapper):
    """Stack consecutive frames for temporal information."""
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Update observation space
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # gymnasium returns (obs, info)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)  # gymnasium API
        self.frames.append(obs)
        # Combine terminated and truncated into done for compatibility
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
    # Create base environment
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)

    # Apply wrappers
    env = SkipFrame(env, skip=4)
    env = AtariPreprocessing(env, grayscale=True, resize_shape=(84, 84), normalize=True)
    env = FrameStack(env, num_stack=4)

    return env


env = create_atari_env(args.env, render_mode='human' if args.render else None)


# ================================================================================
# QUANTUM CIRCUIT COMPONENTS
# ================================================================================
def sim14_circuit(params, wires, layers=1):
    """Sim14 quantum circuit ansatz."""
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


def apply_unitaries_pl(base_states, unitary_params, qnode_state, coeffs):
    """Apply linear combination of unitaries (fully vectorized)."""
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
    """QSVT polynomial state preparation."""
    acc = poly_coeffs[0] * base_states
    working_register = base_states

    for c in poly_coeffs[1:]:
        working_register = apply_unitaries_pl(
            working_register, unitary_params, qnode_state, lcu_coeffs
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

        # Convolutional layers (Nature DQN architecture)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate conv output size: 4×84×84 → 32×20×20 → 64×9×9 → 64×7×7
        conv_output_size = 64 * 7 * 7

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim * 4)  # output_dim features × 4 frames
        )
        self.output_dim = output_dim

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        batch_size = x.shape[0]
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        # Reshape to (batch, 4, output_dim) for time-series format
        x = x.view(batch_size, 4, self.output_dim)
        return x


# ================================================================================
# QUANTUM TIME-SERIES TRANSFORMER
# ================================================================================
class QuantumTSTransformerRL(nn.Module):
    """Quantum Time-Series Transformer for Atari."""
    def __init__(self, n_qubits: int, n_timesteps: int, degree: int,
                 n_ansatz_layers: int, feature_dim: int, output_dim: int,
                 dropout: float, device):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.device = device

        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1

        # Classical layers
        self.feature_projection = nn.Linear(feature_dim, self.n_rots)
        self.dropout = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()
        self.output_ff = nn.Sequential(
            nn.Linear(3 * n_qubits, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        # Trainable quantum parameters
        self.n_poly_coeffs = self.degree + 1
        self.poly_coeffs = nn.Parameter(torch.rand(self.n_poly_coeffs))
        self.mix_coeffs = nn.Parameter(
            torch.rand(self.n_timesteps, dtype=torch.complex64)
        )
        self.qff_params = nn.Parameter(torch.rand(self.qff_n_rots))

        # PennyLane device and QNodes
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _timestep_state_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            sim14_circuit(params, wires=self.n_qubits, layers=self.n_ansatz_layers)
            return qml.state()

        self.timestep_state_qnode = _timestep_state_qnode

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _qff_qnode_expval(initial_state, params):
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            sim14_circuit(params, wires=self.n_qubits, layers=1)
            observables = ([qml.PauliX(i) for i in range(self.n_qubits)] +
                          [qml.PauliY(i) for i in range(self.n_qubits)] +
                          [qml.PauliZ(i) for i in range(self.n_qubits)])
            return [qml.expval(op) for op in observables]

        self.qff_qnode_expval = _qff_qnode_expval

    def forward(self, x):
        # x: (batch, n_timesteps, feature_dim)
        bsz = x.shape[0]

        x = self.feature_projection(self.dropout(x))
        timestep_params = self.rot_sigm(x)

        base_states = torch.zeros(
            bsz, 2 ** self.n_qubits,
            dtype=torch.complex64,
            device=self.device
        )
        base_states[:, 0] = 1.0

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

        exps = self.qff_qnode_expval(
            initial_state=normalized_mixed_timestep,
            params=self.qff_params
        )

        exps = torch.stack(exps, dim=1)
        exps = exps.float()
        op = self.output_ff(exps)
        return op


# ================================================================================
# HYBRID NETWORK
# ================================================================================
class QuantumTransformerAtariNet(nn.Module):
    """Complete Q-Network: CNN + Quantum Transformer."""
    def __init__(self, action_dim, device, quantum_params):
        super().__init__()

        self.action_dim = action_dim
        self.device = device

        # CNN feature extractor
        self.cnn_extractor = AtariCNNFeatureExtractor(
            output_dim=quantum_params['feature_dim']
        )

        # Build online and target networks
        self.online = self._build_quantum_network(quantum_params)
        self.target = self._build_quantum_network(quantum_params)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        for p in self.target.parameters():
            p.requires_grad = False

    def _build_quantum_network(self, qparams):
        """Build complete quantum network pipeline."""
        return nn.Sequential(
            self.cnn_extractor,
            QuantumTSTransformerRL(
                n_qubits=qparams['n_qubits'],
                n_timesteps=qparams['n_timesteps'],
                degree=qparams['degree'],
                n_ansatz_layers=qparams['n_ansatz_layers'],
                feature_dim=qparams['feature_dim'],
                output_dim=self.action_dim,
                dropout=qparams['dropout'],
                device=self.device
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
# PRIORITIZED EXPERIENCE REPLAY
# ================================================================================
class SumTree:
    """
    Binary sum tree for O(log N) priority-based sampling.
    Used for Prioritized Experience Replay (PER).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find leaf node for given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return total priority sum."""
        return self.tree[0]

    def add(self, priority, data):
        """Add experience with given priority."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """Update priority at given index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get experience for given cumulative sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using Sum Tree.
    Samples experiences based on TD-error priority.

    Reference: Schaul et al. (2016) "Prioritized Experience Replay"
    """
    def __init__(self, capacity, state_shape, device, alpha=0.6, beta_start=0.4,
                 beta_frames=100000, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        self.state_shape = state_shape

        # PER hyperparameters
        self.alpha = alpha  # Priority exponent (0=uniform, 1=full priority)
        self.beta = beta_start  # Importance sampling weight
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon  # Small constant to ensure non-zero priority
        self.max_priority = 1.0

        self.frame_count = 0

    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority (will be updated after learning)."""
        experience = (
            state.copy() if isinstance(state, np.ndarray) else state,
            action,
            reward,
            next_state.copy() if isinstance(next_state, np.ndarray) else next_state,
            done
        )
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """Sample batch with priority-based probability."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Anneal beta towards 1.0
        self.frame_count += 1
        self.beta = min(1.0, self.beta_start +
                        (1.0 - self.beta_start) * self.frame_count / self.beta_frames)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            if data is None:
                # Fallback to random sampling if data is None
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        probs = np.array(priorities) / (self.tree.total() + 1e-10)
        weights = (self.tree.n_entries * probs + 1e-10) ** (-self.beta)
        weights = weights / (weights.max() + 1e-10)  # Normalize

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.BoolTensor(dones).to(self.device),
            torch.FloatTensor(weights).to(self.device),
            indices
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ================================================================================
# DQN AGENT
# ================================================================================
class QuantumAtariAgent:
    """DQN agent for Atari using quantum transformer with advanced features."""
    def __init__(self, action_dim, quantum_params):
        self.action_dim = action_dim
        self.device = args.device
        self.use_per = args.use_per

        # Q-Networks
        self.net = QuantumTransformerAtariNet(
            action_dim, self.device, quantum_params
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Per-sample loss for PER

        # Replay buffer (standard or prioritized)
        state_shape = (4, 84, 84)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=args.memory_size,
                state_shape=state_shape,
                device=self.device,
                alpha=args.per_alpha,
                beta_start=args.per_beta_start,
                beta_frames=args.per_beta_frames,
                epsilon=args.per_epsilon
            )
            print(f"✓ Using Prioritized Experience Replay (α={args.per_alpha}, β₀={args.per_beta_start})")
        else:
            self.memory = ReplayBuffer(args.memory_size, state_shape, self.device)

        # Exploration
        self.exploration_rate = args.exploration_rate_start
        self.exploration_rate_decay = args.exploration_rate_decay
        self.exploration_rate_min = args.exploration_rate_min

        # Training counters
        self.curr_step = 0
        self.sync_every = args.sync_every
        self.burnin = 10000

        # Target update mode
        self.target_update_mode = args.target_update
        self.tau = args.tau
        if self.target_update_mode == "soft":
            print(f"✓ Using soft target updates (τ={self.tau})")

        # Early stopping tracking
        self.best_avg_reward = float('-inf')
        self.best_episode = 0
        self.episodes_without_improvement = 0

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
        """Perform one learning step with PER and soft/hard target updates."""
        # Increment step counter first (before any checks)
        self.curr_step += 1

        if len(self.memory) < self.burnin:
            return None

        if self.curr_step % args.learn_step != 0:
            return None

        # Sample batch (different return format for PER)
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = \
                self.memory.sample(args.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample(args.batch_size)
            weights = torch.ones(args.batch_size, device=self.device)
            indices = None

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

        # Compute TD errors for PER priority update
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # Compute weighted loss (importance sampling for PER)
        element_wise_loss = self.loss_fn(current_q, target_q)
        loss = (element_wise_loss * weights).mean()

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        # Update PER priorities
        if self.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors)

        # Decay exploration
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay
        )

        # Update target network (soft or hard)
        if self.target_update_mode == "soft":
            # Soft update: θ_target = τ*θ_online + (1-τ)*θ_target
            for target_param, online_param in zip(
                self.net.target.parameters(), self.net.online.parameters()
            ):
                target_param.data.copy_(
                    self.tau * online_param.data + (1.0 - self.tau) * target_param.data
                )
        else:
            # Hard update every sync_every steps
            if self.curr_step % self.sync_every == 0:
                self.net.target.load_state_dict(self.net.online.state_dict())

        return loss.item()

    def update_best_model(self, episode, avg_reward, save_dir):
        """
        Track best model based on average reward.
        Returns True if this is a new best, False otherwise.
        """
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            self.best_episode = episode
            self.episodes_without_improvement = 0

            # Save best model checkpoint
            best_path = save_dir / "best_checkpoint.chkpt"
            best_checkpoint = {
                'episode': episode,
                'curr_step': self.curr_step,
                'net_state': self.net.state_dict(),
                'best_avg_reward': avg_reward,
            }
            torch.save(best_checkpoint, best_path)
            print(f"★ New best model! AvgR: {avg_reward:.1f} at Episode {episode}")
            return True
        else:
            self.episodes_without_improvement += 1
            return False

    def should_stop_early(self):
        """Check if training should stop based on patience."""
        return self.episodes_without_improvement >= args.patience

    def save_checkpoint(self, episode, metrics, checkpoint_path):
        """Save checkpoint."""
        checkpoint = {
            'episode': episode,
            'curr_step': self.curr_step,
            'net_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'metrics': metrics,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            # Early stopping state
            'best_avg_reward': self.best_avg_reward,
            'best_episode': self.best_episode,
            'episodes_without_improvement': self.episodes_without_improvement,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved (Episode {episode})")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        if not checkpoint_path.exists():
            print(f"✗ No checkpoint found. Starting from scratch.")
            return 0, {}

        print(f"✓ Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.net.load_state_dict(checkpoint['net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['curr_step']

        # Restore early stopping state
        self.best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
        self.best_episode = checkpoint.get('best_episode', 0)
        self.episodes_without_improvement = checkpoint.get('episodes_without_improvement', 0)

        torch_rng_state = checkpoint['torch_rng_state']
        if torch_rng_state.device != torch.device('cpu'):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        return checkpoint['episode'], checkpoint.get('metrics', {})


# ================================================================================
# TRAINING
# ================================================================================
def train():
    """Main training loop."""
    set_global_seeds(args.seed)

    quantum_params = {
        'n_qubits': args.n_qubits,
        'n_timesteps': args.n_timesteps,
        'degree': args.degree,
        'n_ansatz_layers': args.n_layers,
        'feature_dim': args.feature_dim,
        'dropout': args.dropout
    }

    agent = QuantumAtariAgent(env.action_space.n, quantum_params)

    start_episode = 0
    metrics = {'rewards': [], 'lengths': [], 'losses': []}
    if args.resume and CHECKPOINT_FILE_PATH.exists():
        start_episode, metrics = agent.load_checkpoint(CHECKPOINT_FILE_PATH)
        start_episode += 1

    print("\n" + "="*80)
    print(f"QUANTUM TRANSFORMER - {args.env}")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"  • Game:          {args.env}")
    print(f"  • Action Space:  {env.action_space.n} actions")
    print(f"  • Qubits:        {args.n_qubits}")
    print(f"  • Layers:        {args.n_layers}")
    print(f"  • QSVT Degree:   {args.degree}")
    print(f"  • Learning Rate: {args.lr}")
    print(f"  • Device:        {args.device}")
    print(f"\n🔧 Advanced Features:")
    print(f"  • PER:           {'Enabled' if args.use_per else 'Disabled'}")
    print(f"  • Early Stopping: {'Enabled (patience=' + str(args.patience) + ')' if args.early_stopping else 'Disabled'}")
    print(f"  • Target Update: {args.target_update}" + (f" (τ={args.tau})" if args.target_update == "soft" else f" (every {args.sync_every} steps)"))
    print(f"\n💾 Save Directory: {SAVE_DIR}")
    print(f"   Checkpoint every {args.save_every} episodes")
    print("="*80 + "\n")

    # ================================================================================
    # SIGNAL HANDLING FOR GRACEFUL SHUTDOWN
    # ================================================================================
    current_episode = [start_episode]

    def save_checkpoint_on_signal(signum, frame):
        """Handle SIGTERM/SIGINT by saving checkpoint before exit."""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n\n{'='*80}")
        print(f"⚠️  {signal_name} received! Saving checkpoint before exit...")
        print(f"{'='*80}")
        try:
            agent.save_checkpoint(current_episode[0], metrics, CHECKPOINT_FILE_PATH)
            plot_training_curves(metrics, SAVE_DIR)
            print(f"✓ Emergency checkpoint saved at episode {current_episode[0]}")
            print(f"  To resume: python QuantumTransformerAtari.py --env={args.env} --resume --save-dir={CHECKPOINT_BASE_DIR}")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
        print(f"{'='*80}\n")
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_checkpoint_on_signal)
    signal.signal(signal.SIGINT, save_checkpoint_on_signal)
    print(f"✓ Signal handlers registered (SIGTERM, SIGINT)\n")

    for episode in range(start_episode, args.num_episodes):
        current_episode[0] = episode
        state, info = env.reset()  # gymnasium returns (obs, info)
        episode_reward = 0
        episode_loss = []

        for step in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)  # gymnasium API
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

        # Compute average reward for logging and early stopping
        avg_reward = np.mean(metrics['rewards'][-100:])

        if episode % 10 == 0:
            avg_length = np.mean(metrics['lengths'][-100:])
            avg_loss = np.mean(metrics['losses'][-100:])
            print(
                f"Ep {episode:>4} | R: {episode_reward:>7.1f} | "
                f"AvgR: {avg_reward:>7.1f} | L: {step+1:>4} | "
                f"Loss: {avg_loss:>6.4f} | ε: {agent.exploration_rate:.3f}",
                flush=True
            )

        # Update best model tracking (after min_episodes)
        if episode >= args.min_episodes:
            agent.update_best_model(episode, avg_reward, SAVE_DIR)

            # Check early stopping
            if args.early_stopping and agent.should_stop_early():
                print(f"\n{'='*80}")
                print(f"⏹️  Early stopping triggered!")
                print(f"   No improvement for {args.patience} episodes")
                print(f"   Best AvgR: {agent.best_avg_reward:.1f} at Episode {agent.best_episode}")
                print(f"{'='*80}\n")
                agent.save_checkpoint(episode, metrics, CHECKPOINT_FILE_PATH)
                plot_training_curves(metrics, SAVE_DIR)
                break

        if (episode + 1) % args.save_every == 0 or episode == args.num_episodes - 1:
            agent.save_checkpoint(episode, metrics, CHECKPOINT_FILE_PATH)
            plot_training_curves(metrics, SAVE_DIR)

    env.close()
    print("\n🏁 Training finished!")
    if agent.best_avg_reward > float('-inf'):
        print(f"   Best AvgR: {agent.best_avg_reward:.1f} at Episode {agent.best_episode}")
        print(f"   Best model saved to: {SAVE_DIR / 'best_checkpoint.chkpt'}")
    print()


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
