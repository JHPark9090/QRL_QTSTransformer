"""
Quantum Time-Series Transformer for Atari Environments — v4 (Multi-Circuit)
=============================================================================
Multi-circuit architecture: CNN features are split into K chunks, each processed
by an independent quantum circuit with minimal classical projection.

v4 improvements over v3:
  1. Multi-circuit (K chips): CNN output (batch, 4, 128) is split into K chunks
     of (batch, 4, chunk_size), each processed by an independent quantum circuit.
  2. Minimal classical head: Raw quantum expectation values are concatenated and
     mapped directly to actions via a single linear layer.
  3. Matches v3 transformer param count (~12K) while distributing computation
     across K independent quantum circuits.

Architecture:
  Atari Frame (210x160 RGB) -> Preprocessing -> 4 Stacked Frames (4x84x84)
  -> CNN Feature Extractor -> chunk(K) -> K x Quantum Circuit -> cat -> Linear -> Q-values

Note: Uses gymnasium (modern gym) with ale-py for Atari environments.
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
        description='Quantum Transformer v4 (Multi-Circuit) for Atari Games'
    )

    # Environment selection
    parser.add_argument("--env", type=str, default="ALE/DonkeyKong-v5",
                        choices=["ALE/DonkeyKong-v5", "ALE/Pacman-v5", "ALE/MarioBros-v5",
                                "ALE/SpaceInvaders-v5", "ALE/Tetris-v5",
                                "ALE/Breakout-v5", "ALE/Pong-v5"],
                        help="Atari game to train on (gymnasium v5 naming)")

    # Quantum parameters
    parser.add_argument("--n-qubits", type=int, default=8,
                        help="Number of qubits per quantum circuit")
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

    # Multi-circuit parameters
    parser.add_argument("--n-chips", type=int, default=16,
                        help="Number of parallel quantum circuits")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Feature chunk size per circuit (should equal n_qubits)")

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

# Validate multi-circuit constraints
assert args.feature_dim == args.n_chips * args.chunk_size, \
    f"feature_dim ({args.feature_dim}) must equal n_chips ({args.n_chips}) * chunk_size ({args.chunk_size})"
assert args.chunk_size == args.n_qubits, \
    f"chunk_size ({args.chunk_size}) must equal n_qubits ({args.n_qubits})"


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
RUN_ID = f"QTransformerV4_{env_name_clean}_Q{args.n_qubits}_L{args.n_layers}_D{args.degree}_K{args.n_chips}_Run{args.log_index}"
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

        # Calculate conv output size: 4x84x84 -> 32x20x20 -> 64x9x9 -> 64x7x7
        conv_output_size = 64 * 7 * 7

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim * 4)  # output_dim features x 4 frames
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
# QUANTUM TIME-SERIES TRANSFORMER CORE (per-circuit, no output_ff)
# ================================================================================
class QuantumTSTransformerRL_Core(nn.Module):
    """Single quantum circuit core for multi-circuit architecture.

    Same QSVT/QFF logic as v3's QuantumTSTransformerRL but:
      - feature_projection maps chunk_size -> n_rots (small projection)
      - PE adapts to chunk_size
      - No output_ff: returns raw expectation values (3 * n_qubits)
    """
    def __init__(self, n_qubits: int, n_timesteps: int, degree: int,
                 n_ansatz_layers: int, chunk_size: int,
                 dropout: float, device):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.device = device

        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1

        # Classical layers — small projection from chunk_size to n_rots
        self.feature_projection = nn.Linear(chunk_size, self.n_rots)
        self.dropout = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()

        # Sinusoidal Positional Encoding adapted to chunk_size
        pe = torch.zeros(n_timesteps, chunk_size)
        pos = torch.arange(n_timesteps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, chunk_size, 2).float()
                        * -(math.log(10000.0) / chunk_size))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:chunk_size // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, chunk_size)

        # Trainable quantum parameters
        self.n_poly_coeffs = self.degree + 1
        self.poly_coeffs = nn.Parameter(torch.rand(self.n_poly_coeffs))
        self.mix_coeffs = nn.Parameter(
            torch.rand(self.n_timesteps, dtype=torch.complex64)
        )
        self.qff_params = nn.Parameter(torch.rand(self.qff_n_rots))

        # PennyLane device and QNodes
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Capture local vars for closure
        _n_qubits = self.n_qubits
        _n_ansatz_layers = self.n_ansatz_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _timestep_state_qnode(initial_state, params):
            qml.StatePrep(initial_state, wires=range(_n_qubits))
            sim14_circuit(params, wires=_n_qubits, layers=_n_ansatz_layers)
            return qml.state()

        self.timestep_state_qnode = _timestep_state_qnode

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
        # x: (batch, n_timesteps, chunk_size)
        bsz = x.shape[0]

        # Positional encoding injection
        x = x + self.pe[:, :x.size(1)]

        x = self.feature_projection(self.dropout(x))
        # 2pi angle scaling: full [-pi, pi] rotation range centered at 0
        timestep_params = (self.rot_sigm(x) - 0.5) * (2 * math.pi)

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
        # Return raw expectation values: (batch, 3 * n_qubits)
        return exps


# ================================================================================
# MULTI-CIRCUIT QUANTUM TRANSFORMER
# ================================================================================
class MultiCircuitQTSTransformer(nn.Module):
    """Multi-circuit Quantum Time-Series Transformer.

    Splits CNN features into K chunks, processes each with an independent
    quantum circuit, concatenates results, and maps to actions.
    """
    def __init__(self, n_qubits: int, n_timesteps: int, degree: int,
                 n_ansatz_layers: int, feature_dim: int, output_dim: int,
                 dropout: float, device, n_chips: int, chunk_size: int):
        super().__init__()

        self.n_chips = n_chips
        self.chunk_size = chunk_size
        self.n_qubits = n_qubits

        # K independent quantum circuits
        self.circuits = nn.ModuleList([
            QuantumTSTransformerRL_Core(
                n_qubits=n_qubits,
                n_timesteps=n_timesteps,
                degree=degree,
                n_ansatz_layers=n_ansatz_layers,
                chunk_size=chunk_size,
                dropout=dropout,
                device=device
            )
            for _ in range(n_chips)
        ])

        # Minimal output head: concatenated expectation values -> actions
        total_expvals = n_chips * 3 * n_qubits
        self.output_head = nn.Linear(total_expvals, output_dim)

    def forward(self, x):
        # x: (batch, n_timesteps, feature_dim)
        chunks = torch.chunk(x, self.n_chips, dim=2)  # K chunks of (batch, T, chunk_size)

        circuit_outputs = []
        for i, chunk in enumerate(chunks):
            out = self.circuits[i](chunk)  # (batch, 3 * n_qubits)
            circuit_outputs.append(out)

        # Concatenate all circuit outputs
        combined = torch.cat(circuit_outputs, dim=1)  # (batch, K * 3 * n_qubits)

        # Single linear layer to actions
        return self.output_head(combined)


# ================================================================================
# HYBRID NETWORK
# ================================================================================
class QuantumTransformerAtariNet_V4(nn.Module):
    """Complete Q-Network: CNN + Multi-Circuit Quantum Transformer."""
    def __init__(self, action_dim, device, quantum_params):
        super().__init__()

        self.action_dim = action_dim
        self.device = device

        # Build online and target networks (each with its own CNN)
        self.online = self._build_network(quantum_params)
        self.target = self._build_network(quantum_params)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        for p in self.target.parameters():
            p.requires_grad = False

    def _build_network(self, qparams):
        """Build complete quantum network pipeline with its own CNN."""
        return nn.Sequential(
            AtariCNNFeatureExtractor(
                output_dim=qparams['feature_dim']
            ),
            MultiCircuitQTSTransformer(
                n_qubits=qparams['n_qubits'],
                n_timesteps=qparams['n_timesteps'],
                degree=qparams['degree'],
                n_ansatz_layers=qparams['n_ansatz_layers'],
                feature_dim=qparams['feature_dim'],
                output_dim=self.action_dim,
                dropout=qparams['dropout'],
                device=self.device,
                n_chips=qparams['n_chips'],
                chunk_size=qparams['chunk_size']
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
# DQN AGENT
# ================================================================================
class QuantumAtariAgent_V4:
    """DQN agent for Atari using multi-circuit quantum transformer."""
    def __init__(self, action_dim, quantum_params):
        self.action_dim = action_dim
        self.device = args.device

        # Q-Networks
        self.net = QuantumTransformerAtariNet_V4(
            action_dim, self.device, quantum_params
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
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
        """Perform one learning step."""
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

        # Update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        # Decay exploration
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay
        )

        # Sync target network
        if self.curr_step % self.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        return loss.item()

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
            'python_rng_state': random.getstate()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved (Episode {episode})")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        if not checkpoint_path.exists():
            print(f"No checkpoint found. Starting from scratch.")
            return 0, {}

        print(f"Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.net.load_state_dict(checkpoint['net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['curr_step']

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
        'dropout': args.dropout,
        'n_chips': args.n_chips,
        'chunk_size': args.chunk_size
    }

    agent = QuantumAtariAgent_V4(env.action_space.n, quantum_params)

    # Parameter count reporting
    cnn_params = sum(p.numel() for p in agent.net.online[0].parameters())
    multi_circuit = agent.net.online[1]
    total_circuit_params = 0
    total_proj_params = 0
    total_quantum_params = 0
    for i, circuit in enumerate(multi_circuit.circuits):
        proj_p = sum(p.numel() for p in circuit.feature_projection.parameters())
        q_p = circuit.poly_coeffs.numel() + circuit.mix_coeffs.numel() + circuit.qff_params.numel()
        total_proj_params += proj_p
        total_quantum_params += q_p
        total_circuit_params += sum(p.numel() for p in circuit.parameters())
    output_head_params = sum(p.numel() for p in multi_circuit.output_head.parameters())
    transformer_total = sum(p.numel() for p in multi_circuit.parameters())
    total_online = sum(p.numel() for p in agent.net.online.parameters())

    start_episode = 0
    metrics = {'rewards': [], 'lengths': [], 'losses': []}
    if args.resume and CHECKPOINT_FILE_PATH.exists():
        start_episode, metrics = agent.load_checkpoint(CHECKPOINT_FILE_PATH)
        start_episode += 1

    print("\n" + "="*80)
    print(f"QUANTUM TRANSFORMER V4 (MULTI-CIRCUIT) - {args.env}")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Game:              {args.env}")
    print(f"  Action Space:      {env.action_space.n} actions")
    print(f"  Qubits:            {args.n_qubits}")
    print(f"  Layers:            {args.n_layers}")
    print(f"  QSVT Degree:       {args.degree}")
    print(f"  N Chips (K):       {args.n_chips}")
    print(f"  Chunk Size:        {args.chunk_size}")
    print(f"  Feature Dim:       {args.feature_dim}")
    print(f"  Device:            {args.device}")
    print(f"  v4 Features:       Multi-circuit split + minimal head")
    print(f"\nParameter Counts (online network):")
    print(f"  CNN params:          {cnn_params:>12,}")
    print(f"  feature_projection:  {total_proj_params:>12,} ({args.n_chips}x Linear({args.chunk_size},{4*args.n_qubits*args.n_layers}))")
    print(f"  Quantum params:      {total_quantum_params:>12,} ({args.n_chips}x {total_quantum_params//args.n_chips})")
    print(f"  Output head:         {output_head_params:>12,} (Linear({args.n_chips*3*args.n_qubits},{env.action_space.n}))")
    print(f"  Transformer total:   {transformer_total:>12,}")
    print(f"  Total (online):      {total_online:>12,}")
    print(f"\nSave Directory: {SAVE_DIR}")
    print("="*80 + "\n")

    for episode in range(start_episode, args.num_episodes):
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

        if episode % 10 == 0:
            avg_reward = np.mean(metrics['rewards'][-100:])
            avg_length = np.mean(metrics['lengths'][-100:])
            avg_loss = np.mean(metrics['losses'][-100:])
            print(
                f"Ep {episode:>4} | R: {episode_reward:>7.1f} | "
                f"AvgR: {avg_reward:>7.1f} | L: {step+1:>4} | "
                f"Loss: {avg_loss:>6.4f} | eps: {agent.exploration_rate:.3f}"
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
