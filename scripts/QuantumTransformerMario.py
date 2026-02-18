"""
Quantum Time-Series Transformer for Super Mario Bros Reinforcement Learning
============================================================================
This script implements a hybrid quantum-classical RL agent that uses:
1. CNN Feature Extractor: Extracts spatial features from game frames
2. Quantum Transformer: Processes temporal sequences using QSVT
3. DQN Framework: Double Q-learning with experience replay

Architecture Flow:
  Input: 4 stacked frames (4×84×84)
    ↓
  CNN Feature Extractor
    ↓
  Quantum Time-Series Transformer (QSVT)
    ↓
  Q-values for actions
    ↓
  DQN Training Loop
"""

import torch
from torch import nn
from torchvision import transforms as T
import numpy as np
from pathlib import Path
import random, time, datetime, os, signal, sys
import argparse
from math import log2

import pennylane as qml
import matplotlib.pyplot as plt

# Gym for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# Super Mario environment
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


# ================================================================================
# ARGUMENT PARSER
# ================================================================================
def get_args():
    parser = argparse.ArgumentParser(description='Quantum Transformer for Super Mario RL')

    # Quantum parameters
    parser.add_argument("--n-qubits", type=int, default=6,
                        help="Number of qubits in quantum circuit")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of ansatz layers in quantum circuit")
    parser.add_argument("--degree", type=int, default=2,
                        help="Degree of QSVT polynomial")
    parser.add_argument("--n-timesteps", type=int, default=4,
                        help="Number of timesteps (must match frame stack)")
    parser.add_argument("--feature-dim", type=int, default=128,
                        help="Feature dimension after CNN extraction")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # RL parameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--exploration-rate-decay', type=float, default=0.99999975)
    parser.add_argument('--gamma', type=float, default=0.9,
                        help="Discount factor")
    parser.add_argument('--learn-step', type=int, default=3,
                        help="Learn every N steps")
    parser.add_argument('--num-episodes', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=0.00025,
                        help="Learning rate")

    # Training parameters
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log-index", type=int, default=1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Base directory for saving checkpoints and results")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N episodes (default: 5)")

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
RUN_ID = f"QTransformerMario_Q{args.n_qubits}_L{args.n_layers}_D{args.degree}_Run{args.log_index}"
if args.save_dir:
    CHECKPOINT_BASE_DIR = Path(args.save_dir)
else:
    CHECKPOINT_BASE_DIR = Path("SuperMarioCheckpoints")
SAVE_DIR = CHECKPOINT_BASE_DIR / RUN_ID
REPLAY_BUFFER_DIR = SAVE_DIR / "replay_buffer"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPLAY_BUFFER_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE_PATH = SAVE_DIR / "latest_checkpoint.chkpt"


# ================================================================================
# ENVIRONMENT SETUP
# ================================================================================
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb',
                                     apply_api_compatibility=True)

# Limit action space to: 0. walk right, 1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])


class SkipFrame(gym.Wrapper):
    """Skip frames for faster training."""
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB frames to grayscale."""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """Resize frames to 84x84."""
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
            T.Normalize(0, 255)
        ])
        observation = transforms(observation).squeeze(0)
        return observation


# Apply environment wrappers
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)

if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

env.reset()


# ================================================================================
# QUANTUM CIRCUIT COMPONENTS
# ================================================================================
def sim14_circuit(params, wires, layers=1):
    """
    Implements the 'sim14' circuit from Sim et al. (2019) using PennyLane.
    Batch-aware and handles both 1D and 2D parameter tensors.
    """
    is_batched = params.ndim == 2
    param_idx = 0

    for _ in range(layers):
        # Layer 1: RY rotations
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        # Layer 2: CRX gates (forward)
        for i in range(wires - 1, -1, -1):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i + 1) % wires])
            param_idx += 1

        # Layer 3: RY rotations
        for i in range(wires):
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.RY(angle, wires=i)
            param_idx += 1

        # Layer 4: CRX gates (counter)
        wire_order = [wires - 1] + list(range(wires - 1))
        for i in wire_order:
            angle = params[:, param_idx] if is_batched else params[param_idx]
            qml.CRX(angle, wires=[i, (i - 1) % wires])
            param_idx += 1


def apply_unitaries_pl(base_states, unitary_params, qnode_state, coeffs):
    """
    Applies a linear combination of unitaries to a batch of states.
    Fully vectorized for maximum efficiency.
    """
    bsz, n_timesteps, n_rots = unitary_params.shape
    n_qbs = int(log2(base_states.shape[1]))

    # Flatten timestep parameters into single batch
    flat_params = unitary_params.reshape(bsz * n_timesteps, n_rots)

    # Repeat base states to match flattened parameters
    repeated_base_states = base_states.repeat_interleave(n_timesteps, dim=0)

    # Execute QNode once with entire batch (highly parallelized)
    evolved_states = qnode_state(
        initial_state=repeated_base_states,
        params=flat_params
    )

    # Reshape results to include timestep dimension
    evolved_states_reshaped = evolved_states.reshape(bsz, n_timesteps, 2**n_qbs)

    # Ensure complex dtype consistency
    evolved_states_reshaped = evolved_states_reshaped.to(torch.complex64)
    coeffs = coeffs.to(torch.complex64)

    # Apply LCU coefficients via efficient tensor multiplication
    lcs = torch.einsum('bti,bt->bi', evolved_states_reshaped, coeffs)

    return lcs


def evaluate_polynomial_state_pl(base_states, unitary_params, qnode_state,
                                  n_qbs, lcu_coeffs, poly_coeffs):
    """Simulates the QSVT polynomial state preparation."""
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
class CNNFeatureExtractor(nn.Module):
    """
    Extracts spatial features from stacked game frames.
    Reduces 4×84×84 images to feature vectors.
    """
    def __init__(self, output_dim=128):
        super().__init__()

        # Convolutional layers (similar to DQN architecture)
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
    """
    Quantum Time-Series Transformer adapted for Reinforcement Learning.
    Uses QSVT (Quantum Singular Value Transformation) for temporal processing.
    """
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
        self.output_ff = nn.Linear(3 * n_qubits, output_dim)

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

        # Project features through quantum rotation parameters
        x = self.feature_projection(self.dropout(x))
        timestep_params = self.rot_sigm(x)

        # Initialize base quantum states (all |0⟩)
        base_states = torch.zeros(
            bsz, 2 ** self.n_qubits,
            dtype=torch.complex64,
            device=self.device
        )
        base_states[:, 0] = 1.0

        # Apply QSVT polynomial transformation
        mixed_timestep = evaluate_polynomial_state_pl(
            base_states,
            timestep_params,
            self.timestep_state_qnode,
            self.n_qubits,
            self.mix_coeffs.repeat(bsz, 1),
            self.poly_coeffs
        )

        # Normalize the quantum state
        norm = torch.linalg.vector_norm(mixed_timestep, dim=1, keepdim=True)
        normalized_mixed_timestep = mixed_timestep / (norm + 1e-9)

        # Measure observables (X, Y, Z on all qubits)
        exps = self.qff_qnode_expval(
            initial_state=normalized_mixed_timestep,
            params=self.qff_params
        )

        exps = torch.stack(exps, dim=1)
        exps = exps.float()

        # Final output layer to Q-values
        op = self.output_ff(exps)
        return op  # (batch, output_dim) - Q-values for each action


# ================================================================================
# HYBRID QUANTUM-CLASSICAL NETWORK
# ================================================================================
class QuantumTransformerMarioNet(nn.Module):
    """
    Complete Q-Network combining CNN feature extraction and
    quantum time-series transformer.
    """
    def __init__(self, input_dim_tuple, output_dim_actions, device, quantum_params):
        super().__init__()

        C, H, W = input_dim_tuple
        self.output_dim_actions = output_dim_actions
        self.device = device

        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            output_dim=quantum_params['feature_dim']
        )

        # Build online and target networks
        self.online = self._build_quantum_network(quantum_params)
        self.target = self._build_quantum_network(quantum_params)

        # Initialize target network with online weights
        self.target.load_state_dict(self.online.state_dict())

        # Freeze target network
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
                output_dim=self.output_dim_actions,
                dropout=qparams['dropout'],
                device=self.device
            )
        )

    def forward(self, state_batch, model):
        """Forward pass through either online or target network."""
        if model == "online":
            return self.online(state_batch)
        else:
            return self.target(state_batch)


# ================================================================================
# MARIO AGENT (DQN)
# ================================================================================
class Mario:
    """
    Mario agent using Double Q-Learning with quantum transformer network.
    """
    def __init__(self, state_dim, action_dim, save_dir, replay_buffer_path,
                 quantum_params):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.replay_buffer_path = replay_buffer_path
        self.device = args.device

        # Initialize quantum network
        self.net = QuantumTransformerMarioNet(
            self.state_dim,
            self.action_dim,
            self.device,
            quantum_params
        ).to(self.device)

        # Epsilon-greedy exploration
        self.exploration_rate = 1.0
        self.exploration_rate_decay = args.exploration_rate_decay
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # Experience replay buffer
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                100000,
                scratch_dir=str(self.replay_buffer_path),
                device=torch.device("cpu")
            )
        )

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.burnin = 1e4  # Min experiences before training
        self.learn_every = args.learn_step
        self.sync_every = 1e4  # Sync target network every N steps
        self.save_every_episodes = args.save_every

    def act(self, state):
        """Select action using epsilon-greedy policy."""
        # Exploration: random action
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # Exploitation: best action from Q-network
        else:
            # Preprocess state
            if isinstance(state, tuple):
                state = state[0]
            if not isinstance(state, torch.Tensor):
                state_np = np.array(state, dtype=np.float32)
                state = torch.tensor(
                    state_np, device=self.device, dtype=torch.float32
                ).unsqueeze(0)
            elif state.device != self.device or state.dtype != torch.float32:
                state = state.to(
                    device=self.device, dtype=torch.float32
                ).unsqueeze(0)
            else:
                state = state.unsqueeze(0)

            if state.ndim == 3:
                state = state.unsqueeze(0)

            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # Decay exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Store experience in replay buffer."""
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state_np = np.array(first_if_tuple(state), dtype=np.float32)
        next_state_np = np.array(first_if_tuple(next_state), dtype=np.float32)

        state_t = torch.tensor(state_np, dtype=torch.float32)
        next_state_t = torch.tensor(next_state_np, dtype=torch.float32)
        action_t = torch.tensor([action], dtype=torch.int64)
        reward_t = torch.tensor([reward], dtype=torch.float32)
        done_t = torch.tensor([done], dtype=torch.bool)

        experience = TensorDict({
            "state": state_t,
            "next_state": next_state_t,
            "action": action_t,
            "reward": reward_t,
            "done": done_t
        }, batch_size=[])

        self.memory.add(experience)

    def recall(self):
        """Sample batch from replay buffer."""
        batch = self.memory.sample(self.batch_size).to(self.device)

        state = batch.get("state").float()
        next_state = batch.get("next_state").float()
        action = batch.get("action").squeeze(-1)
        reward = batch.get("reward").squeeze(-1).float()
        done = batch.get("done").squeeze(-1).bool()

        return state, next_state, action, reward, done

    def td_estimate(self, state, action):
        """Compute Q-value estimates for current states."""
        current_Q_values = self.net(state, model="online")
        current_Q_selected = current_Q_values.gather(
            1, action.unsqueeze(1)
        ).squeeze(1)
        return current_Q_selected

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """Compute TD target using Double Q-Learning."""
        # Select best action using online network
        next_state_Q_online = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q_online, axis=1)

        # Evaluate action using target network
        next_Q_target = self.net(next_state, model="target")
        best_next_q = next_Q_target.gather(
            1, best_action.unsqueeze(1)
        ).squeeze(1)

        return (reward + (1 - done.float()) * self.gamma * best_next_q).float()

    def update_Q_online(self, td_estimate, td_target):
        """Update online network using TD error."""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Synchronize target network with online network."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save_checkpoint(self, episode, logger_state_dict, checkpoint_path):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'curr_step': self.curr_step,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'logger_state': logger_state_dict,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved (Episode {episode}, Step {self.curr_step})")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        if not checkpoint_path.exists():
            print(f"✗ No checkpoint found. Starting from scratch.")
            return 0, None

        print(f"✓ Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curr_step = checkpoint['curr_step']
        self.exploration_rate = checkpoint['exploration_rate']

        torch_rng_state = checkpoint['torch_rng_state']
        if torch_rng_state.device != torch.device('cpu'):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        # Move optimizer state to correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        completed_episode = checkpoint['episode']
        logger_state_to_load = checkpoint.get('logger_state', None)
        print(f"✓ Resuming from Episode {completed_episode + 1} (Step {self.curr_step})")

        return completed_episode, logger_state_to_load

    def learn(self):
        """Perform one learning step."""
        # Sync target network periodically
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Wait until enough experiences collected
        if self.curr_step < self.burnin:
            return None, None

        # Learn only every N steps
        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample batch and update network
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss) if td_est is not None else (0.0, loss)


# ================================================================================
# METRIC LOGGER
# ================================================================================
class MetricLogger:
    """Logs training metrics and generates plots."""
    def __init__(self, save_dir, resume_log_file_exists=False):
        self.save_log_path = save_dir / "log.txt"
        log_mode = "a" if resume_log_file_exists else "w"

        with open(self.save_log_path, log_mode) as f:
            if log_mode == "w":
                f.write(
                    f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                    f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                    f"{'TimeDelta':>15}{'Time':>20}\n"
                )

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        """Log metrics for a single step."""
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None and q is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """Log metrics for completed episode."""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)

        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)

        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.init_episode()

    def init_episode(self):
        """Reset episode trackers."""
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        """Record and print episode statistics."""
        mean_ep_reward = np.round(
            np.mean(self.ep_rewards[-100:]) if self.ep_rewards else 0.0, 3
        )
        mean_ep_length = np.round(
            np.mean(self.ep_lengths[-100:]) if self.ep_lengths else 0.0, 3
        )
        mean_ep_loss = np.round(
            np.mean(self.ep_avg_losses[-100:]) if self.ep_avg_losses else 0.0, 3
        )
        mean_ep_q = np.round(
            np.mean(self.ep_avg_qs[-100:]) if self.ep_avg_qs else 0.0, 3
        )

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Ep {episode:>6} | Step {step:>8} | ε {epsilon:.3f} | "
            f"Reward {mean_ep_reward:>6.1f} | Length {mean_ep_length:>6.1f} | "
            f"Loss {mean_ep_loss:>6.3f} | Q {mean_ep_q:>6.2f} | "
            f"Time {time_since_last_record:>6.1f}s"
        )

        with open(self.save_log_path, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}"
                f"{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        self._plot_metrics()

    def _plot_metrics(self):
        """Generate and save metric plots."""
        metrics_to_plot = {
            "Rewards": (self.moving_avg_ep_rewards, self.ep_rewards_plot),
            "Lengths": (self.moving_avg_ep_lengths, self.ep_lengths_plot),
            "Avg Losses": (self.moving_avg_ep_avg_losses, self.ep_avg_losses_plot),
            "Avg Qs": (self.moving_avg_ep_avg_qs, self.ep_avg_qs_plot)
        }

        for metric_name, (data, save_path) in metrics_to_plot.items():
            if data:
                plt.figure(figsize=(10, 6))
                plt.plot(data, label=f"Moving Avg {metric_name}", linewidth=2)
                plt.title(f"Moving Average of Episode {metric_name}")
                plt.xlabel("Episode")
                plt.ylabel(metric_name)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()

    def get_state_dict(self):
        """Get logger state for checkpointing."""
        return {
            'ep_rewards': self.ep_rewards,
            'ep_lengths': self.ep_lengths,
            'ep_avg_losses': self.ep_avg_losses,
            'ep_avg_qs': self.ep_avg_qs,
            'moving_avg_ep_rewards': self.moving_avg_ep_rewards,
            'moving_avg_ep_lengths': self.moving_avg_ep_lengths,
            'moving_avg_ep_avg_losses': self.moving_avg_ep_avg_losses,
            'moving_avg_ep_avg_qs': self.moving_avg_ep_avg_qs,
            'record_time': self.record_time
        }

    def load_state_dict(self, state_dict):
        """Load logger state from checkpoint."""
        if state_dict:
            self.ep_rewards = state_dict.get('ep_rewards', [])
            self.ep_lengths = state_dict.get('ep_lengths', [])
            self.ep_avg_losses = state_dict.get('ep_avg_losses', [])
            self.ep_avg_qs = state_dict.get('ep_avg_qs', [])
            self.moving_avg_ep_rewards = state_dict.get('moving_avg_ep_rewards', [])
            self.moving_avg_ep_lengths = state_dict.get('moving_avg_ep_lengths', [])
            self.moving_avg_ep_avg_losses = state_dict.get('moving_avg_ep_avg_losses', [])
            self.moving_avg_ep_avg_qs = state_dict.get('moving_avg_ep_avg_qs', [])
            self.record_time = state_dict.get('record_time', time.time())
            print("✓ MetricLogger state loaded")


# ================================================================================
# MAIN TRAINING LOOP
# ================================================================================
if __name__ == '__main__':
    set_global_seeds(args.seed)

    # Define quantum parameters
    quantum_params = {
        'n_qubits': args.n_qubits,
        'n_timesteps': args.n_timesteps,
        'degree': args.degree,
        'n_ansatz_layers': args.n_layers,
        'feature_dim': args.feature_dim,
        'dropout': args.dropout
    }

    print("\n" + "="*80)
    print("QUANTUM TIME-SERIES TRANSFORMER - SUPER MARIO BROS RL")
    print("="*80)
    print(f"\n📊 Quantum Parameters:")
    print(f"  • Qubits:        {args.n_qubits}")
    print(f"  • Ansatz Layers: {args.n_layers}")
    print(f"  • QSVT Degree:   {args.degree}")
    print(f"  • Timesteps:     {args.n_timesteps}")
    print(f"  • Feature Dim:   {args.feature_dim}")
    print(f"\n🎮 Training Parameters:")
    print(f"  • Device:        {args.device}")
    print(f"  • Batch Size:    {args.batch_size}")
    print(f"  • Episodes:      {args.num_episodes}")
    print(f"  • Learning Rate: {args.lr}")
    print(f"  • Gamma:         {args.gamma}")
    print(f"  • Learn Step:    {args.learn_step}")
    print(f"\n💾 Save Directory: {SAVE_DIR}")
    print("="*80 + "\n")

    action_dim = env.action_space.n  # 2 actions: walk right, jump right

    # Initialize agent
    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=action_dim,
        save_dir=SAVE_DIR,
        replay_buffer_path=REPLAY_BUFFER_DIR,
        quantum_params=quantum_params
    )

    # Load checkpoint if resuming
    start_episode = 0
    loaded_logger_state = None
    if args.resume and CHECKPOINT_FILE_PATH.exists():
        print(f"📂 Found existing checkpoint: {CHECKPOINT_FILE_PATH}")
        completed_episode, loaded_logger_state = mario.load_checkpoint(CHECKPOINT_FILE_PATH)
        start_episode = completed_episode + 1
    else:
        print(f"🆕 No checkpoint found. Starting new training.\n")

    # Initialize logger
    logger = MetricLogger(
        save_dir=SAVE_DIR,
        resume_log_file_exists=(args.resume and CHECKPOINT_FILE_PATH.exists())
    )
    if loaded_logger_state:
        logger.load_state_dict(loaded_logger_state)

    total_episodes = args.num_episodes

    # ================================================================================
    # SIGNAL HANDLING FOR GRACEFUL SHUTDOWN
    # ================================================================================
    # Global variables for signal handler
    current_episode = [start_episode]  # Use list to allow modification in nested function
    shutdown_requested = [False]

    def save_checkpoint_on_signal(signum, frame):
        """Handle SIGTERM/SIGINT by saving checkpoint before exit."""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n\n{'='*80}")
        print(f"⚠️  {signal_name} received! Saving checkpoint before exit...")
        print(f"{'='*80}")

        try:
            logger_state_to_save = logger.get_state_dict()
            mario.save_checkpoint(
                episode=current_episode[0],
                logger_state_dict=logger_state_to_save,
                checkpoint_path=CHECKPOINT_FILE_PATH
            )
            print(f"✓ Emergency checkpoint saved at episode {current_episode[0]}")
            print(f"  To resume: python QuantumTransformerMario.py --resume --save-dir={CHECKPOINT_BASE_DIR}")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")

        print(f"{'='*80}\n")
        shutdown_requested[0] = True
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, save_checkpoint_on_signal)
    signal.signal(signal.SIGINT, save_checkpoint_on_signal)
    print(f"✓ Signal handlers registered (SIGTERM, SIGINT)")
    print(f"  Checkpoints saved every {mario.save_every_episodes} episodes")

    print(f"\n🚀 Starting training from episode {start_episode} to {total_episodes-1}")
    print(f"   Current step: {mario.curr_step} | Exploration rate: {mario.exploration_rate:.4f}\n")
    print("-"*80 + "\n")

    # Training loop
    for e in range(start_episode, total_episodes):
        current_episode[0] = e  # Update for signal handler
        state_tuple = env.reset()
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple

        # Episode loop
        while True:
            action = mario.act(state)

            next_state_tuple = env.step(action)
            if len(next_state_tuple) == 5:
                next_obs, reward, done_env, trunc, info = next_state_tuple
            else:
                next_obs, reward, done_env, info = next_state_tuple
                trunc = False

            current_next_state = (
                next_obs[0] if isinstance(next_obs, tuple) and
                not isinstance(next_obs[0], (int, float, bool))
                else next_obs
            )

            episode_done = done_env or trunc

            mario.cache(state, current_next_state, action, reward, episode_done)
            q_val, loss_val = mario.learn()
            logger.log_step(reward, loss_val, q_val)

            state = current_next_state

            if episode_done or info.get("flag_get", False):
                break

        logger.log_episode()

        # Record metrics every episode
        if e % 1 == 0 or e == total_episodes - 1:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

        # Save checkpoint periodically
        if (e > 0 and e % mario.save_every_episodes == 0) or (e == total_episodes - 1):
            logger_state_to_save = logger.get_state_dict()
            mario.save_checkpoint(
                episode=e,
                logger_state_dict=logger_state_to_save,
                checkpoint_path=CHECKPOINT_FILE_PATH
            )

    print("\n" + "="*80)
    print("🏁 Training finished!")
    print(f"📊 Final results saved to: {SAVE_DIR}")
    print("="*80 + "\n")
