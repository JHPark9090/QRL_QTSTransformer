"""
Quantum Time-Series Transformer for Simple RL Environments
===========================================================
This script implements a quantum transformer-based RL agent for environments with
low-dimensional state spaces: CartPole, FrozenLake, MountainCar, Acrobot, etc.

Unlike the Super Mario implementation which needs CNN feature extraction,
these environments have simple state vectors that can be directly processed
by the quantum transformer.

Supported Environments:
- CartPole-v1: 4-dimensional continuous state
- FrozenLake-v1: 16-dimensional one-hot encoded state
- MountainCar-v0: 2-dimensional continuous state
- Acrobot-v1: 6-dimensional continuous state

Architecture:
  State → State History Buffer (n_timesteps) → Quantum Transformer → Q-values
"""

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
import gymnasium as gym  # Using gymnasium (actively maintained replacement for gym)


# ================================================================================
# ARGUMENT PARSER
# ================================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description='Quantum Transformer for Simple RL Environments'
    )

    # Environment selection
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        choices=["CartPole-v1", "FrozenLake-v1",
                                "MountainCar-v0", "Acrobot-v1"],
                        help="RL environment to train on")

    # Quantum parameters
    parser.add_argument("--n-qubits", type=int, default=4,
                        help="Number of qubits in quantum circuit")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of ansatz layers in quantum circuit")
    parser.add_argument("--degree", type=int, default=2,
                        help="Degree of QSVT polynomial")
    parser.add_argument("--n-timesteps", type=int, default=4,
                        help="Number of timesteps in state history")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # RL parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--exploration-rate-start', type=float, default=1.0)
    parser.add_argument('--exploration-rate-decay', type=float, default=0.995)
    parser.add_argument('--exploration-rate-min', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument('--learn-step', type=int, default=1,
                        help="Learn every N steps")
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--memory-size', type=int, default=10000,
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
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Base directory for saving checkpoints and results")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N episodes (default: 10)")

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
env_name_clean = args.env.replace("-", "").replace("v", "")
RUN_ID = f"QTransformer_{env_name_clean}_Q{args.n_qubits}_L{args.n_layers}_D{args.degree}_Run{args.log_index}"
if args.save_dir:
    CHECKPOINT_BASE_DIR = Path(args.save_dir)
else:
    CHECKPOINT_BASE_DIR = Path("SimpleRLCheckpoints")
SAVE_DIR = CHECKPOINT_BASE_DIR / RUN_ID

SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE_PATH = SAVE_DIR / "latest_checkpoint.chkpt"


# ================================================================================
# ENVIRONMENT SETUP
# ================================================================================
def create_env(env_name):
    """Create and configure the environment."""
    env = gym.make(env_name)
    return env

env = create_env(args.env)


# ================================================================================
# STATE PREPROCESSING
# ================================================================================
class StateProcessor:
    """Handles state preprocessing for different environment types."""

    def __init__(self, env, n_timesteps):
        self.env_name = args.env
        self.n_timesteps = n_timesteps

        # Determine state dimensionality
        if hasattr(env.observation_space, 'n'):
            # Discrete state space (e.g., FrozenLake)
            self.state_dim = env.observation_space.n
            self.discrete_state = True
        else:
            # Continuous state space (e.g., CartPole)
            self.state_dim = env.observation_space.shape[0]
            self.discrete_state = False

        # State history buffer
        self.state_history = deque(maxlen=n_timesteps)
        self.reset()

    def reset(self):
        """Reset state history with zeros."""
        self.state_history.clear()
        zero_state = np.zeros(self.state_dim)
        for _ in range(self.n_timesteps):
            self.state_history.append(zero_state)

    def process_state(self, state):
        """Process a single state observation."""
        if self.discrete_state:
            # One-hot encode discrete states
            one_hot = np.zeros(self.state_dim)
            one_hot[state] = 1.0
            return one_hot
        else:
            # Normalize continuous states
            return np.array(state, dtype=np.float32)

    def add_state(self, state):
        """Add a new state to history and return the history tensor."""
        processed = self.process_state(state)
        self.state_history.append(processed)

        # Return as tensor: (n_timesteps, state_dim)
        history_array = np.array(list(self.state_history))
        return torch.tensor(history_array, dtype=torch.float32)


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
    """Applies a linear combination of unitaries (fully vectorized)."""
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
# QUANTUM TIME-SERIES TRANSFORMER
# ================================================================================
class QuantumTSTransformerRL(nn.Module):
    """
    Quantum Time-Series Transformer for simple RL environments.
    """
    def __init__(self, state_dim: int, n_qubits: int, n_timesteps: int,
                 degree: int, n_ansatz_layers: int, output_dim: int,
                 dropout: float, device):
        super().__init__()

        self.state_dim = state_dim
        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.device = device

        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits * 1

        # Classical layers
        self.feature_projection = nn.Linear(state_dim, self.n_rots)
        self.dropout = nn.Dropout(dropout)
        self.rot_sigm = nn.Sigmoid()
        self.output_ff = nn.Sequential(
            nn.Linear(3 * n_qubits, 64),
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
        # x: (batch, n_timesteps, state_dim)
        bsz = x.shape[0]

        # Project features through quantum rotation parameters
        x = self.feature_projection(self.dropout(x))
        timestep_params = self.rot_sigm(x)

        # Initialize base quantum states
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

        # Measure observables
        exps = self.qff_qnode_expval(
            initial_state=normalized_mixed_timestep,
            params=self.qff_params
        )

        exps = torch.stack(exps, dim=1)
        exps = exps.float()

        # Final output layer to Q-values
        op = self.output_ff(exps)
        return op


# ================================================================================
# REPLAY BUFFER
# ================================================================================
class ReplayBuffer:
    """Simple replay buffer for experience replay."""

    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Pre-allocate memory
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
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
class QuantumDQNAgent:
    """DQN agent using quantum transformer network."""

    def __init__(self, state_processor, action_dim, quantum_params):
        self.state_processor = state_processor
        self.action_dim = action_dim
        self.device = args.device

        # Q-Networks
        self.online_net = QuantumTSTransformerRL(
            state_dim=state_processor.state_dim,
            n_qubits=quantum_params['n_qubits'],
            n_timesteps=quantum_params['n_timesteps'],
            degree=quantum_params['degree'],
            n_ansatz_layers=quantum_params['n_ansatz_layers'],
            output_dim=action_dim,
            dropout=quantum_params['dropout'],
            device=self.device
        ).to(self.device)

        self.target_net = QuantumTSTransformerRL(
            state_dim=state_processor.state_dim,
            n_qubits=quantum_params['n_qubits'],
            n_timesteps=quantum_params['n_timesteps'],
            degree=quantum_params['degree'],
            n_ansatz_layers=quantum_params['n_ansatz_layers'],
            output_dim=action_dim,
            dropout=quantum_params['dropout'],
            device=self.device
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=args.lr)
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
        self.sync_every = 100  # Sync target network every N steps

    def select_action(self, state_history):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = state_history.unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """Perform one learning step."""
        if len(self.memory) < args.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(args.batch_size)

        # Compute current Q-values
        current_q_values = self.online_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones.float()) * args.gamma * next_q

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay
        )

        # Sync target network
        self.curr_step += 1
        if self.curr_step % self.sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save_checkpoint(self, episode, metrics, checkpoint_path):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'curr_step': self.curr_step,
            'online_net_state': self.online_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'metrics': metrics,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved (Episode {episode})", flush=True)

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        if not checkpoint_path.exists():
            print(f"✗ No checkpoint found. Starting from scratch.")
            return 0, {}

        print(f"✓ Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.online_net.load_state_dict(checkpoint['online_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['curr_step']

        torch_rng_state = checkpoint['torch_rng_state']
        if torch_rng_state.device != torch.device('cpu'):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        completed_episode = checkpoint['episode']
        metrics = checkpoint.get('metrics', {})
        print(f"✓ Resuming from Episode {completed_episode + 1}")

        return completed_episode, metrics


# ================================================================================
# TRAINING LOOP
# ================================================================================
def train():
    """Main training loop."""
    set_global_seeds(args.seed)

    # Initialize state processor
    state_processor = StateProcessor(env, args.n_timesteps)

    # Define quantum parameters
    quantum_params = {
        'n_qubits': args.n_qubits,
        'n_timesteps': args.n_timesteps,
        'degree': args.degree,
        'n_ansatz_layers': args.n_layers,
        'dropout': args.dropout
    }

    # Initialize agent
    agent = QuantumDQNAgent(
        state_processor,
        action_dim=env.action_space.n,
        quantum_params=quantum_params
    )

    # Load checkpoint if resuming
    start_episode = 0
    metrics = {'rewards': [], 'lengths': [], 'losses': []}
    if args.resume and CHECKPOINT_FILE_PATH.exists():
        start_episode, metrics = agent.load_checkpoint(CHECKPOINT_FILE_PATH)
        start_episode += 1

    print("\n" + "="*80)
    print(f"QUANTUM TRANSFORMER - {args.env}")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"  • Environment:   {args.env}")
    print(f"  • State Dim:     {state_processor.state_dim}")
    print(f"  • Action Dim:    {env.action_space.n}")
    print(f"  • Qubits:        {args.n_qubits}")
    print(f"  • Ansatz Layers: {args.n_layers}")
    print(f"  • QSVT Degree:   {args.degree}")
    print(f"  • Timesteps:     {args.n_timesteps}")
    print(f"  • Device:        {args.device}")
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
            print(f"  To resume: python QuantumTransformerSimpleRL.py --env={args.env} --resume --save-dir={CHECKPOINT_BASE_DIR}")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
        print(f"{'='*80}\n")
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_checkpoint_on_signal)
    signal.signal(signal.SIGINT, save_checkpoint_on_signal)
    print(f"✓ Signal handlers registered (SIGTERM, SIGINT)\n")

    # Training loop
    for episode in range(start_episode, args.num_episodes):
        current_episode[0] = episode
        state, _ = env.reset()  # gymnasium returns (obs, info)
        state_processor.reset()
        state_history = state_processor.add_state(state)

        episode_reward = 0
        episode_loss = []

        for step in range(args.max_steps):
            if args.render:
                env.render()

            # Select and perform action
            action = agent.select_action(state_history)
            next_state, reward, terminated, truncated, info = env.step(action)  # gymnasium API
            done = terminated or truncated
            next_state_history = state_processor.add_state(next_state)

            # Store transition
            agent.store_transition(
                state_history, action, reward, next_state_history, done
            )

            # Learn
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

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(metrics['rewards'][-100:])
            avg_length = np.mean(metrics['lengths'][-100:])
            avg_loss = np.mean(metrics['losses'][-100:])
            print(
                f"Ep {episode:>4} | Reward: {episode_reward:>6.1f} | "
                f"Avg Reward: {avg_reward:>6.1f} | Avg Length: {avg_length:>5.1f} | "
                f"Loss: {avg_loss:>6.4f} | ε: {agent.exploration_rate:.3f}",
                flush=True  # Ensure immediate output to SLURM logs
            )

        # Save checkpoint
        if (episode + 1) % args.save_every == 0 or episode == args.num_episodes - 1:
            agent.save_checkpoint(episode, metrics, CHECKPOINT_FILE_PATH)
            plot_training_curves(metrics, SAVE_DIR)

    env.close()
    print("\n" + "="*80)
    print("🏁 Training finished!")
    print("="*80 + "\n")


# ================================================================================
# PLOTTING
# ================================================================================
def plot_training_curves(metrics, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Rewards
    axes[0].plot(metrics['rewards'], alpha=0.3, color='blue')
    if len(metrics['rewards']) > 0:
        window = min(100, len(metrics['rewards']))
        moving_avg = np.convolve(
            metrics['rewards'],
            np.ones(window)/window,
            mode='valid'
        )
        axes[0].plot(moving_avg, color='blue', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].grid(True, alpha=0.3)

    # Lengths
    axes[1].plot(metrics['lengths'], alpha=0.3, color='green')
    if len(metrics['lengths']) > 0:
        window = min(100, len(metrics['lengths']))
        moving_avg = np.convolve(
            metrics['lengths'],
            np.ones(window)/window,
            mode='valid'
        )
        axes[1].plot(moving_avg, color='green', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Length')
    axes[1].set_title('Episode Lengths')
    axes[1].grid(True, alpha=0.3)

    # Losses
    axes[2].plot(metrics['losses'], alpha=0.3, color='red')
    if len(metrics['losses']) > 0:
        window = min(100, len(metrics['losses']))
        moving_avg = np.convolve(
            metrics['losses'],
            np.ones(window)/window,
            mode='valid'
        )
        axes[2].plot(moving_avg, color='red', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=100, bbox_inches='tight')
    plt.close()


# ================================================================================
# MAIN
# ================================================================================
if __name__ == '__main__':
    train()
