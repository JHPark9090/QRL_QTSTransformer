"""
Classical Time-Series Transformer for Simple RL Environments
==============================================================
Matched classical baseline for Quantum Transformer v5 SimpleRL experiments.
Uses the SAME StateProcessor, replay buffer, DQN training loop,
and RL hyperparameters -- only the transformer core differs.

Architecture:
  State -> State History Buffer (n_timesteps) -> + Sinusoidal PE
  -> Classical Transformer Encoder -> Mean Pooling -> Q-values

Fair comparison design:
  - Same state preprocessing (StateProcessor with one-hot / continuous)
  - Same replay buffer and Double DQN training
  - Same exploration schedule (epsilon-greedy, per-episode decay)
  - Same signal handling for graceful shutdown
  - d_model=32, n_heads=4, d_ff=64 (scaled for small state dims)
"""

import math
import torch
from torch import nn
import numpy as np
from pathlib import Path
from collections import deque
import random, time, datetime, os, signal, sys
import argparse

import matplotlib.pyplot as plt
import gymnasium as gym


# ================================================================================
# ARGUMENT PARSER
# ================================================================================
def get_args():
    parser = argparse.ArgumentParser(
        description='Classical Transformer Baseline for Simple RL Environments'
    )

    # Environment selection
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        choices=["CartPole-v1", "FrozenLake-v1",
                                "MountainCar-v0", "Acrobot-v1"],
                        help="RL environment to train on")

    # Transformer parameters
    parser.add_argument("--d-model", type=int, default=32,
                        help="Transformer model dimension")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n-transformer-layers", type=int, default=1,
                        help="Number of transformer encoder layers")
    parser.add_argument("--d-ff", type=int, default=64,
                        help="Transformer feedforward dimension")
    parser.add_argument("--n-timesteps", type=int, default=4,
                        help="Number of timesteps in state history")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # RL parameters (matched to quantum v5 SimpleRL defaults)
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
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint every N episodes")

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
env_name_clean = args.env.replace("-", "").replace("v", "")
RUN_ID = (f"CTransformer_{env_name_clean}_DM{args.d_model}_H{args.n_heads}"
          f"_L{args.n_transformer_layers}_Run{args.log_index}")
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
    return gym.make(env_name)

env = create_env(args.env)


# ================================================================================
# STATE PREPROCESSING
# ================================================================================
class StateProcessor:
    """Handles state preprocessing for different environment types."""

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
        """Reset state history with zeros."""
        self.state_history.clear()
        zero_state = np.zeros(self.state_dim)
        for _ in range(self.n_timesteps):
            self.state_history.append(zero_state)

    def process_state(self, state):
        """Process a single state observation."""
        if self.discrete_state:
            one_hot = np.zeros(self.state_dim)
            one_hot[state] = 1.0
            return one_hot
        else:
            return np.array(state, dtype=np.float32)

    def add_state(self, state):
        """Add a new state to history and return the history tensor."""
        processed = self.process_state(state)
        self.state_history.append(processed)
        history_array = np.array(list(self.state_history))
        return torch.tensor(history_array, dtype=torch.float32)


# ================================================================================
# CLASSICAL TIME-SERIES TRANSFORMER
# ================================================================================
class ClassicalTSTransformerRL(nn.Module):
    """Classical Time-Series Transformer for simple RL environments.

    Matched baseline for QuantumTSTransformerRL_v5:
      - Same sinusoidal PE (Vaswani et al., 2017)
      - Same input shape: (batch, n_timesteps, state_dim)
      - nn.TransformerEncoder replaces quantum QSVT/QFF
      - Mean pooling over timesteps + output head
    """
    def __init__(self, state_dim: int, d_model: int, n_heads: int,
                 n_transformer_layers: int, d_ff: int, n_timesteps: int,
                 output_dim: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.n_timesteps = n_timesteps

        # Input projection
        self.input_projection = nn.Linear(state_dim, d_model)

        # Sinusoidal Positional Encoding (Vaswani et al., 2017)
        pe = torch.zeros(n_timesteps, d_model)
        pos = torch.arange(n_timesteps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: (batch, n_timesteps, state_dim)
        x = self.input_projection(x)
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_head(x)
        return x


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

        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer. Accepts tensors from StateProcessor."""
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
class ClassicalSimpleRLAgent:
    """DQN agent using classical transformer network."""

    def __init__(self, state_processor, action_dim, transformer_params):
        self.state_processor = state_processor
        self.action_dim = action_dim
        self.device = args.device

        # Q-Networks
        self.online_net = ClassicalTSTransformerRL(
            state_dim=state_processor.state_dim,
            d_model=transformer_params['d_model'],
            n_heads=transformer_params['n_heads'],
            n_transformer_layers=transformer_params['n_transformer_layers'],
            d_ff=transformer_params['d_ff'],
            n_timesteps=transformer_params['n_timesteps'],
            output_dim=action_dim,
            dropout=transformer_params['dropout']
        ).to(self.device)

        self.target_net = ClassicalTSTransformerRL(
            state_dim=state_processor.state_dim,
            d_model=transformer_params['d_model'],
            n_heads=transformer_params['n_heads'],
            n_transformer_layers=transformer_params['n_transformer_layers'],
            d_ff=transformer_params['d_ff'],
            n_timesteps=transformer_params['n_timesteps'],
            output_dim=action_dim,
            dropout=transformer_params['dropout']
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=args.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        state_shape = (transformer_params['n_timesteps'], state_processor.state_dim)
        self.memory = ReplayBuffer(args.memory_size, state_shape, self.device)

        # Exploration
        self.exploration_rate = args.exploration_rate_start
        self.exploration_rate_decay = args.exploration_rate_decay
        self.exploration_rate_min = args.exploration_rate_min

        # Training counters
        self.curr_step = 0
        self.sync_every = 100

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
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Decay exploration rate per step (matches v1 behavior)
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
        print(f"Checkpoint saved (Episode {episode})", flush=True)

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        if not checkpoint_path.exists():
            print(f"No checkpoint found. Starting from scratch.")
            return 0, {}

        print(f"Loading checkpoint...")
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
        print(f"Resuming from Episode {completed_episode + 1}")

        return completed_episode, metrics


# ================================================================================
# TRAINING LOOP
# ================================================================================
def train():
    """Main training loop."""
    set_global_seeds(args.seed)

    state_processor = StateProcessor(env, args.n_timesteps)

    transformer_params = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_transformer_layers': args.n_transformer_layers,
        'd_ff': args.d_ff,
        'n_timesteps': args.n_timesteps,
        'dropout': args.dropout
    }

    agent = ClassicalSimpleRLAgent(
        state_processor,
        action_dim=env.action_space.n,
        transformer_params=transformer_params
    )

    # Parameter count reporting
    total_params = sum(p.numel() for p in agent.online_net.parameters())

    start_episode = 0
    metrics = {'rewards': [], 'lengths': [], 'losses': []}
    if args.resume and CHECKPOINT_FILE_PATH.exists():
        start_episode, metrics = agent.load_checkpoint(CHECKPOINT_FILE_PATH)
        start_episode += 1

    print("\n" + "=" * 80)
    print(f"CLASSICAL TRANSFORMER BASELINE - {args.env}")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Environment:     {args.env}")
    print(f"  State Dim:       {state_processor.state_dim}")
    print(f"  Action Space:    {env.action_space.n} actions")
    print(f"  d_model:         {args.d_model}")
    print(f"  n_heads:         {args.n_heads}")
    print(f"  n_layers:        {args.n_transformer_layers}")
    print(f"  d_ff:            {args.d_ff}")
    print(f"  Timesteps:       {args.n_timesteps}")
    print(f"  Device:          {args.device}")
    print(f"\nTotal Parameters (online): {total_params:,}")
    print(f"\nSave Directory: {SAVE_DIR}")
    print("=" * 80 + "\n")

    # ================================================================================
    # SIGNAL HANDLING FOR GRACEFUL SHUTDOWN
    # ================================================================================
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
    for episode in range(start_episode, args.num_episodes):
        current_episode[0] = episode

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

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(metrics['rewards'][-100:])
            avg_length = np.mean(metrics['lengths'][-100:])
            avg_loss = np.mean(metrics['losses'][-100:])
            print(
                f"Ep {episode:>4} | R: {episode_reward:>6.1f} | "
                f"AvgR: {avg_reward:>6.1f} | L: {step+1:>4} | "
                f"Loss: {avg_loss:>6.4f} | e: {agent.exploration_rate:.3f}",
                flush=True
            )

        # Save checkpoint
        if (episode + 1) % args.save_every == 0 or episode == args.num_episodes - 1:
            agent.save_checkpoint(episode, metrics, CHECKPOINT_FILE_PATH)
            plot_training_curves(metrics, SAVE_DIR)

    env.close()
    print("\n" + "=" * 80)
    print("Training finished!")
    print("=" * 80 + "\n")


# ================================================================================
# PLOTTING
# ================================================================================
def plot_training_curves(metrics, save_dir):
    """Plot and save training curves."""
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
