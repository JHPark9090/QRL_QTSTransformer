"""
Record Video of Quantum RL Agent Playing Super Mario Bros
===========================================================
This script loads a trained quantum RL agent from checkpoint and records
a video of gameplay.

Usage:
    python record_quantum_mario_video.py --checkpoint-dir=SuperMarioCheckpoints/QuantumMario_Run16
"""

import torch
from torch import nn
import numpy as np
from pathlib import Path
import argparse
import sys

# Import the model architecture from QuantumSuperMario.py
# We need to import the classes: QuantumMarioNet, QCNN, Mario
sys.path.append(str(Path(__file__).parent))

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
from torchvision import transforms as T
from gym.spaces import Box

# For video recording
import cv2
from datetime import datetime


# ================================================================================
# ENVIRONMENT WRAPPERS (copied from QuantumSuperMario.py)
# ================================================================================
class SkipFrame(gym.Wrapper):
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
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# ================================================================================
# QUANTUM MODEL (import from QuantumSuperMario.py)
# ================================================================================
# We'll load the model using the checkpoint's state dict
# Import the necessary components
try:
    from QuantumSuperMario import QuantumMarioNet, QCNN
    print("✓ Successfully imported model from QuantumSuperMario.py")
except ImportError:
    print("❌ ERROR: Could not import from QuantumSuperMario.py")
    print("Make sure QuantumSuperMario.py is in the same directory")
    sys.exit(1)


# ================================================================================
# VIDEO RECORDING SETUP
# ================================================================================
def get_args():
    parser = argparse.ArgumentParser(description='Record Quantum Mario Agent')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='SuperMarioCheckpoints/QuantumMario_Run16',
                        help='Directory containing latest_checkpoint.chkpt')
    parser.add_argument('--num-episodes', type=int, default=3,
                        help='Number of episodes to record')
    parser.add_argument('--output-dir', type=str, default='mario_videos',
                        help='Directory to save videos')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run model on')
    return parser.parse_args()


def create_mario_env():
    """Create the Super Mario environment with same wrappers as training."""
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0",
                                        render_mode='rgb_array',
                                        apply_api_compatibility=True)

    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)

    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    return env


class VideoRecorder:
    """Records gameplay to video file."""
    def __init__(self, output_path, fps=30, frame_size=(240, 256)):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, frame_size
        )
        self.frame_count = 0

    def add_frame(self, frame):
        """Add a frame to the video."""
        # frame is (height, width, 3) RGB from env.render()
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize if needed
        if frame_bgr.shape[:2][::-1] != self.frame_size:
            frame_bgr = cv2.resize(frame_bgr, self.frame_size)

        self.writer.write(frame_bgr)
        self.frame_count += 1

    def close(self):
        """Finalize the video file."""
        self.writer.release()
        print(f"✓ Saved video: {self.output_path} ({self.frame_count} frames)")


# ================================================================================
# LOAD CHECKPOINT AND PLAY
# ================================================================================
def load_checkpoint(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    print(f"\n📂 Loading checkpoint from: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"✓ Checkpoint loaded (Episode {checkpoint['episode']})")
    print(f"  • Total steps: {checkpoint['curr_step']}")
    print(f"  • Exploration rate: {checkpoint['exploration_rate']:.4f}")

    return checkpoint


def create_model_from_checkpoint(checkpoint, device):
    """Recreate the model architecture and load weights."""
    print("\n🧠 Creating model architecture...")

    # These should match the architecture used in training
    # Assuming standard QuantumMario configuration
    state_dim = (4, 84, 84)
    action_dim = 2  # right, jump right

    # Try to infer model parameters from checkpoint
    # Default values based on typical QuantumMario setup
    n_qubits = 8
    circuit_depth = 2
    n_chips = 100

    model = QuantumMarioNet(
        state_dim, action_dim, device,
        n_qubits_per_chip=n_qubits,
        circuit_depth_per_chip=circuit_depth,
        num_chips=n_chips
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    print(f"✓ Model created and weights loaded")
    print(f"  • Qubits per chip: {n_qubits}")
    print(f"  • Circuit depth: {circuit_depth}")
    print(f"  • Number of chips: {n_chips}")

    return model


def play_episode(env, model, device, video_recorder=None):
    """Play one episode and optionally record video."""
    state_tuple = env.reset()
    state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 5000:  # Max 5000 steps per episode
        # Render frame for video
        if video_recorder is not None:
            frame = env.render()
            if frame is not None:
                video_recorder.add_frame(frame)

        # Prepare state for model
        if isinstance(state, tuple):
            state = state[0]
        if not isinstance(state, torch.Tensor):
            state_np = np.array(state, dtype=np.float32)
            state_tensor = torch.tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
        else:
            state_tensor = state.to(device=device, dtype=torch.float32).unsqueeze(0)

        if state_tensor.ndim == 3:
            state_tensor = state_tensor.unsqueeze(0)

        # Get action from model (greedy - no exploration)
        with torch.no_grad():
            q_values = model(state_tensor, model="online")
            action = torch.argmax(q_values, axis=1).item()

        # Take action
        next_state_tuple = env.step(action)
        if len(next_state_tuple) == 5:
            next_obs, reward, done_env, trunc, info = next_state_tuple
        else:
            next_obs, reward, done_env, info = next_state_tuple
            trunc = False

        current_next_state = (next_obs[0] if isinstance(next_obs, tuple) and
                            not isinstance(next_obs[0], (int, float, bool))
                            else next_obs)

        done = done_env or trunc or info.get("flag_get", False)

        total_reward += reward
        state = current_next_state
        steps += 1

    return total_reward, steps, info


# ================================================================================
# MAIN
# ================================================================================
def main():
    args = get_args()

    print("="*80)
    print("QUANTUM MARIO - VIDEO RECORDING")
    print("="*80)
    print(f"\n📁 Checkpoint: {args.checkpoint_dir}")
    print(f"🎬 Recording {args.num_episodes} episode(s)")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🖥️  Device: {args.device}")
    print("="*80 + "\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / "latest_checkpoint.chkpt"
    checkpoint = load_checkpoint(checkpoint_path, args.device)

    # Create model
    model = create_model_from_checkpoint(checkpoint, args.device)

    # Create environment
    print("\n🎮 Creating Super Mario environment...")
    env = create_mario_env()
    print("✓ Environment created\n")

    # Record episodes
    print(f"🎬 Starting recording...\n")

    for episode in range(args.num_episodes):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = output_dir / f"quantum_mario_episode_{episode+1}_{timestamp}.mp4"

        print(f"Episode {episode + 1}/{args.num_episodes}")
        print(f"  Recording to: {video_path}")

        # Create video recorder
        recorder = VideoRecorder(video_path, fps=args.fps)

        # Play episode
        reward, steps, info = play_episode(env, model, args.device, recorder)

        # Close video
        recorder.close()

        # Print stats
        print(f"  • Total reward: {reward:.1f}")
        print(f"  • Steps: {steps}")
        print(f"  • X position: {info.get('x_pos', 'N/A')}")
        print(f"  • Flag reached: {info.get('flag_get', False)}")
        print()

    env.close()

    print("="*80)
    print("🎉 VIDEO RECORDING COMPLETE!")
    print("="*80)
    print(f"\nVideos saved in: {output_dir.absolute()}")
    print(f"\nTo view videos:")
    print(f"  ls -lh {output_dir}")
    print(f"  # Copy to local machine and open with video player")
    print()


if __name__ == '__main__':
    main()
