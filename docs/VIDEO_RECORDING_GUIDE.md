# Video Recording Guide for Quantum Mario Agents

This guide explains how to record and access videos of your trained quantum RL agents playing Super Mario Bros.

---

## 📹 Quick Start

### For QCNN Agent (QuantumSuperMario.py)
```bash
sbatch run_record_mario_video.sh
```

### For Transformer Agent (QuantumTransformerMario.py)
```bash
# Edit run_record_mario_video.sh: Change line 44
AGENT_TYPE="TRANSFORMER"  # Change from "QCNN" to "TRANSFORMER"

# Then submit
sbatch run_record_mario_video.sh
```

---

## 🎥 1. Video Access & Retrieval

### ✅ Yes, videos are automatically saved!

**Save locations**:
- **QCNN agent videos**: `/pscratch/sd/j/junghoon/mario_videos_qcnn/`
- **Transformer agent videos**: `/pscratch/sd/j/junghoon/mario_videos_transformer/`

**File naming format**:
```
quantum_mario_episode_1_20250622_143052.mp4
quantum_transformer_mario_ep1_20250622_143052.mp4
```

### How to Access Videos

#### Method 1: SCP (Recommended for Small Files)

**From your local terminal** (not on the cluster):
```bash
# Copy all videos from QCNN agent
scp junghoon@perlmutter.nersc.gov:/pscratch/sd/j/junghoon/mario_videos_qcnn/*.mp4 ~/Downloads/

# Copy all videos from Transformer agent
scp junghoon@perlmutter.nersc.gov:/pscratch/sd/j/junghoon/mario_videos_transformer/*.mp4 ~/Downloads/

# Copy specific video
scp junghoon@perlmutter.nersc.gov:/pscratch/sd/j/junghoon/mario_videos_qcnn/quantum_mario_episode_1_*.mp4 .
```

#### Method 2: Globus (Recommended for Large Files)

1. **Set up Globus endpoints**:
   - Source: Perlmutter endpoint + `/pscratch/sd/j/junghoon/mario_videos_*/`
   - Destination: Your local Globus endpoint

2. **Transfer files** via Globus web interface or CLI

#### Method 3: View on Cluster (if X11 forwarding enabled)

```bash
# SSH with X11 forwarding
ssh -X junghoon@perlmutter.nersc.gov

# Navigate to video directory
cd /pscratch/sd/j/junghoon/mario_videos_qcnn

# Play video (requires VLC or ffplay)
vlc quantum_mario_episode_1_*.mp4
# or
ffplay quantum_mario_episode_1_*.mp4
```

#### Method 4: Web-based File Browser (if available)

Some clusters provide web-based file browsers. Check NERSC documentation.

### Video Specifications

| Property | Value |
|----------|-------|
| **Format** | MP4 (H.264 codec) |
| **Resolution** | 240×256 pixels (Mario's native) |
| **FPS** | 30 (configurable) |
| **File size** | 5-20 MB per episode |
| **Duration** | Varies (until episode ends or 5000 steps) |

---

## 🔄 2. Switching Between Agent Types

### Summary Table

| Agent Type | Script | Checkpoint Example | Model File |
|------------|--------|-------------------|------------|
| **QCNN** | `record_quantum_mario_video.py` | `QuantumMario_Run16` | `QuantumSuperMario.py` |
| **Transformer** | `record_quantum_mario_video_TRANSFORMER.py` | `QTransformerMario_Q6_L2_D2_Run1` | `QuantumTransformerMario.py` |

### Option A: Using the Batch Script (Easiest)

**Edit `run_record_mario_video.sh`**:

**Line 44** - Change agent type:
```bash
# For QCNN agent
AGENT_TYPE="QCNN"

# For Transformer agent
AGENT_TYPE="TRANSFORMER"
```

**Lines 47-61** - Update checkpoint path and parameters:

**For QCNN**:
```bash
if [ "$AGENT_TYPE" == "QCNN" ]; then
    CHECKPOINT_DIR="SuperMarioCheckpoints/QuantumMario_Run16"  # Your checkpoint
    SCRIPT="record_quantum_mario_video.py"
    OUTPUT_DIR="mario_videos_qcnn"
```

**For Transformer**:
```bash
elif [ "$AGENT_TYPE" == "TRANSFORMER" ]; then
    CHECKPOINT_DIR="SuperMarioCheckpoints/QTransformerMario_Q6_L2_D2_Run1"  # Your checkpoint
    SCRIPT="record_quantum_mario_video_TRANSFORMER.py"
    OUTPUT_DIR="mario_videos_transformer"

    # IMPORTANT: These must match your training configuration!
    N_QUBITS=6
    N_LAYERS=2
    DEGREE=2
    FEATURE_DIM=128
```

Then submit:
```bash
sbatch run_record_mario_video.sh
```

### Option B: Running Scripts Directly

#### For QCNN Agent:
```bash
conda activate ./conda-envs/qml_eeg

python record_quantum_mario_video.py \
    --checkpoint-dir=SuperMarioCheckpoints/QuantumMario_Run16 \
    --num-episodes=3 \
    --output-dir=mario_videos_qcnn \
    --device=cuda
```

#### For Transformer Agent:
```bash
conda activate ./conda-envs/qml_eeg

python record_quantum_mario_video_TRANSFORMER.py \
    --checkpoint-dir=SuperMarioCheckpoints/QTransformerMario_Q6_L2_D2_Run1 \
    --num-episodes=3 \
    --output-dir=mario_videos_transformer \
    --n-qubits=6 \
    --n-layers=2 \
    --degree=2 \
    --feature-dim=128 \
    --device=cuda
```

### Key Differences Between Scripts

| Aspect | QCNN Script | Transformer Script |
|--------|-------------|-------------------|
| **Import** | `from QuantumSuperMario import QuantumMarioNet, QCNN` | `from QuantumTransformerMario import QuantumTransformerMarioNet` |
| **Model Parameters** | `n_qubits_per_chip`, `circuit_depth_per_chip`, `num_chips` | `n_qubits`, `n_layers`, `degree`, `feature_dim` |
| **Default Output** | `mario_videos/` | `mario_videos_transformer/` |

### Finding Your Checkpoint

```bash
# List all QCNN checkpoints
ls -d SuperMarioCheckpoints/QuantumMario_*

# List all Transformer checkpoints
ls -d SuperMarioCheckpoints/QTransformerMario_*

# Check specific checkpoint
ls -lh SuperMarioCheckpoints/QuantumMario_Run16/latest_checkpoint.chkpt
```

---

## 📝 3. Batch Script Usage

### File: `run_record_mario_video.sh`

**Features**:
- ✅ Automatically activates conda environment
- ✅ Supports both QCNN and Transformer agents
- ✅ Configurable parameters
- ✅ SLURM job management
- ✅ Error logging

### Configuration Options

**Edit these variables in the script**:

```bash
# Line 44: Choose agent type
AGENT_TYPE="QCNN"          # or "TRANSFORMER"

# Lines 47-61: Checkpoint paths
CHECKPOINT_DIR="SuperMarioCheckpoints/QuantumMario_Run16"

# Lines 64-66: Recording settings
NUM_EPISODES=5             # Number of episodes to record
FPS=30                     # Video frames per second
DEVICE="cuda"              # "cuda" or "cpu"

# Lines 54-58: Transformer parameters (if using Transformer)
N_QUBITS=6
N_LAYERS=2
DEGREE=2
FEATURE_DIM=128
```

### Submitting the Job

```bash
# Basic submission
sbatch run_record_mario_video.sh

# Check job status
squeue -u junghoon

# View output log (while running)
tail -f logs/record_mario_<job_id>.out

# View error log (if failed)
cat logs/record_mario_<job_id>.err
```

### Job Resource Allocation

**Current settings**:
- **Account**: m4138_g (GPU allocation)
- **Time**: 1 hour
- **GPUs**: 1
- **CPUs**: 16
- **Memory**: Shared (sufficient for video recording)

**Adjust if needed** (lines 2-9):
```bash
#SBATCH --time=02:00:00         # Increase for more episodes
#SBATCH --cpus-per-task=32      # More CPUs if needed
#SBATCH --constraint=cpu        # Use CPU nodes if GPU busy
```

### Output

**Console output location**:
```
logs/record_mario_<job_id>.out
```

**Sample successful output**:
```
========================================================================
QUANTUM MARIO VIDEO RECORDING - SLURM JOB
========================================================================
Job ID: 12345678
Node: nid001234
Start time: Thu Jun 22 14:30:52 PDT 2025
========================================================================

✓ Checkpoint loaded (Episode 999)
  • Total steps: 500000
  • Exploration rate: 0.1000

✓ Model created and weights loaded

Episode 1/5
  Recording to: mario_videos_qcnn/quantum_mario_episode_1_20250622_143052.mp4
✓ Saved video: mario_videos_qcnn/quantum_mario_episode_1_20250622_143052.mp4 (1234 frames)
  • Total reward: 245.0
  • Steps: 1234
  • X position: 512
  • Flag reached: False

...

✓ SUCCESS! Videos saved to: mario_videos_qcnn/
```

---

## 🎬 Advanced Usage

### Recording Multiple Checkpoints

Create a script to record from multiple checkpoints:

```bash
#!/bin/bash
# record_multiple.sh

CHECKPOINTS=(
    "SuperMarioCheckpoints/QuantumMario_Run16"
    "SuperMarioCheckpoints/QuantumMario_Run17"
    "SuperMarioCheckpoints/QuantumMario_Run18"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
    python record_quantum_mario_video.py \
        --checkpoint-dir=$checkpoint \
        --num-episodes=1 \
        --output-dir=mario_videos_comparison
done
```

### Custom Video Settings

```bash
# High FPS (smoother video)
python record_quantum_mario_video.py \
    --checkpoint-dir=SuperMarioCheckpoints/QuantumMario_Run16 \
    --fps=60

# More episodes
python record_quantum_mario_video.py \
    --checkpoint-dir=SuperMarioCheckpoints/QuantumMario_Run16 \
    --num-episodes=10

# Custom output directory
python record_quantum_mario_video.py \
    --checkpoint-dir=SuperMarioCheckpoints/QuantumMario_Run16 \
    --output-dir=videos_for_paper
```

### Converting Video Format

```bash
# Convert to GIF (for presentations)
ffmpeg -i mario_videos_qcnn/quantum_mario_episode_1_*.mp4 \
    -vf "fps=15,scale=480:-1:flags=lanczos" \
    mario_episode1.gif

# Compress video (reduce file size)
ffmpeg -i mario_videos_qcnn/quantum_mario_episode_1_*.mp4 \
    -vcodec libx264 -crf 28 \
    mario_episode1_compressed.mp4

# Extract frames as images
ffmpeg -i mario_videos_qcnn/quantum_mario_episode_1_*.mp4 \
    frames/frame_%04d.png
```

---

## 🐛 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'gym'"

**Solution**: Activate conda environment
```bash
source activate ./conda-envs/qml_eeg
```

### Issue 2: "Checkpoint not found"

**Solution**: Verify checkpoint path
```bash
ls -l SuperMarioCheckpoints/QuantumMario_Run16/latest_checkpoint.chkpt
```

### Issue 3: "ImportError: cannot import QuantumMarioNet"

**Solution**: Ensure you're in the correct directory
```bash
cd /pscratch/sd/j/junghoon
ls QuantumSuperMario.py  # Should exist
```

### Issue 4: Model parameter mismatch

**Error**: "Error(s) in loading state_dict"

**Solution**: Ensure quantum parameters match training:
- For QCNN: Check `n_qubits_per_chip`, `circuit_depth_per_chip`, `num_chips`
- For Transformer: Check `n_qubits`, `n_layers`, `degree`, `feature_dim`

**How to find parameters**:
```bash
# Check training script parameters
grep -E "n-qubits|n-layers|degree" SuperMarioCheckpoints/*/log.txt | head -1

# Or check checkpoint
python -c "
import torch
ckpt = torch.load('SuperMarioCheckpoints/QuantumMario_Run16/latest_checkpoint.chkpt', map_location='cpu')
print(ckpt.keys())
"
```

### Issue 5: Video file is empty (0 bytes)

**Cause**: OpenCV couldn't initialize video writer

**Solution**: Check OpenCV installation
```bash
python -c "import cv2; print(cv2.__version__)"

# If missing:
pip install opencv-python
```

### Issue 6: CUDA out of memory

**Solution**: Use CPU
```bash
python record_quantum_mario_video.py \
    --checkpoint-dir=SuperMarioCheckpoints/QuantumMario_Run16 \
    --device=cpu
```

---

## 📊 Video Metadata

### Extracting Video Information

```bash
# Duration and frame count
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 video.mp4

# Detailed info
ffmpeg -i video.mp4

# Frame rate
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 video.mp4
```

### Organizing Videos

```bash
# Create directory structure
mkdir -p videos/{qcnn,transformer}/{successful,failed}

# Move successful runs (flag reached)
# (You'll need to check game info from logs)

# Rename for clarity
mv quantum_mario_episode_1_*.mp4 qcnn_run16_episode1.mp4
```

---

## 📋 Checklist

Before recording videos:

- [ ] Checkpoint exists and is accessible
- [ ] Conda environment activated
- [ ] Correct script selected (QCNN vs Transformer)
- [ ] Quantum parameters match training configuration
- [ ] Output directory has write permissions
- [ ] Sufficient disk space (~20 MB per episode)
- [ ] GPU available (or using `--device=cpu`)

After recording:

- [ ] Videos saved successfully (check file size > 0)
- [ ] Copy videos to local machine
- [ ] Document which checkpoint/run created which video
- [ ] Back up important videos

---

## 🎯 Quick Reference Commands

```bash
# Submit batch job
sbatch run_record_mario_video.sh

# Check job status
squeue -u junghoon

# View job output
tail -f logs/record_mario_<job_id>.out

# List videos
ls -lh mario_videos_*/

# Copy videos to local
scp junghoon@perlmutter.nersc.gov:/pscratch/sd/j/junghoon/mario_videos_*/*.mp4 ~/Downloads/

# Check disk usage
du -sh mario_videos_*/

# Delete old videos (be careful!)
rm mario_videos_*/quantum_mario_episode_*.mp4
```

---

## 📞 Summary

**Three files created**:
1. **`record_quantum_mario_video.py`** - For QCNN agents
2. **`record_quantum_mario_video_TRANSFORMER.py`** - For Transformer agents
3. **`run_record_mario_video.sh`** - Batch script for both

**To record videos**:
1. Edit `run_record_mario_video.sh` (choose agent type, checkpoint)
2. Submit: `sbatch run_record_mario_video.sh`
3. Wait for completion (check logs)
4. Copy videos: `scp ... *.mp4 ~/Downloads/`

**Videos are saved in**:
- `/pscratch/sd/j/junghoon/mario_videos_qcnn/` (QCNN)
- `/pscratch/sd/j/junghoon/mario_videos_transformer/` (Transformer)

---

Happy video recording! 🎮📹✨
