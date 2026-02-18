"""
Quantum RL Setup Verification Script
====================================
This script tests that all dependencies are correctly installed
and provides diagnostic information for troubleshooting.
"""

import sys

print("="*80)
print("QUANTUM RL SETUP VERIFICATION")
print("="*80)
print()

# Test 1: Python version
print("✓ Testing Python version...")
print(f"  Python {sys.version}")
if sys.version_info < (3, 8):
    print("  ⚠️  WARNING: Python 3.8+ recommended")
print()

# Test 2: PyTorch
print("✓ Testing PyTorch...")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print("  ✅ PyTorch OK")
except ImportError as e:
    print(f"  ❌ ERROR: {e}")
    print("  Install: pip install torch torchvision")
print()

# Test 3: PennyLane
print("✓ Testing PennyLane...")
try:
    import pennylane as qml
    print(f"  PennyLane version: {qml.__version__}")

    # Test quantum device
    dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev, interface="torch")
    def test_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    result = test_circuit()
    print(f"  Test circuit executed: result={result}")
    print("  ✅ PennyLane OK")
except ImportError as e:
    print(f"  ❌ ERROR: {e}")
    print("  Install: pip install pennylane")
except Exception as e:
    print(f"  ⚠️  WARNING: PennyLane installed but test failed: {e}")
print()

# Test 4: Gym
print("✓ Testing OpenAI Gym...")
try:
    import gym
    print(f"  Gym version: {gym.__version__}")

    # Test CartPole environment
    env = gym.make("CartPole-v1")
    state = env.reset()
    print(f"  CartPole state shape: {state.shape if hasattr(state, 'shape') else type(state)}")
    env.close()
    print("  ✅ Gym OK")
except ImportError as e:
    print(f"  ❌ ERROR: {e}")
    print("  Install: pip install gym")
except Exception as e:
    print(f"  ⚠️  WARNING: Gym installed but test failed: {e}")
print()

# Test 5: Super Mario Bros
print("✓ Testing Super Mario Bros environment...")
try:
    import gym_super_mario_bros
    import nes_py
    print(f"  gym-super-mario-bros installed")
    print(f"  nes-py installed")

    # Test environment creation
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    print(f"  Super Mario environment created")
    env.close()
    print("  ✅ Super Mario Bros OK")
except ImportError as e:
    print(f"  ⚠️  WARNING: {e}")
    print("  Install: pip install gym-super-mario-bros nes-py")
    print("  (Optional - only needed for Super Mario)")
except Exception as e:
    print(f"  ⚠️  WARNING: Package installed but test failed: {e}")
print()

# Test 6: Numpy
print("✓ Testing NumPy...")
try:
    import numpy as np
    print(f"  NumPy version: {np.__version__}")
    print("  ✅ NumPy OK")
except ImportError as e:
    print(f"  ❌ ERROR: {e}")
    print("  Install: pip install numpy")
print()

# Test 7: Matplotlib
print("✓ Testing Matplotlib...")
try:
    import matplotlib
    print(f"  Matplotlib version: {matplotlib.__version__}")
    print("  ✅ Matplotlib OK")
except ImportError as e:
    print(f"  ❌ ERROR: {e}")
    print("  Install: pip install matplotlib")
print()

# Test 8: TorchRL (for Super Mario)
print("✓ Testing TorchRL...")
try:
    from tensordict import TensorDict
    from torchrl.data import TensorDictReplayBuffer
    print(f"  TensorDict installed")
    print(f"  TorchRL installed")
    print("  ✅ TorchRL OK")
except ImportError as e:
    print(f"  ⚠️  WARNING: {e}")
    print("  Install: pip install tensordict torchrl")
    print("  (Optional - only needed for Super Mario)")
print()

# Summary
print("="*80)
print("SETUP SUMMARY")
print("="*80)
print()
print("Core Dependencies (Required for Simple RL):")
print("  • PyTorch: ", end="")
try:
    import torch
    print("✅")
except:
    print("❌")

print("  • PennyLane: ", end="")
try:
    import pennylane
    print("✅")
except:
    print("❌")

print("  • Gym: ", end="")
try:
    import gym
    print("✅")
except:
    print("❌")

print("  • NumPy: ", end="")
try:
    import numpy
    print("✅")
except:
    print("❌")

print("  • Matplotlib: ", end="")
try:
    import matplotlib
    print("✅")
except:
    print("❌")

print()
print("Optional Dependencies (For Super Mario):")
print("  • gym-super-mario-bros: ", end="")
try:
    import gym_super_mario_bros
    print("✅")
except:
    print("❌")

print("  • nes-py: ", end="")
try:
    import nes_py
    print("✅")
except:
    print("❌")

print("  • TorchRL: ", end="")
try:
    from torchrl.data import TensorDictReplayBuffer
    print("✅")
except:
    print("❌")

print()
print("="*80)
print("RECOMMENDED NEXT STEPS")
print("="*80)
print()

# Check if all core dependencies are installed
try:
    import torch
    import pennylane
    import gym
    import numpy
    import matplotlib

    print("✅ All core dependencies installed!")
    print()
    print("You can now run:")
    print("  1. Simple RL environments:")
    print("     python QuantumTransformerSimpleRL.py --env=CartPole-v1")
    print()

    try:
        import gym_super_mario_bros
        import nes_py
        from torchrl.data import TensorDictReplayBuffer
        print("  2. Super Mario Bros:")
        print("     python QuantumTransformerMario.py")
    except:
        print("  2. Install Super Mario dependencies to run QuantumTransformerMario.py:")
        print("     pip install gym-super-mario-bros nes-py tensordict torchrl")

except ImportError:
    print("⚠️  Some core dependencies are missing!")
    print()
    print("Install missing packages:")
    print("  pip install torch pennylane gym numpy matplotlib")
    print()
    print("Then run this test again:")
    print("  python test_quantum_rl_setup.py")

print()
print("="*80)
print()
