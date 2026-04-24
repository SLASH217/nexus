"""
Google Colab Setup Script for Nexus MARL Training.

Run this at the TOP of your Colab notebook to:
1. Mount Google Drive for persistent checkpoints
2. Set up PYTHONPATH for absolute imports
3. Verify environment readiness

Usage in Colab cell:
    !pip install -q nexus-rl  # If publishing to PyPI
    
    # Then run this setup
    exec(open('/content/colab_setup.py').read())
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("🚀 NEXUS MARL - COLAB SETUP")
print("=" * 70)

# ============================================================================
# Step 1: Mount Google Drive (GHOST PROGRESS FIX)
# ============================================================================
print("\n📁 Step 1: Mounting Google Drive for persistent checkpoints...")

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted at /content/drive")
    DRIVE_AVAILABLE = True
except ImportError:
    print("⚠️  Not running in Google Colab (ImportError). Skipping Drive mount.")
    DRIVE_AVAILABLE = False
except Exception as e:
    print(f"⚠️  Could not mount Google Drive: {e}")
    DRIVE_AVAILABLE = False

# ============================================================================
# Step 2: Set up PYTHONPATH for absolute imports
# ============================================================================
print("\n🔧 Step 2: Configuring PYTHONPATH for absolute imports...")

# Ensure /content is in path for nexus_rl imports
if '/content' not in sys.path:
    sys.path.insert(0, '/content')
    print("✅ Added /content to sys.path")

# Verify the nexus_rl package is discoverable
try:
    import nexus_rl
    print(f"✅ nexus_rl package found at: {nexus_rl.__file__}")
except ImportError as e:
    print(f"❌ ERROR: Could not import nexus_rl. Make sure it's in /content/nexus_rl/")
    print(f"   Error: {e}")
    sys.exit(1)

# ============================================================================
# Step 3: Set up training output directory (GHOST PROGRESS FIX)
# ============================================================================
print("\n💾 Step 3: Setting up persistent checkpoint directory...")

if DRIVE_AVAILABLE:
    # Use Google Drive for checkpoints
    checkpoint_dir = "/content/drive/MyDrive/nexus_training_v1"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✅ Checkpoint directory: {checkpoint_dir}")
    print(f"   (This will persist even if Colab disconnects)")
else:
    # Fallback to /content (will be lost if Colab disconnects)
    checkpoint_dir = "/content/nexus_rl_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"⚠️  Using /content/{checkpoint_dir} (WARNING: will be lost on disconnect)")
    print(f"   Consider mounting Google Drive for persistence")

# ============================================================================
# Step 4: Verify critical imports
# ============================================================================
print("\n🔍 Step 4: Verifying critical imports...")

try:
    from nexus_rl.server.gym_wrapper import create_nexus_env
    print("✅ gym_wrapper imported successfully")
except ImportError as e:
    print(f"❌ ERROR: Could not import gym_wrapper: {e}")
    sys.exit(1)

try:
    from nexus_rl.server import ENVConfig
    print("✅ ENVConfig imported successfully")
except ImportError as e:
    print(f"❌ ERROR: Could not import ENVConfig: {e}")
    sys.exit(1)

try:
    from trl import GRPOTrainer, GRPOConfig
    print("✅ TRL/GRPO imported successfully")
except ImportError:
    print("⚠️  WARNING: TRL not installed. Install with: pip install trl")

# ============================================================================
# Step 5: Quick handshake test
# ============================================================================
print("\n🤝 Step 5: Running environment handshake test...")

try:
    env = create_nexus_env()
    obs, info = env.reset()
    print("✅ Environment handshake SUCCESSFUL")
    print(f"   First observation length: {len(obs)} characters")
    print(f"   First 200 chars: {obs[:200]}...")
except Exception as e:
    print(f"❌ ERROR: Handshake failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Step 6: Export configuration
# ============================================================================
print("\n⚙️  Step 6: Exporting configuration for training...")

CONFIG = {
    "checkpoint_dir": checkpoint_dir,
    "drive_available": DRIVE_AVAILABLE,
    "python_path": sys.path[:3],
}

print("\n" + "=" * 70)
print("✅ SETUP COMPLETE - Ready for training!")
print("=" * 70)
print(f"\n📌 Key Settings:")
print(f"   Checkpoints → {checkpoint_dir}")
print(f"   Drive Access → {'✅ Enabled' if DRIVE_AVAILABLE else '⚠️  Disabled'}")
print(f"\n🚀 Next: Run your training cell with GRPOConfig pointing to checkpoint_dir")
print("\n   Example:")
print(f"   training_args = GRPOConfig(")
print(f"       output_dir='{checkpoint_dir}',")
print(f"       # ... rest of config")
print(f"   )")

# ============================================================================
# Return configuration for notebook use
# ============================================================================
print(f"\n💾 Available in notebook: CONFIG = {CONFIG}")
