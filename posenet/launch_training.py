#!/usr/bin/env python3
"""
Launch script for multi-GPU PoseNet training using Accelerate.

Usage:
    python launch_training.py
    
Or with custom config:
    accelerate launch --config_file accelerate_config.yaml train_accelerated.py
"""

import subprocess
import sys
import os

def main():
    """Launch multi-GPU training with accelerate"""
    
    # Check if accelerate is available
    try:
        import accelerate
        print(f"Using accelerate version: {accelerate.__version__}")
    except ImportError:
        print("Error: accelerate is not installed. Please install it with:")
        print("pip install accelerate")
        sys.exit(1)
    
    # Check if config file exists
    config_file = "accelerate_config.yaml"
    if not os.path.exists(config_file):
        print(f"Warning: {config_file} not found. Using default accelerate config.")
        config_file = None
    
    # Prepare the command
    cmd = ["accelerate", "launch"]
    
    if config_file:
        cmd.extend(["--config_file", config_file])
    else:
        # Use command line arguments for multi-GPU setup
        cmd.extend([
            "--multi_gpu",
            "--num_processes", "4",
            "--mixed_precision", "fp16"
        ])
    
    cmd.append("train_accelerated.py")
    
    print("Launching multi-GPU training with command:")
    print(" ".join(cmd))
    print("-" * 50)
    
    # Launch the training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 