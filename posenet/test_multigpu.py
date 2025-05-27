#!/usr/bin/env python3
"""
Test script for multi-GPU PoseNet training setup.
This script verifies that all components work correctly before starting full training.
"""

import torch
import sys
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_improved import ImprovedPoseNet, PoseNet
from dataset.tartanair import TartanAirDataset

def test_model_forward():
    """Test model forward pass with dummy data"""
    print("Testing model forward pass...")
    
    # Test original PoseNet
    model_orig = PoseNet()
    
    # Test improved PoseNet with different backbones
    model_resnet50 = ImprovedPoseNet(backbone='resnet50', hidden_dim=256, use_attention=True)
    model_resnet34 = ImprovedPoseNet(backbone='resnet34', hidden_dim=256, use_attention=True)
    model_vgg16 = ImprovedPoseNet(backbone='vgg16', hidden_dim=256, use_attention=False)
    
    # Create dummy input
    batch_size, seq_len = 2, 6
    dummy_input = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    models = {
        'Original PoseNet': model_orig,
        'ResNet50': model_resnet50,
        'ResNet34': model_resnet34,
        'VGG16': model_vgg16
    }
    
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                t_pred, q_pred = model(dummy_input)
            
            print(f"✓ {name}: Output shapes - Translation: {t_pred.shape}, Rotation: {q_pred.shape}")
            
            # Test loss computation
            dummy_t_gt = torch.randn_like(t_pred)
            dummy_q_gt = torch.randn_like(q_pred)
            dummy_q_gt = torch.nn.functional.normalize(dummy_q_gt, p=2, dim=2)
            
            if hasattr(model, 'pose_loss') and 'ImprovedPoseNet' in str(type(model)):
                loss, loss_dict = model.pose_loss(t_pred, q_pred, dummy_t_gt, dummy_q_gt)
                print(f"  Loss components: {list(loss_dict.keys())}")
            else:
                loss = model.pose_loss(t_pred, q_pred, dummy_t_gt, dummy_q_gt)
                print(f"  Loss: {loss.item():.6f}")
            
        except Exception as e:
            print(f"✗ {name}: Error - {str(e)}")
    
    print()

def test_accelerator_setup():
    """Test Accelerator initialization"""
    print("Testing Accelerator setup...")
    
    try:
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16"
        )
        
        print(f"✓ Accelerator initialized successfully")
        print(f"  Device: {accelerator.device}")
        print(f"  Number of processes: {accelerator.num_processes}")
        print(f"  Mixed precision: {accelerator.mixed_precision}")
        print(f"  Is main process: {accelerator.is_main_process}")
        
        return accelerator
        
    except Exception as e:
        print(f"✗ Accelerator initialization failed: {str(e)}")
        return None

def test_data_loading():
    """Test data loading with dummy dataset"""
    print("Testing data loading...")
    
    try:
        # Create dummy dataset
        batch_size = 4
        seq_len = 6
        num_samples = 20
        
        # Generate dummy data
        images = torch.randn(num_samples, seq_len, 3, 224, 224)
        translations = torch.randn(num_samples, seq_len-1, 3)
        rotations = torch.randn(num_samples, seq_len-1, 4)
        rotations = torch.nn.functional.normalize(rotations, p=2, dim=2)
        
        dataset = TensorDataset(images, translations, rotations)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        # Test one batch
        for batch_idx, (imgs, trans, rots) in enumerate(dataloader):
            print(f"✓ Data loading successful")
            print(f"  Batch {batch_idx}: Images {imgs.shape}, Translations {trans.shape}, Rotations {rots.shape}")
            if batch_idx == 0:  # Only test first batch
                break
        
        return dataloader
        
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        return None

def test_multi_gpu_training():
    """Test multi-GPU training setup"""
    print("Testing multi-GPU training setup...")
    
    accelerator = test_accelerator_setup()
    if accelerator is None:
        return False
    
    dataloader = test_data_loading()
    if dataloader is None:
        return False
    
    try:
        # Initialize model and optimizer
        model = ImprovedPoseNet(backbone='resnet34', hidden_dim=128, use_attention=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Prepare with accelerator
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        
        print(f"✓ Model and optimizer prepared for multi-GPU")
        
        # Test one training step
        model.train()
        for batch_idx, (images, translations, rotations) in enumerate(dataloader):
            optimizer.zero_grad()
            
            t_pred, q_pred = model(images)
            loss, loss_dict = model.pose_loss(t_pred, q_pred, translations, rotations)
            
            accelerator.backward(loss)
            optimizer.step()
            
            print(f"✓ Training step successful")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Device: {images.device}")
            
            break  # Only test one step
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-GPU training test failed: {str(e)}")
        return False

def test_checkpoint_saving():
    """Test checkpoint saving and loading"""
    print("Testing checkpoint functionality...")
    
    try:
        accelerator = Accelerator()
        model = ImprovedPoseNet(backbone='resnet34', hidden_dim=128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        model, optimizer = accelerator.prepare(model, optimizer)
        
        # Create test directory
        test_dir = "test_checkpoint"
        os.makedirs(test_dir, exist_ok=True)
        
        # Save state
        accelerator.save_state(test_dir)
        print(f"✓ Checkpoint saved to {test_dir}")
        
        # Load state
        accelerator.load_state(test_dir)
        print(f"✓ Checkpoint loaded from {test_dir}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        print(f"✓ Test checkpoint cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PoseNet Multi-GPU Setup Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Run tests
    tests = [
        ("Model Forward Pass", test_model_forward),
        ("Multi-GPU Training", test_multi_gpu_training),
        ("Checkpoint Saving", test_checkpoint_saving),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your multi-GPU setup is ready.")
        print("\nTo start training, run:")
        print("  python launch_training.py")
        print("  or")
        print("  accelerate launch --config_file accelerate_config.yaml train_improved.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Make sure you have:")
        print("  - Multiple GPUs available")
        print("  - accelerate installed (pip install accelerate)")
        print("  - Proper CUDA setup")

if __name__ == "__main__":
    main() 