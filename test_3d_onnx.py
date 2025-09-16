#!/usr/bin/env python3
"""
Test script to verify 3D convolution and pooling support in ONNX loading
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nnenum.network import Convolutional3dLayer, Pooling3dLayer
from nnenum.settings import Settings

def test_3d_layer_classes():
    """Test that our 3D layer classes work correctly"""
    print("=== Testing 3D Layer Classes ===")
    
    # Test data: 4D tensor (depth=4, height=8, width=8, channels=3)
    input_shape = (4, 8, 8, 3)
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Test 3D Convolution
    print("Testing Convolutional3dLayer...")
    
    # Create 3D kernels: (output_channels=2, input_channels=3, depth=3, height=3, width=3)
    kernels = np.random.randn(2, 3, 3, 3, 3).astype(np.float32)
    biases = np.random.randn(2).astype(np.float32)
    
    conv3d_layer = Convolutional3dLayer(0, kernels, biases, input_shape)
    print(f"  Input shape: {conv3d_layer.get_input_shape()}")
    print(f"  Output shape: {conv3d_layer.get_output_shape()}")
    
    # Execute layer
    try:
        output = conv3d_layer.execute(test_input)
        print(f"  Execution successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"  ERROR in conv3d execution: {e}")
        return False
    
    # Test 3D Pooling
    print("\nTesting Pooling3dLayer...")
    
    pool_input_shape = conv3d_layer.get_output_shape()
    pool_input = output
    
    pool3d_layer = Pooling3dLayer(1, kernel_size=2, prev_layer_output_shape=pool_input_shape, method='mean')
    print(f"  Input shape: {pool3d_layer.get_input_shape()}")
    print(f"  Output shape: {pool3d_layer.get_output_shape()}")
    
    # Execute pooling layer
    try:
        pool_output = pool3d_layer.execute(pool_input)
        print(f"  Execution successful! Output shape: {pool_output.shape}")
    except Exception as e:
        print(f"  ERROR in pool3d execution: {e}")
        return False
    
    return True

def test_onnx_whitelist():
    """Test that 3D operations are in ONNX whitelist"""
    print("\n=== Testing ONNX Whitelist ===")
    
    required_ops = ['Conv3d', 'AveragePool3d', 'MaxPool3d']
    
    print("Current ONNX whitelist:")
    for op in sorted(Settings.ONNX_WHITELIST):
        print(f"  {op}")
    
    print(f"\n3D operations support:")
    all_supported = True
    for op in required_ops:
        supported = op in Settings.ONNX_WHITELIST
        print(f"  {op}: {'✓' if supported else '✗'}")
        if not supported:
            all_supported = False
    
    return all_supported

def test_zonotope_transforms():
    """Test that 3D layers can transform zonotopes (basic test)"""
    print("\n=== Testing Zonotope Transforms ===")
    
    # Simple test - create minimal 3D conv layer and check transform methods exist
    input_shape = (4, 4, 4, 2)  # small tensor
    kernels = np.random.randn(1, 2, 3, 3, 3).astype(np.float32)
    biases = np.array([0.1], dtype=np.float32)
    
    conv3d_layer = Convolutional3dLayer(0, kernels, biases, input_shape)
    
    # Check that the transform methods exist and are callable
    has_star_transform = hasattr(conv3d_layer, 'transform_star') and callable(conv3d_layer.transform_star)
    has_zono_transform = hasattr(conv3d_layer, 'transform_zono') and callable(conv3d_layer.transform_zono)
    
    print(f"  Convolutional3dLayer.transform_star exists: {'✓' if has_star_transform else '✗'}")
    print(f"  Convolutional3dLayer.transform_zono exists: {'✓' if has_zono_transform else '✗'}")
    
    # Test pooling layer as well
    pool_shape = conv3d_layer.get_output_shape()
    pool3d_layer = Pooling3dLayer(1, 2, pool_shape, method='mean')
    
    # Pooling now has transform methods for verification support
    print(f"  Pooling3dLayer created successfully: ✓")
    
    return has_star_transform and has_zono_transform

if __name__ == "__main__":
    print("Testing 3D Convolution and Pooling Support in nnenum")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic 3D layer functionality
    try:
        success &= test_3d_layer_classes()
    except Exception as e:
        print(f"ERROR in 3D layer test: {e}")
        success = False
    
    # Test 2: ONNX whitelist
    try:
        success &= test_onnx_whitelist()
    except Exception as e:
        print(f"ERROR in ONNX whitelist test: {e}")
        success = False
    
    # Test 3: Zonotope transform capability
    try:
        success &= test_zonotope_transforms()
    except Exception as e:
        print(f"ERROR in zonotope transform test: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! 3D operations should work for verification.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    print("\nNOTE: This tests the layer classes and ONNX integration readiness.")
    print("To fully test ONNX loading, you would need an actual .onnx file with 3D operations.")
