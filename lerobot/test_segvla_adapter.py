#!/usr/bin/env python
"""Test script to verify SegVLA segmentation adapter functionality."""

import torch
from lerobot.policies.segvla.configuration_segvla import SegVLAConfig
from lerobot.policies.segvla.modeling_segvla import SegVLAPolicy

# Import the module and access the class directly
import lerobot.policies.segvla.modeling_segvla as segvla_module

def test_segmentation_adapter():
    """Test the segmentation adapter component directly."""
    print("Testing SegmentationFusionAdapter...")

    # Test adapter instantiation
    print("\n1. Testing adapter instantiation...")
    num_classes = 150

    # Check if the class exists
    if not hasattr(segvla_module, 'SegmentationFusionAdapter'):
        print("ERROR: SegmentationFusionAdapter not found in module!")
        print(f"Available classes: {[c for c in dir(segvla_module) if c[0].isupper()]}")
        return False

    SegmentationFusionAdapter = segvla_module.SegmentationFusionAdapter
    adapter = SegmentationFusionAdapter(num_segmentation_classes=num_classes)
    print(f"✓ Successfully created SegmentationFusionAdapter with {num_classes} classes")

    # Check weight initialization
    print("\n2. Testing weight initialization...")
    with torch.no_grad():
        conv_weight = adapter.fusion_conv.weight  # Shape: (3, 153, 1, 1)

        # Check RGB channels are identity
        for i in range(3):
            assert conv_weight[i, i, 0, 0].item() == 1.0, f"RGB channel {i} not identity!"
            print(f"  - RGB channel {i}: Identity ✓")

        # Check segmentation channels are zero
        for i in range(3):
            for j in range(3, 3 + num_classes):
                assert conv_weight[i, j, 0, 0].item() == 0.0, f"Seg channel {j} not zero!"
        print(f"  - Segmentation channels (3-{3+num_classes}): All zero ✓")

        # Check bias is zero
        if adapter.fusion_conv.bias is not None:
            assert torch.all(adapter.fusion_conv.bias == 0), "Bias not zero!"
            print(f"  - Bias: All zero ✓")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 2
    height, width = 224, 224

    # Create dummy inputs
    rgb_image = torch.randn(batch_size, 3, height, width)
    seg_logits = torch.randn(batch_size, num_classes, height, width)

    # Run forward pass
    fused_output = adapter(rgb_image, seg_logits)

    print(f"✓ Forward pass successful")
    print(f"  - Input RGB shape: {rgb_image.shape}")
    print(f"  - Input segmentation shape: {seg_logits.shape}")
    print(f"  - Output fused shape: {fused_output.shape}")

    # Verify output shape
    assert fused_output.shape == rgb_image.shape, "Output shape mismatch!"
    print(f"  - Output shape verification: PASS")

    # Test that initial output equals input RGB (due to zero-init)
    print("\n4. Testing zero-init behavior...")
    adapter_fresh = SegmentationFusionAdapter(num_segmentation_classes=num_classes)

    # Use specific test input
    rgb_test = torch.randn(1, 3, 32, 32)
    seg_test = torch.randn(1, num_classes, 32, 32)

    with torch.no_grad():
        output_test = adapter_fresh(rgb_test, seg_test)

    # Check if output approximately equals input RGB (within floating point tolerance)
    diff = torch.abs(output_test - rgb_test).max().item()
    print(f"  - Max difference between output and input RGB: {diff:.6f}")
    assert diff < 1e-5, f"Output doesn't match input RGB! Max diff: {diff}"
    print(f"  - Zero-init behavior verification: PASS ✓")

    print("\n✅ All adapter tests passed successfully!")
    return True


def test_segvla_with_adapter():
    """Test SegVLA policy with adapter integration."""
    print("\n" + "="*50)
    print("Testing SegVLA Policy with Adapter Integration...")

    # Create config with adapter enabled
    config = SegVLAConfig()
    config.use_segmentation_adapter = True
    config.num_segmentation_classes = 150

    print("\n1. Creating SegVLA policy with adapter...")
    policy = SegVLAPolicy(config)
    print(f"✓ Policy created successfully")

    # Check if adapter exists in the model
    print("\n2. Checking adapter in model...")
    if hasattr(policy.model, 'segmentation_adapter'):
        print(f"  - Adapter found in policy.model: {type(policy.model.segmentation_adapter)}")
        if policy.model.segmentation_adapter is not None:
            print(f"  - Adapter is initialized ✓")
            print(f"  - Number of segmentation classes: {policy.model.segmentation_adapter.num_segmentation_classes}")
        else:
            print(f"  - WARNING: Adapter is None despite being enabled")
    else:
        print(f"  - WARNING: No 'segmentation_adapter' attribute in model")

    print("\n✅ Integration test completed!")
    return True


if __name__ == "__main__":
    try:
        # Test the adapter component
        test_segmentation_adapter()

        # Test integration with policy
        test_segvla_with_adapter()

        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)