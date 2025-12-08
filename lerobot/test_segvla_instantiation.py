#!/usr/bin/env python
"""Test script to verify SegVLA policy can be instantiated."""

import torch
from lerobot.policies.segvla.configuration_segvla import SegVLAConfig
from lerobot.policies.segvla.modeling_segvla import SegVLAPolicy

def test_segvla_instantiation():
    """Test that SegVLA policy can be instantiated with default config."""
    print("Testing SegVLA policy instantiation...")

    # Create default config
    config = SegVLAConfig()
    print(f"Created SegVLAConfig")

    # Test without segmentation adapter
    print("\n1. Testing without segmentation adapter...")
    config.use_segmentation_adapter = False
    policy = SegVLAPolicy(config)
    print(f"✓ Successfully instantiated SegVLAPolicy without adapter")
    print(f"  - Policy name: {policy.name}")
    print(f"  - Segmentation adapter: None (disabled)")

    # Test with segmentation adapter
    print("\n2. Testing with segmentation adapter...")
    config.use_segmentation_adapter = True
    config.num_segmentation_classes = 150
    policy_with_seg = SegVLAPolicy(config)
    print(f"✓ Successfully instantiated SegVLAPolicy with adapter")
    print(f"  - Policy name: {policy_with_seg.name}")
    print(f"  - Segmentation adapter enabled: {config.use_segmentation_adapter}")

    # Test the segmentation adapter
    if config.use_segmentation_adapter and hasattr(policy_with_seg.model, 'segmentation_adapter') and policy_with_seg.model.segmentation_adapter is not None:
        print("\n3. Testing segmentation adapter forward pass...")
        batch_size = 2
        height, width = 224, 224

        # Create dummy inputs
        rgb_image = torch.randn(batch_size, 3, height, width)
        seg_logits = torch.randn(batch_size, config.num_segmentation_classes, height, width)

        # Run forward pass
        fused_output = policy_with_seg.model.segmentation_adapter(rgb_image, seg_logits)
        print(f"✓ Segmentation adapter forward pass successful")
        print(f"  - Input RGB shape: {rgb_image.shape}")
        print(f"  - Input segmentation shape: {seg_logits.shape}")
        print(f"  - Output fused shape: {fused_output.shape}")

        # Verify output shape matches input RGB shape
        assert fused_output.shape == rgb_image.shape, "Output shape mismatch!"
        print(f"  - Output shape verification: PASS")

    print("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_segvla_instantiation()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)