"""SegVLA Policy - SmolVLA with Segmentation Adapter"""

from lerobot.policies.segvla.configuration_segvla import SegVLAConfig
from lerobot.policies.segvla.modeling_segvla import SegVLAPolicy

# Import processor to trigger registration
from lerobot.policies.segvla.processor_segvla import SegVLANewLineProcessor

__all__ = [
    "SegVLAConfig",
    "SegVLAPolicy",
    "SegVLANewLineProcessor",
]
