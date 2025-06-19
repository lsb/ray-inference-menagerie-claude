import os
from typing import Type
from model_zoo.actors.base import HFModelActor


def make() -> Type[HFModelActor]:
    """Factory function to create the appropriate model actor.
    
    Returns:
        The actor class for the specified model
    """
    model_name = os.environ.get("MODEL_NAME", "").lower()
    
    # Import here to avoid circular dependencies
    if "clip" in model_name:
        from model_zoo.actors.clip import CLIPActor
        return CLIPActor
    elif "grounding" in model_name or "sam2" in model_name:
        from model_zoo.actors.grounding_sam2 import GroundingDINO_SAM2_Actor
        return GroundingDINO_SAM2_Actor
    elif "qwen" in model_name:
        from model_zoo.actors.qwen_vl import QwenVLActor
        return QwenVLActor
    else:
        raise ValueError(f"Unknown model name: {model_name}")