import ray
import torch
from PIL import Image
from typing import Dict, Any
import base64
import io
import logging
from transformers import CLIPProcessor, CLIPModel

from model_zoo.actors.base import HFModelActor

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class CLIPActor(HFModelActor):
    """CLIP model actor for image-text similarity."""
    
    async def _load_model(self):
        """Load CLIP model and processor."""
        # For now, use default model. In production, load from weights_uri
        model_name = "openai/clip-vit-base-patch32"
        
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"CLIP model loaded on {self.device}")
    
    async def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate image-text similarity.
        
        Args:
            payload: Dict with keys:
                - image_b64: Base64 encoded image
                - text: Text to compare with image
                
        Returns:
            Dict with similarity score
        """
        if not await self.ready():
            raise RuntimeError("Model not ready")
        
        try:
            # Decode image
            image_bytes = base64.b64decode(payload["image_b64"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Get text
            text = payload["text"]
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                similarity = probs.cpu().numpy()[0, 0].item()
            
            return {"similarity": float(similarity)}
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise