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
        # Check if we have a local weights file
        if self.weights_uri and self.weights_uri.endswith('.pytorch'):
            # Load from local weights file
            logger.info(f"Loading CLIP model from local weights: {self.weights_uri}")
            
            # Load the model architecture first
            model_name = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # Load the weights from local file
            if self.weights_uri.startswith("file://"):
                weights_path = self.weights_uri[7:]  # Remove file:// prefix
            else:
                weights_path = self.weights_uri
                
            # Load state dict from the .pytorch file
            state_dict = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            # Load from HuggingFace or use default
            model_name = "openai/clip-vit-base-patch32"
            logger.info(f"Loading CLIP model from HuggingFace: {model_name}")
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
                # Get raw similarity logits from CLIP 
                # CLIP outputs logits scaled by learnable temperature parameter
                # We'll use the raw cosine similarity before temperature scaling
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                
                # Compute cosine similarity (dot product of normalized vectors)
                cosine_similarity = torch.sum(image_embeds * text_embeds, dim=-1)
                raw_similarity = cosine_similarity.cpu().numpy()[0].item()
                
                # Convert from [-1, 1] to [0, 1] range for API consistency
                normalized_similarity = (raw_similarity + 1.0) / 2.0
            
            return {"similarity": float(normalized_similarity)}
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise