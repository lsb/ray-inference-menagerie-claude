import ray
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class HFModelActor(ABC):
    """Base class for Hugging Face model actors.
    
    Each actor uses exactly one GPU and implements async inference.
    """
    
    def __init__(self, model_name: str, weights_uri: str):
        """Initialize the model actor.
        
        Args:
            model_name: Name of the model
            weights_uri: GCS URI to model weights
        """
        self.model_name = model_name
        self.weights_uri = weights_uri
        self.model = None
        self.processor = None
        logger.info(f"Initializing {self.__class__.__name__} for {model_name}")
    
    @abstractmethod
    async def _load_model(self):
        """Load model and processor from weights URI."""
        pass
    
    async def ready(self) -> bool:
        """Check if the model is ready for inference.
        
        Returns:
            True if model is loaded and ready
        """
        if self.model is None:
            try:
                await self._load_model()
                logger.info(f"{self.model_name} loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load {self.model_name}: {e}")
                return False
        return True
    
    @abstractmethod
    async def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the provided payload.
        
        Args:
            payload: Input data for inference
            
        Returns:
            Dictionary containing inference results
        """
        pass