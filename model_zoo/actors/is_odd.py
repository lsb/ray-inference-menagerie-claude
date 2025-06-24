import ray
import logging
from typing import Dict, Any
from model_zoo.actors.base import HFModelActor

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0)  # No GPU needed for this simple demo
class IsOddActor(HFModelActor):
    """Demo actor that determines if a number is odd. Useful for measuring Ray overhead."""
    
    async def _load_model(self):
        """No model to load - this is a simple computational actor."""
        logger.info("IsOdd demo actor loaded (no model required)")
        # Set model to a dummy value to satisfy base class check
        self.model = "no-model-needed"
    
    async def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if a number is odd.
        
        Args:
            payload: Dict with keys:
                - number: Integer to check
                
        Returns:
            Dict with is_odd boolean result
        """
        if not await self.ready():
            raise RuntimeError("Actor not ready")
        
        try:
            # Get the number from payload
            number = payload.get("number")
            
            if number is None:
                raise ValueError("'number' field is required in payload")
            
            # Convert to int if needed
            if isinstance(number, str):
                number = int(number)
            elif not isinstance(number, int):
                raise ValueError(f"Number must be an integer, got {type(number)}")
            
            # Simple modulo operation to check if odd
            is_odd = bool(number % 2 == 1)
            
            logger.debug(f"Computed: {number} is {'odd' if is_odd else 'even'}")
            
            return {
                "is_odd": is_odd,
                "number": number,
                "result": "odd" if is_odd else "even"
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise