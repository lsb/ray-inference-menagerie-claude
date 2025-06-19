import ray
import os
import time
import logging
from model_zoo.actors import factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main driver function that initializes Ray and creates the model actor."""
    # Connect to Ray cluster
    ray.init(address="auto")
    logger.info("Connected to Ray cluster")
    
    # Get configuration from environment
    model_name = os.environ.get("MODEL_NAME", "model")
    weights_uri = os.environ.get("WEIGHTS_URI", "")
    
    logger.info(f"Creating actor for model: {model_name}")
    logger.info(f"Weights URI: {weights_uri}")
    
    # Create the appropriate actor
    actor_class = factory.make()
    actor = actor_class.options(
        name=model_name,
        num_gpus=1,
        max_concurrency=10,
        lifetime="detached"
    ).remote(model_name, weights_uri)
    
    # Wait for actor to be ready
    logger.info("Waiting for actor to be ready...")
    ready = False
    while not ready:
        try:
            ready = ray.get(actor.ready.remote(), timeout=10)
            if ready:
                logger.info(f"Actor {model_name} is ready for inference")
            else:
                logger.warning("Actor not ready yet, retrying...")
                time.sleep(5)
        except Exception as e:
            logger.error(f"Error checking actor readiness: {e}")
            time.sleep(5)
    
    # Keep driver alive
    logger.info("Driver ready, keeping process alive...")
    try:
        while True:
            time.sleep(30)
            # Periodically check actor health
            try:
                is_ready = ray.get(actor.ready.remote(), timeout=5)
                if not is_ready:
                    logger.warning("Actor reported not ready")
            except Exception as e:
                logger.error(f"Health check failed: {e}")
    except KeyboardInterrupt:
        logger.info("Driver shutting down...")


if __name__ == "__main__":
    main()