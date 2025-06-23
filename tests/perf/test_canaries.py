"""Performance tests for model actors."""
import pytest
import ray
import time
import asyncio
from tests.fixtures import TEST_IMAGE_B64
from model_zoo.actors.clip import CLIPActor
from model_zoo.actors.grounding_sam2 import GroundingDINO_SAM2_Actor
from model_zoo.actors.qwen_vl import QwenVLActor


@pytest.fixture(scope="module")
def ray_fixture():
    """Initialize Ray for testing."""
    ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_clip_actor_performance(ray_fixture):
    """Test CLIP actor initialization and inference performance."""
    # Create actor with local weights
    from pathlib import Path
    weights_path = Path(__file__).parent.parent.parent / "clip-vit-base-patch32.pytorch"
    actor = CLIPActor.remote("clip-test", str(weights_path))
    
    # Test initialization
    start_time = time.time()
    ready = await actor.ready.remote()
    init_time = time.time() - start_time
    
    assert ready, "CLIP actor failed to initialize"
    assert init_time < 30, f"CLIP initialization took {init_time:.2f}s (> 30s)"
    
    # Test inference
    payload = {
        "image_b64": TEST_IMAGE_B64,
        "text": "a red square with a white center"
    }
    
    start_time = time.time()
    result = await actor.infer.remote(payload)
    inference_time = time.time() - start_time
    
    assert "similarity" in result
    assert isinstance(result["similarity"], float)
    assert 0 <= result["similarity"] <= 1
    assert inference_time < 5, f"CLIP inference took {inference_time:.2f}s (> 5s)"


@pytest.mark.asyncio
async def test_grounding_sam2_actor_performance(ray_fixture):
    """Test Grounding DINO + SAM2 actor performance."""
    # Create actor
    actor = GroundingDINO_SAM2_Actor.remote("grounding-sam2-test", "gs://fake/weights")
    
    # Test initialization
    start_time = time.time()
    ready = await actor.ready.remote()
    init_time = time.time() - start_time
    
    assert ready, "Grounding DINO + SAM2 actor failed to initialize"
    assert init_time < 30, f"Initialization took {init_time:.2f}s (> 30s)"
    
    # Test inference
    payload = {
        "image_b64": TEST_IMAGE_B64,
        "text_prompt": "white square"
    }
    
    start_time = time.time()
    result = await actor.infer.remote(payload)
    inference_time = time.time() - start_time
    
    assert "mask_png_b64" in result
    assert isinstance(result["mask_png_b64"], str)
    assert inference_time < 5, f"Inference took {inference_time:.2f}s (> 5s)"


@pytest.mark.asyncio
async def test_qwen_vl_actor_performance(ray_fixture):
    """Test Qwen VL actor performance."""
    # Create actor
    actor = QwenVLActor.remote("qwen-vl-test", "gs://fake/weights")
    
    # Test initialization
    start_time = time.time()
    ready = await actor.ready.remote()
    init_time = time.time() - start_time
    
    assert ready, "Qwen VL actor failed to initialize"
    assert init_time < 30, f"Initialization took {init_time:.2f}s (> 30s)"
    
    # Test inference
    payload = {
        "image_b64": TEST_IMAGE_B64,
        "question": "What color is the background?"
    }
    
    start_time = time.time()
    result = await actor.infer.remote(payload)
    inference_time = time.time() - start_time
    
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert inference_time < 5, f"Inference took {inference_time:.2f}s (> 5s)"


@pytest.mark.asyncio
async def test_all_actors_concurrent(ray_fixture):
    """Test all actors running concurrently."""
    # Create all actors
    from pathlib import Path
    weights_path = Path(__file__).parent.parent.parent / "clip-vit-base-patch32.pytorch"
    clip_actor = CLIPActor.remote("clip-concurrent", str(weights_path))
    grounding_actor = GroundingDINO_SAM2_Actor.remote("grounding-concurrent", "gs://fake/weights")
    qwen_actor = QwenVLActor.remote("qwen-concurrent", "gs://fake/weights")
    
    # Wait for all to be ready
    ready_tasks = [
        clip_actor.ready.remote(),
        grounding_actor.ready.remote(),
        qwen_actor.ready.remote()
    ]
    
    start_time = time.time()
    ready_results = await asyncio.gather(*[ray.get(task) for task in ready_tasks])
    total_init_time = time.time() - start_time
    
    assert all(ready_results), "Some actors failed to initialize"
    assert total_init_time < 60, f"Concurrent init took {total_init_time:.2f}s (> 60s)"
    
    # Run concurrent inference
    inference_tasks = [
        clip_actor.infer.remote({"image_b64": TEST_IMAGE_B64, "text": "test"}),
        grounding_actor.infer.remote({"image_b64": TEST_IMAGE_B64, "text_prompt": "object"}),
        qwen_actor.infer.remote({"image_b64": TEST_IMAGE_B64, "question": "What is this?"})
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*[ray.get(task) for task in inference_tasks])
    total_inference_time = time.time() - start_time
    
    assert len(results) == 3
    assert "similarity" in results[0]
    assert "mask_png_b64" in results[1]
    assert "answer" in results[2]
    assert total_inference_time < 10, f"Concurrent inference took {total_inference_time:.2f}s (> 10s)"