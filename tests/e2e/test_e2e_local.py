#!/usr/bin/env python3
"""End-to-end test with Ray actors and real models."""
import ray
import asyncio
import base64
import time
from pathlib import Path
from PIL import Image
import io


# Test image paths  
FIXTURES_DIR = Path(__file__).parent.parent.parent / "test_images" / "fixtures"
TEST_IMAGES = {
    "cat_office": FIXTURES_DIR / "cat_office_typing.jpg",
    "dog_office": FIXTURES_DIR / "dog_office_typing.jpg", 
    "cat_mountain": FIXTURES_DIR / "cat_mountain_sunrise.jpg",
    "dog_mountain": FIXTURES_DIR / "dog_mountain_sunrise.jpg"
}


def load_image_as_b64(image_path: Path) -> str:
    """Load image file and encode as base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def test_clip_actor_e2e():
    """Test CLIP actor end-to-end with Ray remote actors."""
    print("🚀 Starting End-to-End Ray Actor Test")
    print(f"Ray version: {ray.__version__}")
    
    # Initialize Ray
    print("\n1. Initializing Ray...")
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(num_cpus=4, num_gpus=0)  # CPU only for this test
    print(f"   ✓ Ray initialized with {ray.cluster_resources()}")
    
    # Import and create CLIP actor
    print("\n2. Creating CLIP actor...")
    from model_zoo.actors.clip import CLIPActor
    
    # Create actor with CPU-only config (override GPU requirement for testing)
    # Use local weights file
    weights_path = Path(__file__).parent.parent.parent / "clip-vit-base-patch32.pytorch"
    actor = CLIPActor.options(num_gpus=0).remote("clip-e2e-test", str(weights_path))
    print("   ✓ CLIP actor created")
    
    # Test actor readiness
    print("\n3. Testing actor readiness...")
    start_time = time.time()
    ready = await actor.ready.remote()
    load_time = time.time() - start_time
    print(f"   ✓ Actor ready: {ready} (load time: {load_time:.2f}s)")
    
    if not ready:
        print("   ✗ Actor failed to load!")
        return False
    
    # Test inference with photorealistic images
    print("\n4. Testing inference with photorealistic images...")
    
    results = {}
    for img_name, img_path in TEST_IMAGES.items():
        if not img_path.exists():
            print(f"   ! Skipping {img_name} - file not found")
            continue
            
        print(f"\n   Testing {img_name}...")
        
        # Load image
        image_b64 = load_image_as_b64(img_path)
        
        # Test with appropriate text prompts
        if "cat" in img_name:
            # Test cat image with cat and dog prompts
            cat_payload = {"image_b64": image_b64, "text": "a cat"}
            dog_payload = {"image_b64": image_b64, "text": "a dog"}
            
            start_time = time.time()
            cat_result = await actor.infer.remote(cat_payload)
            cat_time = time.time() - start_time
            
            start_time = time.time()
            dog_result = await actor.infer.remote(dog_payload)
            dog_time = time.time() - start_time
            
            cat_sim = cat_result["similarity"]
            dog_sim = dog_result["similarity"] 
            
            print(f"     Cat similarity: {cat_sim:.3f} ({cat_time:.2f}s)")
            print(f"     Dog similarity: {dog_sim:.3f} ({dog_time:.2f}s)")
            print(f"     ✓ Classification: {'CORRECT' if cat_sim > dog_sim else 'INCORRECT'}")
            
            results[img_name] = {
                "cat_similarity": cat_sim,
                "dog_similarity": dog_sim,
                "correct": cat_sim > dog_sim,
                "inference_time": (cat_time + dog_time) / 2
            }
            
        elif "dog" in img_name:
            # Test dog image with cat and dog prompts
            cat_payload = {"image_b64": image_b64, "text": "a cat"}
            dog_payload = {"image_b64": image_b64, "text": "a dog"}
            
            start_time = time.time()
            cat_result = await actor.infer.remote(cat_payload)
            cat_time = time.time() - start_time
            
            start_time = time.time()
            dog_result = await actor.infer.remote(dog_payload)
            dog_time = time.time() - start_time
            
            cat_sim = cat_result["similarity"]
            dog_sim = dog_result["similarity"]
            
            print(f"     Cat similarity: {cat_sim:.3f} ({cat_time:.2f}s)")
            print(f"     Dog similarity: {dog_sim:.3f} ({dog_time:.2f}s)")
            print(f"     ✓ Classification: {'CORRECT' if dog_sim > cat_sim else 'INCORRECT'}")
            
            results[img_name] = {
                "cat_similarity": cat_sim,
                "dog_similarity": dog_sim,
                "correct": dog_sim > cat_sim,
                "inference_time": (cat_time + dog_time) / 2
            }
    
    # Test concurrent inference
    print("\n5. Testing concurrent inference...")
    if len(TEST_IMAGES) >= 2:
        concurrent_payloads = []
        for img_name, img_path in list(TEST_IMAGES.items())[:2]:
            if img_path.exists():
                image_b64 = load_image_as_b64(img_path)
                concurrent_payloads.append({
                    "image_b64": image_b64, 
                    "text": "a cat" if "cat" in img_name else "a dog"
                })
        
        if len(concurrent_payloads) >= 2:
            start_time = time.time()
            concurrent_results = await asyncio.gather(*[
                actor.infer.remote(payload) for payload in concurrent_payloads
            ])
            concurrent_time = time.time() - start_time
            
            print(f"   ✓ Concurrent inference: {len(concurrent_results)} requests in {concurrent_time:.2f}s")
            print(f"   ✓ Average time per request: {concurrent_time/len(concurrent_results):.2f}s")
    
    # Summary
    print("\n6. 🎯 End-to-End Test Summary:")
    print("=" * 50)
    
    total_tests = len(results)
    correct_tests = sum(1 for r in results.values() if r["correct"])
    avg_inference_time = sum(r["inference_time"] for r in results.values()) / len(results) if results else 0
    
    print(f"Total tests: {total_tests}")
    print(f"Correct classifications: {correct_tests}/{total_tests}")
    print(f"Accuracy: {correct_tests/total_tests*100:.1f}%" if total_tests > 0 else "No tests")
    print(f"Average inference time: {avg_inference_time:.2f}s")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for img_name, result in results.items():
        status = "✅" if result["correct"] else "❌"
        print(f"  {status} {img_name}: {result['inference_time']:.2f}s")
    
    # Cleanup
    print(f"\n7. Cleaning up...")
    ray.shutdown()
    print("   ✓ Ray shutdown complete")
    
    print(f"\n🎉 End-to-End Test Complete!")
    return correct_tests == total_tests


async def main():
    """Run the full end-to-end test."""
    success = await test_clip_actor_e2e()
    if success:
        print("\n✅ ALL TESTS PASSED! Ray actors working perfectly with photorealistic images.")
    else:
        print("\n❌ Some tests failed. Check the results above.")
    return success


if __name__ == "__main__":
    # Check that test images exist
    missing_images = [name for name, path in TEST_IMAGES.items() if not path.exists()]
    if missing_images:
        print(f"❌ Missing test images: {missing_images}")
        print("Run the image generation script first:")
        print("python scripts/generate_sd_test_images.py --output-dir test_images/fixtures")
        exit(1)
    
    # Run the test
    result = asyncio.run(main())
    exit(0 if result else 1)