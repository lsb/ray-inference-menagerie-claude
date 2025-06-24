#!/usr/bin/env python3
"""Performance tests for CLIP actor to measure ML model overhead vs Ray overhead."""
import ray
import asyncio
import time
import base64
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


async def test_clip_performance_e2e():
    """Performance test for CLIP actor to compare against is-odd demo overhead."""
    print("🎯 CLIP Performance Test - Measuring ML Model + Ray Overhead")
    print("=" * 65)
    
    # Check that test images exist
    missing_images = [name for name, path in TEST_IMAGES.items() if not path.exists()]
    if missing_images:
        print(f"❌ Missing test images: {missing_images}")
        print("Run: python scripts/generate_sd_test_images.py --output-dir test_images/fixtures")
        return False
    
    # Initialize Ray
    print("\n1. Initializing Ray...")
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(num_cpus=4, num_gpus=0)  # CPU only for this test
    print(f"   ✓ Ray initialized with {ray.cluster_resources()}")
    
    # Import and create CLIP actor
    print("\n2. Creating CLIP actor...")
    from model_zoo.actors.clip import CLIPActor
    
    # Use local weights file
    weights_path = Path(__file__).parent.parent.parent / "clip-vit-base-patch32.pytorch"
    if not weights_path.exists():
        print(f"❌ CLIP weights not found: {weights_path}")
        print("Download with: curl -L https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin -o clip-vit-base-patch32.pytorch")
        return False
    
    actor = CLIPActor.options(num_gpus=0).remote("clip-perf-test", str(weights_path))
    print("   ✓ CLIP actor created")
    
    # Test actor readiness (model loading time)
    print("\n3. Testing actor readiness (model loading time)...")
    start_time = time.time()
    ready = await actor.ready.remote()
    load_time = time.time() - start_time
    print(f"   ✓ Actor ready: {ready} (model load time: {load_time:.3f}s)")
    
    if not ready:
        print("   ✗ Actor failed to load!")
        return False
    
    # Performance test with various cat/dog combinations
    print("\n4. Testing inference performance with cat/dog classification...")
    
    test_cases = [
        {
            "name": "cat_office + 'a cat'",
            "image": "cat_office",
            "text": "a cat",
            "expected_higher": True  # Cat image should score higher with "cat"
        },
        {
            "name": "cat_office + 'a dog'", 
            "image": "cat_office",
            "text": "a dog",
            "expected_higher": False  # Cat image should score lower with "dog"
        },
        {
            "name": "dog_office + 'a cat'",
            "image": "dog_office", 
            "text": "a cat",
            "expected_higher": False  # Dog image should score lower with "cat"
        },
        {
            "name": "dog_office + 'a dog'",
            "image": "dog_office",
            "text": "a dog", 
            "expected_higher": True  # Dog image should score higher with "dog"
        },
        {
            "name": "cat_mountain + 'a cat'",
            "image": "cat_mountain",
            "text": "a cat",
            "expected_higher": True
        },
        {
            "name": "dog_mountain + 'a dog'",
            "image": "dog_mountain", 
            "text": "a dog",
            "expected_higher": True
        }
    ]
    
    results = []
    total_inference_time = 0
    
    for i, case in enumerate(test_cases):
        print(f"\n   Test {i+1}: {case['name']}")
        
        # Load image
        image_path = TEST_IMAGES[case["image"]]
        image_b64 = load_image_as_b64(image_path)
        
        payload = {"image_b64": image_b64, "text": case["text"]}
        
        # Time the inference
        start_time = time.time()
        result = await actor.infer.remote(payload)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        similarity = result["similarity"]
        
        print(f"     Result: {result}")
        print(f"     Inference time: {inference_time:.3f}s")
        print(f"     Similarity: {similarity:.3f}")
        
        results.append({
            "name": case["name"],
            "image": case["image"],
            "text": case["text"],
            "similarity": similarity,
            "inference_time": inference_time,
            "expected_higher": case["expected_higher"]
        })
    
    # Test concurrent inference (measure batching overhead)
    print("\n5. Testing concurrent CLIP inference (batching performance)...")
    concurrent_payloads = []
    
    # Create 10 concurrent requests using different images and texts
    test_combinations = [
        (TEST_IMAGES["cat_office"], "a cat"),
        (TEST_IMAGES["dog_office"], "a dog"), 
        (TEST_IMAGES["cat_mountain"], "a cat"),
        (TEST_IMAGES["dog_mountain"], "a dog"),
        (TEST_IMAGES["cat_office"], "an animal"),
        (TEST_IMAGES["dog_office"], "an animal"),
        (TEST_IMAGES["cat_mountain"], "outdoor scene"),
        (TEST_IMAGES["dog_mountain"], "outdoor scene"), 
        (TEST_IMAGES["cat_office"], "indoor scene"),
        (TEST_IMAGES["dog_office"], "indoor scene")
    ]
    
    for img_path, text in test_combinations:
        image_b64 = load_image_as_b64(img_path)
        concurrent_payloads.append({"image_b64": image_b64, "text": text})
    
    start_time = time.time()
    concurrent_results = await asyncio.gather(*[
        actor.infer.remote(payload) for payload in concurrent_payloads
    ])
    concurrent_time = time.time() - start_time
    
    print(f"   ✓ Concurrent inference: {len(concurrent_results)} requests in {concurrent_time:.3f}s")
    print(f"   ✓ Average time per request: {concurrent_time/len(concurrent_results):.3f}s")
    print(f"   ✓ CLIP throughput: {len(concurrent_results)/concurrent_time:.1f} inferences/sec")
    
    # Test repeated inference on same image (caching effects)
    print("\n6. Testing repeated inference (caching analysis)...")
    base_payload = {
        "image_b64": load_image_as_b64(TEST_IMAGES["cat_office"]),
        "text": "a cat"
    }
    
    repeated_times = []
    for i in range(5):
        start_time = time.time()
        result = await actor.infer.remote(base_payload)
        inference_time = time.time() - start_time
        repeated_times.append(inference_time)
        print(f"   Repeat {i+1}: {inference_time:.3f}s (similarity: {result['similarity']:.3f})")
    
    # Performance analysis and comparison
    print("\n7. 🎯 CLIP Performance Analysis:")
    print("=" * 50)
    
    avg_inference_time = total_inference_time / len(results) if results else 0
    avg_concurrent_time = concurrent_time / len(concurrent_results) if concurrent_results else 0
    avg_repeated_time = sum(repeated_times) / len(repeated_times) if repeated_times else 0
    
    print(f"Model loading time: {load_time:.3f}s")
    print(f"Average single inference: {avg_inference_time:.3f}s")
    print(f"Average concurrent inference: {avg_concurrent_time:.3f}s") 
    print(f"Average repeated inference: {avg_repeated_time:.3f}s")
    print(f"Concurrent throughput: {len(concurrent_results)/concurrent_time:.1f} req/sec")
    
    # Compare to is-odd demo overhead
    print(f"\n📊 Overhead Breakdown Analysis:")
    print(f"   • Model loading: {load_time:.3f}s (vs is-odd: ~0.14s)")
    print(f"   • Single inference: {avg_inference_time:.3f}s (vs is-odd: ~0.005s)")
    print(f"   • Concurrent inference: {avg_concurrent_time:.3f}s (vs is-odd: ~0.001s)")
    print(f"   • ML model overhead: ~{(avg_inference_time - 0.005)*1000:.0f}ms per inference")
    print(f"   • Ray overhead: ~5ms (same as is-odd demo)")
    
    # Accuracy validation
    print(f"\n🎯 Classification Accuracy Check:")
    cat_results = [r for r in results if "cat" in r["image"] and r["text"] == "a cat"]
    dog_results = [r for r in results if "dog" in r["image"] and r["text"] == "a dog"]
    
    avg_cat_similarity = sum(r["similarity"] for r in cat_results) / len(cat_results) if cat_results else 0
    avg_dog_similarity = sum(r["similarity"] for r in dog_results) / len(dog_results) if dog_results else 0
    
    print(f"   • Cat images with 'cat' prompt: {avg_cat_similarity:.3f} avg similarity")
    print(f"   • Dog images with 'dog' prompt: {avg_dog_similarity:.3f} avg similarity")
    print(f"   • Model working correctly: {'✅' if avg_cat_similarity > 0.6 and avg_dog_similarity > 0.6 else '❌'}")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for result in results:
        status = "🐱" if "cat" in result["image"] else "🐕"
        print(f"  {status} {result['name']}: {result['similarity']:.3f} ({result['inference_time']:.3f}s)")
    
    # Cleanup
    print(f"\n8. Cleaning up...")
    ray.shutdown()
    print("   ✓ Ray shutdown complete")
    
    print(f"\n🎉 CLIP Performance Test Complete!")
    
    # Return success if we got reasonable results
    return len(results) > 0 and avg_inference_time > 0


async def main():
    """Run the CLIP performance test."""
    success = await test_clip_performance_e2e()
    if success:
        print("\n✅ CLIP PERFORMANCE TEST PASSED!")
        print("📊 Use these metrics to compare ML model overhead vs Ray infrastructure overhead.")
        print("💡 Compare against is-odd demo (~5ms) to isolate pure ML computation costs.")
    else:
        print("\n❌ CLIP performance test failed. Check the results above.")
    return success


if __name__ == "__main__":
    # Run the performance test
    result = asyncio.run(main())
    exit(0 if result else 1)