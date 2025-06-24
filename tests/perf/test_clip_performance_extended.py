#!/usr/bin/env python3
"""Extended CLIP performance test with 50+ iterations per test case."""
import ray
import asyncio
import time
import statistics
import base64
from pathlib import Path
from PIL import Image
import io
import json


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


async def test_clip_performance_extended():
    """Extended CLIP performance test with 50+ iterations per test case."""
    print("🎯 CLIP Extended Performance Test - 50+ Iterations Per Test Case")
    print("=" * 70)
    
    # Check that test images exist
    missing_images = [name for name, path in TEST_IMAGES.items() if not path.exists()]
    if missing_images:
        print(f"❌ Missing test images: {missing_images}")
        return False
    
    # Initialize Ray
    print("\n1. Initializing Ray...")
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(num_cpus=4, num_gpus=0)
    print(f"   ✓ Ray initialized with {ray.cluster_resources()}")
    
    # Import and create CLIP actor
    print("\n2. Creating CLIP actor...")
    from model_zoo.actors.clip import CLIPActor
    
    weights_path = Path(__file__).parent.parent.parent / "clip-vit-base-patch32.pytorch"
    if not weights_path.exists():
        print(f"❌ CLIP weights not found: {weights_path}")
        return False
    
    actor = CLIPActor.options(num_gpus=0).remote("clip-extended-perf", str(weights_path))
    print("   ✓ CLIP actor created")
    
    # Test actor readiness
    print("\n3. Testing actor readiness (model loading)...")
    start_time = time.time()
    ready = await actor.ready.remote()
    load_time = time.time() - start_time
    print(f"   ✓ Actor ready: {ready} (model load time: {load_time:.3f}s)")
    
    if not ready:
        return False
    
    # Warm up the model
    print("\n4. Warming up CLIP model with 10 initial inferences...")
    warmup_times = []
    test_payload = {
        "image_b64": load_image_as_b64(TEST_IMAGES["cat_office"]),
        "text": "a cat"
    }
    
    for i in range(10):
        start_time = time.time()
        result = await actor.infer.remote(test_payload)
        warmup_times.append(time.time() - start_time)
    
    warmup_avg = statistics.mean(warmup_times)
    print(f"   ✓ Warmup complete: avg {warmup_avg*1000:.2f}ms")
    
    # Define test cases
    test_cases = [
        # Cat images with various prompts
        {"name": "cat_office + 'a cat'", "image": "cat_office", "text": "a cat"},
        {"name": "cat_office + 'a dog'", "image": "cat_office", "text": "a dog"},
        {"name": "cat_office + 'cat typing'", "image": "cat_office", "text": "cat typing"},
        {"name": "cat_office + 'office scene'", "image": "cat_office", "text": "office scene"},
        
        # Dog images with various prompts  
        {"name": "dog_office + 'a dog'", "image": "dog_office", "text": "a dog"},
        {"name": "dog_office + 'a cat'", "image": "dog_office", "text": "a cat"},
        {"name": "dog_office + 'dog typing'", "image": "dog_office", "text": "dog typing"},
        {"name": "dog_office + 'office scene'", "image": "dog_office", "text": "office scene"},
        
        # Mountain scenes
        {"name": "cat_mountain + 'outdoor cat'", "image": "cat_mountain", "text": "outdoor cat"},
        {"name": "dog_mountain + 'outdoor dog'", "image": "dog_mountain", "text": "outdoor dog"},
        {"name": "cat_mountain + 'mountain landscape'", "image": "cat_mountain", "text": "mountain landscape"},
        {"name": "dog_mountain + 'mountain landscape'", "image": "dog_mountain", "text": "mountain landscape"},
    ]
    
    # Run 50 iterations for each test case
    print("\n5. Running 50 iterations for each test case...")
    ITERATIONS_PER_CASE = 50
    all_results = {}
    
    for case_idx, case in enumerate(test_cases):
        print(f"\n   Test Case {case_idx + 1}/{len(test_cases)}: {case['name']}")
        
        # Prepare payload
        image_b64 = load_image_as_b64(TEST_IMAGES[case["image"]])
        payload = {"image_b64": image_b64, "text": case["text"]}
        
        # Run iterations
        case_times = []
        case_similarities = []
        
        for i in range(ITERATIONS_PER_CASE):
            start_time = time.time()
            result = await actor.infer.remote(payload)
            inference_time = time.time() - start_time
            
            case_times.append(inference_time)
            case_similarities.append(result["similarity"])
            
            # Progress indicator every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"     Progress: {i + 1}/{ITERATIONS_PER_CASE} iterations")
        
        # Calculate statistics for this case
        avg_time = statistics.mean(case_times)
        median_time = statistics.median(case_times)
        min_time = min(case_times)
        max_time = max(case_times)
        stdev_time = statistics.stdev(case_times)
        p95_time = sorted(case_times)[int(0.95 * len(case_times))]
        
        avg_similarity = statistics.mean(case_similarities)
        stdev_similarity = statistics.stdev(case_similarities) if len(set(case_similarities)) > 1 else 0
        
        all_results[case["name"]] = {
            "times": case_times,
            "similarities": case_similarities,
            "avg_time": avg_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
            "stdev_time": stdev_time,
            "p95_time": p95_time,
            "avg_similarity": avg_similarity,
            "stdev_similarity": stdev_similarity
        }
        
        print(f"     ✓ Complete: avg {avg_time*1000:.2f}ms, similarity {avg_similarity:.3f}")
    
    # Large-scale concurrent test
    print("\n6. Testing large-scale concurrent inference...")
    print("   Running 100 concurrent CLIP inferences...")
    
    # Create 100 diverse payloads
    concurrent_payloads = []
    for i in range(100):
        image_key = list(TEST_IMAGES.keys())[i % len(TEST_IMAGES)]
        text_options = ["a cat", "a dog", "an animal", "indoor scene", "outdoor scene", "typing", "mountain", "office"]
        text = text_options[i % len(text_options)]
        
        concurrent_payloads.append({
            "image_b64": load_image_as_b64(TEST_IMAGES[image_key]),
            "text": text
        })
    
    # Run concurrent test
    concurrent_start = time.time()
    concurrent_results = await asyncio.gather(*[
        actor.infer.remote(payload) for payload in concurrent_payloads
    ])
    concurrent_total_time = time.time() - concurrent_start
    concurrent_avg_time = concurrent_total_time / len(concurrent_results)
    
    print(f"   ✓ 100 concurrent inferences in {concurrent_total_time:.2f}s")
    print(f"   ✓ Average time per request: {concurrent_avg_time*1000:.2f}ms")
    print(f"   ✓ Throughput: {len(concurrent_results)/concurrent_total_time:.1f} req/s")
    
    # Aggregate statistics across all test cases
    print("\n7. 📊 Aggregate Performance Analysis")
    print("=" * 60)
    
    all_times = []
    for result in all_results.values():
        all_times.extend(result["times"])
    
    total_inferences = len(all_times)
    overall_avg = statistics.mean(all_times)
    overall_median = statistics.median(all_times)
    overall_min = min(all_times)
    overall_max = max(all_times)
    overall_stdev = statistics.stdev(all_times)
    
    # Calculate percentiles
    sorted_times = sorted(all_times)
    p50 = sorted_times[int(0.50 * len(sorted_times))]
    p90 = sorted_times[int(0.90 * len(sorted_times))]
    p95 = sorted_times[int(0.95 * len(sorted_times))]
    p99 = sorted_times[int(0.99 * len(sorted_times))]
    
    print(f"Total CLIP inferences: {total_inferences}")
    print(f"Model load time: {load_time:.3f}s")
    
    print(f"\nLatency Statistics (milliseconds):")
    print(f"  Average:     {overall_avg*1000:6.2f}ms")
    print(f"  Median:      {overall_median*1000:6.2f}ms")
    print(f"  Min:         {overall_min*1000:6.2f}ms")
    print(f"  Max:         {overall_max*1000:6.2f}ms")
    print(f"  Std Dev:     {overall_stdev*1000:6.2f}ms")
    
    print(f"\nLatency Percentiles:")
    print(f"  50th (P50):  {p50*1000:6.2f}ms")
    print(f"  90th (P90):  {p90*1000:6.2f}ms")
    print(f"  95th (P95):  {p95*1000:6.2f}ms")
    print(f"  99th (P99):  {p99*1000:6.2f}ms")
    
    # Per-test-case summary
    print("\n8. 📋 Per-Test-Case Summary (50 iterations each)")
    print("=" * 80)
    print(f"{'Test Case':<40} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Similarity':<12} {'Std Dev':<10}")
    print("-" * 80)
    
    for case_name, result in all_results.items():
        print(f"{case_name:<40} "
              f"{result['avg_time']*1000:8.2f}  "
              f"{result['p95_time']*1000:8.2f}  "
              f"{result['avg_similarity']:10.3f}  "
              f"{result['stdev_similarity']:8.4f}")
    
    # Classification accuracy check
    print("\n9. 🎯 Classification Accuracy Analysis")
    cat_cat_scores = []
    cat_dog_scores = []
    dog_dog_scores = []
    dog_cat_scores = []
    
    for case_name, result in all_results.items():
        avg_sim = result["avg_similarity"]
        if "cat_" in case_name and "'a cat'" in case_name:
            cat_cat_scores.append(avg_sim)
        elif "cat_" in case_name and "'a dog'" in case_name:
            cat_dog_scores.append(avg_sim)
        elif "dog_" in case_name and "'a dog'" in case_name:
            dog_dog_scores.append(avg_sim)
        elif "dog_" in case_name and "'a cat'" in case_name:
            dog_cat_scores.append(avg_sim)
    
    if cat_cat_scores and cat_dog_scores and dog_dog_scores and dog_cat_scores:
        avg_cat_cat = statistics.mean(cat_cat_scores)
        avg_cat_dog = statistics.mean(cat_dog_scores)
        avg_dog_dog = statistics.mean(dog_dog_scores)
        avg_dog_cat = statistics.mean(dog_cat_scores)
        
        print(f"Cat images with 'a cat': {avg_cat_cat:.3f} avg similarity")
        print(f"Cat images with 'a dog': {avg_cat_dog:.3f} avg similarity")
        print(f"Dog images with 'a dog': {avg_dog_dog:.3f} avg similarity")
        print(f"Dog images with 'a cat': {avg_dog_cat:.3f} avg similarity")
        
        cat_accuracy = avg_cat_cat > avg_cat_dog
        dog_accuracy = avg_dog_dog > avg_dog_cat
        print(f"\nClassification accuracy: {'✅' if cat_accuracy and dog_accuracy else '❌'}")
    
    # Save detailed results to JSON
    print("\n10. Saving detailed results...")
    output_file = Path("clip_performance_results.json")
    
    results_data = {
        "summary": {
            "total_inferences": total_inferences,
            "model_load_time": load_time,
            "avg_latency_ms": overall_avg * 1000,
            "median_latency_ms": overall_median * 1000,
            "p95_latency_ms": p95 * 1000,
            "p99_latency_ms": p99 * 1000,
            "concurrent_throughput_rps": len(concurrent_results) / concurrent_total_time
        },
        "per_test_case": {
            case_name: {
                "avg_time_ms": result["avg_time"] * 1000,
                "median_time_ms": result["median_time"] * 1000,
                "p95_time_ms": result["p95_time"] * 1000,
                "avg_similarity": result["avg_similarity"],
                "iterations": ITERATIONS_PER_CASE
            }
            for case_name, result in all_results.items()
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"   ✓ Results saved to {output_file}")
    
    # Cleanup
    print(f"\n11. Cleaning up...")
    ray.shutdown()
    print("   ✓ Ray shutdown complete")
    
    print(f"\n🎉 CLIP Extended Performance Test Complete!")
    print(f"✅ {total_inferences} total inferences analyzed")
    print(f"📊 Median latency: {overall_median*1000:.2f}ms")
    
    return True


async def main():
    """Run the extended CLIP performance test."""
    success = await test_clip_performance_extended()
    if success:
        print("\n✅ CLIP EXTENDED PERFORMANCE TEST PASSED!")
        print("📊 Comprehensive performance metrics collected over 600+ iterations.")
    else:
        print("\n❌ Test failed.")
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)