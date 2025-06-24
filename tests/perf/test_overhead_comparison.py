#!/usr/bin/env python3
"""Comparative performance test between is-odd demo and CLIP to analyze overhead breakdown."""
import ray
import asyncio
import time
import base64
from pathlib import Path
import statistics


# Test image paths  
FIXTURES_DIR = Path(__file__).parent.parent.parent / "test_images" / "fixtures"
TEST_IMAGES = {
    "cat_office": FIXTURES_DIR / "cat_office_typing.jpg",
    "dog_office": FIXTURES_DIR / "dog_office_typing.jpg"
}


def load_image_as_b64(image_path: Path) -> str:
    """Load image file and encode as base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def benchmark_is_odd_actor():
    """Benchmark the is-odd demo actor for baseline Ray overhead."""
    print("🔢 Benchmarking Is-Odd Demo Actor...")
    
    from model_zoo.actors.is_odd import IsOddActor
    
    # Create actor
    actor = IsOddActor.options(num_gpus=0).remote("is-odd-benchmark", "no-weights")
    
    # Measure load time
    start_time = time.time()
    ready = await actor.ready.remote()
    load_time = time.time() - start_time
    
    if not ready:
        return None
    
    # Single inference benchmark
    single_times = []
    for i in range(10):
        payload = {"number": i + 1}
        start_time = time.time()
        result = await actor.infer.remote(payload)
        inference_time = time.time() - start_time
        single_times.append(inference_time)
    
    # Concurrent inference benchmark  
    concurrent_payloads = [{"number": i} for i in range(20, 30)]
    start_time = time.time()
    concurrent_results = await asyncio.gather(*[
        actor.infer.remote(payload) for payload in concurrent_payloads
    ])
    concurrent_total_time = time.time() - start_time
    concurrent_avg_time = concurrent_total_time / len(concurrent_results)
    
    return {
        "name": "Is-Odd Demo",
        "load_time": load_time,
        "single_avg": statistics.mean(single_times),
        "single_median": statistics.median(single_times),
        "single_min": min(single_times),
        "single_max": max(single_times),
        "concurrent_avg": concurrent_avg_time,
        "concurrent_total": concurrent_total_time,
        "concurrent_count": len(concurrent_results),
        "throughput": len(concurrent_results) / concurrent_total_time
    }


async def benchmark_clip_actor():
    """Benchmark the CLIP actor for ML model + Ray overhead."""
    print("🖼️  Benchmarking CLIP Actor...")
    
    # Check weights and images exist
    weights_path = Path(__file__).parent.parent.parent / "clip-vit-base-patch32.pytorch"
    if not weights_path.exists():
        print("   ❌ CLIP weights not found")
        return None
    
    missing_images = [name for name, path in TEST_IMAGES.items() if not path.exists()]
    if missing_images:
        print(f"   ❌ Missing test images: {missing_images}")
        return None
    
    from model_zoo.actors.clip import CLIPActor
    
    # Create actor
    actor = CLIPActor.options(num_gpus=0).remote("clip-benchmark", str(weights_path))
    
    # Measure load time
    start_time = time.time()
    ready = await actor.ready.remote()
    load_time = time.time() - start_time
    
    if not ready:
        return None
    
    # Prepare test payloads
    test_payloads = [
        {"image_b64": load_image_as_b64(TEST_IMAGES["cat_office"]), "text": "a cat"},
        {"image_b64": load_image_as_b64(TEST_IMAGES["dog_office"]), "text": "a dog"},
        {"image_b64": load_image_as_b64(TEST_IMAGES["cat_office"]), "text": "a dog"},
        {"image_b64": load_image_as_b64(TEST_IMAGES["dog_office"]), "text": "a cat"},
        {"image_b64": load_image_as_b64(TEST_IMAGES["cat_office"]), "text": "an animal"},
        {"image_b64": load_image_as_b64(TEST_IMAGES["dog_office"]), "text": "an animal"}
    ]
    
    # Single inference benchmark
    single_times = []
    for payload in test_payloads:
        start_time = time.time()
        result = await actor.infer.remote(payload)
        inference_time = time.time() - start_time
        single_times.append(inference_time)
    
    # Concurrent inference benchmark
    concurrent_payloads = test_payloads * 2  # 12 concurrent requests
    start_time = time.time()
    concurrent_results = await asyncio.gather(*[
        actor.infer.remote(payload) for payload in concurrent_payloads
    ])
    concurrent_total_time = time.time() - start_time
    concurrent_avg_time = concurrent_total_time / len(concurrent_results)
    
    return {
        "name": "CLIP",
        "load_time": load_time,
        "single_avg": statistics.mean(single_times),
        "single_median": statistics.median(single_times),
        "single_min": min(single_times),
        "single_max": max(single_times),
        "concurrent_avg": concurrent_avg_time,
        "concurrent_total": concurrent_total_time,
        "concurrent_count": len(concurrent_results),
        "throughput": len(concurrent_results) / concurrent_total_time
    }


async def test_overhead_comparison():
    """Compare is-odd demo vs CLIP to analyze overhead breakdown."""
    print("⚡ Ray Actor Overhead Comparison")
    print("=" * 50)
    print("Comparing is-odd demo (minimal computation) vs CLIP (ML model)")
    print("to isolate Ray infrastructure overhead vs ML computation overhead")
    print()
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(num_cpus=4, num_gpus=0)
    print(f"✓ Ray initialized with {ray.cluster_resources()}")
    print()
    
    # Run benchmarks
    is_odd_results = await benchmark_is_odd_actor()
    clip_results = await benchmark_clip_actor()
    
    if not is_odd_results or not clip_results:
        print("❌ Benchmark failed - missing dependencies")
        return False
    
    # Comparative analysis
    print("\n📊 PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    
    # Format results table
    print(f"{'Metric':<25} {'Is-Odd Demo':<15} {'CLIP':<15} {'Difference':<15}")
    print("-" * 70)
    
    # Load time comparison
    load_diff = clip_results["load_time"] - is_odd_results["load_time"]
    print(f"{'Model Load Time':<25} {is_odd_results['load_time']:<15.3f}s {clip_results['load_time']:<15.3f}s +{load_diff:.3f}s")
    
    # Single inference comparison
    single_diff = clip_results["single_avg"] - is_odd_results["single_avg"]
    print(f"{'Single Inference (avg)':<25} {is_odd_results['single_avg']:<15.3f}s {clip_results['single_avg']:<15.3f}s +{single_diff:.3f}s")
    
    single_median_diff = clip_results["single_median"] - is_odd_results["single_median"]
    print(f"{'Single Inference (median)':<25} {is_odd_results['single_median']:<15.3f}s {clip_results['single_median']:<15.3f}s +{single_median_diff:.3f}s")
    
    # Concurrent inference comparison
    concurrent_diff = clip_results["concurrent_avg"] - is_odd_results["concurrent_avg"]
    print(f"{'Concurrent Inference':<25} {is_odd_results['concurrent_avg']:<15.3f}s {clip_results['concurrent_avg']:<15.3f}s +{concurrent_diff:.3f}s")
    
    # Throughput comparison
    throughput_ratio = clip_results["throughput"] / is_odd_results["throughput"]
    print(f"{'Throughput':<25} {is_odd_results['throughput']:<15.1f}/s {clip_results['throughput']:<15.1f}/s {throughput_ratio:.2f}x")
    
    print()
    
    # Overhead breakdown analysis
    print("🔍 OVERHEAD BREAKDOWN ANALYSIS")
    print("=" * 40)
    
    ray_overhead = is_odd_results["single_avg"]  # Pure Ray overhead from trivial computation
    ml_overhead = single_diff  # Additional overhead from ML model
    total_overhead = clip_results["single_avg"]
    
    print(f"Ray Infrastructure Overhead: {ray_overhead:.3f}s ({ray_overhead/total_overhead*100:.1f}%)")
    print(f"ML Model Computation Overhead: {ml_overhead:.3f}s ({ml_overhead/total_overhead*100:.1f}%)")
    print(f"Total CLIP Inference Time: {total_overhead:.3f}s (100%)")
    print()
    
    # Performance insights
    print("💡 PERFORMANCE INSIGHTS")
    print("=" * 30)
    
    if ray_overhead > ml_overhead:
        print("• Ray infrastructure overhead dominates performance")
        print("• Consider batching multiple requests for better efficiency")
    else:
        print("• ML model computation dominates performance")
        print("• Ray overhead is relatively small compared to model computation")
    
    concurrency_improvement = is_odd_results["single_avg"] / is_odd_results["concurrent_avg"]
    clip_concurrency_improvement = clip_results["single_avg"] / clip_results["concurrent_avg"]
    
    print(f"• Concurrency improves is-odd performance by {concurrency_improvement:.1f}x")
    print(f"• Concurrency improves CLIP performance by {clip_concurrency_improvement:.1f}x")
    
    if load_diff > 2.0:
        print(f"• CLIP model loading takes {load_diff:.1f}s longer (expected for ML models)")
    
    # Recommendations
    print(f"\n🚀 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 40)
    
    if ray_overhead > 0.01:  # > 10ms
        print("• Ray overhead is significant - consider:")
        print("  - Using concurrent/batched requests")
        print("  - Keeping actors alive longer")
        print("  - Optimizing payload serialization")
    
    if ml_overhead > 0.05:  # > 50ms  
        print("• ML model overhead is significant - consider:")
        print("  - Model optimization (quantization, pruning)")
        print("  - GPU acceleration for larger batches")
        print("  - Model caching strategies")
    
    # Detailed metrics
    print(f"\n📋 DETAILED METRICS")
    print("=" * 25)
    
    for name, results in [("Is-Odd Demo", is_odd_results), ("CLIP", clip_results)]:
        print(f"\n{name}:")
        print(f"  Load time: {results['load_time']:.3f}s")
        print(f"  Single inference: {results['single_min']:.3f}s - {results['single_max']:.3f}s (avg: {results['single_avg']:.3f}s)")
        print(f"  Concurrent: {results['concurrent_count']} requests in {results['concurrent_total']:.3f}s")
        print(f"  Throughput: {results['throughput']:.1f} requests/second")
    
    # Cleanup
    ray.shutdown()
    print(f"\n✓ Benchmark complete - Ray shutdown")
    
    return True


async def main():
    """Run the overhead comparison test."""
    success = await test_overhead_comparison()
    if success:
        print("\n🎉 OVERHEAD COMPARISON COMPLETE!")
        print("📊 Results show the breakdown between Ray infrastructure and ML model costs.")
    else:
        print("\n❌ Overhead comparison failed.")
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)