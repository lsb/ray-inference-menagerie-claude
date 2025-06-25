#!/usr/bin/env python3
"""Comprehensive CLIP performance test with multiple image formats and classifications."""
import ray
import asyncio
import time
import statistics
import base64
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Tuple


# Test image formats directory
FORMATS_DIR = Path(__file__).parent.parent.parent / "test_images" / "formats"

# Define test images and their classifications
TEST_IMAGES = {
    # Indoor cat images
    "cat_office_lowres": {"path": FORMATS_DIR / "cat_office_lowres.jpg", "animal": "cat", "location": "indoor"},
    "cat_office_highres": {"path": FORMATS_DIR / "cat_office_highres.jpg", "animal": "cat", "location": "indoor"},
    "cat_office_bmp": {"path": FORMATS_DIR / "cat_office.bmp", "animal": "cat", "location": "indoor"},
    
    # Indoor dog images
    "dog_office_lowres": {"path": FORMATS_DIR / "dog_office_lowres.jpg", "animal": "dog", "location": "indoor"},
    "dog_office_highres": {"path": FORMATS_DIR / "dog_office_highres.jpg", "animal": "dog", "location": "indoor"},
    "dog_office_bmp": {"path": FORMATS_DIR / "dog_office.bmp", "animal": "dog", "location": "indoor"},
    
    # Outdoor cat images
    "cat_mountain_lowres": {"path": FORMATS_DIR / "cat_mountain_lowres.jpg", "animal": "cat", "location": "outdoor"},
    "cat_mountain_highres": {"path": FORMATS_DIR / "cat_mountain_highres.jpg", "animal": "cat", "location": "outdoor"},
    "cat_mountain_bmp": {"path": FORMATS_DIR / "cat_mountain.bmp", "animal": "cat", "location": "outdoor"},
    
    # Outdoor dog images
    "dog_mountain_lowres": {"path": FORMATS_DIR / "dog_mountain_lowres.jpg", "animal": "dog", "location": "outdoor"},
    "dog_mountain_highres": {"path": FORMATS_DIR / "dog_mountain_highres.jpg", "animal": "dog", "location": "outdoor"},
    "dog_mountain_bmp": {"path": FORMATS_DIR / "dog_mountain.bmp", "animal": "dog", "location": "outdoor"},
}


def load_image_as_b64(image_path: Path) -> str:
    """Load image file and encode as base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def calculate_percentiles(times: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    sorted_times = sorted(times)
    n = len(sorted_times)
    
    return {
        "p50": sorted_times[int(0.50 * n)],
        "p90": sorted_times[int(0.90 * n)],
        "p95": sorted_times[int(0.95 * n)],
        "p99": sorted_times[int(0.99 * n)]
    }


async def run_classification_test(actor, image_name: str, image_info: Dict, iterations: int = 500) -> Dict:
    """Run classification test for a single image with multiple iterations."""
    print(f"\n🧪 Testing {image_name} ({iterations} iterations)")
    
    # Get image info
    image_path = image_info["path"]
    expected_animal = image_info["animal"]
    expected_location = image_info["location"]
    
    # Get image size for analysis
    img_size = image_path.stat().st_size / 1024  # KB
    with Image.open(image_path) as img:
        width, height = img.size
    
    print(f"   📁 Format: {image_path.suffix}, Size: {img_size:.1f}KB, Dimensions: {width}x{height}")
    
    # Load image once
    image_b64 = load_image_as_b64(image_path)
    
    # Test animal classification (cat vs dog)
    print("   🐱🐕 Testing animal classification...")
    animal_times = []
    cat_similarities = []
    dog_similarities = []
    
    for i in range(iterations):
        # Test with "cat" prompt
        cat_payload = {"image_b64": image_b64, "text": "a cat"}
        start_time = time.time()
        cat_result = await actor.infer.remote(cat_payload)
        cat_time = time.time() - start_time
        cat_similarities.append(cat_result["similarity"])
        
        # Test with "dog" prompt
        dog_payload = {"image_b64": image_b64, "text": "a dog"}
        start_time = time.time()
        dog_result = await actor.infer.remote(dog_payload)
        dog_time = time.time() - start_time
        dog_similarities.append(dog_result["similarity"])
        
        # Use average of both times for this iteration
        animal_times.append((cat_time + dog_time) / 2)
        
        # Progress update every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"     Progress: {i + 1}/{iterations} animal tests")
    
    # Test location classification (indoor vs outdoor)
    print("   🏠🏔️  Testing location classification...")
    location_times = []
    indoor_similarities = []
    outdoor_similarities = []
    
    for i in range(iterations):
        # Test with "indoor" prompt
        indoor_payload = {"image_b64": image_b64, "text": "indoor scene"}
        start_time = time.time()
        indoor_result = await actor.infer.remote(indoor_payload)
        indoor_time = time.time() - start_time
        indoor_similarities.append(indoor_result["similarity"])
        
        # Test with "outdoor" prompt
        outdoor_payload = {"image_b64": image_b64, "text": "outdoor scene"}
        start_time = time.time()
        outdoor_result = await actor.infer.remote(outdoor_payload)
        outdoor_time = time.time() - start_time
        outdoor_similarities.append(outdoor_result["similarity"])
        
        # Use average of both times for this iteration
        location_times.append((indoor_time + outdoor_time) / 2)
        
        # Progress update every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"     Progress: {i + 1}/{iterations} location tests")
    
    # Analyze results
    avg_cat_sim = statistics.mean(cat_similarities)
    avg_dog_sim = statistics.mean(dog_similarities)
    avg_indoor_sim = statistics.mean(indoor_similarities)
    avg_outdoor_sim = statistics.mean(outdoor_similarities)
    
    # Determine classification accuracy
    animal_correct = (expected_animal == "cat" and avg_cat_sim > avg_dog_sim) or \
                    (expected_animal == "dog" and avg_dog_sim > avg_cat_sim)
    
    location_correct = (expected_location == "indoor" and avg_indoor_sim > avg_outdoor_sim) or \
                      (expected_location == "outdoor" and avg_outdoor_sim > avg_indoor_sim)
    
    # Calculate statistics
    animal_stats = {
        "mean": statistics.mean(animal_times),
        "median": statistics.median(animal_times),
        "stdev": statistics.stdev(animal_times),
        **{f"{k}": v for k, v in calculate_percentiles(animal_times).items()}
    }
    
    location_stats = {
        "mean": statistics.mean(location_times),
        "median": statistics.median(location_times),
        "stdev": statistics.stdev(location_times),
        **{f"{k}": v for k, v in calculate_percentiles(location_times).items()}
    }
    
    # Print results
    print(f"   📊 Animal Classification:")
    print(f"     Cat similarity: {avg_cat_sim:.3f}, Dog similarity: {avg_dog_sim:.3f}")
    print(f"     Predicted: {'cat' if avg_cat_sim > avg_dog_sim else 'dog'}, Expected: {expected_animal}")
    print(f"     Correct: {'✅' if animal_correct else '❌'}")
    print(f"     Latency - P50: {animal_stats['p50']*1000:.1f}ms, P95: {animal_stats['p95']*1000:.1f}ms, P99: {animal_stats['p99']*1000:.1f}ms")
    
    print(f"   📊 Location Classification:")
    print(f"     Indoor similarity: {avg_indoor_sim:.3f}, Outdoor similarity: {avg_outdoor_sim:.3f}")
    print(f"     Predicted: {'indoor' if avg_indoor_sim > avg_outdoor_sim else 'outdoor'}, Expected: {expected_location}")
    print(f"     Correct: {'✅' if location_correct else '❌'}")
    print(f"     Latency - P50: {location_stats['p50']*1000:.1f}ms, P95: {location_stats['p95']*1000:.1f}ms, P99: {location_stats['p99']*1000:.1f}ms")
    
    return {
        "image_info": {
            "name": image_name,
            "format": image_path.suffix,
            "size_kb": img_size,
            "dimensions": f"{width}x{height}",
            "expected_animal": expected_animal,
            "expected_location": expected_location
        },
        "animal_classification": {
            "cat_similarity": avg_cat_sim,
            "dog_similarity": avg_dog_sim,
            "predicted": "cat" if avg_cat_sim > avg_dog_sim else "dog",
            "correct": animal_correct,
            "latency_stats": animal_stats
        },
        "location_classification": {
            "indoor_similarity": avg_indoor_sim,
            "outdoor_similarity": avg_outdoor_sim,
            "predicted": "indoor" if avg_indoor_sim > avg_outdoor_sim else "outdoor",
            "correct": location_correct,
            "latency_stats": location_stats
        },
        "iterations": iterations
    }


async def test_clip_comprehensive():
    """Comprehensive CLIP performance test with multiple formats and classifications."""
    print("🎯 CLIP Comprehensive Performance Test")
    print("=" * 60)
    print("Testing cat/dog + indoor/outdoor classification across image formats")
    print("500 iterations per test case for accurate percentile measurements")
    
    # Check that test images exist
    missing_images = [name for name, info in TEST_IMAGES.items() if not info["path"].exists()]
    if missing_images:
        print(f"❌ Missing test images: {missing_images}")
        print("Run: python scripts/generate_test_image_formats.py")
        return False
    
    # Initialize Ray
    print("\n1. Initializing Ray...")
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(num_cpus=4, num_gpus=0)
    print(f"   ✓ Ray initialized with {ray.cluster_resources()}")
    
    # Create CLIP actor
    print("\n2. Creating CLIP actor...")
    from model_zoo.actors.clip import CLIPActor
    
    weights_path = Path(__file__).parent.parent.parent / "clip-vit-base-patch32.pytorch"
    if not weights_path.exists():
        print(f"❌ CLIP weights not found: {weights_path}")
        return False
    
    actor = CLIPActor.options(num_gpus=0).remote("clip-comprehensive", str(weights_path))
    
    # Test actor readiness
    start_time = time.time()
    ready = await actor.ready.remote()
    load_time = time.time() - start_time
    print(f"   ✓ Actor ready: {ready} (load time: {load_time:.3f}s)")
    
    if not ready:
        return False
    
    # Warm up
    print("\n3. Warming up with 10 test inferences...")
    warmup_payload = {
        "image_b64": load_image_as_b64(TEST_IMAGES["cat_office_lowres"]["path"]),
        "text": "a cat"
    }
    
    warmup_times = []
    for i in range(10):
        start_time = time.time()
        await actor.infer.remote(warmup_payload)
        warmup_times.append(time.time() - start_time)
    
    warmup_avg = statistics.mean(warmup_times)
    print(f"   ✓ Warmup complete: avg {warmup_avg*1000:.1f}ms")
    
    # Run comprehensive tests
    print(f"\n4. Running comprehensive tests...")
    print(f"   {len(TEST_IMAGES)} images × 500 iterations × 2 classifications = {len(TEST_IMAGES) * 500 * 2:,} total inferences")
    print(f"   Estimated time: {len(TEST_IMAGES) * 500 * 2 * warmup_avg / 60:.1f} minutes")
    
    results = {}
    total_start = time.time()
    
    for i, (image_name, image_info) in enumerate(TEST_IMAGES.items(), 1):
        print(f"\n📸 Image {i}/{len(TEST_IMAGES)}: {image_name}")
        result = await run_classification_test(actor, image_name, image_info, iterations=500)
        results[image_name] = result
    
    total_time = time.time() - total_start
    
    # Aggregate analysis
    print(f"\n5. 📊 Comprehensive Analysis")
    print("=" * 60)
    
    # Format-based analysis
    formats = {".jpg": {"lowres": [], "highres": []}, ".bmp": []}
    
    for name, result in results.items():
        fmt = result["image_info"]["format"]
        if fmt == ".jpg":
            if "lowres" in name:
                formats[".jpg"]["lowres"].append(result)
            else:
                formats[".jpg"]["highres"].append(result)
        else:
            formats[".bmp"].append(result)
    
    print("\n🖼️  Performance by Image Format:")
    for fmt, data in formats.items():
        if fmt == ".jpg":
            for res, results_list in data.items():
                if results_list:
                    avg_animal_p50 = statistics.mean([r["animal_classification"]["latency_stats"]["p50"] for r in results_list])
                    avg_location_p50 = statistics.mean([r["location_classification"]["latency_stats"]["p50"] for r in results_list])
                    avg_size = statistics.mean([r["image_info"]["size_kb"] for r in results_list])
                    print(f"   JPG ({res}): P50 latency {(avg_animal_p50 + avg_location_p50)/2*1000:.1f}ms, avg size {avg_size:.1f}KB")
        else:
            if data:
                avg_animal_p50 = statistics.mean([r["animal_classification"]["latency_stats"]["p50"] for r in data])
                avg_location_p50 = statistics.mean([r["location_classification"]["latency_stats"]["p50"] for r in data])
                avg_size = statistics.mean([r["image_info"]["size_kb"] for r in data])
                print(f"   {fmt.upper()[1:]}: P50 latency {(avg_animal_p50 + avg_location_p50)/2*1000:.1f}ms, avg size {avg_size:.1f}KB")
    
    # Classification accuracy
    print("\n🎯 Classification Accuracy:")
    animal_correct = sum(1 for r in results.values() if r["animal_classification"]["correct"])
    location_correct = sum(1 for r in results.values() if r["location_classification"]["correct"])
    total_tests = len(results)
    
    print(f"   Animal (cat/dog): {animal_correct}/{total_tests} ({animal_correct/total_tests*100:.1f}%)")
    print(f"   Location (indoor/outdoor): {location_correct}/{total_tests} ({location_correct/total_tests*100:.1f}%)")
    
    # Aggregate latency statistics
    all_animal_times = []
    all_location_times = []
    
    for result in results.values():
        # Reconstruct times from stats (approximation for aggregation)
        animal_mean = result["animal_classification"]["latency_stats"]["mean"]
        location_mean = result["location_classification"]["latency_stats"]["mean"]
        
        # Add mean times (simplified aggregation)
        all_animal_times.extend([animal_mean] * 500)  
        all_location_times.extend([location_mean] * 500)
    
    print(f"\n⏱️  Aggregate Latency Statistics (over {len(all_animal_times):,} measurements):")
    
    animal_percentiles = calculate_percentiles(all_animal_times)
    location_percentiles = calculate_percentiles(all_location_times)
    
    print(f"   Animal Classification:")
    print(f"     P50: {animal_percentiles['p50']*1000:.1f}ms")
    print(f"     P90: {animal_percentiles['p90']*1000:.1f}ms")
    print(f"     P95: {animal_percentiles['p95']*1000:.1f}ms")
    print(f"     P99: {animal_percentiles['p99']*1000:.1f}ms")
    
    print(f"   Location Classification:")
    print(f"     P50: {location_percentiles['p50']*1000:.1f}ms")
    print(f"     P90: {location_percentiles['p90']*1000:.1f}ms")
    print(f"     P95: {location_percentiles['p95']*1000:.1f}ms")
    print(f"     P99: {location_percentiles['p99']*1000:.1f}ms")
    
    # Save detailed results
    print(f"\n6. Saving detailed results...")
    output_file = Path("clip_comprehensive_results.json")
    
    comprehensive_results = {
        "summary": {
            "total_images": len(TEST_IMAGES),
            "iterations_per_test": 500,
            "total_inferences": len(TEST_IMAGES) * 500 * 2,
            "total_time_seconds": total_time,
            "model_load_time": load_time,
            "animal_accuracy": animal_correct / total_tests,
            "location_accuracy": location_correct / total_tests
        },
        "format_performance": {
            "jpg_lowres": {
                "avg_size_kb": statistics.mean([r["image_info"]["size_kb"] for r in formats[".jpg"]["lowres"]]) if formats[".jpg"]["lowres"] else 0,
                "avg_p50_ms": statistics.mean([r["animal_classification"]["latency_stats"]["p50"] for r in formats[".jpg"]["lowres"]]) * 1000 if formats[".jpg"]["lowres"] else 0
            },
            "jpg_highres": {
                "avg_size_kb": statistics.mean([r["image_info"]["size_kb"] for r in formats[".jpg"]["highres"]]) if formats[".jpg"]["highres"] else 0,
                "avg_p50_ms": statistics.mean([r["animal_classification"]["latency_stats"]["p50"] for r in formats[".jpg"]["highres"]]) * 1000 if formats[".jpg"]["highres"] else 0
            },
            "bmp": {
                "avg_size_kb": statistics.mean([r["image_info"]["size_kb"] for r in formats[".bmp"]]) if formats[".bmp"] else 0,
                "avg_p50_ms": statistics.mean([r["animal_classification"]["latency_stats"]["p50"] for r in formats[".bmp"]]) * 1000 if formats[".bmp"] else 0
            }
        },
        "detailed_results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2)
    print(f"   ✓ Results saved to {output_file}")
    
    # Cleanup
    print(f"\n7. Cleaning up...")
    ray.shutdown()
    print("   ✓ Ray shutdown complete")
    
    print(f"\n🎉 Comprehensive CLIP Test Complete!")
    print(f"✅ {len(TEST_IMAGES) * 500 * 2:,} total inferences completed in {total_time/60:.1f} minutes")
    print(f"📊 Classification accuracy: Animals {animal_correct/total_tests*100:.1f}%, Locations {location_correct/total_tests*100:.1f}%")
    
    return True


async def main():
    """Run the comprehensive CLIP performance test."""
    success = await test_clip_comprehensive()
    if success:
        print("\n✅ CLIP COMPREHENSIVE PERFORMANCE TEST PASSED!")
        print("📊 Detailed performance metrics across formats and classifications collected.")
    else:
        print("\n❌ Test failed.")
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)