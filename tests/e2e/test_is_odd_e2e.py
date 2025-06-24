#!/usr/bin/env python3
"""End-to-end test for the is-odd demo actor to measure Ray overhead."""
import ray
import asyncio
import time


async def test_is_odd_actor_e2e():
    """Test is-odd actor end-to-end to measure Ray overhead."""
    print("🎯 Testing Is-Odd Demo Actor - Measuring Ray Overhead")
    print("=" * 55)
    
    # Initialize Ray
    print("\n1. Initializing Ray...")
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(num_cpus=4, num_gpus=0)  # CPU only for this demo
    print(f"   ✓ Ray initialized with {ray.cluster_resources()}")
    
    # Import and create is-odd actor
    print("\n2. Creating is-odd demo actor...")
    from model_zoo.actors.is_odd import IsOddActor
    
    actor = IsOddActor.options(num_gpus=0).remote("is-odd-demo", "no-weights-needed")
    print("   ✓ Is-odd actor created")
    
    # Test actor readiness
    print("\n3. Testing actor readiness...")
    start_time = time.time()
    ready = await actor.ready.remote()
    load_time = time.time() - start_time
    print(f"   ✓ Actor ready: {ready} (load time: {load_time:.3f}s)")
    
    if not ready:
        print("   ✗ Actor failed to load!")
        return False
    
    # Test inference with various numbers
    print("\n4. Testing inference with various numbers...")
    
    test_cases = [
        {"number": 1, "expected": True},
        {"number": 2, "expected": False},
        {"number": 7, "expected": True},
        {"number": 10, "expected": False},
        {"number": 999, "expected": True},
        {"number": 1000, "expected": False},
    ]
    
    results = []
    total_inference_time = 0
    
    for i, case in enumerate(test_cases):
        print(f"\n   Test {i+1}: number={case['number']}")
        
        payload = {"number": case["number"]}
        
        start_time = time.time()
        result = await actor.infer.remote(payload)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        is_odd = result["is_odd"]
        expected = case["expected"]
        correct = is_odd == expected
        
        print(f"     Result: {result}")
        print(f"     Expected odd: {expected}, Got: {is_odd} ({'✓' if correct else '✗'})")
        print(f"     Inference time: {inference_time:.3f}s")
        
        results.append({
            "number": case["number"],
            "expected": expected,
            "actual": is_odd,
            "correct": correct,
            "inference_time": inference_time
        })
    
    # Test concurrent inference (measure overhead)
    print("\n5. Testing concurrent inference (overhead measurement)...")
    concurrent_payloads = [{"number": i} for i in range(10, 20)]
    
    start_time = time.time()
    concurrent_results = await asyncio.gather(*[
        actor.infer.remote(payload) for payload in concurrent_payloads
    ])
    concurrent_time = time.time() - start_time
    
    print(f"   ✓ Concurrent inference: {len(concurrent_results)} requests in {concurrent_time:.3f}s")
    print(f"   ✓ Average time per request: {concurrent_time/len(concurrent_results):.3f}s")
    print(f"   ✓ Ray overhead per call: ~{concurrent_time/len(concurrent_results)*1000:.1f}ms")
    
    # Summary
    print("\n6. 🎯 Is-Odd Demo Test Summary:")
    print("=" * 40)
    
    total_tests = len(results)
    correct_tests = sum(1 for r in results if r["correct"])
    avg_inference_time = total_inference_time / len(results) if results else 0
    
    print(f"Total tests: {total_tests}")
    print(f"Correct results: {correct_tests}/{total_tests}")
    print(f"Accuracy: {correct_tests/total_tests*100:.1f}%" if total_tests > 0 else "No tests")
    print(f"Average inference time: {avg_inference_time:.3f}s")
    print(f"Ray overhead per call: ~{avg_inference_time*1000:.1f}ms")
    
    # Ray overhead analysis
    print(f"\n🔍 Ray Overhead Analysis:")
    print(f"   • Actor creation: {load_time:.3f}s")
    print(f"   • Single inference: {avg_inference_time:.3f}s")
    print(f"   • Concurrent inference: {concurrent_time/len(concurrent_results):.3f}s per call")
    print(f"   • Ray call overhead: ~{avg_inference_time*1000:.1f}ms (trivial computation)")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for i, result in enumerate(results):
        status = "✅" if result["correct"] else "❌"
        print(f"  {status} {result['number']} is {'odd' if result['expected'] else 'even'}: {result['inference_time']:.3f}s")
    
    # Cleanup
    print(f"\n7. Cleaning up...")
    ray.shutdown()
    print("   ✓ Ray shutdown complete")
    
    print(f"\n🎉 Is-Odd Demo Test Complete!")
    return correct_tests == total_tests


async def main():
    """Run the is-odd demo test."""
    success = await test_is_odd_actor_e2e()
    if success:
        print("\n✅ ALL TESTS PASSED! Is-odd demo actor working perfectly.")
        print("📊 Use these metrics to understand Ray overhead for simple computations.")
    else:
        print("\n❌ Some tests failed. Check the results above.")
    return success


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(main())
    exit(0 if result else 1)