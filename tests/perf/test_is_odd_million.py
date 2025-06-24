#!/usr/bin/env python3
"""Performance test for is-odd demo actor with 1 million iterations."""
import ray
import asyncio
import time
import statistics
from pathlib import Path


async def test_is_odd_million_iterations():
    """Test is-odd actor with 1 million iterations to measure Ray overhead precisely."""
    print("🎯 Is-Odd Million Iteration Performance Test")
    print("=" * 50)
    print("Running 1,000,000 iterations to measure precise Ray overhead")
    
    # Initialize Ray
    print("\n1. Initializing Ray...")
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(num_cpus=4, num_gpus=0)
    print(f"   ✓ Ray initialized with {ray.cluster_resources()}")
    
    # Import and create is-odd actor
    print("\n2. Creating is-odd demo actor...")
    from model_zoo.actors.is_odd import IsOddActor
    
    actor = IsOddActor.options(num_gpus=0).remote("is-odd-million", "no-weights")
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
    
    # Warm up the actor
    print("\n4. Warming up actor with 1,000 initial calls...")
    warmup_start = time.time()
    warmup_times = []
    
    for i in range(1000):
        payload = {"number": i}
        start_time = time.time()
        result = await actor.infer.remote(payload)
        warmup_times.append(time.time() - start_time)
        
        # Verify correctness
        expected = bool(i % 2 == 1)
        assert result["is_odd"] == expected
    
    warmup_time = time.time() - warmup_start
    warmup_avg = statistics.mean(warmup_times)
    print(f"   ✓ Warmup complete: 1,000 calls in {warmup_time:.2f}s (avg: {warmup_avg*1000:.2f}ms)")
    
    # Million iteration performance test
    print("\n5. Running 1 MILLION inference iterations...")
    print("   Progress updates every 100,000 iterations")
    
    MILLION_ITERATIONS = 1_000_000
    batch_size = 10_000  # Process in batches for progress updates
    num_batches = MILLION_ITERATIONS // batch_size
    
    all_times = []
    total_start_time = time.time()
    
    for batch_idx in range(num_batches):
        batch_times = []
        batch_start = time.time()
        
        # Run batch of inferences
        for i in range(batch_size):
            number = (batch_idx * batch_size) + i
            payload = {"number": number}
            
            start_time = time.time()
            result = await actor.infer.remote(payload)
            inference_time = time.time() - start_time
            batch_times.append(inference_time)
            
            # Verify correctness (spot check every 1000th)
            if i % 1000 == 0:
                expected = bool(number % 2 == 1)
                assert result["is_odd"] == expected
        
        all_times.extend(batch_times)
        batch_elapsed = time.time() - batch_start
        
        # Progress update every 100,000 iterations (10 batches)
        if (batch_idx + 1) % 10 == 0:
            completed = (batch_idx + 1) * batch_size
            progress = completed / MILLION_ITERATIONS * 100
            batch_avg = statistics.mean(batch_times)
            batch_throughput = batch_size / batch_elapsed
            
            print(f"   {progress:3.0f}% ({completed:,}/{MILLION_ITERATIONS:,}) - "
                  f"Batch avg: {batch_avg*1000:.2f}ms, "
                  f"Throughput: {batch_throughput:.0f} req/s")
    
    total_elapsed = time.time() - total_start_time
    
    # Calculate comprehensive statistics
    print("\n6. Calculating statistics on 1 million data points...")
    
    avg_time = statistics.mean(all_times)
    median_time = statistics.median(all_times)
    min_time = min(all_times)
    max_time = max(all_times)
    stdev_time = statistics.stdev(all_times)
    
    # Calculate percentiles
    sorted_times = sorted(all_times)
    p50_time = sorted_times[int(0.50 * len(sorted_times))]
    p90_time = sorted_times[int(0.90 * len(sorted_times))]
    p95_time = sorted_times[int(0.95 * len(sorted_times))]
    p99_time = sorted_times[int(0.99 * len(sorted_times))]
    p999_time = sorted_times[int(0.999 * len(sorted_times))]
    
    # Test concurrent inference at scale
    print("\n7. Testing large-scale concurrent inference...")
    concurrent_sizes = [10, 100, 1000]
    concurrent_results = {}
    
    for size in concurrent_sizes:
        payloads = [{"number": i} for i in range(size)]
        
        start_time = time.time()
        results = await asyncio.gather(*[
            actor.infer.remote(payload) for payload in payloads
        ])
        concurrent_time = time.time() - start_time
        
        concurrent_results[size] = {
            "total_time": concurrent_time,
            "avg_time": concurrent_time / size,
            "throughput": size / concurrent_time
        }
        
        print(f"   {size:4d} concurrent: {concurrent_time:.3f}s total, "
              f"{concurrent_time/size*1000:.2f}ms avg, "
              f"{size/concurrent_time:.0f} req/s")
    
    # Performance summary
    print("\n8. 🎯 Million Iteration Performance Summary:")
    print("=" * 60)
    
    print(f"Total iterations: {MILLION_ITERATIONS:,}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Overall throughput: {MILLION_ITERATIONS/total_elapsed:,.0f} requests/second")
    
    print(f"\nLatency Statistics (milliseconds):")
    print(f"  Average:     {avg_time*1000:6.2f}ms")
    print(f"  Median:      {median_time*1000:6.2f}ms")
    print(f"  Min:         {min_time*1000:6.2f}ms")
    print(f"  Max:         {max_time*1000:6.2f}ms")
    print(f"  Std Dev:     {stdev_time*1000:6.2f}ms")
    
    print(f"\nLatency Percentiles (milliseconds):")
    print(f"  50th (P50):  {p50_time*1000:6.2f}ms")
    print(f"  90th (P90):  {p90_time*1000:6.2f}ms")
    print(f"  95th (P95):  {p95_time*1000:6.2f}ms")
    print(f"  99th (P99):  {p99_time*1000:6.2f}ms")
    print(f"  99.9th:      {p999_time*1000:6.2f}ms")
    
    print(f"\n🔍 Ray Overhead Analysis:")
    print(f"  Pure computation: <0.001ms (essentially instant)")
    print(f"  Ray overhead: ~{median_time*1000:.2f}ms (median)")
    print(f"  Overhead range: {min_time*1000:.2f}ms - {p99_time*1000:.2f}ms (min-P99)")
    
    print(f"\n📊 Concurrent Performance Scaling:")
    for size, results in concurrent_results.items():
        speedup = avg_time / results["avg_time"]
        print(f"  {size:4d} concurrent: {speedup:5.1f}x speedup, "
              f"{results['throughput']:6.0f} req/s")
    
    # Distribution analysis
    print(f"\n📈 Latency Distribution Analysis:")
    
    # Count requests in different latency buckets
    buckets = {
        "<1ms": sum(1 for t in all_times if t < 0.001),
        "1-2ms": sum(1 for t in all_times if 0.001 <= t < 0.002),
        "2-5ms": sum(1 for t in all_times if 0.002 <= t < 0.005),
        "5-10ms": sum(1 for t in all_times if 0.005 <= t < 0.010),
        "10-20ms": sum(1 for t in all_times if 0.010 <= t < 0.020),
        ">20ms": sum(1 for t in all_times if t >= 0.020),
    }
    
    for bucket, count in buckets.items():
        percentage = count / MILLION_ITERATIONS * 100
        print(f"  {bucket:8s}: {count:8,} ({percentage:5.1f}%)")
    
    # Cleanup
    print(f"\n9. Cleaning up...")
    ray.shutdown()
    print("   ✓ Ray shutdown complete")
    
    print(f"\n🎉 Million Iteration Test Complete!")
    print(f"✅ Ray overhead precisely measured: {median_time*1000:.2f}ms median latency")
    
    return True


async def main():
    """Run the million iteration performance test."""
    success = await test_is_odd_million_iterations()
    if success:
        print("\n✅ IS-ODD MILLION ITERATION TEST PASSED!")
        print("📊 Precise Ray overhead measurements collected over 1M iterations.")
    else:
        print("\n❌ Test failed.")
    return success


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(main())
    exit(0 if result else 1)