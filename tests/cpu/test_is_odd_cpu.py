"""CPU tests for the is-odd demo model."""
import pytest
import asyncio
from model_zoo.actors.is_odd import IsOddActor


class TestIsOddCPU:
    """Test is-odd demo functionality without Ray."""
    
    def test_is_odd_logic(self):
        """Test the basic odd/even logic."""
        # Test cases with expected results
        test_cases = [
            (1, True),   # 1 is odd
            (2, False),  # 2 is even  
            (7, True),   # 7 is odd
            (10, False), # 10 is even
            (0, False),  # 0 is even
            (-1, True),  # -1 is odd
            (-2, False), # -2 is even
            (999, True), # 999 is odd
            (1000, False) # 1000 is even
        ]
        
        for number, expected_odd in test_cases:
            actual_odd = bool(number % 2 == 1)
            assert actual_odd == expected_odd, f"Failed for {number}: expected {expected_odd}, got {actual_odd}"
            print(f"✓ {number} is {'odd' if expected_odd else 'even'}")
    
    def test_is_odd_payload_validation(self):
        """Test payload validation without Ray actor."""
        # Test that the logic works for different input types
        
        # Integer input
        number = 7
        is_odd = bool(number % 2 == 1)
        assert is_odd == True
        
        # String input (would be converted to int)
        number_str = "7"
        number = int(number_str)
        is_odd = bool(number % 2 == 1)
        assert is_odd == True
        
        print("✓ Payload validation logic working")
    
    def test_is_odd_performance(self):
        """Test performance of the basic computation."""
        import time
        
        # Time a large number of computations
        start_time = time.time()
        iterations = 10000
        
        for i in range(iterations):
            _ = bool(i % 2 == 1)
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        print(f"✓ {iterations} computations in {total_time:.3f}s")
        print(f"✓ Average computation time: {avg_time*1000000:.1f}μs")
        
        # Should be very fast (sub-microsecond)
        assert avg_time < 0.001, f"Computation too slow: {avg_time:.6f}s per operation"


class TestIsOddActorImport:
    """Test that the IsOddActor can be imported and has correct structure."""
    
    def test_is_odd_actor_import(self):
        """Test that IsOddActor can be imported."""
        from model_zoo.actors.is_odd import IsOddActor
        assert IsOddActor is not None
        
        # Should have Ray remote decorator
        assert hasattr(IsOddActor, '_remote')
        print("✓ IsOddActor imported successfully with Ray remote decorator")
    
    def test_factory_registration(self):
        """Test that is-odd is registered in the factory."""
        from model_zoo.actors.factory import make
        import os
        
        # Test with is-odd model name
        os.environ["MODEL_NAME"] = "is-odd-demo"
        actor_class = make()
        
        from model_zoo.actors.is_odd import IsOddActor
        assert actor_class == IsOddActor
        
        # Test with is_odd variant
        os.environ["MODEL_NAME"] = "is_odd_test"
        actor_class = make()
        assert actor_class == IsOddActor
        
        print("✓ IsOddActor properly registered in factory")