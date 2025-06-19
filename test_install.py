#!/usr/bin/env python3
"""Test script to verify installation works."""
import sys
import subprocess

def test_basic_import():
    """Test basic imports work."""
    try:
        import model_zoo
        print("✓ model_zoo imports successfully")
        
        import model_zoo.cli
        print("✓ model_zoo.cli imports successfully")
        
        from model_zoo.actors.base import HFModelActor
        print("✓ HFModelActor imports successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_cli_help():
    """Test CLI help command."""
    try:
        # Test module import
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, '.'); from model_zoo.cli import app; app(['--help'])"
        ], capture_output=True, text=True, timeout=10)
        
        if "Model Zoo CLI" in result.stdout or "Usage:" in result.stdout:
            print("✓ CLI help works via module")
            return True
        else:
            print(f"✗ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Model Zoo installation...")
    
    success = True
    success &= test_basic_import()
    success &= test_cli_help()
    
    if success:
        print("\n🎉 All tests passed! Installation is working.")
        print("You can now run: python -m model_zoo.cli --help")
    else:
        print("\n❌ Some tests failed. Check dependencies.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())