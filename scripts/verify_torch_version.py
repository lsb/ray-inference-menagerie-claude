#!/usr/bin/env python3
"""Verify PyTorch version and compatibility with model zoo."""
import sys

def verify_torch_version():
    """Check PyTorch version and compatibility."""
    print("🔍 Verifying PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check minimum version
        major, minor = map(int, torch.__version__.split('.')[:2])
        if major < 2 or (major == 2 and minor < 7):
            print(f"❌ PyTorch version {torch.__version__} is too old. Need >= 2.7.1")
            return False
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("ℹ️  CUDA not available - CPU mode only")
        
        # Check torchvision
        try:
            import torchvision
            print(f"✅ Torchvision version: {torchvision.__version__}")
        except ImportError:
            print("❌ Torchvision not installed")
            return False
        
        # Test basic operations
        print("\n🧪 Testing basic PyTorch operations...")
        
        # CPU tensor test
        x = torch.randn(10, 10)
        y = torch.matmul(x, x.T)
        print(f"✅ CPU tensor operations working: {y.shape}")
        
        # Test autograd
        x = torch.randn(5, requires_grad=True)
        y = x.sum()
        y.backward()
        print(f"✅ Autograd working: gradient shape {x.grad.shape}")
        
        # Test transformers compatibility
        print("\n🤝 Testing transformers compatibility...")
        try:
            from transformers import CLIPModel
            # Just import, don't load model
            print("✅ Transformers CLIP import successful")
        except ImportError as e:
            print(f"❌ Transformers import failed: {e}")
            return False
        
        print("\n🎉 PyTorch 2.7.1+ verification complete!")
        print("✅ All compatibility checks passed")
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        print("Install with: pip install torch>=2.7.1 torchvision>=0.18.1")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = verify_torch_version()
    sys.exit(0 if success else 1)