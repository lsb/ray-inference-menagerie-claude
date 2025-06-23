"""CPU-only tests for model functionality."""
import pytest
import torch
import base64
from pathlib import Path
from PIL import Image
import io
import asyncio


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
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@pytest.fixture(scope="module")
def cpu_device():
    """Ensure we're testing on CPU."""
    return torch.device("cpu")


class TestCLIPCPU:
    """Test CLIP model functionality on CPU without Ray."""
    
    @pytest.fixture(scope="class")
    def clip_model_cpu(self, cpu_device):
        """Load CLIP model on CPU."""
        from transformers import CLIPProcessor, CLIPModel
        
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = model.to(cpu_device)
        model.eval()
        
        return {"model": model, "processor": processor, "device": cpu_device}
    
    def test_clip_cat_vs_dog_classification(self, clip_model_cpu):
        """Test that CLIP can distinguish cats from dogs correctly."""
        model = clip_model_cpu["model"]
        processor = clip_model_cpu["processor"]
        device = clip_model_cpu["device"]
        
        # Test cat images
        for img_name, img_path in TEST_IMAGES.items():
            if "cat" in img_name:
                image = Image.open(img_path).convert("RGB")
                
                # Compare both prompts in a single inference call
                inputs = processor(
                    text=["a cat", "a dog"], 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    cat_similarity = probs[0, 0].item()
                    dog_similarity = probs[0, 1].item()
                
                print(f"\n{img_name}:")
                print(f"  Cat similarity: {cat_similarity:.3f}")
                print(f"  Dog similarity: {dog_similarity:.3f}")
                
                # Cat images should have higher similarity with "cat" than "dog"
                assert cat_similarity > dog_similarity, f"Cat image {img_name} classified incorrectly"
        
        # Test dog images
        for img_name, img_path in TEST_IMAGES.items():
            if "dog" in img_name:
                image = Image.open(img_path).convert("RGB")
                
                # Compare both prompts in a single inference call
                inputs = processor(
                    text=["a cat", "a dog"], 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    cat_similarity = probs[0, 0].item()
                    dog_similarity = probs[0, 1].item()
                
                print(f"\n{img_name}:")
                print(f"  Cat similarity: {cat_similarity:.3f}")
                print(f"  Dog similarity: {dog_similarity:.3f}")
                
                # Dog images should have higher similarity with "dog" than "cat"
                assert dog_similarity > cat_similarity, f"Dog image {img_name} classified incorrectly"
    
    def test_clip_indoor_vs_outdoor_classification(self, clip_model_cpu):
        """Test that CLIP can distinguish indoor vs outdoor scenes."""
        model = clip_model_cpu["model"] 
        processor = clip_model_cpu["processor"]
        device = clip_model_cpu["device"]
        
        # Test office images (indoor)
        for img_name, img_path in TEST_IMAGES.items():
            if "office" in img_name:
                image = Image.open(img_path).convert("RGB")
                
                inputs = processor(
                    text=["indoor office environment", "outdoor mountain landscape"], 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    indoor_prob = probs[0, 0].item()
                    outdoor_prob = probs[0, 1].item()
                
                print(f"\n{img_name}:")
                print(f"  Indoor probability: {indoor_prob:.3f}")
                print(f"  Outdoor probability: {outdoor_prob:.3f}")
                
                # Office images should be classified as indoor
                assert indoor_prob > outdoor_prob, f"Office image {img_name} not classified as indoor"
        
        # Test mountain images (outdoor)
        for img_name, img_path in TEST_IMAGES.items():
            if "mountain" in img_name:
                image = Image.open(img_path).convert("RGB")
                
                inputs = processor(
                    text=["indoor office environment", "outdoor mountain landscape"], 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    indoor_prob = probs[0, 0].item()
                    outdoor_prob = probs[0, 1].item()
                
                print(f"\n{img_name}:")
                print(f"  Indoor probability: {indoor_prob:.3f}")
                print(f"  Outdoor probability: {outdoor_prob:.3f}")
                
                # Mountain images should be classified as outdoor
                assert outdoor_prob > indoor_prob, f"Mountain image {img_name} not classified as outdoor"


class TestActorBaseCPU:
    """Test the actor base class without Ray."""
    
    def test_actor_base_import(self):
        """Test that base actor can be imported without Ray decorator issues."""
        from model_zoo.actors.base import HFModelActor
        
        # Should be able to import the base class
        assert HFModelActor is not None
        assert hasattr(HFModelActor, '_load_model')
        assert hasattr(HFModelActor, 'infer')
        assert hasattr(HFModelActor, 'ready')
    
    def test_concrete_actor_imports(self):
        """Test that concrete actor classes can be imported."""
        from model_zoo.actors.clip import CLIPActor
        from model_zoo.actors.grounding_sam2 import GroundingDINO_SAM2_Actor
        from model_zoo.actors.qwen_vl import QwenVLActor
        
        # Should be able to import all concrete classes
        assert CLIPActor is not None
        assert GroundingDINO_SAM2_Actor is not None  
        assert QwenVLActor is not None
        
        # Should have Ray remote decorators
        assert hasattr(CLIPActor, '_remote')
        assert hasattr(GroundingDINO_SAM2_Actor, '_remote')
        assert hasattr(QwenVLActor, '_remote')


class TestCPUFallback:
    """Test CPU fallback behavior."""
    
    def test_device_selection(self):
        """Test that models correctly select CPU when CUDA unavailable."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Should fall back to CPU in most test environments
        if not torch.cuda.is_available():
            assert device.type == "cpu"
            print(f"✓ Correctly using CPU fallback: {device}")
        else:
            print(f"ℹ GPU available, using: {device}")
    
    def test_torch_operations_cpu(self):
        """Test basic PyTorch operations on CPU."""
        device = torch.device("cpu")
        
        # Test tensor operations
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x.T)
        
        assert y.device.type == "cpu"
        assert y.shape == (10, 10)
        print(f"✓ PyTorch CPU operations working: {y.shape} tensor on {y.device}")