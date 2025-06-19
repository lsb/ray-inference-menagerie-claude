#!/usr/bin/env python3
"""Generate test images using Stable Diffusion for Model Zoo testing."""
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_pipeline():
    """Set up Stable Diffusion pipeline."""
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading Stable Diffusion 1.5 on {device}...")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # Disable for testing
        requires_safety_checker=False
    )
    pipeline = pipeline.to(device)
    
    # Enable memory efficient attention if available
    if hasattr(pipeline, "enable_memory_efficient_attention"):
        pipeline.enable_memory_efficient_attention()
    
    # Enable attention slicing for lower memory usage
    pipeline.enable_attention_slicing()
    
    logger.info("Pipeline loaded successfully")
    return pipeline


def generate_test_images(output_dir: Path, skip_existing: bool = True):
    """Generate all test images using Stable Diffusion.
    
    Args:
        output_dir: Directory to save images
        skip_existing: Skip generation if image already exists
    """
    # Test image specifications
    test_specs = [
        {
            "filename": "cat_office_typing.jpg",
            "prompt": "a cat typing at a computer in a modern office, professional lighting, detailed, photorealistic",
            "description": "Cat typing in office (indoor cat scene)"
        },
        {
            "filename": "dog_office_typing.jpg", 
            "prompt": "a dog typing at a computer in a modern office, professional lighting, detailed, photorealistic",
            "description": "Dog typing in office (indoor dog scene)"
        },
        {
            "filename": "cat_mountain_sunrise.jpg",
            "prompt": "a cat sitting on a mountain peak at sunrise, beautiful landscape, golden hour lighting, detailed, photorealistic",
            "description": "Cat on mountain at sunrise (outdoor cat scene)"
        },
        {
            "filename": "dog_mountain_sunrise.jpg",
            "prompt": "a dog sitting on a mountain peak at sunrise, beautiful landscape, golden hour lighting, detailed, photorealistic", 
            "description": "Dog on mountain at sunrise (outdoor dog scene)"
        }
    ]
    
    # Set up pipeline
    pipeline = setup_pipeline()
    
    # Generation parameters
    generation_params = {
        "height": 512,
        "width": 512,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device=pipeline.device).manual_seed(42)  # Fixed seed for reproducibility
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for spec in test_specs:
        output_path = output_dir / spec["filename"]
        
        if skip_existing and output_path.exists():
            logger.info(f"✓ Skipping {spec['filename']} (already exists)")
            continue
        
        logger.info(f"Generating: {spec['description']}")
        logger.info(f"Prompt: {spec['prompt']}")
        
        try:
            # Generate image
            result = pipeline(spec["prompt"], **generation_params)
            image = result.images[0]
            
            # Save image
            image.save(output_path, quality=95)
            logger.info(f"✓ Saved {output_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to generate {spec['filename']}: {e}")
            continue
    
    logger.info(f"\n✓ Test image generation complete!")
    logger.info(f"Images saved to: {output_dir}")
    logger.info("\nGenerated images:")
    for spec in test_specs:
        output_path = output_dir / spec["filename"]
        if output_path.exists():
            logger.info(f"  - {spec['filename']}: {spec['description']}")


def main():
    """Generate Stable Diffusion test images."""
    parser = argparse.ArgumentParser(description="Generate test images using Stable Diffusion 1.5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_images/sd_generated"),
        help="Output directory for generated images (default: test_images/sd_generated)"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate images even if they already exist"
    )
    parser.add_argument(
        "--check-requirements",
        action="store_true",
        help="Check if required packages are installed"
    )
    
    args = parser.parse_args()
    
    if args.check_requirements:
        try:
            import diffusers
            import torch
            logger.info("✓ Required packages (diffusers, torch) are available")
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            else:
                logger.info("ℹ CUDA not available, will use CPU (slower)")
            return 0
        except ImportError as e:
            logger.error(f"✗ Missing required package: {e}")
            logger.error("Install with: pip install diffusers torch torchvision")
            return 1
    
    try:
        # Check dependencies
        import diffusers
        import torch
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.error("Install with: pip install diffusers torch torchvision")
        return 1
    
    # Generate images
    generate_test_images(args.output_dir, skip_existing=not args.regenerate)
    return 0


if __name__ == "__main__":
    exit(main())