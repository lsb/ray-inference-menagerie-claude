#!/usr/bin/env python3
"""Generate test images using Stable Diffusion XL for Model Zoo testing."""
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_pipeline():
    """Set up Stable Diffusion XL pipeline."""
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    device = "cpu"  # Force CPU usage as requested
    
    logger.info(f"Loading Stable Diffusion XL on {device}...")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU
        safety_checker=None,  # Disable for testing
        requires_safety_checker=False,
        use_safetensors=True
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
            "prompt": "a photorealistic domestic cat sitting at a computer desk in a modern office, paws on keyboard, professional lighting, high resolution, ultra detailed, realistic fur texture, sharp focus, professional photography",
            "description": "Cat typing in office (indoor cat scene)"
        },
        {
            "filename": "dog_office_typing.jpg", 
            "prompt": "a photorealistic dog sitting at a computer desk in a modern office, paws on keyboard, professional lighting, high resolution, ultra detailed, realistic fur texture, sharp focus, professional photography",
            "description": "Dog typing in office (indoor dog scene)"
        },
        {
            "filename": "cat_mountain_sunrise.jpg",
            "prompt": "a photorealistic domestic cat sitting on a rocky mountain peak at sunrise, beautiful natural landscape, golden hour lighting, high resolution, ultra detailed, realistic fur texture, sharp focus, landscape photography",
            "description": "Cat on mountain at sunrise (outdoor cat scene)"
        },
        {
            "filename": "dog_mountain_sunrise.jpg",
            "prompt": "a photorealistic dog sitting on a rocky mountain peak at sunrise, beautiful natural landscape, golden hour lighting, high resolution, ultra detailed, realistic fur texture, sharp focus, landscape photography", 
            "description": "Dog on mountain at sunrise (outdoor dog scene)"
        }
    ]
    
    # Set up pipeline
    pipeline = setup_pipeline()
    
    # Generation parameters optimized for photorealistic images
    generation_params = {
        "height": 1024,  # High resolution as requested
        "width": 1024,
        "num_inference_steps": 20,  # Fast generation with 20 steps
        "guidance_scale": 8.0,  # Higher guidance for better prompt adherence
        "num_images_per_prompt": 1,
        "generator": torch.Generator(device=pipeline.device).manual_seed(12345)  # New seed for regeneration
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
    parser = argparse.ArgumentParser(description="Generate test images using Stable Diffusion XL")
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
            logger.info("ℹ Using CPU for Stable Diffusion XL generation (as requested)")
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