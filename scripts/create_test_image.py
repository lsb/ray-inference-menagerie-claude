#!/usr/bin/env python3
"""Create test images for Model Zoo inference testing."""
import argparse
from PIL import Image
import numpy as np
from pathlib import Path


def create_red_square_with_white_center(output_path: str = "test_image.jpg", size: int = 224):
    """Create a red square with white center for testing.
    
    Args:
        output_path: Path to save the image
        size: Image size (square)
    """
    # Create a red square
    img = Image.new('RGB', (size, size), color='red')
    pixels = img.load()
    
    # Add white center
    center_start = size // 3
    center_end = 2 * size // 3
    
    for i in range(center_start, center_end):
        for j in range(center_start, center_end):
            pixels[i, j] = (255, 255, 255)
    
    img.save(output_path)
    print(f"✓ Created {output_path} - red square with white center")
    return output_path


def create_gradient_image(output_path: str = "gradient_test.jpg", size: int = 224):
    """Create a gradient test image.
    
    Args:
        output_path: Path to save the image
        size: Image size (square)
    """
    # Create gradient array
    gradient = np.zeros((size, size, 3), dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            gradient[i, j] = [
                int(255 * i / size),  # Red increases top to bottom
                int(255 * j / size),  # Green increases left to right
                128                    # Blue constant
            ]
    
    img = Image.fromarray(gradient)
    img.save(output_path)
    print(f"✓ Created {output_path} - gradient test image")
    return output_path


def create_shapes_image(output_path: str = "shapes_test.jpg", size: int = 224):
    """Create an image with multiple shapes for detection testing.
    
    Args:
        output_path: Path to save the image
        size: Image size (square)
    """
    # Create white background
    img = Image.new('RGB', (size, size), color='white')
    pixels = img.load()
    
    # Add red square in top-left
    for i in range(20, 70):
        for j in range(20, 70):
            pixels[i, j] = (255, 0, 0)
    
    # Add blue circle in top-right (approximate)
    center_x, center_y = size - 45, 45
    radius = 25
    for i in range(size):
        for j in range(size):
            if (i - center_x)**2 + (j - center_y)**2 <= radius**2:
                pixels[i, j] = (0, 0, 255)
    
    # Add green triangle in bottom (approximate)
    for i in range(size // 2, size - 20):
        for j in range(size // 2 - (i - size // 2), size // 2 + (i - size // 2)):
            if 20 < j < size - 20:
                pixels[i, j] = (0, 255, 0)
    
    img.save(output_path)
    print(f"✓ Created {output_path} - multiple shapes for detection")
    return output_path


def main():
    """Generate test images for Model Zoo."""
    parser = argparse.ArgumentParser(description="Create test images for Model Zoo inference")
    parser.add_argument(
        "--type", 
        choices=["simple", "gradient", "shapes", "all"],
        default="simple",
        help="Type of test image to create"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default=None,
        help="Output filename (default: auto-generated based on type)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Image size in pixels (default: 224)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = Path("test_images")
    output_dir.mkdir(exist_ok=True)
    
    if args.type == "simple" or args.type == "all":
        output = args.output or str(output_dir / "test_image.jpg")
        create_red_square_with_white_center(output, args.size)
    
    if args.type == "gradient" or args.type == "all":
        output = args.output or str(output_dir / "gradient_test.jpg")
        create_gradient_image(output, args.size)
    
    if args.type == "shapes" or args.type == "all":
        output = args.output or str(output_dir / "shapes_test.jpg")
        create_shapes_image(output, args.size)
    
    if args.type == "all":
        print(f"\n✓ Created all test images in {output_dir}/")
        print("  - test_image.jpg: Red square with white center (good for CLIP)")
        print("  - gradient_test.jpg: Color gradient (good for general testing)")
        print("  - shapes_test.jpg: Multiple shapes (good for detection models)")


if __name__ == "__main__":
    main()