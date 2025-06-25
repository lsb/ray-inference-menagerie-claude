#!/usr/bin/env python3
"""Generate test images in multiple formats: low-res JPG, high-res JPG, and BMP."""
from pathlib import Path
from PIL import Image
import shutil


def generate_image_formats():
    """Generate test images in different formats for performance testing."""
    
    # Paths
    fixtures_dir = Path(__file__).parent.parent / "test_images" / "fixtures"
    formats_dir = Path(__file__).parent.parent / "test_images" / "formats"
    
    # Create formats directory
    formats_dir.mkdir(exist_ok=True)
    
    # Source images
    source_images = {
        "cat_office": fixtures_dir / "cat_office_typing.jpg",
        "dog_office": fixtures_dir / "dog_office_typing.jpg",
        "cat_mountain": fixtures_dir / "cat_mountain_sunrise.jpg",
        "dog_mountain": fixtures_dir / "dog_mountain_sunrise.jpg"
    }
    
    # Check all source images exist
    missing = [name for name, path in source_images.items() if not path.exists()]
    if missing:
        print(f"❌ Missing source images: {missing}")
        print("Run: python scripts/generate_sd_test_images.py --output-dir test_images/fixtures")
        return False
    
    print("📸 Generating test images in multiple formats...")
    
    for name, source_path in source_images.items():
        print(f"\nProcessing {name}...")
        
        # Load original image
        img = Image.open(source_path)
        orig_width, orig_height = img.size
        print(f"  Original size: {orig_width}x{orig_height}")
        
        # 1. Low-res JPG (256x256, quality 70)
        low_res = img.resize((256, 256), Image.Resampling.LANCZOS)
        low_res_path = formats_dir / f"{name}_lowres.jpg"
        low_res.save(low_res_path, "JPEG", quality=70, optimize=True)
        low_res_size = low_res_path.stat().st_size / 1024
        print(f"  ✓ Low-res JPG: 256x256, {low_res_size:.1f}KB")
        
        # 2. High-res JPG (original size, quality 95)
        high_res_path = formats_dir / f"{name}_highres.jpg"
        img.save(high_res_path, "JPEG", quality=95)
        high_res_size = high_res_path.stat().st_size / 1024
        print(f"  ✓ High-res JPG: {orig_width}x{orig_height}, {high_res_size:.1f}KB")
        
        # 3. BMP (uncompressed, 512x512)
        bmp = img.resize((512, 512), Image.Resampling.LANCZOS)
        bmp_path = formats_dir / f"{name}.bmp"
        bmp.save(bmp_path, "BMP")
        bmp_size = bmp_path.stat().st_size / 1024
        print(f"  ✓ BMP: 512x512, {bmp_size:.1f}KB")
    
    # Summary
    print("\n✅ Image format generation complete!")
    print(f"📁 Images saved to: {formats_dir}")
    
    # List all generated files
    print("\n📋 Generated files:")
    for fmt in ["lowres.jpg", "highres.jpg", ".bmp"]:
        files = list(formats_dir.glob(f"*{fmt}"))
        total_size = sum(f.stat().st_size for f in files) / 1024 / 1024
        print(f"  {fmt}: {len(files)} files, {total_size:.1f}MB total")
    
    return True


if __name__ == "__main__":
    success = generate_image_formats()
    exit(0 if success else 1)