"""Test fixtures for image data."""
import base64
import io
from PIL import Image
import numpy as np


def create_test_image(width=224, height=224, color=(255, 0, 0)):
    """Create a simple test image.
    
    Args:
        width: Image width
        height: Image height
        color: RGB color tuple
        
    Returns:
        Base64 encoded PNG image
    """
    # Create a simple colored image with a square in the middle
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    img_array[:, :] = color
    
    # Add a white square in the middle
    center_x, center_y = width // 2, height // 2
    square_size = min(width, height) // 4
    img_array[
        center_y - square_size:center_y + square_size,
        center_x - square_size:center_x + square_size
    ] = (255, 255, 255)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_b64


# Pre-generated test fixtures
TEST_IMAGE_B64 = create_test_image()
TEST_IMAGE_SMALL_B64 = create_test_image(64, 64)
TEST_IMAGE_LARGE_B64 = create_test_image(512, 512)