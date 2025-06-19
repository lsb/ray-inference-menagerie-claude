#!/bin/bash
# Example CLI usage for model zoo

echo "Model Zoo CLI Examples"
echo "====================="

# Initialize a model family
echo "1. Initialize CLIP model family:"
echo "model-zoo init clip --target k3d"

# Deploy a model
echo -e "\n2. Deploy CLIP model:"
echo "model-zoo deploy clip-vit-base \\"
echo "  --weights gs://my-bucket/clip/weights \\"
echo "  --gpu nvidia-tesla-t4 \\"
echo "  --target k3d"

# List deployed models
echo -e "\n3. List deployed models:"
echo "model-zoo list"

# View logs
echo -e "\n4. View model logs:"
echo "model-zoo logs clip-vit-base --tail"

# Create test image and run inference
echo -e "\n5. Create test image and run inference:"
echo "python scripts/create_test_image.py --type simple"
echo "model-zoo infer clip-vit-base \\"
echo "  --file test_images/test_image.jpg \\"
echo "  --text 'a red square with a white center'"

# Deploy Grounding DINO + SAM2
echo -e "\n6. Deploy Grounding DINO + SAM2:"
echo "model-zoo deploy grounding-sam2 \\"
echo "  --weights gs://my-bucket/grounding-sam2/weights \\"
echo "  --gpu nvidia-tesla-a100 \\"
echo "  --target gke"

# Run segmentation
echo -e "\n7. Run segmentation:"
echo "model-zoo infer grounding-sam2 \\"
echo "  --file test_images/shapes_test.jpg \\"
echo "  --text-prompt 'red square'"

# Deploy Qwen VL
echo -e "\n8. Deploy Qwen VL:"
echo "model-zoo deploy qwen-vl-chat \\"
echo "  --weights gs://my-bucket/qwen-vl/weights \\"
echo "  --gpu nvidia-tesla-a100 \\"
echo "  --target gke"

# Run VQA
echo -e "\n9. Run visual question answering:"
echo "model-zoo infer qwen-vl-chat \\"
echo "  --file test_images/shapes_test.jpg \\"
echo "  --question 'What shapes and colors do you see?'"

# Delete a model
echo -e "\n10. Delete a model:"
echo "model-zoo delete clip-vit-base --yes"

echo -e "\nFor more help: model-zoo --help"