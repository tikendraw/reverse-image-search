import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def ensure_test_image(test_images_dir):
    """Helper function to ensure we have at least one test image"""
    test_dir = test_images_dir / "0"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_image = test_dir / "test.png"
    
    if not test_image.exists():
        # Create a small test image
        img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
        img.save(test_image)
    return test_image

def test_embedding_generation(embedding_model, test_images_dir):
    """Test generating embeddings for single and multiple images"""
    test_image = ensure_test_image(test_images_dir)
    
    # Test single image embedding
    embedding = embedding_model.embed_image(test_image)
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    
    # Test batch embedding
    image_paths = [test_image] * 3
    embeddings, bad_images = embedding_model.batch_embed_images(image_paths)
    assert len(embeddings) == len(image_paths)
    assert len(bad_images) == 0

def test_image_loading(embedding_model, test_images_dir):
    """Test image loading functionality"""
    test_image = ensure_test_image(test_images_dir)

    # Test with path string
    loaded = embedding_model.load_image(str(test_image))
    assert loaded is not None
    assert isinstance(loaded, torch.Tensor)  # Returns tensor, not numpy array

    # Test with Path object
    loaded = embedding_model.load_image(test_image)
    assert loaded is not None
    
    # Test with PIL Image
    img = Image.open(test_image)
    loaded = embedding_model.load_image(img)
    assert loaded is not None
    
    # Test with numpy array
    img_array = np.array(Image.open(test_image))
    loaded = embedding_model.load_image(img_array)
    assert loaded is not None

def test_invalid_inputs(embedding_model):
    """Test handling of invalid inputs"""
    # Test non-existent file
    result = embedding_model.load_image("nonexistent.jpg")
    assert result is None

    # Test with a very small image
    # Now a small image is processed because filtering happens externally.
    invalid_img = Image.fromarray(np.zeros((2, 2), dtype=np.uint8))
    result = embedding_model.load_image(invalid_img)
    assert result is not None  # Small images will be processed here

def test_different_image_types(embedding_model, test_images_dir):
    """Test handling different image types and formats"""
    for digit_dir in ["0", "3", "4", "6", "7", "8", "9"]:
        path = test_images_dir / digit_dir
        if not path.exists():
            continue
            
        for img_path in path.glob("*.png"):
            embedding = embedding_model.embed_image(img_path)
            assert isinstance(embedding, list)
            assert len(embedding) > 0

def test_batch_processing(embedding_model, test_images_dir):
    """Test batch processing behavior"""
    # Test with different batch sizes
    image_paths = []
    for digit_dir in ["0", "3", "4"]:
        path = test_images_dir / digit_dir
        if path.exists():
            image_paths.extend(list(path.glob("*.png")))
    
    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        embeddings, bad_images = embedding_model.batch_embed_images(image_paths, batch_size=batch_size)
        assert len(embeddings) == len(image_paths)
        assert len(bad_images) == 0
