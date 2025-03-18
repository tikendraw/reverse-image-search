from pathlib import Path

import pytest
from PIL import Image

from v2.embedding_store import list_images


def test_list_images(test_images_dir):
    """Test listing images from directories"""
    # Ensure test directory exists and has images
    test_dir = test_images_dir / "0"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_image = test_dir / "test.png"
    
    # Create a test image if it doesn't exist
    if not test_image.exists():
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
        img.save(test_image)
    
    # Test single directory
    images = list_images(test_dir)
    assert len(images) > 0
    assert all(Path(img).suffix.lower() in [".png", ".jpg", ".jpeg"] for img in images)

    # Test recursive
    images = list_images(test_images_dir, recursive=True)
    assert len(images) > 0
    assert all(Path(img).suffix.lower() in [".png", ".jpg", ".jpeg"] for img in images)

def test_embedding_store_crud(db, test_images_dir):
    """Test Create, Read, Update, Delete operations of EmbeddingStore"""
    # Ensure test directory exists and has an image
    test_dir = test_images_dir / "0"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_image = test_dir / "test.png"
    
    if not test_image.exists():
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
        img.save(test_image)
    
    # Test adding images
    result = db.update_images(dir_path=str(test_dir))
    assert result["added"] > 0
    assert result["updated"] == 0

    # Test querying similar images
    results = db.get_similar_images(str(test_image), k=5)
    assert len(results["uris"]) == 1
    assert len(results["uris"][0]) > 0  # At least one similar image

    # Test updating images
    result = db.update_images(dir_path=str(test_dir))
    assert result["added"] == 0  # No new images added
    
    # Test deleting images
    deleted_count = db.delete_images(str(test_dir))
    assert deleted_count > 0
    
    # Verify deletion
    results = db.get_similar_images(str(test_image), k=5)
    assert len(results.get("uris", [[]])[0]) == 0

def test_batch_processing(db, test_images_dir):
    """Test batch processing of images"""
    # Create test images
    test_dir = test_images_dir / "0"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_images = []
    
    for i in range(5):
        test_image = test_dir / f"test_{i}.png"
        if not test_image.exists():
            import numpy as np
            from PIL import Image
            img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
            img.save(test_image)
        test_images.append(test_image)
    
    # Test batch update
    result = db.update_images(image_paths=[str(p) for p in test_images])
    assert result["added"] == len(test_images)
    
    # Test with small batch size
    db.delete_images(str(test_dir))
    result = db.update_images(image_paths=[str(p) for p in test_images], batch_size=2)
    assert result["added"] == len(test_images)

def test_invalid_images(db, test_images_dir):
    """Test handling of invalid images"""
    # Create a non-image file
    invalid_file = test_images_dir / "not_an_image.txt"
    invalid_file.touch()
    
    try:
        result = db.update_images(image_paths=[str(invalid_file)])
        assert result["added"] == 0  # Should not add invalid images
    finally:
        if invalid_file.exists():
            invalid_file.unlink()
