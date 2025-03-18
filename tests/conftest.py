import os
from pathlib import Path

import pytest

from v2.embed_model import EfficientNetEmbeddingFunction
from v2.embedding_store import EmbeddingStore


@pytest.fixture
def test_images_dir():
    return Path(__file__).parent / "test_images"

@pytest.fixture
def test_db_dir(tmp_path):
    """Create a temporary directory for the test database"""
    db_dir = tmp_path / "test_db"
    db_dir.mkdir(exist_ok=True)
    return str(db_dir)

@pytest.fixture
def embedding_model():
    """Create an embedding model instance"""
    return EfficientNetEmbeddingFunction(device="cpu")

@pytest.fixture
def db(test_db_dir, embedding_model):
    """Create a test database instance"""
    return EmbeddingStore(
        save_dir=test_db_dir,
        embedding_model=embedding_model,
        collection_name="test_collection"
    )
