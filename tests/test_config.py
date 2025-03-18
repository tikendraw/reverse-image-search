from pathlib import Path

import pytest

from v2.config import Config, load_config, save_config


@pytest.fixture
def test_config_file(tmp_path):
    return tmp_path / "test_config.json"

def test_config_creation():
    """Test Config dataclass creation"""
    config = Config(folders_embedded=[], batch_size=16, num_similar_images=20, n_cols=5)
    assert config.folders_embedded == []
    assert config.batch_size == 16
    assert config.num_similar_images == 20
    assert config.n_cols == 5

def test_save_and_load_config(test_config_file):
    """Test saving and loading configuration"""
    # Test saving config
    save_config(
        path=test_config_file,
        folder_embedded=["/test/path1", "/test/path2"],
        batch_size=32,
        num_similar_images=10,
        n_cols=4
    )
    
    # Test loading config
    config = load_config(test_config_file)
    assert isinstance(config, dict)
    assert len(config["folders_embedded"]) == 2
    assert config["batch_size"] == 32
    assert config["num_similar_images"] == 10
    assert config["n_cols"] == 4

def test_config_with_empty_values(test_config_file):
    """Test configuration with empty or default values"""
    # Test saving with minimal parameters
    save_config(path=test_config_file)
    
    # Check defaults
    config = load_config(test_config_file)
    assert isinstance(config["folders_embedded"], list)
    assert len(config["folders_embedded"]) == 0
    assert config["batch_size"] == 16
    assert config["num_similar_images"] == 10
    assert config["n_cols"] == 5

def test_config_update(test_config_file):
    """Test updating existing configuration"""
    # Create initial config
    save_config(
        path=test_config_file,
        folder_embedded=["/test/path1"],
        batch_size=16
    )
    
    # Update with new values
    config = Config(
        folders_embedded=["/test/path1", "/test/path2"],
        batch_size=32,
        num_similar_images=15,
        n_cols=6
    )
    save_config(path=test_config_file, config=config)
    
    # Verify updates
    loaded_config = load_config(test_config_file)
    assert len(loaded_config["folders_embedded"]) == 2
    assert loaded_config["batch_size"] == 32
    assert loaded_config["num_similar_images"] == 15
    assert loaded_config["n_cols"] == 6
