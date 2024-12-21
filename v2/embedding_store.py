import hashlib
import json
import logging
import os
import random
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from tqdm import tqdm

from v2.basevectordb import BaseVectorDB
from v2.utils import list_items_in_dir

BATCH_SIZE = 16
INCLUDE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

list_images = partial(list_items_in_dir, include_extensions=INCLUDE_IMAGE_EXTENSIONS)

def get_unique_id() -> str:
    return f"{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d%H%M%S.%f')}_{random.randint(0, 99999)}"

class EmbeddingStore(BaseVectorDB):
    """
    A class to manage image embeddings using ChromaDB with a caching mechanism.
    """
    _client: PersistentClient = None  # Class-level client to avoid redundant creation

    def __init__(
        self,
        save_dir: str,
        embedding_model: EmbeddingFunction,
        collection_name: str = "my_collection",
        cache_file: str = "image_cache.json",
    ):
        """
        Initializes the EmbeddingStore.

        Args:
            save_dir: Directory to save the ChromaDB database and cache file.
            embedding_model: The embedding function to use.
            collection_name: Name of the ChromaDB collection.
            cache_file: Name of the cache file.
        """
        self._save_dir = save_dir
        self.image_loader = ImageLoader()
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.cache_file = Path(self._save_dir) / cache_file
        self.image_cache = self._load_cache()
        self._setup()

    def _setup(self):
        """Sets up the ChromaDB client and gets or creates the collection."""
        if EmbeddingStore._client is None:
            EmbeddingStore._client = PersistentClient(path=self._save_dir)
        self.collection = EmbeddingStore._client.get_or_create_collection(
            name=self.collection_name, embedding_function=self.embedding_model,data_loader=self.image_loader
        )

    def _load_cache(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """Loads the image cache from disk."""
        try:
            return json.load(open(self.cache_file, "r"))
        except FileNotFoundError:
            return {}

    def _save_cache(self):
        """Saves the image cache to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self.image_cache, f)

    def _get_file_hash(self, file_path: str) -> str:
        """Generates the MD5 hash of a file."""
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5()
            for chunk in iter(lambda: f.read(8192), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def _update_cache(self, image_paths: List[str], image_ids: List[str], file_hashes: List[str], file_mtimes: List[float]):
        """Updates the image cache with new or updated image information."""
        for path, id_, hash_, mtime in zip(image_paths, image_ids, file_hashes, file_mtimes):
            self.image_cache[path] = {"id": id_, "hash": hash_, "mtime": mtime}
        self._save_cache()

    def _delete_from_cache(self, image_paths: List[str]):
        """Deletes entries from the image cache."""
        for path in image_paths:
            try: 
                self.image_cache.pop(path, None)
            except KeyError:
                # logging.error(f"Error deleting {path} from cache.")
                pass
        self._save_cache()

    def _get_updated_images(self, image_paths: List[str]) -> Dict[str, List]:
        """Identifies new and updated images based on cache and file metadata."""
        new_images, updated_images = {"paths": [], "ids": [], "hashes": [], "mtimes": []}, {"paths": [], "ids": [], "hashes": [], "mtimes": []}

        for path in image_paths:
            file_hash = self._get_file_hash(path)
            file_mtime = os.path.getmtime(path)

            if path not in self.image_cache:
                new_images["paths"].append(path)
                new_images["ids"].append(get_unique_id())
                new_images["hashes"].append(file_hash)
                new_images["mtimes"].append(file_mtime)
            elif (self.image_cache[path]["hash"] != file_hash or self.image_cache[path]["mtime"] != file_mtime):
                updated_images["paths"].append(path)
                updated_images["ids"].append(self.image_cache[path]["id"])
                updated_images["hashes"].append(file_hash)
                updated_images["mtimes"].append(file_mtime)

        return {"new": new_images, "updated": updated_images}

    def _process_batches(self, paths: List[str], ids: List[str], hashes: List[str], mtimes: List[float], batch_size: int, operation: str):
        """Processes embeddings in batches for adding or updating."""
        for i in tqdm(range(0, len(paths), batch_size), desc=f"{operation.capitalize()} embeddings"):
            batch_paths = paths[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            embeddings, bad_images = self.embed_images(batch_paths, batch_size=batch_size)
            valid_indices = [idx for idx in range(len(batch_paths)) if idx not in bad_images.values()]

            current_ids = [batch_ids[idx] for idx in valid_indices]
            current_uris = [batch_paths[idx] for idx in valid_indices]
            current_embeddings = [embeddings[idx] for idx in valid_indices]

            try:
                assert len(current_ids) == len(current_uris) == len(current_embeddings), "Lengths of ids, uris, and embeddings should be the same."
                assert len(current_ids) > 0, "No valid images found in the batch."
            except AssertionError as e:
                logging.error(f"Error during {operation} operation: {e}")
                continue

            if operation == "add":
                self.collection.add(
                    ids=current_ids,
                    uris=current_uris,
                    embeddings=current_embeddings,
                )
            elif operation == "update":
                self.collection.update(
                    ids=current_ids,
                    uris=current_uris,
                    embeddings=current_embeddings,
                )

        self._update_cache(paths, ids, hashes, mtimes)

    def embed_images(self, image_paths: List[str], batch_size: int = BATCH_SIZE):
        """Embeds a batch of images using the provided embedding model."""
        return self.embedding_model.batch_embed_images(image_paths, batch_size=batch_size)

    def update_images(self,dir_path: str = None, recursive: bool = False, image_paths: List[str] = None, batch_size: int = BATCH_SIZE) -> Dict[str, int]:
        """
        Updates embeddings for new or modified images in the specified directory or list of paths.

        Args:
            dir_path: Directory to scan for images.
            recursive: Whether to scan directories recursively.
            image_paths: A list of specific image paths to update.
            batch_size: The batch size for processing embeddings.

        Returns:
            A dictionary containing the number of added and updated embeddings.
        """
        if dir_path:
            self._remove_deleted_from_cache_and_db(dir_path, recursive) # Clean up deleted images first

        image_paths = image_paths or list_images(dir_path, recursive=recursive)
        added_count = 0
        updated_count = 0

        if not image_paths:
            return {"added": 0, "updated": 0}

        updates = self._get_updated_images(image_paths)
        for action, images in updates.items():
            if images["paths"]:
                self._process_batches(
                    paths=images["paths"],
                    ids=images["ids"],
                    hashes=images["hashes"],
                    mtimes=images["mtimes"],
                    batch_size=batch_size,
                    operation="add" if action == "new" else "update",
                )
                if action == "new":
                    added_count += len(images["paths"])
                else:
                    updated_count += len(images["paths"])

        self._save_cache()
        return {"added": added_count, "updated": updated_count}

    def get_similar_images(self, image: str, k: int = 5) -> dict:
        """
        Retrieves similar images to the given image.

        Args:
            image: Path to the query image.
            k: Number of similar images to retrieve.

        Returns:
            A dictionary containing the query results.
        """
        try:
            return self.collection.query(
                query_uris=image, include=["uris", "distances"], n_results=k
            )
        except ValueError as e:
            logging.error(f"Error during similarity query: {e}")
            return {}

    def _get_images_from_cache(self, dir_path: str, recursive: bool = False) -> List[str]:
        """Retrieves image paths from the cache for a given directory."""
        if recursive:
            return [path for path in self.image_cache.keys() if Path(dir_path) in Path(path).parents]
        else:
            return [path for path in self.image_cache.keys() if Path(dir_path) == Path(path).parent]

    def delete_collection(self):
        """Delete the current collection from the client."""
        self._client.delete_collection(self.collection_name)
        
    def delete_images(self, dir_path: str, recursive: bool = False) -> int:
        """
        Deletes embeddings for images in the specified directory.

        Args:
            dir_path: Directory to delete embeddings from.
            recursive: Whether to delete embeddings in subdirectories.

        Returns:
            The number of embeddings deleted.
        """
        
        self._remove_deleted_from_cache_and_db(dir_path, recursive) 
        all_images = list_images(dir_path, recursive=recursive)

        
        to_delete_ids = [self.image_cache[path]["id"] for path in all_images]

        if to_delete_ids:
            self.collection.delete(ids=to_delete_ids)
            self._delete_from_cache(all_images)
            return len(to_delete_ids)
        return 0

    def _remove_deleted_from_cache_and_db(self, dir_path: str, recursive: bool = False):
        """Removes embeddings from the cache and database for files that no longer exist."""
        cached_images = self._get_images_from_cache(dir_path, recursive)
        existing_images = list_images(dir_path, recursive=recursive)
        deleted_images = [img for img in cached_images if img not in existing_images]

        if deleted_images:
            to_delete_ids = [self.image_cache[path]["id"] for path in deleted_images if path in self.image_cache]
            if to_delete_ids:
                self.collection.delete(ids=to_delete_ids)
                logging.info(f"Deleted {len(to_delete_ids)} embeddings for removed images.")
            self._delete_from_cache(deleted_images)
            logging.info(f"Removed {len(deleted_images)} deleted images from cache.")