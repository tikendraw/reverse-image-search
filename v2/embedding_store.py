import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from chromadb import PersistentClient
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from tqdm import tqdm

from v2.basevectordb import BaseVectorDB

BATCH_SIZE = 16


class EmbeddingStore(BaseVectorDB):
    def __init__(
        self,
        save_dir: str,
        embedding_model: EmbeddingFunction,
        collection_name: str = "my_collection",
        cache_file: str = "image_cache.json",
    ):
        self._save_dir = save_dir
        self.image_loader = ImageLoader()
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.cache_file = Path(self._save_dir) / cache_file
        self.image_cache = self.load_cache()
        self.setup()

    def setup(self):
        self._client = self._initialize_client()
        self.collection = self._get_or_create_collection()

    def _initialize_client(self) -> PersistentClient:
        return PersistentClient(path=self._save_dir)

    def _get_or_create_collection(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_model,
            data_loader=self.image_loader,
        )

    def load_cache(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """Load image cache from the cache file if it exists, otherwise return an empty dict."""
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def save_cache(self):
        """Save the current image cache to the cache file."""
        with open(self.cache_file, "w") as f:
            json.dump(self.image_cache, f)

    def get_file_hash(self, file_path: str) -> str:
        """Generate an MD5 hash for a given file."""
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5()
            for chunk in iter(lambda: f.read(8192), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def get_updated_image_paths(
        self,
        image_paths: List[Union[str, Path]] = None,
    ) -> Dict[str, List[tuple[str, str]]]:
        """Return dict containing new and updated image paths based on file hash and modification time."""

        new_images = {
            "image_paths": [],
            "image_ids": [],
            "file_hashes": [],
            "file_mtimes": [],
        }

        updated_images = {
            "image_paths": [],
            "image_ids": [],
            "file_hashes": [],
            "file_mtimes": [],
        }

        for image_path in image_paths:
            file_hash = self.get_file_hash(image_path)
            file_mtime = os.path.getmtime(image_path)

            if image_path not in self.image_cache:
                image_id = str(uuid.uuid4())
                new_images["image_paths"].append((image_path))
                new_images["image_ids"].append((image_id))
                new_images["file_hashes"].append((file_hash))
                new_images["file_mtimes"].append((file_mtime))

            elif (self.image_cache[image_path]["hash"] != file_hash) or (
                self.image_cache[image_path]["mtime"] != file_mtime
            ):
                image_id = self.image_cache[image_path]["id"]
                updated_images["image_paths"].append((image_path))
                updated_images["image_ids"].append((image_id))
                updated_images["file_hashes"].append((file_hash))
                updated_images["file_mtimes"].append((file_mtime))

        return {"new_image_paths": new_images, "updated_image_paths": updated_images}

    def _update_cache(
        self, image_paths: str, image_ids: str, file_hashes: str, file_mtimes: float
    ):
        """Update the image cache with new hash and modification time."""
        for image_path, image_id, file_hash, file_mtime in zip(
            image_paths, image_ids, file_hashes, file_mtimes
        ):
            self.image_cache[image_path] = {
                "id": image_id,
                "hash": file_hash,
                "mtime": file_mtime,
            }

    def _delete_cache(self, image_paths: List[str]):
        """Delete images from the image cache."""
        for image_path in image_paths:
            if image_path in self.image_cache:
                del self.image_cache[image_path]

    def update_images(
        self, image_paths: List[Union[str, Path]] = None, batch_size: int = BATCH_SIZE
    ):
        """Update image embeddings in the collection."""
        paths_info = self.get_updated_image_paths(image_paths=image_paths)
        new_images, updated_images = (
            paths_info["new_image_paths"],
            paths_info["updated_image_paths"],
        )

        if new_images:
            image_ids = new_images["image_ids"]
            image_paths = new_images["image_paths"]
            file_hashes = new_images["file_hashes"]
            file_mtimes = new_images["file_mtimes"]
            self._add_images(
                image_paths,
                image_ids=image_ids,
                batch_size=batch_size,
                image_hashes=file_hashes,
                image_mtime=file_mtimes,
            )
            print(f"Added {len(new_images)} new images.")

        if updated_images:
            image_ids = updated_images["image_ids"]
            image_paths = updated_images["image_paths"]
            file_hashes = updated_images["file_hashes"]
            file_mtimes = updated_images["file_mtimes"]
            self._update_embeddings(
                image_paths,
                image_ids,
                batch_size=batch_size,
                image_hashes=file_hashes,
                image_mtime=file_mtimes,
            )
            print(f"Updated {len(updated_images)} existing images.")

        self.save_cache()

    def _update_embeddings(
        self,
        image_paths: List[Union[str, Path]],
        image_ids: List[str],
        image_hashes: List[str],
        image_mtime: List[float],
        batch_size: int = BATCH_SIZE,
    ):
        """Update embeddings for given image paths and IDs."""
        for i in tqdm(
            range(0, len(image_paths), batch_size), desc="Updating embeddings"
        ):
            batch_paths = image_paths[i : i + batch_size]
            batch_ids = image_ids[i : i + batch_size]

            embeddings = self.embed_images(batch_paths, batch_size=batch_size)
            self.collection.update(
                ids=batch_ids, uris=batch_paths, embeddings=embeddings
            )
        self._update_cache(
            image_paths=image_paths,
            image_ids=image_ids,
            file_hashes=image_hashes,
            file_mtimes=image_mtime,
        )

    def embed_images(self, image_paths: List[str], batch_size: int = BATCH_SIZE):
        """Generate embeddings for a list of image paths."""
        return self.embedding_model.batch_embed_images(
            image_paths, batch_size=batch_size
        )

    def add_images(
        self,
        image_paths: List[str] = None,
        image_ids: List[str] = None,
        image_hashes: List[str] = None,
        image_mtime: List[float] = None,
        batch_size: int = BATCH_SIZE,
    ):
        """Add images to the collection from paths or a directory."""

        self._add_images(
            image_paths,
            image_ids=image_ids,
            batch_size=batch_size,
            image_hashes=image_hashes,
            image_mtime=image_mtime,
        )

    def _add_images(
        self,
        image_paths: List[str],
        image_ids: List[str] = None,
        image_hashes: List[str] = None,
        image_mtime: List[float] = None,
        batch_size: int = BATCH_SIZE,
    ):
        """Helper method to handle image addition logic."""
        if image_ids is None:
            image_ids = [str(uuid.uuid4()) for _ in image_paths]

        if image_hashes is None:
            image_hashes = [self.get_file_hash(path) for path in image_paths]

        if image_mtime is None:
            image_mtime = [os.path.getmtime(path) for path in image_paths]

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Adding images"):
            batch_paths = image_paths[i : i + batch_size]
            batch_ids = image_ids[i : i + batch_size]
            embeddings = self.embed_images(batch_paths, batch_size=batch_size)
            self.collection.add(ids=batch_ids, uris=batch_paths, embeddings=embeddings)

        self._update_cache(
            image_paths=image_paths,
            image_ids=image_ids,
            file_hashes=image_hashes,
            file_mtimes=image_mtime,
        )

        self.save_cache()

    def get_n_similar_images(self, image: str, k: int = 5):
        """Return the top `k` similar images for a given image."""
        return self.collection.query(
            query_uris=image,
            include=["uris", "distances", "metadatas", "embeddings", "documents"],
            n_results=k,
        )

    def delete_collection(self):
        """Delete the current collection from the client."""
        self._client.delete_collection(self.collection_name)

    def delete_embeddings(self, image_paths: list[str] = None) -> None:
        """Deletes embeddings from the database."""
        image_paths = list(set(image_paths))
        out = self.collection.query(
            query_uris=image_paths, include=["uris"], n_results=1
        )

        uris = np.ravel(out["uris"]).tolist()
        ids = np.ravel(out["ids"]).tolist()

        delete_ids = [idd for idd, uri in zip(ids, uris) if uri in image_paths]
        self.collection.delete(ids=delete_ids)
        self._delete_cache(image_paths=image_paths)
        print(f"deleted {len(delete_ids)} embeddings!")
