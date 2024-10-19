import os
import hashlib
import uuid
import json
from pathlib import Path
from typing import Dict, List, Union
from v2.basevectordb import BaseVectorDB
from chromadb import PersistentClient
from chromadb.utils.data_loaders import ImageLoader
import numpy as np
from chromadb.api.types import EmbeddingFunction
from tqdm import tqdm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")  # keep it lowercase
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

    def _list_images_in_dir(self, image_dir: str) -> List[str]:
        """Return a list of image paths in a directory with valid extensions."""
        return  [
            str(p.absolute())
            for p in Path(image_dir).glob("**/*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def get_updated_image_paths(
        self, 
        image_paths: List[Union[str, Path]] = None, 
        image_dir: Union[str, Path] = None
    ) -> Dict[str, List[tuple[str, str]]]:
        """Return dict containing new and updated image paths based on file hash and modification time."""
        if image_paths is None and image_dir:
            image_paths = self._list_images_in_dir(image_dir)

        new_images, updated_images = [], []

        for image_path in image_paths:
            file_hash = self.get_file_hash(image_path)
            file_mtime = os.path.getmtime(image_path)

            if image_path not in self.image_cache:
                image_id = str(uuid.uuid4())
                new_images.append((image_id, image_path))
                self._update_cache(image_path, image_id, file_hash, file_mtime)
            elif (
                self.image_cache[image_path]["hash"] != file_hash
                or self.image_cache[image_path]["mtime"] != file_mtime
            ):
                image_id = self.image_cache[image_path]["id"]
                updated_images.append((image_id, image_path))
                self._update_cache(image_path, image_id, file_hash, file_mtime)

        return {"new_image_paths": new_images, "updated_image_paths": updated_images}

    def _update_cache(self, image_path: str, image_id: str, file_hash: str, file_mtime: float):
        """Update the image cache with new hash and modification time."""
        self.image_cache[image_path] = {"id": image_id, "hash": file_hash, "mtime": file_mtime}

    def update_images(self, image_paths: List[Union[str, Path]] = None, batch_size: int = BATCH_SIZE):
        """Update image embeddings in the collection."""
        paths_info = self.get_updated_image_paths(image_paths=image_paths)
        new_images, updated_images = paths_info["new_image_paths"], paths_info["updated_image_paths"]

        if new_images:
            self._add_images([path for _, path in new_images], [id for id, _ in new_images], batch_size=batch_size)
            print(f"Added {len(new_images)} new images.")

        if updated_images:
            self.update_embeddings([path for _, path in updated_images], [id for id, _ in updated_images], batch_size=batch_size)
            print(f"Updated {len(updated_images)} existing images.")

        self.save_cache()

    def update_embeddings(self, image_paths: List[Union[str, Path]], image_ids: List[str], batch_size: int = BATCH_SIZE):
        """Update embeddings for given image paths and IDs."""
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Updating embeddings"):
            batch_paths = image_paths[i: i + batch_size]
            batch_ids = image_ids[i: i + batch_size]
            embeddings = self.embed_images(batch_paths, batch_size=batch_size)
            self.collection.update(ids=batch_ids, uris=batch_paths, embeddings=embeddings)

    def embed_images(self, image_paths: List[str], batch_size: int = BATCH_SIZE):
        """Generate embeddings for a list of image paths."""
        return self.embedding_model.batch_embed_images(image_paths, batch_size=batch_size)

    def add_images(
        self,
        image_paths: List[str] = None,
        image_dir: Union[str, Path] = None,
        image_ids: List[str] = None,
        batch_size: int = BATCH_SIZE
    ):
        """Add images to the collection from paths or a directory."""
        if image_paths is None and image_dir:
            image_paths = self._list_images_in_dir(image_dir)

        self._add_images(image_paths, image_ids=image_ids, batch_size=batch_size)

    def _add_images(self, image_paths: List[str], image_ids: List[str] = None, batch_size: int = BATCH_SIZE):
        """Helper method to handle image addition logic."""
        if image_ids is None:
            image_ids = [str(uuid.uuid4()) for _ in image_paths]

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Adding images"):
            batch_paths = image_paths[i: i + batch_size]
            batch_ids = image_ids[i: i + batch_size]
            embeddings = self.embed_images(batch_paths, batch_size=batch_size)
            self.collection.add(ids=batch_ids, uris=batch_paths, embeddings=embeddings)

            for id, path in zip(batch_ids, batch_paths):
                file_hash = self.get_file_hash(path)
                file_mtime = os.path.getmtime(path)
                self._update_cache(path, id, file_hash, file_mtime)

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
        
    def delete_embeddings(self, image_dir:str|Path=None, image_paths:list[str]=None) -> None:
        """Deletes embeddings from the database."""
        if image_dir:
            if isinstance(image_dir, str):
                image_dir = Path(image_dir)
                assert image_dir.is_dir(), f"The directory '{image_dir}' does not exist."
                
            image_paths = [str(i.absolute()) for i in image_dir.iterdir()]
        
        out = self.collection.query(query_uris=image_paths, include=['uris'], n_results=1)
        
        uris =  np.ravel(out['uris']).tolist()
        ids =   np.ravel(out['ids']).tolist()
        
        delete_ids = [idd for idd, uri in zip(ids, uris) if uri in image_paths]
        self.collection.delete(ids=delete_ids)
        print(f'deleted {len(delete_ids)} embeddins!')