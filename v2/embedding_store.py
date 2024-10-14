import uuid

from pathlib import Path
from basevectordb import BaseVectorDB
from chromadb import PersistentClient

from chromadb.utils.data_loaders import ImageLoader
from chromadb.api.types import EmbeddingFunction

import os
import hashlib
import uuid
from typing import Dict, List
from pathlib import Path
import json
from chromadb import PersistentClient
from chromadb.utils.data_loaders import ImageLoader

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")  # keep it lower


class EmbeddingStore(BaseVectorDB):
    def __init__(
        self,
        save_dir: str | Path,
        embedding_model: EmbeddingFunction,
        collection_name: str = "my_collection",
        cache_file: str = "image_cache.json",
    ):
        self._save_dir = save_dir
        self.image_loader = ImageLoader()
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.cache_file = Path(save_dir) / cache_file
        self.image_cache = self.load_cache()
        self.setup()

    def setup(self):
        self._client = PersistentClient(path=self._save_dir)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_model,
            data_loader=self.image_loader,
        )

    def delete_collection(self):
        self._client.delete_collection(self.collection_name)

    def embed_images(self, image_paths: list[str], batch_size=16):
        embeddings = self.embedding_model.batch_embed_images(
            image_paths, batch_size=batch_size
        )
        return embeddings

    def load_cache(self) -> Dict[str, Dict[str, str | float]]:
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.image_cache, f)

    def get_file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        return file_hash.hexdigest()
    
    
    def get_updated_image_paths(
        self, 
        image_paths: list[str | Path] = None,
        image_dir: str = None,
    ) -> dict:
        
        """ returns dict with keys "new_image_paths" and "updated_image_paths" """
        
        if not image_paths:
            image_paths = [
                str(p)
                for p in Path(image_dir).glob("**/*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        
        new_images = []
        updated_images = []

        for image_path in image_paths:
            file_hash = self.get_file_hash(image_path)
            file_mtime = os.path.getmtime(image_path)

            if image_path not in self.image_cache:
                image_id = str(uuid.uuid4())
                new_images.append((image_id, image_path))
                self.image_cache[image_path] = {
                    "id": image_id,
                    "hash": file_hash,
                    "mtime": file_mtime,
                }
            elif (
                self.image_cache[image_path]["hash"] != file_hash
                or self.image_cache[image_path]["mtime"] != file_mtime
            ):
                image_id = self.image_cache[image_path]["id"]
                updated_images.append((image_id, image_path))
                self.image_cache[image_path]["hash"] = file_hash
                self.image_cache[image_path]["mtime"] = file_mtime
        
        return {
            "new_image_paths": new_images, 
            "updated_image_paths": updated_images,
        }
            
    def update_images(
        self,
        image_paths: list[str | Path] = None,
        batch_size: int = 16,
    ):


        if new_images:
            self.add_images(
                [path for _, path in new_images],
                [id for id, _ in new_images],
                batch_size=batch_size,
            )
            print(f"Added {len(new_images)} new images.")

        if updated_images:
            self.update_embeddings(
                [path for _, path in updated_images], [id for id, _ in updated_images]
            )
            print(f"Updated {len(updated_images)} existing images.")

        self.save_cache()

    def update_embeddings(self, image_paths: List[str | Path], image_ids: list[str]):
        embeddings = self.embed_images(image_paths)
        self.collection.update(ids=image_ids, uris=image_paths, embeddings=embeddings)

    def get_n_similar_images(self, image, k=5):
        return self.collection.query(
            query_uris=image,
            include=["uris", "distances", "metadatas", "embeddings", "documents"],
            n_results=k,
        )

    def add_images(
        self,
        image_paths: list[str] = None,
        image_dir: str | Path = None,
        image_ids: list[str] = None,
        **kwargs,
    ):

        if image_paths is None and os.path.isdir(image_dir):
            image_paths = [
                str(p)
                for p in Path(image_dir).glob("**/*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]

        self._add_images(image_paths, image_ids, **kwargs)

    def _add_images(
        self, image_paths: list[str], image_ids: list[str] = None, **kwargs
    ):
        batch_size = kwargs.pop("batch_size", 16)

        if image_ids is None:
            image_ids = [str(uuid.uuid4()) for _ in image_paths]

        embeddings = self.embed_images(image_paths, batch_size=batch_size)

        self.collection.add(
            ids=image_ids, uris=image_paths, embeddings=embeddings, **kwargs
        )

        for id, path in zip(image_ids, image_paths):
            file_hash = self.get_file_hash(path)
            file_mtime = os.path.getmtime(path)
            self.image_cache[path] = {"id": id, "hash": file_hash, "mtime": file_mtime}

        self.save_cache()
