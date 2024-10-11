import uuid

from pathlib import Path
from basevectordb import BaseVectorDB
from chromadb import PersistentClient

from chromadb.utils.data_loaders import ImageLoader
from chromadb.api.types import EmbeddingFunction


class ChromaDB(BaseVectorDB):
    def __init__(
        self,
        save_dir: str | Path,
        embedding_model: EmbeddingFunction,
        collection_name: str = "my_collection",
    ):
        self._save_dir = save_dir
        self.image_loader = ImageLoader()
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.setup()

    def setup(self):
        self._client = PersistentClient(path=self._save_dir)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_model,
            data_loader=self.image_loader,
        )

    def get_n_similar_images(self, image, k=5):
        return self.collection.query(
            query_uris=image,
            include=["uris", "distances", "metadatas", "embeddings", "documents"],
            n_results=k,
        )

    def add_images(self, image_paths: list[str]):
        self._add_images(image_paths)

    def _add_images(self, image_paths: list[str]):
        ids = [uuid.uuid4().hex for i in range(len(image_paths))]
        embeddings = self.embed_images(image_paths)
        self.collection.add(ids=ids, uris=image_paths, embeddings=embeddings)
        print("done")

    def embed_images(self, image_paths: list[str]):
        embeddings = self.embedding_model.batch_embed_images(image_paths)
        return embeddings