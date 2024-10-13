from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from PIL import Image
import torch
import numpy as np


class EfficientNetEmbeddingFunction(EmbeddingFunction[Documents]):
    """To use this EmbeddingFunction, you must have the google.generativeai Python package installed and have a PaLM API key."""

    def __init__(self, model_name: str = "google/efficientnet-b0", device="cuda"):
        self.device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self._model_name = model_name
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        # self.model = EfficientNetModel.from_pretrained(model_name).to(self.device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def load_image(self, image: str | Path | Image.Image | np.ndarray):
        """
        Loads an image and processes it using the model's image processor.
        Reusable helper function for single and batch image processing.
        """

        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image)

        image = self.image_processor(images=image, return_tensors="pt").to(self.device)

        return image["pixel_values"]

    def _embed(self, pixel_values):
        """
        Helper function to perform model inference and return the embeddings.
        """
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output

    def embed_image(self, image):
        """
        Embeds a single image by first loading and processing it, then passing it to the model.
        """
        pixel_values = self.load_image(image)
        embeddings = self._embed(pixel_values)
        return embeddings.cpu().numpy().tolist()

    def batch_embed_images(
        self, images: list[str | Path | Image.Image | np.ndarray], batch_size: int = 32
    ):
        """
        Embeds a batch of images, processing them in batches of the specified batch_size.
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            processed_images = [self.load_image(image) for image in batch]
            batched_images = torch.cat(processed_images).to(self.device)
            batch_embeddings = self._embed(batched_images)
            all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())

        return all_embeddings

    def __call__(
        self, images: list[Image.Image | Path | str], batch_size: int = 8
    ) -> Embeddings:
        return self.batch_embed_images(images, batch_size=batch_size)
