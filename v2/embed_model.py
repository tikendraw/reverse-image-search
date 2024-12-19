from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from PIL import Image
import torch
import logging

class EfficientNetEmbeddingFunction(EmbeddingFunction[Documents]):
    """To use this EmbeddingFunction, you must have the transformers Python package installed."""

    def __init__(self, model_name: str = "google/efficientnet-b0", device="cuda"):
        self.device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self._model_name = model_name
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def load_image(self, image: str | Path ):
        """
        Loads an image and processes it using the model's image processor.
        Reusable helper function for single and batch image processing.
        """
        try:
            if isinstance(image, str) or isinstance(image, Path):
                image = Image.open(image)
            else:
                logging.error('Image type not supported(knowingly), use str or pathlib.Path ')
                return None

            if image.mode != "RGB":
                image = image.convert("RGB")

            image = self.image_processor(images=image, return_tensors="pt").to(self.device)

            return image["pixel_values"]
        except OSError as e:
            logging.error(f"Error processing image {image}: {e}")
            return None

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
        if pixel_values is not None:
            embeddings = self._embed(pixel_values)
            return embeddings.cpu().numpy().tolist()
        else:
            logging.error(f"Failed to load image: {image}")
        return None

    def batch_embed_images(
        self, images: list[str | Path], batch_size: int = 32
    ):
        """
        Embeds a batch of images, processing them in batches of the specified batch_size.
        """
        all_embeddings = []
        bad_images = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            processed_images = [self.load_image(image) for image in batch]
            bad_images = [image for image, pixel_values in zip(batch, processed_images) if pixel_values is None]
            processed_images = [pixel_values for pixel_values in processed_images if pixel_values is not None]
            
            if not processed_images:
                continue
            
            try:
              batched_images = torch.cat(processed_images).to(self.device)
              batch_embeddings = self._embed(batched_images)
              all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            except Exception as e:
              logging.error(f"Error during batch embedding, skipping batch: {e}")

        if bad_images:
          logging.warning(f"The following images failed to process : {', '.join(bad_images)}")
        return all_embeddings

    def __call__(
        self, images: list[Path | str], batch_size: int = 8
    ) -> Embeddings:
        return self.batch_embed_images(images, batch_size=batch_size)