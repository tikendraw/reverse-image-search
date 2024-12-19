import logging
from pathlib import Path

import torch
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from PIL import Image
import torch
import numpy as np


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
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image,Image.Image):
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
        if not (pixel_values is None):
            embeddings = self._embed(pixel_values)
            return embeddings.cpu().numpy().tolist()
        else:
            logging.error("Failed to load image.")

    def batch_embed_images(
        self, images: list[str | Path], batch_size: int = 32
    ):
        """
        Embeds a batch of images, processing them in batches of the specified batch_size.
        """
        all_embeddings = []
        bad_images={}
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            processed_images = [self.load_image(image) for image in batch]
            for num, (iurl, ipro) in enumerate(zip(batch, processed_images)):
                if ipro is None:
                    bad_images[iurl]=num
                    logging.error(f"Failed processing the image: {iurl}")
            
            try: 
                processed_images=[i for i in processed_images if i is not None]
                batched_images = torch.cat(processed_images).to(self.device)
                batch_embeddings = self._embed(batched_images)
                all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            except Exception as e:
                logging.error(f"Error while creating Embeddings for batch of images: {', '.join(batch)}", exc_info=True)
                
        return all_embeddings, bad_images
    
    def __call__(
        self, images: list[str | Path | Image.Image | np.ndarray], batch_size: int = 32
    ):
        """
        Embeds a batch of images, processing them in batches of the specified batch_size.
        """
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            processed_images = [self.load_image(image) for image in batch]
            for num, (iurl, ipro) in enumerate(zip(batch, processed_images)):
                if ipro is None:
                    logging.error(f"Failed processing the image: {iurl}")
            
            processed_images=[i for i in processed_images if i is not None]
            batched_images = torch.cat(processed_images).to(self.device)
            batch_embeddings = self._embed(batched_images)
            all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())

        if bad_images:
          logging.warning(f"The following images failed to process : {', '.join(bad_images)}")
        return all_embeddings

    def __call__(
        self, images: list[Image.Image | Path | str], batch_size: int = 8
    ) -> Embeddings:
        return self.batch_embed_images(images, batch_size=batch_size)
